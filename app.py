import html as _html
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Literal

import pandas as pd
import requests
import streamlit as st
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from text_processing import (
    RE_PAGE_NUM as _RE_PAGE_NUM,
    RE_HSPACE as _RE_HSPACE,
    RE_NEWLINES as _RE_NEWLINES,
    RE_SENTENCES as _RE_SENTENCES,
    clean_text,
    extract_pdf_text,
    extract_docx_text,
    extract_plain_text,
    _hard_wrap,
    split_into_chunks as _split_into_chunks_impl,
)

# Optional: load .env file when running locally (no-op inside Docker)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from checker import (
    PairCheckResult,
    check_pair,
    checker_markdown_block,
    get_checker_settings,
    make_check_key,
    should_retry_translation,
)
from infrastructure.ollama_client import (
    chat as _ollama_chat,
    load_model as _ollama_load,
    session as _http_session,
    stream_chat as _ollama_stream,
)
from infrastructure.translation_history import (
    MongoTranslationHistoryStore,
    build_history_document,
    build_session_restore_state,
)
from tts_component import render_tts_button

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------
# Configuration
# -----------------------------

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
_AVAILABLE_MODELS_RAW = os.getenv(
    "AVAILABLE_OLLAMA_MODELS",
    "qwen2.5:7b,qwen2.5:14b",
)
AVAILABLE_OLLAMA_MODELS: List[str] = [
    m.strip() for m in _AVAILABLE_MODELS_RAW.split(",") if m.strip()
]
if not AVAILABLE_OLLAMA_MODELS:
    AVAILABLE_OLLAMA_MODELS = [OLLAMA_MODEL]
_keep_alive_raw = os.getenv("OLLAMA_KEEP_ALIVE", "-1")
try:
    OLLAMA_KEEP_ALIVE: int = int(_keep_alive_raw)
except ValueError:
    OLLAMA_KEEP_ALIVE = -1  # malformed value (e.g. "-1m") corrected to -1
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
OLLAMA_TOP_P = float(os.getenv("OLLAMA_TOP_P", "0.9"))
OLLAMA_REQUEST_TIMEOUT = int(os.getenv("OLLAMA_REQUEST_TIMEOUT", "240"))
DEFAULT_MAX_CHARS = int(os.getenv("MAX_CHARS_PER_CHUNK", "2200"))
TRANSLATION_CACHE_MAX_ENTRIES = int(os.getenv("TRANSLATION_CACHE_MAX_ENTRIES", "50"))

# qwen2.5:3b can drift or truncate on large structured JSON responses.
# Guardrails keep requests within a safer operating envelope.
QWEN25_3B_SAFE_MAX_CHARS = 1200
QWEN25_3B_SAFE_NUM_PREDICT_CAP = 3200

# Persistent HTTP session lives in infrastructure.ollama_client (imported above).

# Pre-compiled regex patterns — aliases from text_processing.

Difficulty = Literal["A1", "A2", "B1", "B2", "C1", "C2"]


# -----------------------------
# Bounded LRU session cache
# -----------------------------

from collections import OrderedDict


class _BoundedCache:
    """LRU-style dict capped at *max_entries*.

    Oldest entry is evicted when the cap is reached.  Accessing an entry
    moves it to most-recently-used position.  All data stays in
    st.session_state — no global/cross-user state.
    """

    def __init__(self, max_entries: int) -> None:
        self._max = max(1, max_entries)
        self._data: OrderedDict = OrderedDict()

    def get(self, key: object) -> object:
        """Return value or None; promotes key to MRU position."""
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: object, value: object) -> None:
        """Insert/update key; evicts LRU entry when over capacity."""
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        while len(self._data) > self._max:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


def _inline_schema(schema: dict) -> dict:
    """
    Resolve all $ref/$defs references in a JSON Schema so the result is fully
    inlined.  Ollama's structured-output format field does not support $ref.
    """
    defs = schema.get("$defs", {})

    def resolve(node: object) -> object:
        if isinstance(node, dict):
            if "$ref" in node:
                ref_name = node["$ref"].split("/")[-1]
                return resolve(defs[ref_name])
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return resolve(schema)  # type: ignore[return-value]


# -----------------------------
# Structured output schemas
# -----------------------------

class VocabularyItem(BaseModel):
    spanish: str = Field(description="Spanish word or phrase")
    english: str = Field(description="English meaning")
    note: str = ""


_VALID_DIFFICULTIES = ("A1", "A2", "B1", "B2", "C1", "C2")


class ReadingPair(BaseModel):
    english: str
    spanish: str
    literal_spanish: str = ""
    vocabulary: List[VocabularyItem] = Field(default_factory=list)
    grammar_notes: List[str] = Field(default_factory=list)
    comprehension_question_spanish: str = ""
    difficulty: Difficulty = "B1"
    # Runtime-only flags used by the UI to mark checker-applied fixes.
    corrected_by_checker: bool = Field(default=False, exclude=True)
    correction_note: str = Field(default="", exclude=True)
    correction_reason: str = Field(default="", exclude=True)
    original_spanish_before_correction: str = Field(default="", exclude=True)

    @model_validator(mode="before")
    @classmethod
    def normalize_common_key_typos(cls, v: object) -> object:
        """Normalize frequent model key typos before field validation."""
        if isinstance(v, dict) and "spanish" not in v and isinstance(v.get("spanished"), str):
            fixed = dict(v)
            fixed["spanish"] = fixed.pop("spanished")
            logger.warning("Normalized malformed pair key: 'spanished' -> 'spanish'.")
            return fixed
        return v

    @field_validator("difficulty", mode="before")
    @classmethod
    def coerce_difficulty(cls, v: object) -> str:
        """Normalise difficulty values emitted by smaller models.

        qwen2.5:3b sometimes returns lowercase ("b1"), plain level numbers
        ("3"), or prose labels ("Intermediate").  Try an uppercase exact match
        first; fall back to a substring scan; default to "B1".
        """
        if isinstance(v, str):
            upper = v.strip().upper()
            if upper in _VALID_DIFFICULTIES:
                return upper
            # Substring scan: "b1 (intermediate)" → "B1"
            for level in _VALID_DIFFICULTIES:
                if level in upper:
                    return level
        logger.warning("Unrecognised difficulty value %r — defaulting to 'B1'.", v)
        return "B1"


class TranslationResponse(BaseModel):
    title: str = ""
    summary_english: str = Field(default="", description="A brief summary of the text written in ENGLISH only. Never use Spanish here.")
    summary_spanish: str = Field(default="", description="Un breve resumen del texto escrito únicamente en ESPAÑOL. Never use English here.")
    pairs: List[ReadingPair]
    # Runtime-only: populated by translate_chunk when pairs are dropped or fields
    # are empty. Not part of the LLM schema; excluded from all serialisation.
    parse_warnings: List[str] = Field(default_factory=list, exclude=True)


# Module-level schema constant — computed once at startup.
# _TRANSLATION_SCHEMA is kept for future structured-output use (schema-constrained
# sampling). The prompt currently uses _TRANSLATION_EXAMPLE_STR for brevity.
_TRANSLATION_SCHEMA: dict = _inline_schema(TranslationResponse.model_json_schema())

# Compact example used in the prompt instead of the full JSON Schema.
# A worked-example is clearer and far shorter (~80 tokens vs ~500 for the schema),
# leaving significantly more room in num_ctx for output.
_TRANSLATION_EXAMPLE_STR: str = json.dumps(
    {
        "title": "Lesson title",
        "summary_english": "One or two sentence English summary of the passage.",
        "summary_spanish": "Resumen de una o dos oraciones en español.",
        "pairs": [
            {
                "english": "He ran quickly to the store before it closed.",
                "spanish": "Corrió deprisa a la tienda antes de que cerrara.",
                "literal_spanish": "Él corrió rápidamente a la tienda antes que ella cerrara.",
                "vocabulary": [
                    {"spanish": "deprisa", "english": "quickly, fast", "note": "common in Spain and Latin America"}
                ],
                "grammar_notes": ["'antes de que' requires the subjunctive; 'cerrara' is imperfect subjunctive of cerrar."],
                "comprehension_question_spanish": "¿Por qué corrió él a la tienda?",
                "difficulty": "B1",
            }
        ],
    },
    ensure_ascii=False,
)

# Single-pair example used in the retranslation prompt.
_RETRANSLATE_PAIR_EXAMPLE_STR: str = json.dumps(
    {
        "english": "He ran quickly to the store before it closed.",
        "spanish": "Corrió deprisa a la tienda antes de que cerrara.",
        "literal_spanish": "Él corrió rápidamente a la tienda antes que ella cerrara.",
        "vocabulary": [
            {"spanish": "deprisa", "english": "quickly, fast", "note": "common in Spain and Latin America"}
        ],
        "grammar_notes": ["'antes de que' requires the subjunctive; 'cerrara' is imperfect subjunctive of cerrar."],
        "comprehension_question_spanish": "¿Por qué corrió él a la tienda?",
        "difficulty": "B1",
    },
    ensure_ascii=False,
)

# Spanish signal tokens used by _fix_english_summary heuristic.
_STRONG_SPANISH: frozenset = frozenset([
    "el", "la", "los", "las", "en", "de", "que", "es", "un", "una",
    "del", "se", "con", "por", "para", "como", "más", "pero", "su",
])


# -----------------------------
# Page setup
# -----------------------------

st.set_page_config(
    page_title="Spanish Parallel Reader",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 7rem;
    }

    .small-muted {
        font-size: 0.88rem;
        color: #6b7280;
    }

    /* Guarantee full passage text is always visible */
    .passage-text {
        line-height: 1.75;
        overflow: visible;
        white-space: normal;
        overflow-wrap: break-word;
        margin: 0 0 0.25rem 0;
    }

    /* Fixed progress bar shown during generation/checking/correction. */
    .sticky-progress-wrap {
        position: fixed;
        left: 0;
        right: 0;
        bottom: 0;
        z-index: 999999;
        padding: 0.6rem 1rem 0.75rem 1rem;
        background: linear-gradient(
            to top,
            rgba(255, 255, 255, 0.98),
            rgba(255, 255, 255, 0.88)
        );
        border-top: 1px solid rgba(148, 163, 184, 0.35);
        box-shadow: 0 -8px 24px rgba(15, 23, 42, 0.08);
        pointer-events: none;
    }

    .sticky-progress-card {
        max-width: min(1100px, calc(100vw - 2rem));
        margin: 0 auto;
        padding: 0.5rem 0.75rem;
        border-radius: 0.85rem;
        background: rgba(248, 250, 252, 0.96);
        border: 1px solid rgba(203, 213, 225, 0.8);
        pointer-events: auto;
    }

    .sticky-progress-text {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        font-size: 0.88rem;
        color: #334155;
        margin-bottom: 0.35rem;
    }

    .sticky-progress-track {
        height: 0.55rem;
        border-radius: 999px;
        background: #e2e8f0;
        overflow: hidden;
    }

    .sticky-progress-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #2563eb, #06b6d4);
        transition: width 180ms ease;
    }

    /* Floating jump link for returning to the Study tabs. */
    .floating-study-link {
        position: fixed;
        right: 1rem;
        bottom: 5.6rem;
        z-index: 999998;
        padding: 0.55rem 0.8rem;
        border-radius: 999px;
        background: #0f172a;
        color: #ffffff !important;
        font-size: 0.86rem;
        font-weight: 700;
        text-decoration: none !important;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.25);
    }

    .floating-study-link:hover {
        background: #1e293b;
        color: #ffffff !important;
        text-decoration: none !important;
    }

    #study-tabs {
        scroll-margin-top: 1rem;
    }

    @media (max-width: 640px) {
        .sticky-progress-wrap {
            padding: 0.45rem 0.5rem 0.55rem 0.5rem;
        }

        .sticky-progress-card {
            max-width: calc(100vw - 1rem);
        }

        .sticky-progress-text {
            font-size: 0.78rem;
        }

        .floating-study-link {
            right: 0.7rem;
            bottom: 5.2rem;
            font-size: 0.78rem;
            padding: 0.45rem 0.65rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("📚 Spanish Parallel Reader")
st.caption("Local-first Spanish study with Streamlit + Ollama")


# -----------------------------
# Session state
# -----------------------------

if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""

if "results" not in st.session_state:
    st.session_state.results = []

if "checker_results" not in st.session_state:
    # Maps cache_key -> PairCheckResult. Populated after each translation event.
    st.session_state.checker_results = {}

if "_cached_markdown_key" not in st.session_state:
    st.session_state._cached_markdown_key = None
    st.session_state._cached_markdown = None

if "_translation_cache" not in st.session_state:
    # LRU-bounded cache: (chunk, model, level, style, region, fidelity, flags, temp) → TranslationResponse.
    # Capped at TRANSLATION_CACHE_MAX_ENTRIES to bound session memory.
    st.session_state._translation_cache = _BoundedCache(TRANSLATION_CACHE_MAX_ENTRIES)

if "_enrich_idx" not in st.session_state:
    # Index into st.session_state.results of a chunk pending on-demand enrichment.
    # Set by the "⚡ Add enrichments" button; cleared after enrichment completes.
    st.session_state._enrich_idx = None

if "_spanish_focus_idx" not in st.session_state:
    st.session_state._spanish_focus_idx = 0

_env_include_enrichments_default = os.getenv(
    "TRANSLATION_INCLUDE_ENRICHMENTS", "true"
).strip().lower() not in ("false", "0", "no", "off")
_env_checker_enabled_default = os.getenv(
    "CHECKER_ENABLED", "true"
).strip().lower() not in ("false", "0", "no", "off")
_env_checker_mode_default = os.getenv("CHECKER_MODE", "smart").strip().lower() or "smart"
_env_checker_llm_default = os.getenv(
    "CHECKER_LLM_ENABLED", "true"
).strip().lower() not in ("false", "0", "no", "off")
_env_checker_require_pass_default = os.getenv(
    "CHECKER_REQUIRE_PASS", "false"
).strip().lower() not in ("false", "0", "no", "off")
_env_checker_detailed_default = os.getenv(
    "CHECKER_DETAILED_DIAGNOSTICS", "false"
).strip().lower() not in ("false", "0", "no", "off")
_default_model_choice = (
    OLLAMA_MODEL if OLLAMA_MODEL in AVAILABLE_OLLAMA_MODELS else AVAILABLE_OLLAMA_MODELS[0]
)

_session_defaults = {
    "settings_mode": "Simple",
    "selected_model": _default_model_choice,
    "input_mode": "Paste text",
    "level": "B1 intermediate",
    "region": "Neutral",
    "style": "Learner-friendly Spanish",
    "fidelity": "Balanced",
    "max_chars": DEFAULT_MAX_CHARS,
    "chunks_to_process": 2,
    "temperature": OLLAMA_TEMPERATURE,
    "show_audio_controls": True,
    "skip_enrichments": not _env_include_enrichments_default,
    "include_literal": False,
    "include_vocab": True,
    "include_grammar": True,
    "checker_enabled_ui": _env_checker_enabled_default,
    "checker_mode_ui": _env_checker_mode_default,
    "checker_require_pass_ui": _env_checker_require_pass_default,
    "checker_model_ui": os.getenv("CHECKER_MODEL", "").strip(),
    "checker_llm_ui": _env_checker_llm_default,
    "checker_detailed_ui": _env_checker_detailed_default,
    "start_chunk": 1,
    "source_text_input": "",
    "history_selected_id": "",
    "history_rename_value": "",
    "_history_selected_id_prev": "",
    "_history_current_id": "",
    "_history_current_label": "",
    "_history_current_session_hash": "",
    "_history_current_source_hash": "",
    "_history_loaded_notice": "",
    "_source_uploaded_filename": "",
}

for _state_key, _state_value in _session_defaults.items():
    if _state_key not in st.session_state:
        st.session_state[_state_key] = _state_value


# -----------------------------
# Text helpers
# -----------------------------

def set_source_text(text: str) -> None:
    cleaned = clean_text(text) if text else ""

    if cleaned != st.session_state.raw_text:
        st.session_state.raw_text = cleaned
        st.session_state.results = []
        st.session_state.checker_results = {}
        st.session_state._translation_cache.clear()
        _invalidate_export_caches()
        _clear_loaded_history_tracking()


def _invalidate_export_caches() -> None:
    st.session_state._cached_markdown_key = None
    st.session_state._cached_markdown = None
    st.session_state.pop("_cached_extra_exports_key", None)
    st.session_state.pop("_cached_bilingual_csv", None)
    st.session_state.pop("_cached_spanish_text", None)
    st.session_state.pop("_cached_anki_csv", None)


def _clear_loaded_history_tracking() -> None:
    st.session_state._history_current_id = ""
    st.session_state._history_current_label = ""
    st.session_state._history_current_session_hash = ""
    st.session_state._history_current_source_hash = ""


def _reset_study_controls() -> None:
    st.session_state.reader_search = ""
    st.session_state.reader_corrected_only = False
    st.session_state.reader_difficulty_filter = []
    st.session_state.pop("spanish_first_mode", None)
    st.session_state._spanish_focus_idx = 0
    st.session_state._enrich_idx = None


def _clear_current_lesson(*, clear_source: bool) -> None:
    if clear_source:
        st.session_state.raw_text = ""
        st.session_state.source_text_input = ""
        st.session_state._source_uploaded_filename = ""
    st.session_state.results = []
    st.session_state.checker_results = {}
    st.session_state._translation_cache.clear()
    _invalidate_export_caches()
    _reset_study_controls()
    st.session_state.pop("_last_upload_key", None)
    _clear_loaded_history_tracking()


@st.cache_resource
def get_history_store() -> MongoTranslationHistoryStore:
    return MongoTranslationHistoryStore()


def _history_option_label(item: dict) -> str:
    updated_at = item.get("updated_at")
    if hasattr(updated_at, "strftime"):
        updated_label = updated_at.strftime("%Y-%m-%d %H:%M")
    else:
        updated_label = str(updated_at or "")
    details = [
        item.get("label") or item.get("title") or "Untitled",
        f"{item.get('pair_count', 0)} passages",
        item.get("model") or "model unknown",
        updated_label or "no timestamp",
    ]
    preview = item.get("source_preview") or ""
    return " | ".join(part for part in details if part) + (f" | {preview}" if preview else "")


def _apply_history_restore(restore_state: dict[str, object]) -> None:
    _clear_current_lesson(clear_source=False)
    st.session_state.raw_text = str(restore_state.get("raw_text", ""))
    st.session_state.results = list(restore_state.get("results", []))
    st.session_state.checker_results = dict(restore_state.get("checker_results", {}))
    st.session_state.source_text_input = st.session_state.raw_text
    st.session_state.input_mode = str(restore_state.get("input_mode", "Paste text")) or "Paste text"
    st.session_state._source_uploaded_filename = str(
        restore_state.get("history_uploaded_filename", "")
    )
    _reset_study_controls()

    translation_settings = dict(restore_state.get("translation_settings", {}))
    checker_settings_state = dict(restore_state.get("checker_settings", {}))

    restored_model = str(translation_settings.get("selected_model", st.session_state.selected_model))
    st.session_state.selected_model = restored_model or st.session_state.selected_model
    st.session_state.level = str(translation_settings.get("level", st.session_state.level))
    st.session_state.region = str(translation_settings.get("region", st.session_state.region))
    st.session_state.style = str(translation_settings.get("style", st.session_state.style))
    st.session_state.fidelity = str(translation_settings.get("fidelity", st.session_state.fidelity))
    st.session_state.temperature = float(
        translation_settings.get("temperature", st.session_state.temperature)
    )
    st.session_state.max_chars = int(translation_settings.get("max_chars", st.session_state.max_chars))
    st.session_state.start_chunk = max(
        1,
        int(translation_settings.get("start_chunk", st.session_state.start_chunk)),
    )
    st.session_state.chunks_to_process = max(
        1,
        int(translation_settings.get("chunks_to_process", st.session_state.chunks_to_process)),
    )
    st.session_state.include_literal = bool(
        translation_settings.get("include_literal", st.session_state.include_literal)
    )
    st.session_state.include_vocab = bool(
        translation_settings.get("include_vocab", st.session_state.include_vocab)
    )
    st.session_state.include_grammar = bool(
        translation_settings.get("include_grammar", st.session_state.include_grammar)
    )
    st.session_state.skip_enrichments = bool(
        translation_settings.get(
            "skip_enrichments",
            not (
                st.session_state.include_literal
                or st.session_state.include_vocab
                or st.session_state.include_grammar
            ),
        )
    )
    st.session_state.show_audio_controls = bool(
        translation_settings.get(
            "show_audio_controls",
            st.session_state.show_audio_controls,
        )
    )

    st.session_state.checker_enabled_ui = bool(
        checker_settings_state.get("enabled", st.session_state.checker_enabled_ui)
    )
    st.session_state.checker_mode_ui = str(
        checker_settings_state.get("mode", st.session_state.checker_mode_ui)
    )
    st.session_state.checker_model_ui = str(
        checker_settings_state.get("checker_model", st.session_state.checker_model_ui)
    )
    st.session_state.checker_require_pass_ui = bool(
        checker_settings_state.get(
            "require_pass_before_export",
            st.session_state.checker_require_pass_ui,
        )
    )
    st.session_state.checker_detailed_ui = bool(
        checker_settings_state.get(
            "detailed_diagnostics",
            st.session_state.checker_detailed_ui,
        )
    )
    st.session_state.checker_llm_ui = bool(
        checker_settings_state.get(
            "llm_checker_enabled",
            st.session_state.checker_llm_ui,
        )
    )

    st.session_state._history_current_id = str(restore_state.get("history_document_id", ""))
    st.session_state._history_current_label = str(restore_state.get("history_label", ""))
    st.session_state._history_current_session_hash = str(
        restore_state.get("history_session_hash", "")
    )
    st.session_state._history_current_source_hash = str(
        restore_state.get("history_source_hash", "")
    )


def _build_translation_settings_snapshot(
    *,
    input_mode: str,
    selected_model: str,
    level: str,
    region: str,
    style: str,
    fidelity: str,
    temperature: float,
    max_chars: int,
    start_chunk: int,
    chunks_to_process: int,
    include_literal: bool,
    include_vocab: bool,
    include_grammar: bool,
    skip_enrichments: bool,
    show_audio_controls: bool,
) -> dict[str, object]:
    return {
        "input_mode": input_mode,
        "selected_model": selected_model,
        "level": level,
        "region": region,
        "style": style,
        "fidelity": fidelity,
        "temperature": float(temperature),
        "max_chars": int(max_chars),
        "start_chunk": int(start_chunk),
        "chunks_to_process": int(chunks_to_process),
        "chunks_processed": len(st.session_state.results),
        "include_literal": bool(include_literal),
        "include_vocab": bool(include_vocab),
        "include_grammar": bool(include_grammar),
        "skip_enrichments": bool(skip_enrichments),
        "show_audio_controls": bool(show_audio_controls),
    }


def _build_checker_settings_snapshot(checker_settings) -> dict[str, object]:
    return {
        "enabled": bool(checker_settings.enabled),
        "mode": checker_settings.mode,
        "checker_model": checker_settings.model,
        "require_pass_before_export": bool(checker_settings.require_pass),
        "detailed_diagnostics": bool(checker_settings.detailed_diagnostics),
        "llm_checker_enabled": bool(checker_settings.llm_enabled),
    }


def _save_history_snapshot(
    history_store: MongoTranslationHistoryStore,
    *,
    input_mode: str,
    selected_model: str,
    level: str,
    region: str,
    style: str,
    fidelity: str,
    temperature: float,
    max_chars: int,
    start_chunk: int,
    chunks_to_process: int,
    include_literal: bool,
    include_vocab: bool,
    include_grammar: bool,
    skip_enrichments: bool,
    show_audio_controls: bool,
    checker_settings,
) -> None:
    if not history_store.enabled or not st.session_state.results:
        return

    document = build_history_document(
        user_id=history_store.user_id,
        source_text=st.session_state.raw_text,
        input_mode=input_mode,
        uploaded_filename=st.session_state.get("_source_uploaded_filename", ""),
        results=st.session_state.results,
        checker_results=st.session_state.checker_results,
        translation_settings=_build_translation_settings_snapshot(
            input_mode=input_mode,
            selected_model=selected_model,
            level=level,
            region=region,
            style=style,
            fidelity=fidelity,
            temperature=temperature,
            max_chars=max_chars,
            start_chunk=start_chunk,
            chunks_to_process=chunks_to_process,
            include_literal=include_literal,
            include_vocab=include_vocab,
            include_grammar=include_grammar,
            skip_enrichments=skip_enrichments,
            show_audio_controls=show_audio_controls,
        ),
        checker_settings=_build_checker_settings_snapshot(checker_settings),
        label=st.session_state.get("_history_current_label", ""),
        save_source_text=history_store.save_source_text,
    )
    saved_document = history_store.save_history(
        document,
        preferred_history_id=st.session_state.get("_history_current_id", ""),
    )
    if saved_document:
        st.session_state._history_current_id = saved_document.get("_id", "")
        st.session_state._history_current_label = saved_document.get(
            "label", st.session_state.get("_history_current_label", "")
        )
        st.session_state._history_current_session_hash = saved_document.get("session_hash", "")
        st.session_state._history_current_source_hash = saved_document.get("source_hash", "")

def tts_lang_from_region(region: str) -> str:
    """Map app translation preference to a Spanish TTS locale."""
    mapping = {
        "Neutral": "es-MX",
        "Latin American": "es-MX",
        "European / Spain": "es-ES",
    }
    return mapping.get(region, "es-MX")


def _elapsed_suffix(start_time: float) -> str:
    return f" in {time.monotonic() - start_time:.1f}s"


def _overall_progress(chunk_position: int, chunk_count: int, phase_fraction: float) -> float:
    return min(((chunk_position - 1) + phase_fraction) / max(chunk_count, 1), 1.0)

class StickyProgress:
    """Small fixed-position progress UI with a Streamlit-like .progress API."""

    def __init__(self) -> None:
        self._slot = st.empty()

    def progress(self, value: float, text: str = "") -> None:
        fraction = max(0.0, min(float(value), 1.0))
        percent = int(round(fraction * 100))
        safe_text = _html.escape(str(text or "Working…"))

        self._slot.markdown(
            f"""
            <div class="sticky-progress-wrap" role="status" aria-live="polite">
                <div class="sticky-progress-card">
                    <div class="sticky-progress-text">
                        <span>{safe_text}</span>
                        <strong>{percent}%</strong>
                    </div>
                    <div
                        class="sticky-progress-track"
                        role="progressbar"
                        aria-valuemin="0"
                        aria-valuemax="100"
                        aria-valuenow="{percent}"
                        aria-label="Translation progress"
                    >
                        <div class="sticky-progress-fill" style="width: {percent}%;"></div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def empty(self) -> None:
        self._slot.empty()

@st.cache_data
def split_into_chunks(text: str, max_chars: int) -> List[str]:
    return _split_into_chunks_impl(text, max_chars)


def _is_qwen25_3b_model(model_name: str) -> bool:
    """Return True when the selected model is qwen2.5:3b (any tag variant)."""
    return model_name.strip().lower().startswith("qwen2.5:3b")


# -----------------------------
# Ollama helpers
# -----------------------------

@st.cache_data(ttl=120)
def check_ollama(model: str = OLLAMA_MODEL):
    try:
        response = _http_session.get(
            f"{OLLAMA_HOST}/api/tags",
            timeout=5,
        )
        response.raise_for_status()

        models = [
            m.get("name", "")
            for m in response.json().get("models", [])
        ]

        if model in models:
            return True, f"Connected to Ollama using {model}."

        pull_hint = (
            f"Run: `ollama pull {model}` to download it. "
            f"Available: {', '.join(models) or 'none'}"
        )
        return (
            False,
            f"Ollama is reachable, but **{model}** is not listed yet. {pull_hint}",
        )

    except Exception as exc:
        return False, f"Could not reach Ollama at {OLLAMA_HOST}: {exc}"


@st.cache_resource
def warmup_model(model: str) -> None:
    """Pin the model into Ollama GPU memory without generating any tokens.

    Uses POST /api/generate with an empty prompt — loads weights immediately
    without wasting time on token generation.  @st.cache_resource ensures
    this runs at most once per model per app restart.  Executes in a daemon
    thread so it never blocks the UI.
    """
    import threading

    def _ping() -> None:
        try:
            _ollama_load(OLLAMA_HOST, model, keep_alive=OLLAMA_KEEP_ALIVE, timeout=30.0)
            logger.info("warmup_model: %s pinned in memory", model)
        except Exception as exc:
            logger.debug("warmup_model: skipped (%s)", exc)

    threading.Thread(target=_ping, daemon=True).start()


def translate_chunk(
    chunk: str,
    level: str,
    style: str,
    region: str,
    fidelity: str,
    include_literal: bool,
    include_vocab: bool,
    include_grammar: bool,
    temperature: float,
    on_token: object = None,
    model: str | None = None,
) -> TranslationResponse:
    # NOTE: format is set to "json" (general JSON mode) rather than a JSON Schema
    # object because some models do not support Ollama structured-output
    # (schema-constrained sampling). The prompt contains the full schema
    # description so the model returns schema-conformant JSON; Pydantic
    # validation handles minor deviations.

    system = (
        "You are a professional English-to-Spanish translator and Spanish language tutor. "
        "Prioritize accurate meaning transfer, natural Spanish, register preservation, and learner usefulness. "
        "Return only valid JSON matching the provided schema. "
        "Do not include Markdown, XML, chain-of-thought, or commentary outside JSON. "
        "Do not add facts not present in the source."
    )

    user = f"""
Create a Spanish parallel-reader lesson from the English text below.

Learner level: {level}
Spanish region preference: {region}
Translation style: {style}
Translation fidelity: {fidelity}
Include literal Spanish: {include_literal}
Include vocabulary: {include_vocab}
Include grammar notes: {include_grammar}

Rules:
- Preserve the original English in each pair.
- Translate into natural Spanish suitable for the selected level and region.
- If fidelity is "Closest meaning", preserve source meaning and nuance over simplification.
- If fidelity is "Simpler learner wording", simplify wording without changing facts.
- If fidelity is "Preserve literary style", preserve imagery, rhythm, and tone when possible.
- summary_english MUST be written entirely in English. Never write summary_english in Spanish,
  Portuguese, or any other language. It is an English-language summary for an English-speaking
  learner. Write it as if explaining the text to someone who has not read it yet. If uncertain,
  begin with "Summary: " followed by an English description of the main idea.
- summary_spanish MUST be written in Spanish. It is a Spanish-language summary for reading practice.
- If literal Spanish is disabled, set literal_spanish to an empty string.
- If literal Spanish is enabled, literal_spanish must be a word-for-word rendering of the English source into Spanish, preserving English word order even when it sounds awkward. It should differ visibly from the polished spanish field.
- If vocabulary is disabled, use an empty vocabulary list.
- If grammar notes are disabled, use an empty grammar_notes list.
- Use CEFR values only: A1, A2, B1, B2, C1, C2.
- You MUST use the exact field names shown in the example below. Do not rename fields.
- CRITICAL — complete text required: Every string field must contain the complete, fully written-out text.
  Do NOT end any field value with "...", "…", "etc.", "[continues]", or any other abbreviation.
  Do NOT summarise or shorten the English source — copy it verbatim into the "english" field.
  If a field does not apply, use an empty string "" or empty list [], never an ellipsis.
- Output ONLY a single valid JSON object that follows this structure exactly.
  Replace each placeholder description with real content for the passage:

{_TRANSLATION_EXAMPLE_STR}

TEXT:
{chunk}
"""

    _model = model or OLLAMA_MODEL
    num_predict = _estimate_num_predict(
        len(chunk),
        include_literal,
        include_vocab,
        include_grammar,
        model_name=_model,
    )
    payload = {
        "model": _model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": True,
        "format": "json",
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": temperature,
            "top_p": OLLAMA_TOP_P,
            "num_predict": num_predict,
            "num_ctx": _dynamic_num_ctx(len(chunk), num_predict),
            # Prevents the model from looping on tokens (e.g. "ch ch ch…").
            # 1.15 is a mild penalty safe for JSON output; higher values can
            # distort token probabilities and break structured output.
            "repeat_penalty": 1.15,
        },
    }

    t0 = time.monotonic()
    try:
        content, t_first_token = _ollama_stream(
            OLLAMA_HOST, payload, OLLAMA_REQUEST_TIMEOUT, on_token
        )
    except requests.exceptions.ConnectionError as exc:
        raise requests.exceptions.ConnectionError(
            f"Cannot reach Ollama at {OLLAMA_HOST}. "
            "Check that the container is running and healthy."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise requests.exceptions.Timeout(
            "Ollama did not respond within the timeout. "
            "Try reducing the chunk size (Max chars per chunk slider) or using a faster model."
        ) from exc

    elapsed = time.monotonic() - t0
    logger.info(
        "translate_chunk: model=%s len_in=%d len_out=%d ttft=%.2fs total=%.2fs num_predict=%d",
        _model, len(chunk), len(content),
        t_first_token or 0.0, elapsed, num_predict,
    )

    # Strip markdown code fences that some Ollama versions emit
    _content_stripped = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.MULTILINE)
    _content_stripped = re.sub(r"```\s*$", "", _content_stripped.strip(), flags=re.MULTILINE).strip()
    if _content_stripped != content:
        logger.debug("Stripped markdown fences from model output before parsing.")
        content = _content_stripped

    try:
        result = TranslationResponse.model_validate_json(content)
    except ValidationError:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            extracted = content[start:end]
            try:
                result = TranslationResponse.model_validate_json(extracted)
            except ValidationError as inner_exc:
                # Last-resort: the model sometimes emits a string (e.g. a
                # schema snippet like "`vocabulary`: [ { … } ]") as a pairs
                # element instead of a proper object.  Parse the raw JSON,
                # drop any non-object entries from pairs, and re-validate.
                try:
                    raw = json.loads(extracted)
                except json.JSONDecodeError:
                    logger.error(
                        "TranslationResponse parse failed — invalid JSON. "
                        "Raw content (first 300 chars): %s",
                        content[:300],
                    )
                    raise inner_exc
                raw_pairs = raw.get("pairs", [])
                # Pass 1: drop non-dict elements (e.g. bare strings the model
                # emits when it confuses schema text with data).
                dict_pairs = [p for p in raw_pairs if isinstance(p, dict)]
                # Pass 2: validate each pair individually; drop any that fail.
                # This catches dicts with wrong value types, e.g. the model
                # echoing schema metadata ({"english": {"type": "string"}, …})
                # as a pair — which is a dict but fails Pydantic field checks.
                valid_pairs: list[dict] = []
                for idx, p in enumerate(dict_pairs):
                    try:
                        ReadingPair.model_validate(p)
                        valid_pairs.append(p)
                    except ValidationError as pair_exc:
                        logger.warning(
                            "Dropped pairs[%d] — %d field error(s): %s",
                            idx,
                            pair_exc.error_count(),
                            pair_exc.errors(include_url=False),
                        )
                if not valid_pairs:
                    logger.error(
                        "TranslationResponse parse failed — no valid pairs after "
                        "per-pair filtering. Raw content (first 300 chars): %s",
                        content[:300],
                    )
                    raise inner_exc
                total_dropped = len(raw_pairs) - len(valid_pairs)
                if total_dropped:
                    logger.warning(
                        "Dropped %d pair(s) from model output after validation "
                        "(non-dict: %d, schema-echo/type-error: %d).",
                        total_dropped,
                        len(raw_pairs) - len(dict_pairs),
                        len(dict_pairs) - len(valid_pairs),
                    )
                raw["pairs"] = valid_pairs
                try:
                    result = TranslationResponse.model_validate(raw)
                except ValidationError:
                    logger.error(
                        "TranslationResponse parse failed after per-pair filtering. "
                        "Raw content (first 300 chars): %s",
                        content[:300],
                    )
                    raise inner_exc
                if total_dropped:
                    result.parse_warnings.append(
                        f"{total_dropped} pair(s) were dropped during parsing "
                        "(malformed model output \u2014 see logs for details)."
                    )
        else:
            logger.error(
                "TranslationResponse parse failed — no JSON object found. "
                "Raw (first 300 chars): %s",
                content[:300],
            )
            raise

    # Post-process: fix summary_english if the model wrote it in Spanish.
    result.summary_english = _fix_english_summary(result.summary_english, chunk)

    # Post-parse omission check: flag pairs with empty source or translation.
    for _pidx, _pair in enumerate(result.pairs):
        if not _pair.english.strip():
            result.parse_warnings.append(
                f"Pair {_pidx + 1}: English field is empty \u2014 source text may have been dropped."
            )
        if not _pair.spanish.strip():
            result.parse_warnings.append(
                f"Pair {_pidx + 1}: Spanish field is empty \u2014 translation is missing."
            )

    # Post-process: strip any trailing ellipsis the model added as abbreviation.
    result.summary_english = _strip_ellipsis(result.summary_english)
    result.summary_spanish = _strip_ellipsis(result.summary_spanish)
    for pair in result.pairs:
        pair.english = _strip_ellipsis(pair.english)
        pair.spanish = _strip_ellipsis(pair.spanish)
        pair.literal_spanish = _strip_ellipsis(pair.literal_spanish)
        pair.comprehension_question_spanish = _strip_ellipsis(pair.comprehension_question_spanish)
        pair.grammar_notes = [_strip_ellipsis(n) for n in pair.grammar_notes]

    return result


def _estimate_num_predict(
    chunk_len: int,
    include_literal: bool,
    include_vocab: bool,
    include_grammar: bool,
    model_name: str = "",
) -> int:
    """Adaptive num_predict budget based on chunk size and enabled features."""
    base = min(chunk_len * 3, 4000)
    if include_literal:
        base += 600
    if include_vocab:
        base += 800
    if include_grammar:
        base += 600
    cap = 5000
    if _is_qwen25_3b_model(model_name):
        cap = QWEN25_3B_SAFE_NUM_PREDICT_CAP
    # Hard cap: smaller models (3b) cannot reliably generate very long JSON and
    # will ramble to the token limit if given too much budget, producing ~3× the
    # input length as unparseable output.
    return min(max(base, 800), cap)


def _dynamic_num_ctx(chunk_len: int, num_predict: int) -> int:
    """Smallest sufficient num_ctx for this chunk; always a power of 2 in [2048, OLLAMA_NUM_CTX]."""
    # ~4 chars/token; system + user prompt overhead ~600 tokens.
    estimated = (chunk_len // 4) + num_predict + 600
    ctx = 2048
    while ctx < estimated:
        ctx <<= 1
    return min(ctx, OLLAMA_NUM_CTX)


def _strip_ellipsis(text: str) -> str:
    """Remove trailing '...' / '…' abbreviation markers that models add when truncating.

    Strips only from the *end* of the string so legitimate mid-text ellipses
    (e.g. quoted omissions "he said … the attacks") are preserved.
    """
    s = text.rstrip()
    changed = True
    while changed:
        changed = False
        for marker in ("...", "…"):
            if s.endswith(marker):
                s = s[: -len(marker)].rstrip()
                changed = True
    return s


def _norm_text_for_compare(text: str) -> str:
    """Normalise whitespace/case to compare whether two strings meaningfully differ."""
    return " ".join((text or "").split()).strip().casefold()


def _fix_english_summary(summary: str, source_chunk: str) -> str:
    """
    Return summary unchanged if it looks English.
    If it looks Spanish (heuristic signal ratio > 12%), derive a plain-text
    fallback from the first 1-2 sentences of the original English source.
    """
    if not summary:
        return summary
    tokens = re.findall(r"\b[a-z\u00e0-\u00ff]+\b", summary.lower())
    if not tokens:
        return summary
    sp_ratio = sum(1 for t in tokens if t in _STRONG_SPANISH) / len(tokens)
    if sp_ratio > 0.12:
        logger.warning(
            "summary_english appears to be Spanish (signal ratio=%.2f); "
            "falling back to source extraction.",
            sp_ratio,
        )
        sentences = _RE_SENTENCES.split(source_chunk.strip())
        return " ".join(sentences[:2]).strip()
    return summary


# -----------------------------
# Pair retranslation
# -----------------------------

def retranslate_pair(
    pair: ReadingPair,
    check_result: "PairCheckResult",
    level: str,
    style: str,
    region: str,
    fidelity: str,
    include_literal: bool,
    include_vocab: bool,
    include_grammar: bool,
    temperature: float,
    model: str | None = None,
    corrected_spanish_hint: str = "",
) -> ReadingPair:
    """
    Re-translate a single pair that failed quality checks.

    Passes the original English, the rejected Spanish, and the specific issues
    back to the model so it can produce a corrected translation.
    If corrected_spanish_hint is provided (from the checker), it is included as
    a reference so the model can use it as a strong basis.
    The original English is always preserved verbatim in the returned pair.
    """
    all_issues = [
        issue
        for group in (
            check_result.faithfulness_issues,
            check_result.hallucination_issues,
            check_result.omission_issues,
            check_result.label_issues,
            check_result.language_quality_issues,
            check_result.unsupported_claims,
        )
        for issue in group
    ]
    issues_text = (
        "\n".join(f"- {i}" for i in all_issues)
        if all_issues
        else "- General translation quality issue detected."
    )

    system = (
        "You are a professional English-to-Spanish translator and Spanish language tutor. "
        "Prioritize accurate meaning transfer, natural Spanish, register preservation, and learner usefulness. "
        "Return only valid JSON matching the provided schema. "
        "Do not include Markdown, XML, chain-of-thought, or commentary outside JSON. "
        "Do not add facts not present in the source."
    )

    _hint_block = (
        f"\nQUALITY CHECKER SUGGESTED CORRECTION (use as a strong reference):\n{corrected_spanish_hint}\n"
        if corrected_spanish_hint and corrected_spanish_hint.strip()
        else ""
    )
    user = f"""
A previous Spanish translation of the English sentence below was rejected by a quality checker.
Produce a corrected translation that fixes every listed issue.

Learner level: {level}
Spanish region preference: {region}
Translation style: {style}
Translation fidelity: {fidelity}
Include literal Spanish: {include_literal}
Include vocabulary: {include_vocab}
Include grammar notes: {include_grammar}

ENGLISH SOURCE (copy verbatim into the "english" field):
{pair.english}

PREVIOUS SPANISH TRANSLATION (rejected — do NOT copy or reuse it):
{pair.spanish}

ISSUES FOUND IN PREVIOUS TRANSLATION:
{issues_text}{_hint_block}
Rules:
- Fix every issue listed above.
- The "english" field must contain the original English text exactly as shown above.
- Translate into natural Spanish suitable for the selected level and region.
- If fidelity is "Closest meaning", preserve source meaning and nuance over simplification.
- If fidelity is "Simpler learner wording", simplify wording without changing facts.
- If fidelity is "Preserve literary style", preserve imagery, rhythm, and tone when possible.
- If literal Spanish is disabled ({not include_literal}), set literal_spanish to an empty string.
- If literal Spanish is enabled, literal_spanish must be a word-for-word rendering of the English into Spanish, preserving English word order even when it sounds awkward.
- If vocabulary is disabled ({not include_vocab}), use an empty vocabulary list.
- If grammar notes are disabled ({not include_grammar}), use an empty grammar_notes list.
- Use CEFR values only: A1, A2, B1, B2, C1, C2.
- CRITICAL — complete text required: every string field must be fully written out.
  Do NOT end any field with "...", "…", or any abbreviation.
- Output ONLY a single valid JSON object matching this structure exactly:

{_RETRANSLATE_PAIR_EXAMPLE_STR}
"""

    _model = model or OLLAMA_MODEL
    num_predict = _estimate_num_predict(
        len(pair.english),
        include_literal,
        include_vocab,
        include_grammar,
        model_name=_model,
    )

    payload = {
        "model": _model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "format": "json",
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": temperature,
            "top_p": OLLAMA_TOP_P,
            "num_predict": num_predict,
            "num_ctx": OLLAMA_NUM_CTX,
        },
    }

    resp = _ollama_chat(OLLAMA_HOST, payload, OLLAMA_REQUEST_TIMEOUT)
    content = resp

    # Strip markdown code fences that some Ollama versions emit (same as translate_chunk).
    _content_stripped = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.MULTILINE)
    _content_stripped = re.sub(r"```\s*$", "", _content_stripped.strip(), flags=re.MULTILINE).strip()
    if _content_stripped != content:
        logger.debug("Stripped markdown fences from retranslate_pair output before parsing.")
        content = _content_stripped

    try:
        new_pair = ReadingPair.model_validate_json(content)
    except ValidationError:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            new_pair = ReadingPair.model_validate_json(content[start:end])
        else:
            raise

    # Post-process: strip ellipsis abbreviations.
    new_pair.spanish = _strip_ellipsis(new_pair.spanish)
    new_pair.literal_spanish = _strip_ellipsis(new_pair.literal_spanish)
    new_pair.comprehension_question_spanish = _strip_ellipsis(
        new_pair.comprehension_question_spanish
    )
    new_pair.grammar_notes = [_strip_ellipsis(n) for n in new_pair.grammar_notes]

    # Always preserve the original English verbatim — the model must not alter it.
    new_pair.english = pair.english

    return new_pair


# -----------------------------
# Checker UI helpers
# -----------------------------

def _render_check_badge(result: PairCheckResult) -> None:
    """Compact one-line status for a checker result. Uses if/else to avoid bare expression."""
    msg = f"Checker: {result.user_facing_summary}"
    if result.severity == "pass":
        st.success(msg)
    elif result.severity == "info":
        st.info(msg)
    elif result.severity == "warning":
        st.warning(msg)
    else:
        st.error(msg)


def _maybe_render_tts_button(
    text: str,
    *,
    lang: str,
    show_audio_controls: bool,
) -> None:
    if show_audio_controls and text.strip():
        render_tts_button(text, lang=lang)


def flatten_pairs(results: List[TranslationResponse]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    global_passage_idx = 0
    for result_idx, result in enumerate(results):
        for pair_idx, pair in enumerate(result.pairs):
            global_passage_idx += 1
            records.append(
                {
                    "result": result,
                    "pair": pair,
                    "result_idx": result_idx,
                    "pair_idx": pair_idx,
                    "global_passage_idx": global_passage_idx,
                }
            )
    return records


def count_corrections(results: List[TranslationResponse]) -> int:
    return sum(
        1
        for result in results
        for pair in result.pairs
        if pair.corrected_by_checker
    )


def collect_checker_status(
    results: List[TranslationResponse],
    checker_results: dict,
    pair_check_keys: dict[int, str],
) -> dict[str, int]:
    counts = {"pass": 0, "info": 0, "warning": 0, "fail": 0, "missing": 0}
    for result in results:
        for pair in result.pairs:
            check_key = pair_check_keys.get(id(pair))
            check_result = checker_results.get(check_key) if check_key else None
            if check_result is None:
                counts["missing"] += 1
                continue
            counts[check_result.severity if check_result.severity in counts else "info"] += 1
    return counts


def filter_pair_records(
    records: list[dict[str, object]],
    query: str,
    corrected_only: bool,
    difficulty_filter: list[str],
) -> list[dict[str, object]]:
    needle = query.strip().lower()
    allowed_difficulties = set(difficulty_filter)
    filtered: list[dict[str, object]] = []

    for record in records:
        pair = record["pair"]
        if not all(
            hasattr(pair, attr)
            for attr in ("english", "spanish", "difficulty", "corrected_by_checker")
        ):
            continue

        if corrected_only and not pair.corrected_by_checker:
            continue
        if allowed_difficulties and pair.difficulty not in allowed_difficulties:
            continue
        if needle:
            haystacks = [pair.english.lower(), pair.spanish.lower()]
            if not any(needle in haystack for haystack in haystacks):
                continue
        filtered.append(record)

    return filtered


def build_blocked_export_items(
    results: List[TranslationResponse],
    checker_results: dict,
    pair_check_keys: dict[int, str],
) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    global_passage_idx = 0

    for result_idx, result in enumerate(results):
        for pair_idx, pair in enumerate(result.pairs):
            global_passage_idx += 1
            check_key = pair_check_keys.get(id(pair))
            check_result = checker_results.get(check_key) if check_key else None

            if check_result is None:
                items.append(
                    {
                        "location": (
                            f"Passage {global_passage_idx} "
                            f"(chunk {result_idx + 1}, item {pair_idx + 1})"
                        ),
                        "severity": "missing",
                        "summary": "Checker result is missing.",
                    }
                )
                continue

            if check_result.passed:
                continue

            items.append(
                {
                    "location": (
                        f"Passage {global_passage_idx} "
                        f"(chunk {result_idx + 1}, item {pair_idx + 1})"
                    ),
                    "severity": check_result.severity,
                    "summary": (
                        check_result.user_facing_summary
                        or check_result.recommended_action
                        or "Checker reported an issue."
                    ),
                }
            )

    return items


def build_bilingual_csv(results: List[TranslationResponse]) -> bytes:
    rows = [
        {
            "English": pair.english,
            "Spanish": pair.spanish,
            "Difficulty": pair.difficulty,
            "Corrected": "yes" if pair.corrected_by_checker else "no",
            "Correction note": pair.correction_note,
        }
        for result in results
        for pair in result.pairs
    ]
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def build_spanish_only_text(results: List[TranslationResponse]) -> bytes:
    lines: list[str] = []
    for result in results:
        if result.title:
            lines.append(result.title)
        lines.extend(pair.spanish for pair in result.pairs)
        lines.append("")
    return "\n".join(lines).encode("utf-8")


def build_anki_csv(results: List[TranslationResponse]) -> bytes:
    rows = []
    for result in results:
        for pair in result.pairs:
            notes = []
            if pair.grammar_notes:
                notes.append("Grammar: " + " | ".join(pair.grammar_notes))
            if pair.vocabulary:
                notes.append(
                    "Vocabulary: "
                    + " | ".join(
                        f"{vocab.spanish} = {vocab.english}" for vocab in pair.vocabulary
                    )
                )
            rows.append(
                {
                    "Front": pair.spanish,
                    "Back": "\n\n".join(part for part in [pair.english, *notes] if part),
                }
            )
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def render_study_status(
    *,
    total_pairs: int,
    total_vocab: int,
    total_corrections: int,
    checker_counts: dict[str, int],
    export_blocked: bool,
    checker_active: bool,
) -> None:
    
    if not checker_active:
        checker_value = "Off"
    elif checker_counts["fail"]:
        checker_value = f"{checker_counts['fail']} blocked"
    elif checker_counts["warning"]:
        checker_value = f"{checker_counts['warning']} warnings"
    elif total_pairs and checker_counts["missing"] == total_pairs:
        checker_value = "Not checked"
    elif checker_counts["missing"]:
        checker_value = f"{checker_counts['missing']} unchecked"
    elif checker_counts["info"]:
        checker_value = f"{checker_counts['info']} notes"
    else:
        checker_value = "Passed"

    with st.container(border=True):
        cols = st.columns(5)
        cols[0].metric("Passages", total_pairs)
        cols[1].metric("Vocabulary", total_vocab)
        cols[2].metric("Corrections", total_corrections)
        cols[3].metric("Checker", checker_value)
        cols[4].metric("Export", "Blocked" if export_blocked else "Ready")
        st.caption(
            "Lesson ready for study. Use filters in Parallel Reader, focus mode in Spanish First, or export once the checker status looks right."
        )


def _clear_reader_filters() -> None:
    st.session_state.reader_search = ""
    st.session_state.reader_corrected_only = False
    st.session_state.reader_difficulty_filter = []


def _render_pair_card(
    pair: ReadingPair,
    *,
    checker_results: dict,
    checker_settings,
    include_literal: bool,
    tts_lang: str,
    show_audio_controls: bool,
    passage_label: str = "",
) -> None:
    with st.container(border=True):
        if passage_label:
            st.caption(passage_label)

        left, right = st.columns(2)

        with left:
            st.markdown("**English**")
            st.markdown(
                f'<p class="passage-text">{_html.escape(pair.english).replace(chr(10), "<br>")}</p>',
                unsafe_allow_html=True,
            )

        with right:
            st.markdown("**Español**")
            st.markdown(
                f'<p class="passage-text">{_html.escape(pair.spanish).replace(chr(10), "<br>")}</p>',
                unsafe_allow_html=True,
            )
            if pair.corrected_by_checker:
                note = pair.correction_note or "Updated after checker correction"
                st.markdown(
                    f'<span style="background:#dcfce7;color:#166534;'
                    f'padding:2px 8px;border-radius:4px;'
                    f'font-size:0.80rem;font-weight:700;margin-right:6px;">'
                    f'Corrected</span>'
                    f'<span class="small-muted">{_html.escape(note)}</span>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                f'<span style="background:#e0f2fe;color:#0369a1;'
                f'padding:2px 8px;border-radius:4px;'
                f'font-size:0.82rem;font-weight:600;">'
                f'{pair.difficulty}</span>',
                unsafe_allow_html=True,
            )
            _maybe_render_tts_button(
                pair.spanish,
                lang=tts_lang,
                show_audio_controls=show_audio_controls,
            )

        if include_literal and pair.literal_spanish:
            with st.expander("Literal Spanish"):
                st.write(pair.literal_spanish)

        if pair.grammar_notes:
            with st.expander("Grammar notes"):
                st.markdown("\n".join(f"- {note}" for note in pair.grammar_notes))

        if pair.comprehension_question_spanish:
            with st.expander("Comprehension question"):
                st.write(pair.comprehension_question_spanish)

        if checker_settings.enabled and checker_settings.mode != "off":
            check_key = make_check_key(
                checker_settings,
                pair.english,
                pair.spanish,
                pair.literal_spanish,
            )
            check_result = checker_results.get(check_key)
            if check_result is not None:
                _render_checker_details(
                    check_result,
                    checker_settings.detailed_diagnostics,
                    tts_lang,
                    show_audio_controls,
                )



def _norm_pair_key(text: str) -> str:
    """Normalize text for matching generated/enriched pairs by source English."""
    return " ".join((text or "").split()).strip().casefold()


def _preserve_checker_corrections(
    existing_result: TranslationResponse,
    enriched_result: TranslationResponse,
) -> TranslationResponse:
    corrected_by_english: dict[str, ReadingPair] = {}

    for old_pair in existing_result.pairs:
        if not old_pair.corrected_by_checker:
            continue

        key = _norm_pair_key(old_pair.english)
        if key and key not in corrected_by_english:
            corrected_by_english[key] = old_pair

    for new_pair in enriched_result.pairs:
        old_pair = corrected_by_english.get(_norm_pair_key(new_pair.english))
        if old_pair is None:
            continue

        new_pair.spanish = old_pair.spanish
        new_pair.corrected_by_checker = True
        new_pair.correction_note = old_pair.correction_note
        new_pair.correction_reason = old_pair.correction_reason
        new_pair.original_spanish_before_correction = (
            old_pair.original_spanish_before_correction
        )

    return enriched_result

def _render_corrections_tab(
    records: list[dict[str, object]],
    *,
    checker_results: dict,
    pair_check_keys: dict[int, str],
) -> None:
    corrected_records = [
        record
        for record in records
        if all(
            hasattr(record["pair"], attr)
            for attr in (
                "english",
                "spanish",
                "corrected_by_checker",
                "correction_note",
                "original_spanish_before_correction",
            )
        )
        and record["pair"].corrected_by_checker
    ]
    if not corrected_records:
        st.info("No checker corrections were applied to this lesson.")
        return

    for record in corrected_records:
        pair = record["pair"]
        passage_idx = int(record["global_passage_idx"])
        check_key = pair_check_keys.get(id(pair))
        check_result = checker_results.get(check_key) if check_key else None

        with st.container(border=True):
            st.markdown(f"**Passage {passage_idx}**")
            st.markdown("**English**")
            st.write(pair.english)
            if pair.original_spanish_before_correction:
                st.markdown("**Before correction**")
                st.write(pair.original_spanish_before_correction)
            st.markdown("**Corrected Spanish**")
            st.write(pair.spanish)
            st.caption(pair.correction_note or "Applied checker correction")
            
            if getattr(pair, "correction_reason", ""):
                st.caption(f"Reason: {pair.correction_reason}")
            
            if check_result is not None and check_result.user_facing_summary:
                st.caption(f"Checker: {check_result.user_facing_summary}")


history_store = get_history_store()
history_items = history_store.list_history()
history_items_by_id = {item.get("_id", ""): item for item in history_items if item.get("_id")}

st.subheader("0. History")
_history_cols = st.columns([4, 0.8, 0.8, 1.2])
with _history_cols[0]:
    history_options = [""] + [item["_id"] for item in history_items if item.get("_id")]
    st.selectbox(
        "Saved translations",
        history_options,
        key="history_selected_id",
        format_func=lambda item_id: (
            "Select a saved translation"
            if not item_id
            else _history_option_label(history_items_by_id.get(item_id, {}))
        ),
    )
selected_history = history_items_by_id.get(st.session_state.history_selected_id, {})
if st.session_state.get("_history_selected_id_prev") != st.session_state.history_selected_id:
    st.session_state.history_rename_value = selected_history.get("label", "") if selected_history else ""
    st.session_state._history_selected_id_prev = st.session_state.history_selected_id
with _history_cols[1]:
    st.write("")
    history_load_clicked = st.button(
        "Load",
        key="history_load_button",
        disabled=not st.session_state.history_selected_id,
    )
with _history_cols[2]:
    st.write("")
    history_delete_clicked = st.button(
        "Delete",
        key="history_delete_button",
        disabled=not st.session_state.history_selected_id,
    )
with _history_cols[3]:
    st.text_input(
        "Rename label",
        key="history_rename_value",
        placeholder="Optional label",
        disabled=not st.session_state.history_selected_id,
    )
history_rename_clicked = st.button(
    "Rename",
    key="history_rename_button",
    disabled=not st.session_state.history_selected_id or not st.session_state.history_rename_value.strip(),
)

if history_load_clicked and st.session_state.history_selected_id:
    loaded_history = history_store.load_history(st.session_state.history_selected_id)
    if loaded_history:
        _apply_history_restore(
            build_session_restore_state(
                loaded_history,
                translation_response_cls=TranslationResponse,
                reading_pair_cls=ReadingPair,
                vocabulary_item_cls=VocabularyItem,
                pair_check_result_cls=PairCheckResult,
            )
        )
        st.session_state._history_loaded_notice = (
            f"Loaded {loaded_history.get('label') or loaded_history.get('title') or 'saved translation'}."
        )
        st.rerun()
    else:
        st.warning("Could not load the selected history entry.")

if history_delete_clicked and st.session_state.history_selected_id:
    if history_store.delete_history(st.session_state.history_selected_id):
        if st.session_state._history_current_id == st.session_state.history_selected_id:
            _clear_loaded_history_tracking()
        st.session_state.history_selected_id = ""
        st.session_state.history_rename_value = ""
        st.session_state._history_loaded_notice = "History entry deleted."
        st.rerun()
    else:
        st.warning("Could not delete the selected history entry.")

if history_rename_clicked and st.session_state.history_selected_id:
    if history_store.rename_history(
        st.session_state.history_selected_id,
        st.session_state.history_rename_value.strip(),
    ):
        if st.session_state._history_current_id == st.session_state.history_selected_id:
            st.session_state._history_current_label = st.session_state.history_rename_value.strip()
        st.session_state._history_loaded_notice = "History label updated."
        st.rerun()
    else:
        st.warning("Could not rename the selected history entry.")

if st.session_state._history_loaded_notice:
    st.caption(st.session_state._history_loaded_notice)

if st.session_state._history_current_label:
    st.caption(f"Current history entry: {st.session_state._history_current_label}")
elif history_store.enabled and not history_items:
    st.caption("MongoDB history is enabled, but no saved translations exist yet.")
else:
    st.caption(history_store.status_message())


def _render_checker_details(
    result: PairCheckResult,
    detailed: bool,
    tts_lang: str,
    show_audio_controls: bool,
) -> None:
    """Render badge + expandable per-issue details for one pair."""
    _render_check_badge(result)

    issue_groups = [
        ("Faithfulness", result.faithfulness_issues),
        ("Hallucination", result.hallucination_issues),
        ("Omissions", result.omission_issues),
        ("Label issues", result.label_issues),
        ("Language quality", result.language_quality_issues),
        ("Unsupported claims", result.unsupported_claims),
    ]
    has_issues = any(items for _, items in issue_groups)
    has_action = bool(result.recommended_action)

    if not has_issues and not has_action:
        return

    with st.expander("Checker details"):
        if result.score is not None:
            st.markdown(f"**Quality score:** {result.score:.2f} / 1.0")

        for label, items in issue_groups:
            if items:
                st.markdown(f"**{label}:**")
                for issue in items:
                    st.markdown(f"- {issue}")

        if has_action:
            st.markdown(f"**Recommended action:** {result.recommended_action}")

        if result.corrected_spanish:
            st.markdown("**Corrected translation:**")
            st.info(result.corrected_spanish)
            _maybe_render_tts_button(
                result.corrected_spanish,
                lang=tts_lang,
                show_audio_controls=show_audio_controls,
            )

        if detailed:
            method = "LLM + deterministic" if result.checked_with_llm else "deterministic only"
            cache_note = " (cache hit)" if result.cache_hit else ""
            trunc_note = " ⚠️ inputs truncated" if result.truncated else ""
            lat = (
                f" | {result.checker_latency_ms:.0f} ms"
                if result.checker_latency_ms is not None
                else ""
            )
            st.markdown(
                f'<span class="small-muted">Method: {method}{cache_note}{trunc_note}{lat}</span>',
                unsafe_allow_html=True,
            )


def render_result_card(
    result: "TranslationResponse",
    *,
    checker_results: dict,
    checker_settings,
    include_literal: bool,
    tts_lang: str,
    show_audio_controls: bool = True,
    result_idx: "int | None" = None,
) -> None:
    """Render one TranslationResponse as a parallel-reader card.

    Called both during progressive rendering (inside the translate loop) and
    inside the Study > Parallel Reader tab.
    """
    if result.title:
        st.markdown(f"### {result.title}")

    for _warn in result.parse_warnings:
        st.warning(_warn)

    if result.summary_english or result.summary_spanish:
        with st.expander("Summary"):
            if result.summary_english:
                st.markdown("**English summary**")
                st.write(result.summary_english)
            if result.summary_spanish:
                st.markdown("**Spanish summary**")
                st.write(result.summary_spanish)
                _maybe_render_tts_button(
                    result.summary_spanish,
                    lang=tts_lang,
                    show_audio_controls=show_audio_controls,
                )

    for pair_idx, pair in enumerate(result.pairs, start=1):
        _render_pair_card(
            pair,
            checker_results=checker_results,
            checker_settings=checker_settings,
            include_literal=include_literal,
            tts_lang=tts_lang,
            show_audio_controls=show_audio_controls,
            passage_label=f"Passage {pair_idx} / {len(result.pairs)}",
        )

    # ── On-demand enrichment button ───────────────────────────────────────────
    # Shown in the Study tab when the chunk was translated without enrichments.
    if result_idx is not None:
        _has_enrichments = any(
            p.vocabulary or p.grammar_notes or p.literal_spanish
            for p in result.pairs
        )
        if not _has_enrichments:
            if st.button(
                "⚡ Add vocab, grammar & literal Spanish",
                key=f"enrich_{result_idx}",
                help="Re-translate this chunk with all enrichments enabled.",
            ):
                st.session_state._enrich_idx = result_idx
                st.rerun()


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:
    st.header("Settings")

    settings_mode = st.radio(
        "Settings mode",
        ["Simple", "Advanced"],
        horizontal=True,
        key="settings_mode",
        help="Simple keeps learner-friendly controls visible. Advanced shows model tuning and checker internals.",
    )
    _advanced_mode = settings_mode == "Advanced"

    _model_options = list(AVAILABLE_OLLAMA_MODELS)
    if st.session_state.selected_model not in _model_options:
        _model_options.insert(0, st.session_state.selected_model)
    _default_idx = (
        _model_options.index(st.session_state.selected_model)
        if st.session_state.selected_model in _model_options
        else 0
    )

    if _advanced_mode:
        st.write("**Model**")
        selected_model = st.selectbox(
            "Ollama model",
            _model_options,
            index=_default_idx,
            key="selected_model",
            help="qwen2.5:3b is fastest on CPU. qwen2.5:7b is the default. qwen2.5:14b is highest quality.",
        )
    else:
        selected_model = st.session_state.selected_model

    _check_started = time.monotonic()
    ok, status = check_ollama(selected_model)
    status = f"{status} (checked{_elapsed_suffix(_check_started)})"

    if ok:
        if _advanced_mode:
            st.success(status)
        else:
            st.caption("Model ready")
        warmup_model(selected_model)
    else:
        st.warning(status)

    
    if _advanced_mode:
        with st.expander("Model guidance"):

            st.markdown(
                """
    Default: `qwen2.5:7b` — fast, low memory, good Spanish quality.

    CPU-only: `qwen2.5:3b` — ~2× faster than 7b on CPU; modest quality reduction.

    Optional: `qwen2.5:14b` — higher quality, requires ~12 GB RAM / 10 GB+ VRAM.

    To change the default, set `OLLAMA_MODEL` in `.env` and restart containers.

    Pull models before starting:
    ```
    ollama pull qwen2.5:3b
    ollama pull qwen2.5:7b
    ollama pull qwen2.5:14b
    ```

    ⚠️ **Hardware:** `qwen2.5:3b` ~3–4 GB RAM · `qwen2.5:7b` ~6–8 GB RAM · `qwen2.5:14b` ~12 GB RAM.

    🔒 **License:** Qwen2.5 is released under Apache 2.0 (commercial use allowed).
    """
            )

    input_mode = st.radio(
        "Input type",
        [
            "Paste text",
            "Upload file",
        ],
        key="input_mode",
    )

    level = st.selectbox(
        "Spanish level",
        [
            "A1 beginner",
            "A2 elementary",
            "B1 intermediate",
            "B2 upper-intermediate",
            "C1 advanced",
        ],
        key="level",
    )

    region = st.selectbox(
        "Spanish preference",
        [
            "Neutral",
            "Latin American",
            "European / Spain",
        ],
        key="region",
    )
    tts_lang = tts_lang_from_region(region)

    style = st.selectbox(
        "Translation style",
        [
            "Natural Spanish",
            "Learner-friendly Spanish",
            "Literal but readable Spanish",
            "Literary when appropriate",
            "Journalistic when appropriate",
        ],
        key="style",
    )

    fidelity = st.selectbox(
        "Translation fidelity",
        [
            "Balanced",
            "Closest meaning",
            "Simpler learner wording",
            "Preserve literary style",
        ],
        key="fidelity",
    )

    if _advanced_mode:
        _chunk_upper_bound = (
            QWEN25_3B_SAFE_MAX_CHARS
            if _is_qwen25_3b_model(selected_model)
            else 4000
        )
        st.session_state.max_chars = min(
            max(800, int(st.session_state.max_chars)),
            _chunk_upper_bound,
        )
        max_chars = st.slider(
            "Max characters per chunk",
            800,
            _chunk_upper_bound,
            step=100,
            key="max_chars",
        )
    else:
        if _is_qwen25_3b_model(selected_model):
            st.session_state.max_chars = min(
                int(st.session_state.max_chars), QWEN25_3B_SAFE_MAX_CHARS
            )
        max_chars = int(st.session_state.max_chars)

    if _is_qwen25_3b_model(selected_model):
        if max_chars > QWEN25_3B_SAFE_MAX_CHARS:
            max_chars = QWEN25_3B_SAFE_MAX_CHARS
        st.info(
            "qwen2.5:3b guardrail active: "
            f"max chunk size {QWEN25_3B_SAFE_MAX_CHARS} chars and "
            f"max output budget {QWEN25_3B_SAFE_NUM_PREDICT_CAP} tokens."
        )

    chunks_to_process = st.slider(
        "Chunks to process",
        1,
        10,
        key="chunks_to_process",
    )

    if _advanced_mode:
        temperature = st.slider(
            "Model temperature",
            0.0,
            0.3,
            step=0.05,
            key="temperature",
            help="Lower = more consistent JSON output. Keep below 0.3 for reliable structured translation.",
        )
    else:
        temperature = float(st.session_state.temperature)

    show_audio_controls = st.checkbox(
        "Show audio controls",
        key="show_audio_controls",
        help="Hide play buttons if you want a cleaner reading view.",
    )

    _skip_enrichments = st.checkbox(
        "⚡ Skip enrichments (faster)",
        key="skip_enrichments",
        help=(
            "Skip literal Spanish, vocabulary, and grammar notes. "
            "The model returns only English + Spanish pairs, which is significantly faster. "
            "Uncheck to re-enable enrichments."
        ),
    )

    include_literal = st.checkbox(
        "Include literal Spanish",
        key="include_literal",
        disabled=_skip_enrichments,
    )
    if _skip_enrichments:
        include_literal = False
    include_vocab = st.checkbox(
        "Include vocabulary",
        key="include_vocab",
        disabled=_skip_enrichments,
    )
    if _skip_enrichments:
        include_vocab = False

    include_grammar = st.checkbox(
        "Include grammar notes",
        key="include_grammar",
        disabled=_skip_enrichments,
    )
    if _skip_enrichments:
        include_grammar = False

    if _is_qwen25_3b_model(selected_model) and not _skip_enrichments:
        st.warning(
            "qwen2.5:3b with enrichments can produce long JSON and parse failures. "
            "For best reliability, enable 'Skip enrichments (faster)' or keep chunks small."
        )

    with st.expander("🔍 Output Checker"):
        # Derive sidebar defaults from env vars so Docker/local overrides take effect
        # on first render. Subsequent renders use Streamlit's widget session state.
        _checker_mode_options = ["instant", "off", "fast", "smart", "strict"]
        _checker_mode_idx = (
            _checker_mode_options.index(st.session_state.checker_mode_ui)
            if st.session_state.checker_mode_ui in _checker_mode_options
            else 3  # default to smart
        )

        checker_enabled_ui = st.checkbox(
            "Enable output checker",
            key="checker_enabled_ui",
        )
        checker_mode_ui = st.selectbox(
            "Checker mode",
            _checker_mode_options,
            index=_checker_mode_idx,
            key="checker_mode_ui",
            help=(
                "instant: translate and show immediately, no checker.\n"
                "off: no checks.\n"
                "fast: show first, run deterministic checks, and only use the model again for severe failures.\n"
                "smart: show first, then deterministic + LLM for risky pairs.\n"
                "strict: deterministic + LLM + retry before showing result."
            ),
        )
        checker_require_pass_ui = st.checkbox(
            "Require checker pass before export",
            key="checker_require_pass_ui",
            help="Block Markdown export for pairs that fail the checker.",
        )
        checker_model_ui = st.session_state.checker_model_ui
        checker_llm_ui = st.session_state.checker_llm_ui
        checker_detailed_ui = st.session_state.checker_detailed_ui
        if _advanced_mode:
            checker_model_ui = st.text_input(
                "Checker model",
                placeholder="defaults to translation model",
                key="checker_model_ui",
                help="Leave blank to use the same model as translation. Set CHECKER_MODEL in .env to make it persistent.",
            )
            checker_llm_ui = st.checkbox(
                "LLM checker enabled",
                key="checker_llm_ui",
                help="Uncheck to use deterministic checks only (no additional model calls).",
            )
            checker_detailed_ui = st.checkbox(
                "Show detailed diagnostics",
                key="checker_detailed_ui",
                help="Show per-issue breakdown. Keep off for a faster, cleaner UI.",
            )

    if st.button("Clear session"):
        _clear_current_lesson(clear_source=True)
        st.rerun()

# -----------------------------
# Checker settings (resolved from sidebar + env)
# -----------------------------

checker_settings = get_checker_settings(
    ollama_host=OLLAMA_HOST,
    ollama_model=OLLAMA_MODEL,
    enabled_override=checker_enabled_ui,
    mode_override=checker_mode_ui,
    model_override=checker_model_ui.strip() or None,
    require_pass_override=checker_require_pass_ui,
    llm_enabled_override=checker_llm_ui,
    detailed_diagnostics_override=checker_detailed_ui,
)


# -----------------------------
# Input
# -----------------------------

st.subheader("1. Add English source text")
st.markdown(
    '<p class="small-muted">Use pasted text or files you own / have permission to process. For news, paste an article excerpt or text you are allowed to use.</p>',
    unsafe_allow_html=True,
)

if input_mode == "Paste text":
    pasted = st.text_area(
        "Paste English text",
        height=280,
        placeholder="Paste a chapter excerpt, essay, article, or other English text...",
        key="source_text_input",
    )

    st.session_state._source_uploaded_filename = ""
    set_source_text(pasted)

    if pasted:
        _est_chunks = len(split_into_chunks(clean_text(pasted), max_chars)) if pasted.strip() else 0
        st.caption(
            f"{len(pasted):,} characters · ~{_est_chunks} chunk(s) at current max-chars setting"
        )

else:
    uploaded = st.file_uploader(
        "Upload PDF, DOCX, TXT, or Markdown",
        type=[
            "pdf",
            "docx",
            "txt",
            "md",
        ],
    )

    if uploaded:
        suffix = uploaded.name.lower().split(".")[-1]
        _upload_key = f"{uploaded.name}:{uploaded.size}"

        if st.session_state.get("_last_upload_key") != _upload_key:
            try:
                size_mb = uploaded.size / 1024 / 1024
                _extract_started = time.monotonic()
                with st.spinner(f"Extracting {suffix.upper()} ({size_mb:.1f} MB)…"):
                    if suffix == "pdf":
                        set_source_text(extract_pdf_text(uploaded))
                    elif suffix == "docx":
                        set_source_text(extract_docx_text(uploaded))
                    else:
                        set_source_text(extract_plain_text(uploaded))
                st.session_state._source_uploaded_filename = uploaded.name
                st.session_state["_last_upload_key"] = _upload_key
                st.success(
                    f"Extracted {len(st.session_state.raw_text):,} characters"
                    f"{_elapsed_suffix(_extract_started)}"
                )

            except Exception as exc:
                st.error(f"Could not extract text: {exc}")
        else:
            st.session_state._source_uploaded_filename = uploaded.name
            st.success(f"Extracted {len(st.session_state.raw_text):,} characters")
    elif st.session_state._source_uploaded_filename:
        st.caption(f"Loaded source file: {st.session_state._source_uploaded_filename}")


chunks = (
    split_into_chunks(
        st.session_state.raw_text,
        max_chars,
    )
    if st.session_state.raw_text
    else []
)

if chunks:
    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Characters",
        f"{len(st.session_state.raw_text):,}",
    )

    c2.metric(
        "Chunks",
        len(chunks),
    )

    c3.metric(
        "Processed results",
        len(st.session_state.results),
    )

    with st.expander("Preview extracted text"):
        st.write(st.session_state.raw_text[:6000])


# -----------------------------
# Processing
# -----------------------------

st.subheader("2. Generate Spanish study view")

start_index = st.number_input(
    "Start at chunk",
    min_value=1,
    max_value=max(
        len(chunks),
        1,
    ),
    value=min(max(1, int(st.session_state.start_chunk)), max(len(chunks), 1)),
    key="start_chunk",
)

# Chunk preview — show content boundaries so the user knows what they're selecting
if chunks:
    _prev_start = int(start_index) - 1
    _prev_chunks = chunks[_prev_start : _prev_start + chunks_to_process]
    _chunk_label = "chunk" if len(_prev_chunks) == 1 else "chunks"
    with st.expander(
        f"Preview selected {_chunk_label} ({len(_prev_chunks)} of {len(chunks)})",
        expanded=True,
    ):
        for _pi, _pc in enumerate(_prev_chunks, start=_prev_start + 1):
            _pc_chars = len(_pc)
            _pc_words = len(_pc.split())
            st.markdown(
                f"**Chunk {_pi}** &nbsp;"
                f'<span class="small-muted">{_pc_chars:,} chars · {_pc_words:,} words</span>',
                unsafe_allow_html=True,
            )
            _limit = 220
            if _pc_chars <= _limit * 2 + 60:
                # Short enough to show in full
                st.caption(_pc)
            else:
                _col_a, _col_b = st.columns(2)
                with _col_a:
                    st.markdown(
                        '<span class="small-muted">▶ Beginning</span>',
                        unsafe_allow_html=True,
                    )
                    # Break cleanly at a word boundary
                    _head = _pc[:_limit].rsplit(" ", 1)[0]
                    st.caption(_head + " …")
                with _col_b:
                    st.markdown(
                        '<span class="small-muted">⏹ End</span>',
                        unsafe_allow_html=True,
                    )
                    _tail = _pc[-_limit:].split(" ", 1)[-1]
                    st.caption("… " + _tail)
            if _pi < _prev_start + len(_prev_chunks):
                st.divider()

_translate_clicked = st.button(
    "Translate selected chunks",
    type="primary",
    disabled=not chunks or not ok,
)

_checker_changed_translation = False

if _translate_clicked:
    _invalidate_export_caches()

    start = int(start_index) - 1
    selected_chunks = chunks[start : start + chunks_to_process]
    failed_chunks = []
    _n_chunks = len(selected_chunks)
    _workflow_started = time.monotonic()
    _progress_bar = StickyProgress()
    _progress_bar.progress(0, text="Preparing translation workflow…")

    for idx, chunk in enumerate(selected_chunks, start=start + 1):
        _ci = idx - start  # 1-based position within the selected range
        _progress_bar.progress(
            _overall_progress(_ci, _n_chunks, 0.0),
            text=f"Chunk {_ci} of {_n_chunks}: preparing…",
        )
        _progress_bar.progress(
            _overall_progress(_ci, _n_chunks, 0.15),
            text=f"Chunk {_ci} of {_n_chunks}: translating…",
        )
        result = None
        _t_chunk_start = time.monotonic()

        with st.status(
            f"Translating chunk {idx}/{len(chunks)} with {selected_model}…",
            expanded=True,
        ) as _status:
            _status.write(f"Sending {len(chunk):,} characters to model…")
            _token_counter = st.empty()

            def _update_counter(n_chars: int) -> None:
                _token_counter.caption(f"Receiving… {n_chars:,} characters")

            _cache_key = (
                chunk, selected_model, level, style, region, fidelity,
                include_literal, include_vocab, include_grammar, temperature,
            )
            try:
                _cached = st.session_state._translation_cache.get(_cache_key)
                if _cached is not None:
                    result = _cached
                    _status.update(
                        label=f"Chunk {idx} — {len(result.pairs)} pair(s) (cached)",
                        state="complete",
                        expanded=False,
                    )
                else:
                    result = translate_chunk(
                        chunk=chunk,
                        level=level,
                        style=style,
                        region=region,
                        fidelity=fidelity,
                        include_literal=include_literal,
                        include_vocab=include_vocab,
                        include_grammar=include_grammar,
                        temperature=temperature,
                        on_token=_update_counter,
                        model=selected_model,
                    )
                    _token_counter.empty()
                    _elapsed = time.monotonic() - _t_chunk_start
                    _status.update(
                        label=(
                            f"Chunk {idx} — {len(result.pairs)} pair(s) "
                            f"in {_elapsed:.1f}s"
                        ),
                        state="complete",
                        expanded=False,
                    )
                    st.session_state._translation_cache.put(_cache_key, result)
                st.session_state.results.append(result)

            except requests.exceptions.ConnectionError as exc:
                _status.update(label=f"Chunk {idx} — connection error", state="error")
                st.error(
                    f"Cannot reach Ollama at **{OLLAMA_HOST}**. "
                    "Check that the container is running and the healthcheck is passing. "
                    f"Detail: {exc}"
                )
                failed_chunks.append(idx)

            except requests.exceptions.Timeout as exc:
                _status.update(label=f"Chunk {idx} — timeout", state="error")
                st.error(
                    f"Chunk {idx}: Ollama did not respond within the timeout. "
                    "Try reducing **Max characters per chunk** in the sidebar, "
                    "or switch to a faster model. "
                    f"Detail: {exc}"
                )
                failed_chunks.append(idx)

            except ValidationError as exc:
                _status.update(label=f"Chunk {idx} — parse error", state="error")
                st.error(
                    f"Chunk {idx}: the model returned unexpected output that could not "
                    "be parsed. Try lowering **Model temperature** (0.05–0.1) or "
                    "reducing chunk size. "
                    f"Detail: {exc}"
                )
                failed_chunks.append(idx)

            except Exception as exc:
                _status.update(label=f"Chunk {idx} — failed", state="error")
                st.error(f"Chunk {idx} failed: {exc}")
                failed_chunks.append(idx)

        # Run checker after successful translation (not on Streamlit rerenders).
        # Smart/strict can retranslate rejected pairs, then rerun so the updated
        # translation is visible in the same session.
        # Pairs are checked concurrently up to CHECKER_BATCH_SIZE workers.
        _eff_mode = checker_settings.mode if checker_settings.enabled else "off"
        _is_strict = (_eff_mode == "strict")
        _checker_changed_translation = False
        _run_checker = result is not None and _eff_mode not in ("off", "instant")

        
        # ── Render immediately only when no checker/correction pass will run ──
        # If checker is active, wait until after checker + retry/correction so
        # the side-by-side Spanish block always reflects the final corrected text.
        if result is not None and not _run_checker:
            st.markdown(
                f'<span class="small-muted">✓ Chunk {idx} of {len(selected_chunks)} translated</span>',
                unsafe_allow_html=True,
            )
            render_result_card(
                result,
                checker_results=st.session_state.checker_results,
                checker_settings=checker_settings,
                include_literal=include_literal,
                tts_lang=tts_lang,
                show_audio_controls=show_audio_controls,
            )


        if _run_checker:
            # fast mode is deterministic-only (no GPU calls): use det_workers.
            # smart / strict may invoke the LLM: respect llm_concurrency.
            _uses_llm = _eff_mode in ("smart", "strict")
            _batch = max(
                checker_settings.llm_concurrency if _uses_llm else checker_settings.det_workers,
                1,
            )
            _checker_started = time.monotonic()
            with st.status(
                f"Checking {len(result.pairs)} pair(s)…",
                expanded=False,
            ) as _chk_status:
                _cached_snapshot = dict(st.session_state.checker_results)
                _checked_pairs = 0
                with ThreadPoolExecutor(max_workers=_batch) as _pool:
                    _futures = [
                        _pool.submit(
                            check_pair,
                            settings=checker_settings,
                            english=_pair.english,
                            spanish=_pair.spanish,
                            literal_spanish=_pair.literal_spanish,
                            pair_index=_pidx,
                            cached_results=_cached_snapshot,
                        )
                        for _pidx, _pair in enumerate(result.pairs)
                    ]
                    for _future in as_completed(_futures):
                        try:
                            _ck, _cr = _future.result()
                            st.session_state.checker_results[_ck] = _cr
                            _checked_pairs += 1
                            _progress_bar.progress(
                                _overall_progress(
                                    _ci,
                                    _n_chunks,
                                    0.55 + (0.30 * _checked_pairs / max(len(result.pairs), 1)),
                                ),
                                text=(
                                    f"Chunk {_ci} of {_n_chunks}: checking pair "
                                    f"{_checked_pairs} of {len(result.pairs)}"
                                ),
                            )
                        except Exception as _fut_exc:
                            logger.warning(
                                "Checker worker raised an exception: %s", _fut_exc
                            )
                _chk_status.update(
                    label=f"Checked {len(result.pairs)} pair(s){_elapsed_suffix(_checker_started)}",
                    state="complete",
                )

            _progress_bar.progress(
                _overall_progress(_ci, _n_chunks, 0.85),
                text=f"Chunk {_ci} of {_n_chunks}: checker complete",
            )

        # Retranslate pairs flagged by the checker.
        if result is not None and _eff_mode in ("fast", "smart", "strict"):
            _pairs_to_retry = []
            for _pidx, _pair in enumerate(result.pairs):
                _ck = make_check_key(
                    checker_settings,
                    _pair.english,
                    _pair.spanish,
                    _pair.literal_spanish,
                )
                _cr = st.session_state.checker_results.get(_ck)
                if _cr is not None and should_retry_translation(checker_settings, _cr):
                    _pairs_to_retry.append((_pidx, _pair, _ck, _cr))

            if _pairs_to_retry:
                _retry_started = time.monotonic()
                with st.status(
                    f"Correcting {len(_pairs_to_retry)} failed pair(s)…",
                    expanded=True,
                ) as _retry_status:
                    _retry_success = 0
                    for _pidx, _pair, _old_ck, _cr in _pairs_to_retry:
                        _retry_status.write(
                            f"Pair {_pidx + 1}: {_cr.user_facing_summary}"
                        )
                        try:
                            _original_spanish = _pair.spanish
                            _used_checker_correction = False

                            # Prefer the checker's explicit corrected Spanish when available.
                            # This avoids losing a valid correction if the separate retry call
                            # fails, times out, or returns invalid JSON.
                            if _cr.corrected_spanish and _cr.corrected_spanish.strip():
                                _direct_fix = _strip_ellipsis(_cr.corrected_spanish)

                                if _direct_fix:
                                    _new_pair = _pair.model_copy(deep=True)
                                    _new_pair.spanish = _direct_fix
                                    _used_checker_correction = True
                                else:
                                    _new_pair = retranslate_pair(
                                        pair=_pair,
                                        check_result=_cr,
                                        level=level,
                                        style=style,
                                        region=region,
                                        fidelity=fidelity,
                                        include_literal=include_literal,
                                        include_vocab=include_vocab,
                                        include_grammar=include_grammar,
                                        temperature=temperature,
                                        model=selected_model,
                                        corrected_spanish_hint="",
                                    )
                            else:
                                _new_pair = retranslate_pair(
                                    pair=_pair,
                                    check_result=_cr,
                                    level=level,
                                    style=style,
                                    region=region,
                                    fidelity=fidelity,
                                    include_literal=include_literal,
                                    include_vocab=include_vocab,
                                    include_grammar=include_grammar,
                                    temperature=temperature,
                                    model=selected_model,
                                    corrected_spanish_hint="",
                                )

                            result.pairs[_pidx] = _new_pair
                            _checker_changed_translation = True
                            # Remove stale checker result and re-check the corrected pair.
                            st.session_state.checker_results.pop(_old_ck, None)
                            _new_ck, _new_cr = check_pair(
                                settings=checker_settings,
                                english=_new_pair.english,
                                spanish=_new_pair.spanish,
                                literal_spanish=_new_pair.literal_spanish,
                                pair_index=_pidx,
                                cached_results={},  # force fresh check
                            )
                            st.session_state.checker_results[_new_ck] = _new_cr

                            _changed_spanish = (
                                _norm_text_for_compare(_new_pair.spanish)
                                != _norm_text_for_compare(_original_spanish)
                            )
                            # Treat only blocking failures as unresolved.
                            # Minor warning/info follow-up suggestions should not
                            # surface as a failed correction attempt in the UI.
                            _blocking_unresolved = (
                                _new_cr.severity == "fail" or not _new_cr.passed
                            )

                            if _changed_spanish:
                                _new_pair.corrected_by_checker = True
                                _new_pair.correction_note = (
                                    "Applied checker correction"
                                    if _used_checker_correction
                                    else "Updated after checker retry"
                                )
                                _new_pair.original_spanish_before_correction = _original_spanish
                                
                                _new_pair.correction_reason = (
                                    _cr.user_facing_summary
                                    or _cr.recommended_action
                                    or ""
                                )

                            if _changed_spanish and not _blocking_unresolved:
                                _retry_success += 1
                            elif _blocking_unresolved:
                                _retry_status.write(
                                    "  ⚠️ Correction did not fully resolve this pair; "
                                    "the latest checked output is still shown."
                                )
                        except Exception as _retry_exc:
                            logger.warning(
                                "Retranslation failed for pair %d: %s",
                                _pidx,
                                _retry_exc,
                            )
                            _retry_status.write(
                                f"  ⚠️ Correction attempt failed: {_retry_exc}"
                            )
                    _retry_label = (
                        f"Corrected {_retry_success}/{len(_pairs_to_retry)} pair(s)"
                        if _retry_success < len(_pairs_to_retry)
                        else f"Corrected {_retry_success} pair(s)"
                    )
                    _retry_status.update(
                        label=f"{_retry_label}{_elapsed_suffix(_retry_started)}",
                        state="complete",
                    )


                if _checker_changed_translation:
                    st.session_state._cached_markdown_key = None
                    st.session_state._cached_markdown = None


                _progress_bar.progress(
                    _overall_progress(_ci, _n_chunks, 0.95),
                    text=f"Chunk {_ci} of {_n_chunks}: corrections complete",
                )
            else:
                _progress_bar.progress(
                    _overall_progress(_ci, _n_chunks, 0.95),
                    text=f"Chunk {_ci} of {_n_chunks}: no corrections needed",
                )

        
        # ── Render checked modes after checker + retry/correction ─────────────
        # column uses result.pairs after any checker-applied        # This covers fast, smart, and strict. Rendering here ensures the
        # correction or retranslation.
        if result is not None and _run_checker:
            st.markdown(
                f'<span class="small-muted">✓ Chunk {idx} of {len(selected_chunks)} checked</span>',
                unsafe_allow_html=True,
            )
            render_result_card(
                result,
                checker_results=st.session_state.checker_results,
                checker_settings=checker_settings,
                include_literal=include_literal,
                tts_lang=tts_lang,
                show_audio_controls=show_audio_controls,
            )

        _progress_bar.progress(
            _overall_progress(_ci, _n_chunks, 1.0),
            text=f"Chunk {_ci} of {_n_chunks}: complete",
        )

    _progress_bar.progress(1.0, text=f"All {_n_chunks} chunk(s) complete ✓")
    _workflow_elapsed = time.monotonic() - _workflow_started

    if failed_chunks:
        st.warning(
            f"{len(failed_chunks)} chunk(s) failed: {failed_chunks}. "
            f"You can retry or continue viewing successful results. Workflow time: {_workflow_elapsed:.1f}s."
        )
    else:
        st.caption(f"Workflow time: {_workflow_elapsed:.1f}s")

    if st.session_state.results and len(failed_chunks) < _n_chunks:
        _save_history_snapshot(
            history_store,
            input_mode=input_mode,
            selected_model=selected_model,
            level=level,
            region=region,
            style=style,
            fidelity=fidelity,
            temperature=temperature,
            max_chars=max_chars,
            start_chunk=int(start_index),
            chunks_to_process=chunks_to_process,
            include_literal=include_literal,
            include_vocab=include_vocab,
            include_grammar=include_grammar,
            skip_enrichments=_skip_enrichments,
            show_audio_controls=show_audio_controls,
            checker_settings=checker_settings,
        )



    # Force Streamlit into the normal post-translation render.
    # During the button-click render, the Study tabs are intentionally skipped
    # by `if st.session_state.results and not _translate_clicked`.
    # Without this rerun, Spanish First / Vocabulary / Export may not appear
    # until the user interacts with the app again.
    if st.session_state.results and len(failed_chunks) < _n_chunks:
        st.rerun()


# ── On-demand enrichment handler ─────────────────────────────────────────────
# Runs when the user clicked "⚡ Add vocab, grammar & literal Spanish" in the
# Study tab. Retranslates the chunk with all enrichments forced on, then
# updates the stored result in-place.
if (
    not _translate_clicked
    and st.session_state._enrich_idx is not None
    and st.session_state.results
):
    _eidx = st.session_state._enrich_idx
    if 0 <= _eidx < len(st.session_state.results):
        _eresult = st.session_state.results[_eidx]
        _echunk = "\n\n".join(p.english for p in _eresult.pairs)
        _enrich_started = time.monotonic()
        with st.spinner(
            f"Enriching chunk {_eidx + 1} of {len(st.session_state.results)}…"
        ):
            try:
                _enriched = translate_chunk(
                    chunk=_echunk,
                    level=level,
                    style=style,
                    region=region,
                    fidelity=fidelity,
                    include_literal=True,
                    include_vocab=True,
                    include_grammar=True,
                    temperature=temperature,
                    model=selected_model,
                )
                
                _merged_enriched = _preserve_checker_corrections(
                    _eresult,
                    _enriched,
                )

                # Remove stale checker results for the pre-enrichment pairs.
                if checker_settings.enabled and checker_settings.mode not in ("off", "instant"):
                    for _old_pair in _eresult.pairs:
                        _old_ck = make_check_key(
                            checker_settings,
                            _old_pair.english,
                            _old_pair.spanish,
                            _old_pair.literal_spanish,
                        )
                        st.session_state.checker_results.pop(_old_ck, None)

                    # Recheck enriched pairs so Study metrics/export readiness stay accurate.
                    for _pidx, _pair in enumerate(_merged_enriched.pairs):
                        _new_ck, _new_cr = check_pair(
                            settings=checker_settings,
                            english=_pair.english,
                            spanish=_pair.spanish,
                            literal_spanish=_pair.literal_spanish,
                            pair_index=_pidx,
                            cached_results={},
                        )
                        st.session_state.checker_results[_new_ck] = _new_cr

                st.session_state.results[_eidx] = _merged_enriched
                _invalidate_export_caches()
                _save_history_snapshot(
                    history_store,
                    input_mode=input_mode,
                    selected_model=selected_model,
                    level=level,
                    region=region,
                    style=style,
                    fidelity=fidelity,
                    temperature=temperature,
                    max_chars=max_chars,
                    start_chunk=int(start_index),
                    chunks_to_process=chunks_to_process,
                    include_literal=True,
                    include_vocab=True,
                    include_grammar=True,
                    skip_enrichments=False,
                    show_audio_controls=show_audio_controls,
                    checker_settings=checker_settings,
                )

            except Exception as _enrich_exc:
                st.error(f"Enrichment failed: {_enrich_exc}")
            else:
                st.success(
                    f"Enriched chunk {_eidx + 1} of {len(st.session_state.results)}"
                    f"{_elapsed_suffix(_enrich_started)}"
                )
    st.session_state._enrich_idx = None
    st.rerun()

# Study tabs — shown on rerenders after translation completes.
# Skipped on the translate-click render because chunks are rendered
# progressively inline above; Study tabs appear on the next interaction.
if st.session_state.results and not _translate_clicked:
    st.subheader("3. Study")

    _flat_records = flatten_pairs(st.session_state.results)
    _total_pairs = sum(len(r.pairs) for r in st.session_state.results)
    _total_vocab = sum(
        len(p.vocabulary)
        for r in st.session_state.results
        for p in r.pairs
    )
    _total_corrections = count_corrections(st.session_state.results)

    # Pre-compute checker keys once per render; reused in tab_reader, tab_export,
    # and the export-blocked check — avoids redundant json.dumps + SHA-256 calls.
    _pair_check_keys: dict[int, str] = {}
    if checker_settings.enabled and checker_settings.mode != "off":
        for _r in st.session_state.results:
            for _p in _r.pairs:
                _pair_check_keys[id(_p)] = make_check_key(
                    checker_settings,
                    _p.english,
                    _p.spanish,
                    _p.literal_spanish,
                )

    _checker_counts = collect_checker_status(
        st.session_state.results,
        st.session_state.checker_results,
        _pair_check_keys,
    )
    _blocked_export_items = build_blocked_export_items(
        st.session_state.results,
        st.session_state.checker_results,
        _pair_check_keys,
    )

    


    _checker_required_for_export = (
        checker_settings.require_pass
        and checker_settings.enabled
        and checker_settings.mode not in ("off", "instant")
    )

    _export_blocked = _checker_required_for_export and bool(_blocked_export_items)



    render_study_status(
        total_pairs=_total_pairs,
        total_vocab=_total_vocab,
        total_corrections=_total_corrections,
        checker_counts=_checker_counts,
        export_blocked=_export_blocked,
        checker_active=checker_settings.enabled and checker_settings.mode not in ("off", "instant"),
    )

    
    st.markdown('<div id="study-tabs"></div>', unsafe_allow_html=True)
    st.markdown(
        '<a class="floating-study-link" href="#study-tabs" title="Jump back to Study tabs">↥ Study tabs</a>',
        unsafe_allow_html=True,
    )

    _tab_labels = [
        f"📖 Parallel Reader ({_total_pairs})",
        f"🇪🇸 Spanish First ({_total_pairs})",
        f"🧠 Vocabulary ({_total_vocab})",
    ]
    if _total_corrections:
        _tab_labels.append(f"🩹 Corrections ({_total_corrections})")
    _tab_labels.append("⬇️ Export")

    _tabs = st.tabs(_tab_labels)
    tab_reader = _tabs[0]
    tab_spanish = _tabs[1]
    tab_vocab = _tabs[2]
    tab_corrections = _tabs[3] if _total_corrections else None
    tab_export = _tabs[-1]

    with tab_reader:
        _filter_cols = st.columns([2.4, 1, 1.2, 0.8])
        with _filter_cols[0]:
            _reader_search = st.text_input(
                "Search passages",
                key="reader_search",
                placeholder="Search English or Spanish text",
            )
        with _filter_cols[1]:
            _reader_corrected_only = st.checkbox(
                "Corrected only",
                key="reader_corrected_only",
            )
        with _filter_cols[2]:
            _reader_difficulty = st.multiselect(
                "Difficulty",
                list(_VALID_DIFFICULTIES),
                key="reader_difficulty_filter",
            )
        with _filter_cols[3]:
            st.write("")
            st.button("Clear", key="reader_clear_filters", on_click=_clear_reader_filters)

        _filtered_records = filter_pair_records(
            _flat_records,
            _reader_search,
            _reader_corrected_only,
            _reader_difficulty,
        )
        if len(_filtered_records) != len(_flat_records):
            st.caption(f"Showing {len(_filtered_records)} of {_total_pairs} passages.")

        if not _filtered_records:
            st.info("No passages matched the current filters.")
        elif not (_reader_search or _reader_corrected_only or _reader_difficulty):
            for _ridx, result in enumerate(st.session_state.results):
                render_result_card(
                    result,
                    checker_results=st.session_state.checker_results,
                    checker_settings=checker_settings,
                    include_literal=include_literal,
                    tts_lang=tts_lang,
                    show_audio_controls=show_audio_controls,
                    result_idx=_ridx,
                )
        else:
            _last_result_idx = None
            for record in _filtered_records:
                _result = record["result"]
                _pair = record["pair"]
                _result_idx = int(record["result_idx"])
                _pair_idx = int(record["pair_idx"])
                _global_passage_idx = int(record["global_passage_idx"])
                if not all(
                    hasattr(_pair, attr)
                    for attr in (
                        "english",
                        "spanish",
                        "literal_spanish",
                        "difficulty",
                        "corrected_by_checker",
                        "correction_note",
                        "grammar_notes",
                        "comprehension_question_spanish",
                    )
                ):
                    continue
                _result_title = getattr(_result, "title", "")
                if _result_idx != _last_result_idx and _result_title:
                    st.markdown(f"### {_result_title}")
                _render_pair_card(
                    _pair,
                    checker_results=st.session_state.checker_results,
                    checker_settings=checker_settings,
                    include_literal=include_literal,
                    tts_lang=tts_lang,
                    show_audio_controls=show_audio_controls,
                    passage_label=(
                        f"Passage {_global_passage_idx} "
                        f"(chunk {_result_idx + 1}, item {_pair_idx + 1})"
                    ),
                )
                _last_result_idx = _result_idx

    with tab_spanish:
        st.caption(
            "Read Spanish first, then reveal English when you need it."
        )
        _spanish_mode = st.radio(
            "Study mode",
            ["Full list", "Focus mode"],
            horizontal=True,
            key="spanish_first_mode",
        )

        if _spanish_mode == "Focus mode" and _flat_records:
            st.session_state._spanish_focus_idx = min(
                st.session_state._spanish_focus_idx,
                len(_flat_records) - 1,
            )
            _focus_record = _flat_records[st.session_state._spanish_focus_idx]
            _focus_result = _focus_record["result"]
            _focus_pair = _focus_record["pair"]
            _focus_global_idx = int(_focus_record["global_passage_idx"])

            if not all(
                hasattr(_focus_pair, attr)
                for attr in (
                    "english",
                    "spanish",
                    "corrected_by_checker",
                    "correction_note",
                )
            ):
                st.warning("Focus mode could not load the selected passage. Try Full list mode.")
            else:
                _nav_cols = st.columns([1, 1.2, 1])
                with _nav_cols[0]:
                    if st.button(
                        "Previous",
                        key="spanish_focus_prev",
                        disabled=st.session_state._spanish_focus_idx <= 0,
                    ):
                        st.session_state._spanish_focus_idx -= 1
                        st.rerun()
                with _nav_cols[1]:
                    st.caption(f"Passage {_focus_global_idx} of {_total_pairs}")
                with _nav_cols[2]:
                    if st.button(
                        "Next",
                        key="spanish_focus_next",
                        disabled=st.session_state._spanish_focus_idx >= len(_flat_records) - 1,
                    ):
                        st.session_state._spanish_focus_idx += 1
                        st.rerun()

                _focus_title = getattr(_focus_result, "title", "")
                if _focus_title:
                    st.markdown(f"### {_focus_title}")
                with st.container(border=True):
                    st.write(_focus_pair.spanish)
                    if _focus_pair.corrected_by_checker:
                        _note = _focus_pair.correction_note or "Updated after checker correction"
                        st.caption(f"✅ Corrected by checker: {_note}")
                    _maybe_render_tts_button(
                        _focus_pair.spanish,
                        lang=tts_lang,
                        show_audio_controls=show_audio_controls,
                    )
                    with st.expander("Reveal English"):
                        st.write(_focus_pair.english)
        else:
            for result in st.session_state.results:
                if result.title:
                    st.markdown(f"### {result.title}")
                for i, pair in enumerate(result.pairs, start=1):
                    st.markdown(
                        f"**Passage {i} / {len(result.pairs)}**",
                    )
                    st.write(pair.spanish)
                    if pair.corrected_by_checker:
                        _note = pair.correction_note or "Updated after checker correction"
                        st.caption(f"✅ Corrected by checker: {_note}")
                    _maybe_render_tts_button(
                        pair.spanish,
                        lang=tts_lang,
                        show_audio_controls=show_audio_controls,
                    )

                    with st.expander("Reveal English"):
                        st.write(pair.english)

    with tab_vocab:
        rows = [
            {
                "Spanish": vocab.spanish,
                "English": vocab.english,
                "Note": vocab.note,
            }
            for result in st.session_state.results
            for pair in result.pairs
            for vocab in pair.vocabulary
        ]

        if rows:
            df = pd.DataFrame(rows).drop_duplicates()

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download vocabulary CSV",
                df.to_csv(index=False).encode("utf-8"),
                "spanish_vocabulary.csv",
                "text/csv",
            )

        else:
            st.info(
                "No vocabulary generated yet. If you translated with "
                "'Skip enrichments' enabled, use the enrichments button in "
                "the Parallel Reader tab to add vocabulary."
            )

    if tab_corrections is not None:
        with tab_corrections:
            _render_corrections_tab(
                _flat_records,
                checker_results=st.session_state.checker_results,
                pair_check_keys=_pair_check_keys,
            )

    with tab_export:
        # Cache the markdown string; rebuild only when results or checker data change.
        
        _mk_cache_key = (
            tuple(
                (
                    pair.english,
                    pair.spanish,
                    pair.literal_spanish,
                    tuple((v.spanish, v.english, v.note) for v in pair.vocabulary),
                    tuple(pair.grammar_notes),
                    pair.comprehension_question_spanish,
                    pair.corrected_by_checker,
                    pair.correction_note,
                    pair.correction_reason,
                )
                for result in st.session_state.results
                for pair in result.pairs
            ),
            len(st.session_state.checker_results),
            include_literal,
            checker_settings.detailed_diagnostics,
        )

        if st.session_state._cached_markdown_key != _mk_cache_key:
            markdown = []

            for result in st.session_state.results:
                if result.title:
                    markdown.append(f"# {result.title}\n")

                if result.summary_english:
                    markdown.append(
                        f"**English summary:** {result.summary_english}\n"
                    )

                if result.summary_spanish:
                    markdown.append(
                        f"**Spanish summary:** {result.summary_spanish}\n"
                    )

                for pair in result.pairs:
                    
                    markdown += [
                        "---\n",
                        "## English\n",
                        pair.english + "\n",
                        "## Español\n",
                        pair.spanish + "\n",
                    ]

                    if pair.corrected_by_checker:
                        markdown += [
                            "> ✅ **Corrected by checker:** "
                            f"{pair.correction_note or 'Applied checker correction'}\n"
                        ]


                    if include_literal and pair.literal_spanish:
                        markdown += [
                            "### Literal Spanish\n",
                            pair.literal_spanish + "\n",
                        ]

                    if pair.grammar_notes:
                        markdown += ["### Grammar notes\n"]
                        markdown += [
                            f"- {note}"
                            for note in pair.grammar_notes
                        ]
                        markdown += [""]

                    if pair.vocabulary:
                        markdown += ["### Vocabulary\n"]
                        markdown += [
                            f"- **{vocab.spanish}** = {vocab.english}. {vocab.note}"
                            for vocab in pair.vocabulary
                        ]
                        markdown += [""]

                    if pair.comprehension_question_spanish:
                        markdown += [
                            "### Comprehension question\n",
                            pair.comprehension_question_spanish + "\n",
                        ]

                    # Append checker result block to markdown export
                    if checker_settings.enabled and checker_settings.mode != "off":
                        _ck = _pair_check_keys.get(id(pair))
                        _cr = st.session_state.checker_results.get(_ck) if _ck else None
                        if _cr is not None:
                            markdown += [
                                "\n",
                                checker_markdown_block(
                                    _cr, checker_settings.detailed_diagnostics
                                ),
                            ]

            st.session_state._cached_markdown = "\n".join(markdown)
            st.session_state._cached_markdown_key = _mk_cache_key
            
        if st.session_state.get("_cached_extra_exports_key") != _mk_cache_key:
            st.session_state._cached_bilingual_csv = build_bilingual_csv(st.session_state.results)
            st.session_state._cached_spanish_text = build_spanish_only_text(st.session_state.results)
            st.session_state._cached_anki_csv = build_anki_csv(st.session_state.results)
            st.session_state._cached_extra_exports_key = _mk_cache_key

        if _export_blocked:
            
            st.error(
                    "Export blocked: one or more pairs did not pass the checker. "
                    "Review the checker warnings in the Parallel Reader tab, or "
                    "disable 'Require checker pass before export' in the sidebar."
                )

            with st.expander("Blocked passages", expanded=True):
                for item in _blocked_export_items:
                    st.markdown(
                        f"- **{item['location']}** — {item['severity']} — {item['summary']}"
                    )

            st.info("You can still view all translations in the Reader tabs above.")

        else:
            st.download_button(
                "Download study notes Markdown",
                (st.session_state._cached_markdown or "").encode("utf-8"),
                "spanish_parallel_reader.md",
                "text/markdown",
            )

            _export_cols = st.columns(3)

            with _export_cols[0]:
                st.download_button(
                    "Download bilingual CSV",
                    st.session_state._cached_bilingual_csv,
                    "spanish_parallel_reader_bilingual.csv",
                    "text/csv",
                )

            with _export_cols[1]:
                st.download_button(
                    "Download Spanish text",
                    st.session_state._cached_spanish_text,
                    "spanish_parallel_reader_es.txt",
                    "text/plain",
                )

            with _export_cols[2]:
                st.download_button(
                    "Download Anki CSV",
                    st.session_state._cached_anki_csv,
                    "spanish_parallel_reader_anki.csv",
                    "text/csv",
                )


else:
    st.info("Add text and translate a chunk to see the study interface.")