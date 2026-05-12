import html as _html
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Literal

import fitz

import pandas as pd
import requests
import streamlit as st
from docx import Document
from pydantic import BaseModel, Field, ValidationError, field_validator

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

# Persistent HTTP session — reuses TCP connections across all Ollama calls.
_http_session = requests.Session()

# Pre-compiled regex patterns — avoids recompilation on every text operation.
_RE_PAGE_NUM = re.compile(r"\n\s*\d+\s*\n")
_RE_HSPACE = re.compile(r"[ \t]+")
_RE_NEWLINES = re.compile(r"\n{3,}")
_RE_SENTENCES = re.compile(r"(?<=[.!?])\s+")

Difficulty = Literal["A1", "A2", "B1", "B2", "C1", "C2"]


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
        padding-bottom: 3rem;
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

if "abort_requested" not in st.session_state:
    st.session_state.abort_requested = False

if "_translating" not in st.session_state:
    st.session_state._translating = False

if "_cached_markdown_key" not in st.session_state:
    st.session_state._cached_markdown_key = None
    st.session_state._cached_markdown = None

if "_translation_cache" not in st.session_state:
    # Maps (chunk, model, level, style, region, fidelity, flags, temp) → TranslationResponse.
    # Prevents re-running inference for identical inputs within a session.
    st.session_state._translation_cache = {}


# -----------------------------
# Text helpers
# -----------------------------

def set_source_text(text: str) -> None:
    cleaned = clean_text(text) if text else ""

    if cleaned != st.session_state.raw_text:
        st.session_state.raw_text = cleaned
        st.session_state.results = []
        st.session_state._translation_cache = {}

def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = _RE_PAGE_NUM.sub("\n", text)
    text = _RE_HSPACE.sub(" ", text)
    text = _RE_NEWLINES.sub("\n\n", text)
    return text.strip()


def tts_lang_from_region(region: str) -> str:
    """Map app translation preference to a Spanish TTS locale."""
    mapping = {
        "Neutral": "es-MX",
        "Latin American": "es-MX",
        "European / Spain": "es-ES",
    }
    return mapping.get(region, "es-MX")


def extract_pdf_text(uploaded_file) -> str:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    pages = []

    for page in doc:
        pages.append(page.get_text("text", sort=True))

    return clean_text("\n\n".join(pages))


def extract_docx_text(uploaded_file) -> str:
    doc = Document(uploaded_file)
    paragraphs = [
        paragraph.text.strip()
        for paragraph in doc.paragraphs
        if paragraph.text.strip()
    ]
    return clean_text("\n\n".join(paragraphs))


def extract_plain_text(uploaded_file) -> str:
    return clean_text(
        uploaded_file.read().decode("utf-8", errors="ignore")
    )


@st.cache_data
def split_into_chunks(text: str, max_chars: int) -> List[str]:
    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            parts = _RE_SENTENCES.split(paragraph)
        else:
            parts = [paragraph]

        for part in parts:
            if len(current) + len(part) + 2 <= max_chars:
                current = (
                    current + "\n\n" + part
                ).strip() if current else part
            else:
                if current:
                    chunks.append(current)
                current = part

    if current:
        chunks.append(current)

    return chunks


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
    """Pre-load the model into Ollama memory with a 1-token request.

    @st.cache_resource ensures this runs at most once per model per app
    restart.  Executes in a daemon thread so it never blocks the UI.
    """
    import threading

    def _ping() -> None:
        try:
            _http_session.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "keep_alive": OLLAMA_KEEP_ALIVE,
                    "options": {"num_predict": 1},
                },
                timeout=30,
            )
            logger.info("warmup_model: %s pre-loaded", model)
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

    num_predict = _estimate_num_predict(
        len(chunk), include_literal, include_vocab, include_grammar
    )

    _model = model or OLLAMA_MODEL
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
    content = ""
    t_first_token: float | None = None
    _last_on_token_len = 0

    try:
        with _http_session.post(
            f"{OLLAMA_HOST}/api/chat",
            json=payload,
            timeout=OLLAMA_REQUEST_TIMEOUT,
            stream=True,
        ) as response:
            if not response.ok:
                try:
                    detail = response.json().get("error", response.text)
                except Exception:
                    detail = response.text
                raise requests.exceptions.HTTPError(
                    f"{response.status_code} {response.reason} — Ollama said: {detail}",
                    response=response,
                )
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                try:
                    chunk_data = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                token = chunk_data.get("message", {}).get("content", "")
                if token and t_first_token is None:
                    t_first_token = time.monotonic() - t0
                content += token
                if on_token is not None and len(content) - _last_on_token_len >= 150:
                    on_token(len(content))
                    _last_on_token_len = len(content)
                if chunk_data.get("done"):
                    break
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
        else:
            logger.error(
                "TranslationResponse parse failed — no JSON object found. "
                "Raw (first 300 chars): %s",
                content[:300],
            )
            raise

    # Post-process: fix summary_english if the model wrote it in Spanish.
    result.summary_english = _fix_english_summary(result.summary_english, chunk)

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
) -> int:
    """Adaptive num_predict budget based on chunk size and enabled features."""
    base = min(chunk_len * 3, 4000)
    if include_literal:
        base += 600
    if include_vocab:
        base += 800
    if include_grammar:
        base += 600
    # Hard cap: smaller models (3b) cannot reliably generate very long JSON and
    # will ramble to the token limit if given too much budget, producing ~3× the
    # input length as unparseable output.  5000 tokens (~15 k chars) is
    # sufficient for any realistic chunk and keeps 3b output bounded.
    return min(max(base, 800), 5000)


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
        len(pair.english), include_literal, include_vocab, include_grammar
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

    resp = _http_session.post(
        f"{OLLAMA_HOST}/api/chat",
        json=payload,
        timeout=OLLAMA_REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    content = resp.json().get("message", {}).get("content", "")

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


def _render_checker_details(
    result: PairCheckResult,
    detailed: bool,
    tts_lang: str,
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
            render_tts_button(result.corrected_spanish, lang=tts_lang)

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


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:
    st.header("Settings")

    st.write("**Model**")
    _default_idx = (
        AVAILABLE_OLLAMA_MODELS.index(OLLAMA_MODEL)
        if OLLAMA_MODEL in AVAILABLE_OLLAMA_MODELS
        else 0
    )
    selected_model = st.selectbox(
        "Ollama model",
        AVAILABLE_OLLAMA_MODELS,
        index=_default_idx,
        help="qwen2.5:3b is fastest on CPU. qwen2.5:7b is the default. qwen2.5:14b is highest quality.",
    )

    ok, status = check_ollama(selected_model)

    if ok:
        st.success(status)
        warmup_model(selected_model)
    else:
        st.warning(status)

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
        index=2,
    )

    region = st.selectbox(
        "Spanish preference",
        [
            "Neutral",
            "Latin American",
            "European / Spain",
        ],
        index=0,
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
        index=1,
    )

    fidelity = st.selectbox(
        "Translation fidelity",
        [
            "Balanced",
            "Closest meaning",
            "Simpler learner wording",
            "Preserve literary style",
        ],
        index=0,
    )

    max_chars = st.slider(
        "Max characters per chunk",
        800,
        4000,
        DEFAULT_MAX_CHARS,
        step=100,
    )

    chunks_to_process = st.slider(
        "Chunks to process",
        1,
        10,
        2,
    )

    temperature = st.slider(
        "Model temperature",
        0.0,
        0.3,
        OLLAMA_TEMPERATURE,
        step=0.05,
        help="Lower = more consistent JSON output. Keep below 0.3 for reliable structured translation.",
    )

    include_literal = st.checkbox(
        "Include literal Spanish",
        value=True,
    )

    include_vocab = st.checkbox(
        "Include vocabulary",
        value=True,
    )

    include_grammar = st.checkbox(
        "Include grammar notes",
        value=True,
    )

    with st.expander("🔍 Output Checker"):
        # Derive sidebar defaults from env vars so Docker/local overrides take effect
        # on first render. Subsequent renders use Streamlit's widget session state.
        _env_checker_enabled = os.getenv("CHECKER_ENABLED", "true").strip().lower() not in (
            "false", "0", "no", "off"
        )
        _env_checker_mode = os.getenv("CHECKER_MODE", "smart").strip().lower()
        _checker_mode_options = ["off", "fast", "smart", "strict"]
        _env_checker_mode_idx = (
            _checker_mode_options.index(_env_checker_mode)
            if _env_checker_mode in _checker_mode_options
            else 2
        )
        _env_checker_llm = os.getenv("CHECKER_LLM_ENABLED", "true").strip().lower() not in (
            "false", "0", "no", "off"
        )
        _env_checker_require_pass = os.getenv("CHECKER_REQUIRE_PASS", "false").strip().lower() not in (
            "false", "0", "no", "off"
        )
        _env_checker_detailed = os.getenv("CHECKER_DETAILED_DIAGNOSTICS", "false").strip().lower() not in (
            "false", "0", "no", "off"
        )

        checker_enabled_ui = st.checkbox(
            "Enable output checker",
            value=_env_checker_enabled,
        )
        checker_mode_ui = st.selectbox(
            "Checker mode",
            _checker_mode_options,
            index=_env_checker_mode_idx,
            help=(
                "off: no checks. "
                "fast: deterministic checks only (no extra model calls). "
                "smart: deterministic + LLM for risky/sampled pairs. "
                "strict: deterministic + LLM for every pair."
            ),
        )
        checker_model_ui = st.text_input(
            "Checker model",
            value="",
            placeholder="defaults to translation model",
            help="Leave blank to use the same model as translation. Set CHECKER_MODEL in .env to make it persistent.",
        )
        checker_require_pass_ui = st.checkbox(
            "Require checker pass before export",
            value=_env_checker_require_pass,
            help="Block Markdown export for pairs that fail the checker.",
        )
        checker_llm_ui = st.checkbox(
            "LLM checker enabled",
            value=_env_checker_llm,
            help="Uncheck to use deterministic checks only (no additional model calls).",
        )
        checker_detailed_ui = st.checkbox(
            "Show detailed diagnostics",
            value=_env_checker_detailed,
            help="Show per-issue breakdown. Keep off for a faster, cleaner UI.",
        )

    if st.button("Clear session"):
        set_source_text("")
        st.session_state.checker_results = {}
        st.session_state._translation_cache = {}
        st.session_state.abort_requested = False
        st.session_state._translating = False
        st.session_state._cached_markdown_key = None
        st.session_state._cached_markdown = None
        st.session_state.pop("_last_upload_key", None)
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
    )

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
                with st.spinner(f"Extracting {suffix.upper()} ({size_mb:.1f} MB)…"):
                    if suffix == "pdf":
                        set_source_text(extract_pdf_text(uploaded))
                    elif suffix == "docx":
                        set_source_text(extract_docx_text(uploaded))
                    else:
                        set_source_text(extract_plain_text(uploaded))
                st.session_state["_last_upload_key"] = _upload_key
                st.success(f"Extracted {len(st.session_state.raw_text):,} characters")

            except Exception as exc:
                st.error(f"Could not extract text: {exc}")
        else:
            st.success(f"Extracted {len(st.session_state.raw_text):,} characters")


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
    value=1,
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

_btn_col, _stop_col = st.columns([5, 1])
_translate_clicked = _btn_col.button(
    "Translate selected chunks",
    type="primary",
    disabled=not chunks or not ok,
)
_stop_clicked = _stop_col.button(
    "⏹ Stop",
    disabled=not st.session_state._translating,
    help="Stops translation after the current chunk finishes.",
)
if _stop_clicked:
    st.session_state.abort_requested = True

if _translate_clicked:
    st.session_state.abort_requested = False
    st.session_state._translating = True
    st.session_state._cached_markdown_key = None  # invalidate cached export

    start = int(start_index) - 1
    selected_chunks = chunks[start : start + chunks_to_process]
    failed_chunks = []

    for idx, chunk in enumerate(selected_chunks, start=start + 1):
        if st.session_state.abort_requested:
            st.warning(f"Translation stopped by user after chunk {idx - 1}.")
            break

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
                    st.session_state._translation_cache[_cache_key] = result
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
        # Pairs are checked concurrently up to CHECKER_BATCH_SIZE workers.
        if result is not None and checker_settings.enabled and checker_settings.mode != "off":
            _batch = max(checker_settings.batch_size, 1)
            with st.status(
                f"Checking {len(result.pairs)} pair(s)…",
                expanded=False,
            ) as _chk_status:
                _cached_snapshot = dict(st.session_state.checker_results)
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
                        except Exception as _fut_exc:
                            logger.warning(
                                "Checker worker raised an exception: %s", _fut_exc
                            )
                _chk_status.update(
                    label=f"Checked {len(result.pairs)} pair(s)",
                    state="complete",
                )

        # Retranslate any pairs the checker marked as failed.
        # Only one retry attempt per pair to avoid infinite loops.
        if result is not None and checker_settings.enabled and checker_settings.mode != "off":
            _pairs_to_retry = []
            for _pidx, _pair in enumerate(result.pairs):
                _ck = make_check_key(
                    checker_settings,
                    _pair.english,
                    _pair.spanish,
                    _pair.literal_spanish,
                )
                _cr = st.session_state.checker_results.get(_ck)
                if _cr is not None and _cr.severity == "fail":
                    _pairs_to_retry.append((_pidx, _pair, _ck, _cr))

            if _pairs_to_retry:
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
                                corrected_spanish_hint=_cr.corrected_spanish,
                            )
                            result.pairs[_pidx] = _new_pair
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
                            _retry_success += 1
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
                        label=_retry_label,
                        state="complete",
                    )

    st.session_state._translating = False

    if failed_chunks:
        st.warning(
            f"{len(failed_chunks)} chunk(s) failed: {failed_chunks}. "
            "You can retry or continue viewing successful results."
        )


# -----------------------------
# Study tabs
# -----------------------------

if st.session_state.results:
    st.subheader("3. Study")

    _total_pairs = sum(len(r.pairs) for r in st.session_state.results)
    _total_vocab = sum(
        len(p.vocabulary)
        for r in st.session_state.results
        for p in r.pairs
    )

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

    tab_reader, tab_spanish, tab_vocab, tab_export = st.tabs(
        [
            f"📖 Parallel Reader ({_total_pairs})",
            f"🇪🇸 Spanish First ({_total_pairs})",
            f"🧠 Vocabulary ({_total_vocab})",
            "⬇️ Export",
        ]
    )

    with tab_reader:
        for result in st.session_state.results:
            if result.title:
                st.markdown(f"### {result.title}")

            if result.summary_english or result.summary_spanish:
                with st.expander("Summary"):
                    if result.summary_english:
                        st.markdown("**English summary**")
                        st.write(result.summary_english)

                    if result.summary_spanish:
                        st.markdown("**Spanish summary**")
                        st.write(result.summary_spanish)
                        render_tts_button(result.summary_spanish, lang=tts_lang)

            for pair in result.pairs:
                with st.container(border=True):
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
                        st.markdown(
                            f'<span style="background:#e0f2fe;color:#0369a1;'
                            f'padding:2px 8px;border-radius:4px;'
                            f'font-size:0.82rem;font-weight:600;">'
                            f'{pair.difficulty}</span>',
                            unsafe_allow_html=True,
                        )
                        render_tts_button(pair.spanish, lang=tts_lang)

                    if include_literal and pair.literal_spanish:
                        with st.expander("Literal Spanish"):
                            st.write(pair.literal_spanish)

                    if pair.grammar_notes:
                        with st.expander("Grammar notes"):
                            st.markdown(
                                "\n".join(f"- {note}" for note in pair.grammar_notes)
                            )

                    if pair.comprehension_question_spanish:
                        with st.expander("Comprehension question"):
                            st.write(pair.comprehension_question_spanish)

                    # Checker result for this pair (only from session_state — no new call)
                    if checker_settings.enabled and checker_settings.mode != "off":
                        _ck = _pair_check_keys.get(id(pair))
                        _cr = st.session_state.checker_results.get(_ck) if _ck else None
                        if _cr is not None:
                            _render_checker_details(
                                _cr,
                                checker_settings.detailed_diagnostics,
                                tts_lang,
                            )

    with tab_spanish:
        st.caption(
            "Read Spanish first, then reveal English when you need it."
        )

        for result in st.session_state.results:
            if result.title:
                st.markdown(f"### {result.title}")
            for i, pair in enumerate(result.pairs, start=1):
                st.markdown(
                    f"**Passage {i} / {len(result.pairs)}**",
                )
                st.write(pair.spanish)
                render_tts_button(pair.spanish, lang=tts_lang)

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
            st.info("No vocabulary generated yet.")

    with tab_export:
        # Cache the markdown string; rebuild only when results or checker data change.
        _mk_cache_key = (
            len(st.session_state.results),
            len(st.session_state.checker_results),
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

        # Determine if export should be blocked
        _export_blocked = checker_settings.require_pass and any(
            not st.session_state.checker_results.get(
                _pair_check_keys.get(id(pair)),
                PairCheckResult(),
            ).passed
            for result in st.session_state.results
            for pair in result.pairs
        )

        if _export_blocked:
            st.error(
                "Export blocked: one or more pairs did not pass the checker. "
                "Review the checker warnings in the Parallel Reader tab, or "
                "disable 'Require checker pass before export' in the sidebar."
            )
            st.info("You can still view all translations in the Reader tabs above.")
        else:
            st.download_button(
                "Download study notes Markdown",
                (st.session_state._cached_markdown or "").encode("utf-8"),
                "spanish_parallel_reader.md",
                "text/markdown",
            )

else:
    st.info("Add text and translate a chunk to see the study interface.")