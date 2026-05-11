"""
checker.py — Layered output quality checker for Spanish Parallel Reader.

Architecture (three layers):
  Layer 1: Deterministic checks — always run, fast Python only, no model call.
  Layer 2: Risk scoring — assigns a 0–1 risk score from Layer 1 findings.
  Layer 3: LLM-based checker — Ollama call, only when needed (mode/risk/sampling).

Performance principles:
  - Results are cached by deterministic hash key in Streamlit session_state.
  - No checker code runs on normal Streamlit rerenders — only inside the
    translate button handler (generation event).
  - Layer 3 is skipped in fast mode unless severity is "fail".
  - Layer 3 is skipped in smart mode for low-risk pairs below the sampling rate.
  - Inputs are truncated to CHECKER_MAX_CHARS before any LLM call.
  - Checker calls use CHECKER_TIMEOUT_SECONDS so they cannot stall the UI.
  - Checker failures are open: the generated translation is always preserved.
  - All data stays local — only the configured Ollama endpoint is used.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import requests
from pydantic import BaseModel, Field

# Persistent HTTP session — reuses TCP connections across all checker calls.
_http_session = requests.Session()

# ── Version stamp — bump to invalidate all cached results ──────────────────────
CHECKER_PROMPT_VERSION = "v1"

# ── Language signal word sets (common function words only) ─────────────────────
# These are intentionally conservative: common, unambiguous tokens per language.
_SPANISH_SIGNALS: frozenset = frozenset([
    "el", "la", "los", "las", "en", "de", "que", "es", "un", "una",
    "del", "se", "con", "por", "para", "como", "más", "pero", "su",
    "lo", "le", "les", "al", "hay", "son", "está", "están", "fue",
    "ser", "también", "esto", "este", "esta", "estas", "estos",
    "yo", "tú", "él", "ella", "nosotros", "ellos", "usted",
    "muy", "bien", "ya", "así", "cuando", "donde", "aunque",
])

_ENGLISH_SIGNALS: frozenset = frozenset([
    "the", "and", "is", "of", "to", "a", "in", "that", "it", "was",
    "he", "she", "they", "we", "you", "are", "have", "with", "this",
    "at", "be", "from", "or", "an", "will", "all", "there", "their",
    "what", "so", "if", "about", "which", "when", "do", "how",
    "not", "but", "by", "on", "can", "been", "were", "has",
])


# ──────────────────────────────────────────────────────────────────────────────
# Settings
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CheckerSettings:
    enabled: bool
    mode: str            # off | fast | smart | strict
    model: str
    sample_rate: float
    max_chars: int
    require_pass: bool
    timeout_seconds: float
    cache_enabled: bool
    detailed_diagnostics: bool
    llm_enabled: bool
    batch_size: int
    ollama_host: str


def get_checker_settings(
    *,
    ollama_host: Optional[str] = None,
    ollama_model: Optional[str] = None,
    # Sidebar overrides (None = read from env)
    enabled_override: Optional[bool] = None,
    mode_override: Optional[str] = None,
    model_override: Optional[str] = None,
    require_pass_override: Optional[bool] = None,
    detailed_diagnostics_override: Optional[bool] = None,
    llm_enabled_override: Optional[bool] = None,
    sample_rate_override: Optional[float] = None,
    max_chars_override: Optional[int] = None,
) -> CheckerSettings:
    """Build CheckerSettings from environment variables, with optional UI overrides."""

    def _bool(key: str, default: bool) -> bool:
        v = os.getenv(key, "").strip().lower()
        if not v:
            return default
        return v not in ("false", "0", "no", "off")

    base_model = ollama_model or os.getenv("OLLAMA_MODEL", "aya-expanse:8b")
    checker_model_env = os.getenv("CHECKER_MODEL", "").strip()
    checker_model = checker_model_env or base_model

    return CheckerSettings(
        enabled=(
            enabled_override
            if enabled_override is not None
            else _bool("CHECKER_ENABLED", True)
        ),
        mode=(
            (mode_override or os.getenv("CHECKER_MODE", "smart")).strip().lower()
        ),
        model=model_override or checker_model,
        sample_rate=(
            sample_rate_override
            if sample_rate_override is not None
            else float(os.getenv("CHECKER_SAMPLE_RATE", "1.0"))
        ),
        max_chars=(
            max_chars_override
            if max_chars_override is not None
            else int(os.getenv("CHECKER_MAX_CHARS", "2500"))
        ),
        require_pass=(
            require_pass_override
            if require_pass_override is not None
            else _bool("CHECKER_REQUIRE_PASS", False)
        ),
        timeout_seconds=float(os.getenv("CHECKER_TIMEOUT_SECONDS", "45")),
        cache_enabled=_bool("CHECKER_CACHE_ENABLED", True),
        detailed_diagnostics=(
            detailed_diagnostics_override
            if detailed_diagnostics_override is not None
            else _bool("CHECKER_DETAILED_DIAGNOSTICS", False)
        ),
        llm_enabled=(
            llm_enabled_override
            if llm_enabled_override is not None
            else _bool("CHECKER_LLM_ENABLED", True)
        ),
        batch_size=int(os.getenv("CHECKER_BATCH_SIZE", "1")),
        ollama_host=(
            ollama_host or os.getenv("OLLAMA_HOST", "http://ollama:11434")
        ).rstrip("/"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic result schema
# ──────────────────────────────────────────────────────────────────────────────

class PairCheckResult(BaseModel):
    """Result of checking one ReadingPair."""

    passed: bool = True
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    severity: str = "pass"  # pass | info | warning | fail
    faithfulness_issues: List[str] = Field(default_factory=list)
    hallucination_issues: List[str] = Field(default_factory=list)
    omission_issues: List[str] = Field(default_factory=list)
    label_issues: List[str] = Field(default_factory=list)
    language_quality_issues: List[str] = Field(default_factory=list)
    unsupported_claims: List[str] = Field(default_factory=list)
    recommended_action: str = ""
    user_facing_summary: str = ""
    checked_with_llm: bool = False
    deterministic_only: bool = True
    truncated: bool = False
    cache_hit: bool = False
    checker_latency_ms: Optional[float] = None


# ──────────────────────────────────────────────────────────────────────────────
# Cache key
# ──────────────────────────────────────────────────────────────────────────────

def make_check_key(
    settings: CheckerSettings,
    english: str,
    spanish: str,
    literal_spanish: str = "",
    vocab_json: str = "",
    grammar_json: str = "",
) -> str:
    """
    Deterministic SHA-256 key for (checker config + content).
    Stable for identical inputs; changes when any input changes.
    """
    payload = json.dumps(
        [
            CHECKER_PROMPT_VERSION,
            settings.model,
            settings.mode,
            english,
            spanish,
            literal_spanish,
            vocab_json,
            grammar_json,
        ],
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# Input truncation
# ──────────────────────────────────────────────────────────────────────────────

def truncate_for_checker(text: str, max_chars: int) -> Tuple[str, bool]:
    """Return (possibly_truncated_text, was_truncated)."""
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars] + "\u2026", True


# ──────────────────────────────────────────────────────────────────────────────
# Internal heuristics
# ──────────────────────────────────────────────────────────────────────────────

def _lang_score(text: str) -> Dict[str, float]:
    """
    Returns {'spanish': float, 'english': float} — fraction of lowercase tokens
    that are strong signals for each language. Approximate only.
    """
    tokens = re.findall(r"\b[a-z\u00e0-\u00ff]+\b", text.lower())
    if not tokens:
        return {"spanish": 0.0, "english": 0.0}
    n = len(tokens)
    sp = sum(1 for t in tokens if t in _SPANISH_SIGNALS)
    en = sum(1 for t in tokens if t in _ENGLISH_SIGNALS)
    return {"spanish": sp / n, "english": en / n}


def _number_set(text: str) -> frozenset:
    """Extract normalised numeric strings from text."""
    raw = re.findall(r"\b\d+(?:[.,]\d+)*\b", text)
    # Normalize: treat comma and period as equivalent decimal separators
    return frozenset(n.replace(",", ".") for n in raw)


def _count_quotes(text: str) -> int:
    return sum(text.count(c) for c in ('"', "\u201c", "\u201d", "\u2018", "\u2019", "'"))


# ──────────────────────────────────────────────────────────────────────────────
# Layer 1: Deterministic checks
# ──────────────────────────────────────────────────────────────────────────────

def deterministic_pair_checks(
    english: str,
    spanish: str,
    literal_spanish: str = "",
) -> PairCheckResult:
    """
    Fast, pure-Python checks with no model call.
    Always runs regardless of mode (unless mode is off, handled by check_pair).
    Findings are approximate — labelled as such where relevant.
    """
    label_issues: List[str] = []
    faithfulness_issues: List[str] = []
    hallucination_issues: List[str] = []
    omission_issues: List[str] = []
    quality_issues: List[str] = []

    severity = "pass"
    score = 1.0

    def _downgrade(sev: str, delta: float) -> None:
        nonlocal severity, score
        score = max(0.0, round(score - delta, 3))
        rank = {"pass": 0, "info": 1, "warning": 2, "fail": 3}
        if rank.get(sev, 0) > rank.get(severity, 0):
            severity = sev

    # 1. Empty-field checks
    if not english or not english.strip():
        label_issues.append("English source field is empty.")
        _downgrade("fail", 0.5)

    if not spanish or not spanish.strip():
        label_issues.append("Spanish translation field is empty.")
        _downgrade("fail", 0.5)

    if not (english and english.strip()) or not (spanish and spanish.strip()):
        return PairCheckResult(
            passed=False,
            score=score,
            severity=severity,
            label_issues=label_issues,
            user_facing_summary="One or more required fields are empty.",
            deterministic_only=True,
        )

    # 2. Language-field swap detection (approximate)
    en_score = _lang_score(english)
    sp_score = _lang_score(spanish)

    if en_score["spanish"] > 0.20 and en_score["english"] < 0.05:
        label_issues.append(
            f"[Approx] English field may contain Spanish text "
            f"(Spanish signal {en_score['spanish']:.0%} vs English {en_score['english']:.0%})."
        )
        _downgrade("warning", 0.30)

    if sp_score["english"] > 0.20 and sp_score["spanish"] < 0.05:
        label_issues.append(
            f"[Approx] Spanish field may contain English text "
            f"(English signal {sp_score['english']:.0%} vs Spanish {sp_score['spanish']:.0%})."
        )
        _downgrade("warning", 0.30)

    # 3. Identity check — unchanged output usually means translation failure
    if spanish.strip() == english.strip():
        faithfulness_issues.append(
            "Spanish output is identical to the English source. "
            "This likely indicates a translation failure."
        )
        _downgrade("fail", 0.50)

    # 4. Number drift
    en_nums = _number_set(english)
    sp_nums = _number_set(spanish)
    missing = sorted(en_nums - sp_nums)
    added = sorted(sp_nums - en_nums)

    if missing:
        omission_issues.append(
            f"[Approx] Number(s) in English not found in Spanish: {', '.join(missing)}."
        )
        _downgrade("warning", min(0.15 * len(missing), 0.40))

    if added:
        hallucination_issues.append(
            f"[Approx] Number(s) in Spanish not found in English: {', '.join(added)}."
        )
        _downgrade("warning", min(0.15 * len(added), 0.40))

    # 5. Quote-count drift (approximate)
    en_q = _count_quotes(english)
    sp_q = _count_quotes(spanish)
    if en_q > 0 and abs(en_q - sp_q) > max(1, en_q // 2):
        faithfulness_issues.append(
            f"[Approx] Quote count changed: English {en_q}, Spanish {sp_q}."
        )
        _downgrade("info", 0.05)

    # 6. Word-count ratio checks
    en_words = len(english.split())
    sp_words = len(spanish.split())
    if en_words > 0:
        ratio = sp_words / en_words
        if ratio < 0.30:
            omission_issues.append(
                f"Spanish ({sp_words} words) is very short vs English ({en_words} words). "
                "Possible truncation or excessive summarisation."
            )
            _downgrade("warning", 0.20)
        elif ratio > 3.50:
            faithfulness_issues.append(
                f"Spanish ({sp_words} words) is much longer than English ({en_words} words). "
                "Possible added or hallucinated content."
            )
            _downgrade("info", 0.10)

    # 7. Literal Spanish vs natural Spanish — identical (low severity only)
    if (
        literal_spanish
        and literal_spanish.strip()
        and literal_spanish.strip() == spanish.strip()
    ):
        quality_issues.append(
            "[Low] Literal Spanish is identical to natural Spanish. "
            "They may validly be similar; worth a quick review."
        )
        _downgrade("info", 0.02)

    # Build summary
    all_issues = (
        label_issues + faithfulness_issues + hallucination_issues
        + omission_issues + quality_issues
    )

    if severity == "pass":
        summary = "All deterministic checks passed."
    elif severity == "info":
        summary = f"{len(all_issues)} minor note(s) from deterministic checks."
    elif severity == "warning":
        summary = f"{len(all_issues)} warning(s) from deterministic checks."
    else:
        summary = f"Deterministic checks found {len(all_issues)} issue(s) including failures."

    return PairCheckResult(
        passed=(severity != "fail"),
        score=score,
        severity=severity,
        faithfulness_issues=faithfulness_issues,
        hallucination_issues=hallucination_issues,
        omission_issues=omission_issues,
        label_issues=label_issues,
        language_quality_issues=quality_issues,
        user_facing_summary=summary,
        deterministic_only=True,
        checked_with_llm=False,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Layer 2: Risk scoring
# ──────────────────────────────────────────────────────────────────────────────

def compute_pair_risk_score(det_result: PairCheckResult) -> float:
    """
    0.0–1.0 risk score derived from deterministic findings.
    Higher = more likely to benefit from an LLM check.
    """
    if det_result.severity == "fail":
        return 1.0
    base = {"warning": 0.60, "info": 0.30, "pass": 0.0}.get(det_result.severity, 0.0)

    boost = (
        min(len(det_result.hallucination_issues) * 0.20, 0.40)
        + min(len(det_result.omission_issues) * 0.15, 0.30)
        + min(len(det_result.label_issues) * 0.25, 0.40)
        + min(len(det_result.faithfulness_issues) * 0.10, 0.20)
    )

    return min(1.0, base + boost)


# ──────────────────────────────────────────────────────────────────────────────
# Layer 3: LLM-check decision
# ──────────────────────────────────────────────────────────────────────────────

def should_run_llm_checker(
    settings: CheckerSettings,
    risk_score: float,
    pair_index: int,
) -> bool:
    """Return True if the LLM checker should be invoked for this pair."""
    if not settings.llm_enabled:
        return False

    if settings.mode == "off":
        return False

    if settings.mode == "fast":
        # LLM only for definite failures in fast mode
        return risk_score >= 1.0

    if settings.mode == "strict":
        return True

    # smart mode: LLM for risky pairs, or sampled
    if risk_score >= 0.60:
        return True

    if settings.sample_rate >= 1.0:
        return True

    # Pseudo-random sampling, seeded by pair index for reproducibility
    rng = random.Random(pair_index)
    return rng.random() < settings.sample_rate


# ──────────────────────────────────────────────────────────────────────────────
# Layer 3: LLM checker
# ──────────────────────────────────────────────────────────────────────────────

_CHECKER_LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "score": {"type": "number"},
        "severity": {"type": "string", "enum": ["pass", "info", "warning", "fail"]},
        "faithfulness_issues": {"type": "array", "items": {"type": "string"}},
        "hallucination_issues": {"type": "array", "items": {"type": "string"}},
        "omission_issues": {"type": "array", "items": {"type": "string"}},
        "label_issues": {"type": "array", "items": {"type": "string"}},
        "language_quality_issues": {"type": "array", "items": {"type": "string"}},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "recommended_action": {"type": "string"},
        "user_facing_summary": {"type": "string"},
    },
    "required": ["passed", "score", "severity", "user_facing_summary"],
}

_CHECKER_SYSTEM = (
    "You are a bilingual English-Spanish translation quality auditor. "
    "The English source is authoritative. "
    "Return ONLY valid JSON. No chain-of-thought. No markdown. No text outside JSON."
)


def _build_checker_prompt(
    english: str,
    spanish: str,
    literal_spanish: str,
    truncated: bool,
) -> str:
    trunc_note = (
        " [Inputs were truncated to fit checker limits — check coverage is partial.]"
        if truncated
        else ""
    )
    lit_block = (
        f"\n\nLITERAL SPANISH (word-for-word rendering, not the primary translation):\n{literal_spanish}"
        if literal_spanish and literal_spanish.strip()
        else ""
    )
    schema_str = json.dumps(_CHECKER_LLM_SCHEMA, ensure_ascii=False)
    return (
        f"Audit this English→Spanish translation.{trunc_note}\n\n"
        f"ENGLISH SOURCE:\n{english}\n\n"
        f"SPANISH TRANSLATION:\n{spanish}{lit_block}\n\n"
        "Auditing rules:\n"
        "- Do not penalise normal translation differences, word-order changes, "
        "or idiomatic Spanish equivalents.\n"
        "- Flag: added factual claims, changed numbers, changed names, changed dates, "
        "changed places, altered quotes, unsupported explanations, or meaningful omissions.\n"
        "- Distinguish serious meaning errors from minor stylistic differences.\n"
        "- Do not rewrite the translation.\n"
        f"- Return ONLY JSON matching:\n{schema_str}"
    )


def ollama_check_pair(
    settings: CheckerSettings,
    english: str,
    spanish: str,
    literal_spanish: str = "",
    truncated: bool = False,
) -> PairCheckResult:
    """
    Call the Ollama model to check one pair.
    Always fails open: if anything goes wrong, returns a 'checker unavailable' result.
    """
    prompt = _build_checker_prompt(english, spanish, literal_spanish, truncated)

    # NOTE: format is set to "json" rather than the schema object — see the
    # same note in translate_chunk in app.py.
    payload = {
        "model": settings.model,
        "messages": [
            {"role": "system", "content": _CHECKER_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
    }

    try:
        t0 = time.monotonic()
        resp = _http_session.post(
            f"{settings.ollama_host}/api/chat",
            json=payload,
            timeout=settings.timeout_seconds,
        )
        resp.raise_for_status()
        latency_ms = round((time.monotonic() - t0) * 1000, 1)

        content = resp.json().get("message", {}).get("content", "")
        data = _parse_checker_json(content)
        if data is None:
            return _checker_unavailable("LLM checker returned non-JSON output.")

        return PairCheckResult(
            passed=bool(data.get("passed", True)),
            score=float(data.get("score", 1.0)),
            severity=str(data.get("severity", "pass")),
            faithfulness_issues=list(data.get("faithfulness_issues", [])),
            hallucination_issues=list(data.get("hallucination_issues", [])),
            omission_issues=list(data.get("omission_issues", [])),
            label_issues=list(data.get("label_issues", [])),
            language_quality_issues=list(data.get("language_quality_issues", [])),
            unsupported_claims=list(data.get("unsupported_claims", [])),
            recommended_action=str(data.get("recommended_action", "")),
            user_facing_summary=str(data.get("user_facing_summary", "")),
            checked_with_llm=True,
            deterministic_only=False,
            truncated=truncated,
            checker_latency_ms=latency_ms,
        )

    except requests.exceptions.Timeout:
        return _checker_unavailable(
            f"Checker timed out after {settings.timeout_seconds}s. "
            "Increase CHECKER_TIMEOUT_SECONDS or switch to fast mode."
        )
    except requests.exceptions.ConnectionError:
        return _checker_unavailable(
            f"Could not connect to Ollama at {settings.ollama_host}."
        )
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return _checker_unavailable(
                f"Checker model '{settings.model}' not found. "
                "Pull it with `ollama pull <model>` or set CHECKER_MODEL in .env."
            )
        # Try to surface the Ollama error body for easier diagnosis
        detail = ""
        if exc.response is not None:
            try:
                detail = exc.response.json().get("error", exc.response.text)
            except Exception:
                detail = exc.response.text
        return _checker_unavailable(
            f"Ollama HTTP error: {exc}" + (f" — {detail}" if detail else "")
        )
    except Exception as exc:  # noqa: BLE001
        return _checker_unavailable(f"Unexpected checker error: {exc}")


def _parse_checker_json(content: str) -> Optional[dict]:
    """Try to parse JSON from LLM output, with fallback extraction."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(content[start:end])
            except json.JSONDecodeError:
                pass
    return None


def _checker_unavailable(reason: str) -> PairCheckResult:
    """Fail-open result used whenever the checker cannot complete."""
    return PairCheckResult(
        passed=True,  # fail open — do not block the user
        score=1.0,
        severity="info",
        user_facing_summary=f"Checker unavailable: {reason}",
        checked_with_llm=False,
        deterministic_only=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main entry: check one reading pair
# ──────────────────────────────────────────────────────────────────────────────

def check_pair(
    settings: CheckerSettings,
    english: str,
    spanish: str,
    literal_spanish: str = "",
    pair_index: int = 0,
    cached_results: Optional[Dict[str, PairCheckResult]] = None,
) -> Tuple[str, PairCheckResult]:
    """
    Full checker pipeline for one ReadingPair.

    Returns (cache_key, PairCheckResult).

    ``cached_results`` should be st.session_state.checker_results.
    If the key is already present there, the cached result is returned immediately
    (cache_hit=True) without any computation.
    """
    key = make_check_key(settings, english, spanish, literal_spanish)

    # Mode = off: skip everything
    if settings.mode == "off" or not settings.enabled:
        return key, PairCheckResult(
            passed=True,
            score=1.0,
            severity="pass",
            user_facing_summary="Checker is disabled.",
            deterministic_only=True,
        )

    # Cache hit: no recomputation
    if settings.cache_enabled and cached_results is not None and key in cached_results:
        cached = cached_results[key]
        return key, cached.model_copy(update={"cache_hit": True})

    t0 = time.monotonic()

    # ── Layer 1: deterministic ────────────────────────────────────────────────
    det = deterministic_pair_checks(english, spanish, literal_spanish)

    # ── Layer 2: risk score ───────────────────────────────────────────────────
    risk = compute_pair_risk_score(det)

    # ── Layer 3: LLM check (conditionally) ───────────────────────────────────
    if should_run_llm_checker(settings, risk, pair_index):
        trunc_en, was_en = truncate_for_checker(english, settings.max_chars)
        trunc_sp, was_sp = truncate_for_checker(spanish, settings.max_chars)
        trunc_lit, _ = truncate_for_checker(literal_spanish, settings.max_chars // 2)
        truncated = was_en or was_sp

        llm = ollama_check_pair(
            settings, trunc_en, trunc_sp, trunc_lit, truncated=truncated
        )

        if "unavailable" in llm.user_facing_summary.lower():
            # LLM failed — keep deterministic result, note unavailability
            note = f" ({llm.user_facing_summary})"
            final = det.model_copy(update={
                "user_facing_summary": det.user_facing_summary + note,
                "truncated": truncated,
                "checker_latency_ms": round((time.monotonic() - t0) * 1000, 1),
            })
        else:
            # Merge: LLM wins on score/severity, deterministic findings are added
            final = llm.model_copy(update={
                "label_issues": _dedup(det.label_issues + llm.label_issues),
                "hallucination_issues": _dedup(
                    det.hallucination_issues + llm.hallucination_issues
                ),
                "omission_issues": _dedup(det.omission_issues + llm.omission_issues),
                "faithfulness_issues": _dedup(
                    det.faithfulness_issues + llm.faithfulness_issues
                ),
                "language_quality_issues": _dedup(
                    det.language_quality_issues + llm.language_quality_issues
                ),
                "deterministic_only": False,
                "truncated": truncated,
                "checker_latency_ms": round((time.monotonic() - t0) * 1000, 1),
            })
    else:
        final = det.model_copy(update={
            "checker_latency_ms": round((time.monotonic() - t0) * 1000, 1),
        })

    return key, final


def _dedup(items: List[str]) -> List[str]:
    """Deduplicate while preserving order."""
    seen: set = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Export helpers
# ──────────────────────────────────────────────────────────────────────────────

def checker_markdown_block(
    result: PairCheckResult,
    detailed: bool = False,
) -> str:
    """
    Return a compact markdown section summarising a checker result.
    Used when building the study notes export.
    """
    if result.severity == "pass" and not detailed:
        return f"> ✅ **Checker:** {result.user_facing_summary}\n"

    badge = {
        "pass": "✅ Passed",
        "info": "ℹ️ Info",
        "warning": "⚠️ Warning",
        "fail": "❌ Failed",
    }.get(result.severity, "❓ Unknown")

    lines = [
        f"> **Checker:** {badge} — score {result.score:.2f}",
        f"> {result.user_facing_summary}",
    ]

    if detailed:
        for label, items in [
            ("Faithfulness", result.faithfulness_issues),
            ("Hallucination", result.hallucination_issues),
            ("Omissions", result.omission_issues),
            ("Label issues", result.label_issues),
            ("Language quality", result.language_quality_issues),
            ("Unsupported claims", result.unsupported_claims),
        ]:
            if items:
                lines.append(f">\n> **{label}:**")
                for issue in items:
                    lines.append(f"> - {issue}")

        if result.recommended_action:
            lines.append(f">\n> **Recommended action:** {result.recommended_action}")

        method = "LLM + deterministic" if result.checked_with_llm else "deterministic only"
        lines.append(f">\n> *Checked via {method}*")

    return "\n".join(lines) + "\n"
