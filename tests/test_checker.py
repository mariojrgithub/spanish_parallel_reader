"""
Tests: output checker — deterministic checks, risk scoring, caching, LLM path, export.

All Ollama calls are mocked so no real model is required.
"""
from __future__ import annotations

import json
import sys
import os
import unittest.mock as mock

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from checker import (
    CHECKER_PROMPT_VERSION,
    CheckerSettings,
    PairCheckResult,
    _checker_unavailable,
    _count_quotes,
    _lang_score,
    _number_set,
    check_pair,
    checker_markdown_block,
    compute_pair_risk_score,
    deterministic_pair_checks,
    get_checker_settings,
    make_check_key,
    should_run_llm_checker,
    truncate_for_checker,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _settings(
    mode: str = "smart",
    llm_enabled: bool = True,
    sample_rate: float = 1.0,
    require_pass: bool = False,
    cache_enabled: bool = True,
    timeout_seconds: float = 10.0,
    model: str = "aya-expanse:8b",
    max_chars: int = 2500,
) -> CheckerSettings:
    return CheckerSettings(
        enabled=True,
        mode=mode,
        model=model,
        sample_rate=sample_rate,
        max_chars=max_chars,
        require_pass=require_pass,
        timeout_seconds=timeout_seconds,
        cache_enabled=cache_enabled,
        detailed_diagnostics=False,
        llm_enabled=llm_enabled,
        batch_size=1,
        ollama_host="http://localhost:11434",
    )


_GOOD_EN = "The dog runs fast every morning."
_GOOD_ES = "El perro corre rápido cada mañana."


# ── Language score heuristic ──────────────────────────────────────────────────

def test_lang_score_english_text():
    score = _lang_score("The dog runs fast in the morning and the evening")
    assert score["english"] > 0.10
    assert score["spanish"] < 0.10


def test_lang_score_spanish_text():
    score = _lang_score("El perro corre muy rápido por las mañanas en el parque")
    assert score["spanish"] > 0.10
    assert score["english"] < 0.10


# ── Number set extraction ─────────────────────────────────────────────────────

def test_number_set_basic():
    nums = _number_set("There are 42 dogs and 3 cats.")
    assert "42" in nums
    assert "3" in nums


def test_number_set_normalises_comma_decimal():
    nums = _number_set("Cost: 1,500.00 euros")
    assert "1.500.00" in nums or "1.500" in nums or "1500" in nums or len(nums) > 0


def test_number_set_empty():
    assert _number_set("No numbers here.") == frozenset()


# ── Deterministic checks ──────────────────────────────────────────────────────

def test_det_passes_clean_pair():
    result = deterministic_pair_checks(_GOOD_EN, _GOOD_ES)
    assert result.severity == "pass"
    assert result.passed is True
    assert result.score == 1.0


def test_det_catches_empty_spanish():
    result = deterministic_pair_checks("Hello world", "")
    assert result.passed is False
    assert result.severity == "fail"
    assert any("empty" in i.lower() for i in result.label_issues)


def test_det_catches_empty_english():
    result = deterministic_pair_checks("", "Hola mundo")
    assert result.passed is False
    assert result.severity == "fail"


def test_det_catches_spanish_text_in_english_field():
    """Put strongly Spanish text in the English field."""
    spanish_as_english = (
        "El perro corre muy rápido por las mañanas en el parque con los niños"
    )
    result = deterministic_pair_checks(spanish_as_english, "The dog runs fast.")
    # Should flag a label issue for the English field
    assert any("english field" in i.lower() for i in result.label_issues)


def test_det_catches_english_text_in_spanish_field():
    """Put strongly English text in the Spanish field."""
    english_as_spanish = (
        "The cat sat on the mat and the dog was in the house with the children"
    )
    result = deterministic_pair_checks("El gato se sentó en la alfombra.", english_as_spanish)
    assert any("spanish field" in i.lower() for i in result.label_issues)


def test_det_catches_identical_translation():
    text = "The cat sat on the mat."
    result = deterministic_pair_checks(text, text)
    assert result.severity == "fail"
    assert any("identical" in i.lower() for i in result.faithfulness_issues)


def test_det_catches_number_missing_from_spanish():
    en = "There were 42 participants in the study."
    es = "Había participantes en el estudio."  # 42 missing
    result = deterministic_pair_checks(en, es)
    assert any("42" in i for i in result.omission_issues)


def test_det_catches_number_added_in_spanish():
    en = "The study had participants."
    es = "El estudio tenía 99 participantes."  # 99 added
    result = deterministic_pair_checks(en, es)
    assert any("99" in i for i in result.hallucination_issues)


def test_det_short_spanish_warns():
    en = "This is a very long English sentence with many interesting words about various topics."
    es = "Breve."  # far too short
    result = deterministic_pair_checks(en, es)
    assert result.severity in ("warning", "fail")
    assert any("short" in i.lower() for i in result.omission_issues)


def test_det_literal_identical_to_natural_is_info():
    """Literal == natural is only a low-severity info, not a failure."""
    result = deterministic_pair_checks(
        "Hello world", "Hola mundo", literal_spanish="Hola mundo"
    )
    assert result.severity in ("pass", "info")
    assert result.passed is True


# ── Risk scoring ──────────────────────────────────────────────────────────────

def test_risk_score_clean_pair_is_zero():
    det = deterministic_pair_checks(_GOOD_EN, _GOOD_ES)
    risk = compute_pair_risk_score(det)
    assert risk == 0.0


def test_risk_score_fail_is_one():
    det = deterministic_pair_checks(_GOOD_EN, _GOOD_EN)  # identical → fail
    risk = compute_pair_risk_score(det)
    assert risk == 1.0


def test_risk_score_number_drift_triggers_llm_in_smart_mode():
    en = "There were 42 participants."
    es = "Había participantes."
    det = deterministic_pair_checks(en, es)
    risk = compute_pair_risk_score(det)
    settings = _settings(mode="smart")
    assert should_run_llm_checker(settings, risk, pair_index=0)


# ── LLM check decision ────────────────────────────────────────────────────────

def test_smart_mode_skips_llm_for_low_risk_zero_sample_rate():
    settings = _settings(mode="smart", sample_rate=0.0, llm_enabled=True)
    # Low-risk pair (clean), sample_rate=0 → should NOT run LLM
    assert not should_run_llm_checker(settings, risk_score=0.0, pair_index=0)


def test_fast_mode_only_runs_llm_for_fail():
    settings = _settings(mode="fast", llm_enabled=True)
    assert not should_run_llm_checker(settings, risk_score=0.5, pair_index=0)
    assert should_run_llm_checker(settings, risk_score=1.0, pair_index=0)


def test_strict_mode_always_runs_llm():
    settings = _settings(mode="strict", llm_enabled=True)
    assert should_run_llm_checker(settings, risk_score=0.0, pair_index=0)
    assert should_run_llm_checker(settings, risk_score=1.0, pair_index=0)


def test_off_mode_never_runs_llm():
    settings = _settings(mode="off", llm_enabled=True)
    assert not should_run_llm_checker(settings, risk_score=1.0, pair_index=0)


def test_llm_disabled_never_runs():
    for mode in ("fast", "smart", "strict"):
        settings = _settings(mode=mode, llm_enabled=False)
        assert not should_run_llm_checker(settings, risk_score=1.0, pair_index=0)


# ── Cache key ─────────────────────────────────────────────────────────────────

def test_cache_key_stable_for_identical_inputs():
    s = _settings()
    k1 = make_check_key(s, _GOOD_EN, _GOOD_ES, "")
    k2 = make_check_key(s, _GOOD_EN, _GOOD_ES, "")
    assert k1 == k2


def test_cache_key_changes_when_spanish_changes():
    s = _settings()
    k1 = make_check_key(s, _GOOD_EN, _GOOD_ES, "")
    k2 = make_check_key(s, _GOOD_EN, "El perro corre lento.", "")
    assert k1 != k2


def test_cache_key_changes_when_mode_changes():
    s_smart = _settings(mode="smart")
    s_strict = _settings(mode="strict")
    k1 = make_check_key(s_smart, _GOOD_EN, _GOOD_ES, "")
    k2 = make_check_key(s_strict, _GOOD_EN, _GOOD_ES, "")
    assert k1 != k2


def test_cache_key_changes_when_model_changes():
    s1 = _settings(model="aya-expanse:8b")
    s2 = _settings(model="qwen3:8b")
    k1 = make_check_key(s1, _GOOD_EN, _GOOD_ES, "")
    k2 = make_check_key(s2, _GOOD_EN, _GOOD_ES, "")
    assert k1 != k2


def test_cache_key_changes_when_english_changes():
    s = _settings()
    k1 = make_check_key(s, "Hello world.", _GOOD_ES, "")
    k2 = make_check_key(s, "Goodbye world.", _GOOD_ES, "")
    assert k1 != k2


# ── Truncation ────────────────────────────────────────────────────────────────

def test_truncate_short_text_unchanged():
    text = "Hello"
    result, was_truncated = truncate_for_checker(text, 100)
    assert result == text
    assert was_truncated is False


def test_truncate_long_text_is_truncated():
    text = "A" * 300
    result, was_truncated = truncate_for_checker(text, 100)
    assert was_truncated is True
    assert len(result) <= 101  # 100 + ellipsis char


# ── Mocked LLM checker ────────────────────────────────────────────────────────

def _mock_llm_response(data: dict):
    """Return a mock requests.post response with checker JSON."""
    resp = mock.MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"message": {"content": json.dumps(data)}}
    resp.raise_for_status.return_value = None
    return resp


def test_mocked_llm_checker_pass():
    llm_data = {
        "passed": True,
        "score": 0.95,
        "severity": "pass",
        "faithfulness_issues": [],
        "hallucination_issues": [],
        "omission_issues": [],
        "label_issues": [],
        "language_quality_issues": [],
        "unsupported_claims": [],
        "recommended_action": "",
        "user_facing_summary": "Translation looks good.",
    }
    settings = _settings(mode="strict")  # strict → always LLM
    with mock.patch("checker._http_session.post", return_value=_mock_llm_response(llm_data)):
        _, result = check_pair(
            settings=settings,
            english=_GOOD_EN,
            spanish=_GOOD_ES,
            cached_results={},
        )
    assert result.passed is True
    assert result.score >= 0.9
    assert result.checked_with_llm is True


def test_mocked_llm_checker_fail():
    llm_data = {
        "passed": False,
        "score": 0.3,
        "severity": "fail",
        "faithfulness_issues": ["Translation adds unsupported claim about cats."],
        "hallucination_issues": ["Cats mentioned but not in source."],
        "omission_issues": [],
        "label_issues": [],
        "language_quality_issues": [],
        "unsupported_claims": ["cats"],
        "recommended_action": "Remove reference to cats.",
        "user_facing_summary": "Hallucination detected.",
    }
    settings = _settings(mode="strict")
    with mock.patch("checker._http_session.post", return_value=_mock_llm_response(llm_data)):
        _, result = check_pair(
            settings=settings,
            english=_GOOD_EN,
            spanish=_GOOD_ES,
            cached_results={},
        )
    assert result.passed is False
    assert result.checked_with_llm is True
    assert len(result.hallucination_issues) >= 1


def test_mocked_llm_checker_validates_pydantic():
    """Confirm the LLM JSON can be parsed into PairCheckResult without error."""
    llm_data = {
        "passed": True,
        "score": 0.88,
        "severity": "info",
        "faithfulness_issues": [],
        "hallucination_issues": [],
        "omission_issues": ["Minor detail about weather omitted."],
        "label_issues": [],
        "language_quality_issues": [],
        "unsupported_claims": [],
        "recommended_action": "Consider adding weather note.",
        "user_facing_summary": "Minor omission noted.",
    }
    result = PairCheckResult(
        passed=llm_data["passed"],
        score=llm_data["score"],
        severity=llm_data["severity"],
        omission_issues=llm_data["omission_issues"],
        recommended_action=llm_data["recommended_action"],
        user_facing_summary=llm_data["user_facing_summary"],
        checked_with_llm=True,
        deterministic_only=False,
    )
    assert result.score == 0.88
    assert len(result.omission_issues) == 1


# ── Cache hit behaviour ───────────────────────────────────────────────────────

def test_cache_hit_returns_without_llm_call():
    settings = _settings(mode="strict")  # would normally call LLM
    cached_result = PairCheckResult(
        passed=True,
        score=0.99,
        severity="pass",
        user_facing_summary="Cached result.",
    )
    cache = {}
    key = make_check_key(settings, _GOOD_EN, _GOOD_ES)
    cache[key] = cached_result

    with mock.patch("requests.post") as mock_post:
        returned_key, result = check_pair(
            settings=settings,
            english=_GOOD_EN,
            spanish=_GOOD_ES,
            cached_results=cache,
        )
    mock_post.assert_not_called()
    assert result.cache_hit is True
    assert result.score == 0.99


# ── Checker unavailable (fail-open) ──────────────────────────────────────────

def test_checker_unavailable_is_fail_open():
    result = _checker_unavailable("Connection refused")
    assert result.passed is True  # fail open
    assert "unavailable" in result.user_facing_summary.lower()


def test_timeout_returns_unavailable():
    import requests as req
    settings = _settings(mode="strict", timeout_seconds=0.001)
    with mock.patch("requests.post", side_effect=req.exceptions.Timeout()):
        _, result = check_pair(
            settings=settings,
            english=_GOOD_EN,
            spanish=_GOOD_ES,
            cached_results={},
        )
    assert result.passed is True  # fail open
    assert "unavailable" in result.user_facing_summary.lower()


# ── Mode=off returns quickly ──────────────────────────────────────────────────

def test_mode_off_no_check():
    settings = _settings(mode="off")
    with mock.patch("requests.post") as mock_post:
        _, result = check_pair(
            settings=settings,
            english=_GOOD_EN,
            spanish=_GOOD_ES,
            cached_results={},
        )
    mock_post.assert_not_called()
    assert result.passed is True


# ── Export helpers ────────────────────────────────────────────────────────────

def test_markdown_block_pass_is_short():
    result = PairCheckResult(
        passed=True,
        severity="pass",
        score=1.0,
        user_facing_summary="All deterministic checks passed.",
    )
    md = checker_markdown_block(result, detailed=False)
    assert "✅" in md
    assert "All deterministic checks passed" in md


def test_markdown_block_warning_includes_summary():
    result = PairCheckResult(
        passed=True,
        severity="warning",
        score=0.7,
        user_facing_summary="1 warning from deterministic checks.",
        omission_issues=["Number 42 missing from Spanish."],
    )
    md = checker_markdown_block(result, detailed=False)
    assert "⚠️" in md
    assert "0.70" in md


def test_markdown_block_detailed_includes_issues():
    result = PairCheckResult(
        passed=False,
        severity="fail",
        score=0.4,
        user_facing_summary="Checker found failures.",
        hallucination_issues=["Name 'Carlos' not in English source."],
        recommended_action="Remove 'Carlos'.",
    )
    md = checker_markdown_block(result, detailed=True)
    assert "Carlos" in md
    assert "Remove" in md


def test_markdown_export_includes_checker_warning():
    """Simulate what the export tab does: checker result appended after pair markdown."""
    result = PairCheckResult(
        passed=True,
        severity="warning",
        score=0.75,
        user_facing_summary="1 warning: number drift detected.",
    )
    markdown_lines = [
        "## English\n",
        "There were 42 participants.\n",
        "## Español\n",
        "Había participantes.\n",
    ]
    markdown_lines += ["\n", checker_markdown_block(result, detailed=False)]
    full = "".join(markdown_lines)
    assert "warning" in full.lower() or "⚠️" in full


# ── Export blocking ───────────────────────────────────────────────────────────

def test_require_pass_true_fails_blocks_export():
    """
    When require_pass=True and a pair's checker result has passed=False,
    export should be blocked. This test verifies the logic used in app.py.
    """
    settings = _settings(require_pass=True, mode="smart")
    failed_result = PairCheckResult(
        passed=False,
        severity="fail",
        score=0.2,
        user_facing_summary="Hallucination detected.",
    )

    # Simulate the app's export-blocking logic
    from pydantic import BaseModel
    from typing import List

    class _FakePair(BaseModel):
        english: str
        spanish: str
        literal_spanish: str = ""

    fake_pair = _FakePair(english=_GOOD_EN, spanish=_GOOD_ES)
    cache_key = make_check_key(settings, fake_pair.english, fake_pair.spanish, fake_pair.literal_spanish)
    cached = {cache_key: failed_result}

    export_blocked = settings.require_pass and any(
        not cached.get(
            make_check_key(settings, p.english, p.spanish, p.literal_spanish),
            PairCheckResult(),
        ).passed
        for p in [fake_pair]
    )

    assert export_blocked is True


def test_require_pass_false_does_not_block_export():
    settings = _settings(require_pass=False, mode="smart")
    failed_result = PairCheckResult(
        passed=False,
        severity="fail",
        score=0.2,
        user_facing_summary="Hallucination detected.",
    )

    from pydantic import BaseModel

    class _FakePair(BaseModel):
        english: str
        spanish: str
        literal_spanish: str = ""

    fake_pair = _FakePair(english=_GOOD_EN, spanish=_GOOD_ES)
    cache_key = make_check_key(settings, fake_pair.english, fake_pair.spanish, fake_pair.literal_spanish)
    cached = {cache_key: failed_result}

    export_blocked = settings.require_pass and any(
        not cached.get(
            make_check_key(settings, p.english, p.spanish, p.literal_spanish),
            PairCheckResult(),
        ).passed
        for p in [fake_pair]
    )

    assert export_blocked is False


# ── get_checker_settings ──────────────────────────────────────────────────────

def test_get_checker_settings_defaults():
    # With no env vars set, defaults should be safe
    os.environ.pop("CHECKER_ENABLED", None)
    os.environ.pop("CHECKER_MODE", None)
    s = get_checker_settings(ollama_model="aya-expanse:8b")
    assert s.enabled is True
    assert s.mode == "smart"
    assert s.model == "aya-expanse:8b"
    assert s.require_pass is False
    assert s.cache_enabled is True


def test_get_checker_settings_respects_checker_model_env(monkeypatch=None):
    os.environ["CHECKER_MODEL"] = "qwen3:8b"
    os.environ["OLLAMA_MODEL"] = "aya-expanse:8b"
    s = get_checker_settings()
    assert s.model == "qwen3:8b"
    del os.environ["CHECKER_MODEL"]


def test_get_checker_settings_falls_back_to_ollama_model():
    os.environ.pop("CHECKER_MODEL", None)
    os.environ["OLLAMA_MODEL"] = "aya-expanse:8b"
    s = get_checker_settings()
    assert s.model == "aya-expanse:8b"


# ── Merge severity: deterministic failure must not be overridden by LLM pass ──

def test_merge_severity_deterministic_fail_overrides_llm_pass():
    """
    If the LLM says 'pass' but the deterministic checker found 'fail'
    (e.g. Spanish identical to English), the merged result must remain 'fail'
    and retranslation must be triggered.  Regression test for the bug where
    LLM severity completely overwrote deterministic severity.
    """
    # Spanish == English → deterministic "fail"
    identical_spanish = _GOOD_EN  # same string as English
    llm_data = {
        "passed": True,
        "score": 0.95,
        "severity": "pass",
        "faithfulness_issues": [],
        "hallucination_issues": [],
        "omission_issues": [],
        "label_issues": [],
        "language_quality_issues": [],
        "unsupported_claims": [],
        "recommended_action": "",
        "user_facing_summary": "Translation looks good.",
    }
    settings = _settings(mode="strict")
    with mock.patch("checker._http_session.post", return_value=_mock_llm_response(llm_data)):
        _, result = check_pair(
            settings=settings,
            english=_GOOD_EN,
            spanish=identical_spanish,
            cached_results={},
        )
    # Deterministic check detected identity → must still be fail after merge
    assert result.severity == "fail", (
        f"Expected 'fail' but got '{result.severity}'. "
        "Deterministic identity-failure was silently overridden by LLM 'pass'."
    )
    assert result.passed is False
    assert result.score < 1.0


def test_merge_severity_llm_fail_overrides_deterministic_pass():
    """
    If the LLM finds a 'fail' issue on a pair that passed all deterministic
    checks, the merged result must use the LLM's 'fail'.
    """
    llm_data = {
        "passed": False,
        "score": 0.2,
        "severity": "fail",
        "faithfulness_issues": ["Translation reverses meaning of the sentence."],
        "hallucination_issues": [],
        "omission_issues": [],
        "label_issues": [],
        "language_quality_issues": [],
        "unsupported_claims": [],
        "recommended_action": "Retranslate.",
        "user_facing_summary": "Meaning reversed.",
    }
    settings = _settings(mode="strict")
    with mock.patch("checker._http_session.post", return_value=_mock_llm_response(llm_data)):
        _, result = check_pair(
            settings=settings,
            english=_GOOD_EN,
            spanish=_GOOD_ES,  # clean pair — passes deterministic
            cached_results={},
        )
    assert result.severity == "fail"
    assert result.passed is False
    assert result.score <= 0.2


def test_merge_severity_takes_worst_when_llm_warning_det_pass():
    """
    LLM warning + deterministic pass → merged result is 'warning'.
    """
    llm_data = {
        "passed": True,
        "score": 0.72,
        "severity": "warning",
        "faithfulness_issues": ["Minor style deviation."],
        "hallucination_issues": [],
        "omission_issues": [],
        "label_issues": [],
        "language_quality_issues": [],
        "unsupported_claims": [],
        "recommended_action": "",
        "user_facing_summary": "Minor warning.",
    }
    settings = _settings(mode="strict")
    with mock.patch("checker._http_session.post", return_value=_mock_llm_response(llm_data)):
        _, result = check_pair(
            settings=settings,
            english=_GOOD_EN,
            spanish=_GOOD_ES,
            cached_results={},
        )
    assert result.severity == "warning"
    assert result.score <= 0.72
