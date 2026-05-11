"""
Tests for translate_chunk error-handling paths.
All Ollama HTTP calls are mocked — no real Ollama instance required.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from app import (
    TranslationResponse,
    translate_chunk,
    _fix_english_summary,
    _estimate_num_predict,
    _STRONG_SPANISH,
)


# ---------------------------------------------------------------------------
# Helper — build a minimal valid TranslationResponse as a JSON string
# ---------------------------------------------------------------------------

def _make_valid_json_response() -> str:
    data = {
        "title": "Test",
        "summary_english": "A short English summary.",
        "summary_spanish": "Un breve resumen en español.",
        "pairs": [
            {
                "english": "Hello world.",
                "spanish": "Hola mundo.",
                "literal_spanish": "",
                "vocabulary": [],
                "grammar_notes": [],
                "comprehension_question_spanish": "",
            }
        ],
    }
    return json.dumps(data)


def _fake_streaming_response(content: str, status_code: int = 200):
    """Produce a mock response whose iter_lines() yields streamed chunks."""
    lines = []
    for char in content:
        chunk = {"message": {"content": char}, "done": False}
        lines.append(json.dumps(chunk).encode("utf-8"))
    lines.append(json.dumps({"message": {"content": ""}, "done": True}).encode("utf-8"))

    mock_resp = MagicMock()
    mock_resp.ok = status_code < 400
    mock_resp.status_code = status_code
    mock_resp.reason = "OK" if status_code < 400 else "Bad Request"
    mock_resp.iter_lines.return_value = iter(lines)
    mock_resp.__enter__ = lambda self: self
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


_DEFAULT_KWARGS = dict(
    level="B1",
    style="Natural",
    region="Spain",
    fidelity="Closest meaning",
    include_literal=False,
    include_vocab=False,
    include_grammar=False,
    temperature=0.05,
)


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------

class TestTranslateChunkSuccess:
    def test_returns_translation_response(self, monkeypatch):
        translate_chunk.clear()  # clear st.cache_data cache between tests
        mock_resp = _fake_streaming_response(_make_valid_json_response())
        with patch("app._http_session") as mock_session:
            mock_session.post.return_value = mock_resp
            result = translate_chunk(chunk="Hello world.", **_DEFAULT_KWARGS)
        assert isinstance(result, TranslationResponse)
        assert len(result.pairs) == 1
        assert result.pairs[0].english == "Hello world."

    def test_summary_english_unchanged_when_english(self, monkeypatch):
        translate_chunk.clear()
        mock_resp = _fake_streaming_response(_make_valid_json_response())
        with patch("app._http_session") as mock_session:
            mock_session.post.return_value = mock_resp
            result = translate_chunk(chunk="Hello world.", **_DEFAULT_KWARGS)
        assert result.summary_english == "A short English summary."


# ---------------------------------------------------------------------------
# Error: connection failure
# ---------------------------------------------------------------------------

class TestTranslateChunkConnectionError:
    def test_raises_connection_error(self):
        translate_chunk.clear()
        with patch("app._http_session") as mock_session:
            mock_session.post.side_effect = requests.exceptions.ConnectionError("refused")
            with pytest.raises(requests.exceptions.ConnectionError, match="Cannot reach Ollama"):
                translate_chunk(chunk="Hello.", **_DEFAULT_KWARGS)


# ---------------------------------------------------------------------------
# Error: timeout
# ---------------------------------------------------------------------------

class TestTranslateChunkTimeout:
    def test_raises_timeout(self):
        translate_chunk.clear()
        with patch("app._http_session") as mock_session:
            mock_session.post.side_effect = requests.exceptions.Timeout("timed out")
            with pytest.raises(requests.exceptions.Timeout, match="did not respond"):
                translate_chunk(chunk="Hello.", **_DEFAULT_KWARGS)


# ---------------------------------------------------------------------------
# Error: HTTP 4xx/5xx from Ollama
# ---------------------------------------------------------------------------

class TestTranslateChunkHTTPError:
    def test_raises_http_error_on_non_ok_response(self):
        translate_chunk.clear()
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 400
        mock_resp.reason = "Bad Request"
        mock_resp.json.return_value = {"error": "model not found"}
        mock_resp.__enter__ = lambda self: self
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("app._http_session") as mock_session:
            mock_session.post.return_value = mock_resp
            with pytest.raises(requests.exceptions.HTTPError, match="Ollama said"):
                translate_chunk(chunk="Hello.", **_DEFAULT_KWARGS)


# ---------------------------------------------------------------------------
# _estimate_num_predict
# ---------------------------------------------------------------------------

class TestEstimateNumPredict:
    def test_minimum_budget(self):
        result = _estimate_num_predict(10, False, False, False)
        assert result == 600  # minimum floor

    def test_maximum_budget(self):
        result = _estimate_num_predict(10_000, True, True, True)
        assert result == 4000  # ceiling

    def test_vocab_adds_600(self):
        base = _estimate_num_predict(100, False, False, False)
        with_vocab = _estimate_num_predict(100, False, True, False)
        assert with_vocab >= base + 600 or with_vocab == 4000

    def test_grammar_adds_400(self):
        base = _estimate_num_predict(100, False, False, False)
        with_grammar = _estimate_num_predict(100, False, False, True)
        assert with_grammar >= base + 400 or with_grammar == 4000


# ---------------------------------------------------------------------------
# _fix_english_summary
# ---------------------------------------------------------------------------

class TestFixEnglishSummary:
    def test_leaves_good_english_alone(self):
        english = "This is a short summary of the passage in English."
        result = _fix_english_summary(english, "Hello world.")
        assert result == english

    def test_replaces_spanish_looking_summary(self):
        spanish_like = "El texto es sobre la vida de un hombre que se llama Juan."
        source = "John was a man who lived in the mountains."
        result = _fix_english_summary(spanish_like, source)
        # Should fall back to source extraction (first 2 sentences)
        assert "Juan" not in result
        assert result  # not empty

    def test_empty_summary_returned_as_is(self):
        result = _fix_english_summary("", "Hello world.")
        assert result == ""

    def test_no_crash_on_non_alpha_summary(self):
        result = _fix_english_summary("12345 ??? !!!", "Hello world.")
        assert isinstance(result, str)
