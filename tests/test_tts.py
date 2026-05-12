"""
Tests for tts_component.

All tests use unittest.mock — no real browser, Streamlit server, or network
calls are made.  streamlit.components.v1.html is patched at the module level
used by tts_component so the import chain is correctly intercepted.

tts_component is imported inside each test/class rather than at module level
so that the Streamlit stub installed by conftest.py is in place first.
"""
from __future__ import annotations

import json
import importlib
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_tts():
    """Return tts_component, importing it fresh if needed."""
    import tts_component
    return tts_component


def _call_html_arg(mock_html) -> str:
    """Return the positional html string passed to the mocked components.html."""
    return mock_html.call_args[0][0]


# ---------------------------------------------------------------------------
# render_tts_button — no-op conditions
# ---------------------------------------------------------------------------

class TestNoOp:
    def test_empty_string_does_not_call_html(self):
        tts = _get_tts()
        with patch("tts_component.components.html") as mock_html:
            tts.render_tts_button("")
            mock_html.assert_not_called()

    def test_whitespace_only_does_not_call_html(self):
        tts = _get_tts()
        with patch("tts_component.components.html") as mock_html:
            tts.render_tts_button("   \n\t  ")
            mock_html.assert_not_called()

    def test_enable_tts_false_does_not_call_html(self):
        tts = _get_tts()
        with patch.object(tts, "ENABLE_TTS", False):
            with patch("tts_component.components.html") as mock_html:
                tts.render_tts_button("Hola mundo")
                mock_html.assert_not_called()

    def test_enable_tts_false_wins_over_valid_text(self):
        """Even non-empty text must not trigger audio when TTS is disabled."""
        tts = _get_tts()
        with patch.object(tts, "ENABLE_TTS", False):
            with patch("tts_component.components.html") as mock_html:
                tts.render_tts_button("Buenos días, ¿cómo estás?")
                mock_html.assert_not_called()


# ---------------------------------------------------------------------------
# render_tts_button — renders correctly
# ---------------------------------------------------------------------------

class TestRenders:
    def test_valid_text_calls_html_once(self):
        tts = _get_tts()
        with patch.object(tts, "ENABLE_TTS", True):
            with patch("tts_component.components.html") as mock_html:
                tts.render_tts_button("Hola mundo")
                mock_html.assert_called_once()

    def test_text_is_json_encoded_in_output(self):
        """Text must appear as a JSON string literal so quotes/backslashes are safe."""
        tts = _get_tts()
        text = "Hola mundo"
        with patch.object(tts, "ENABLE_TTS", True):
            with patch("tts_component.components.html") as mock_html:
                tts.render_tts_button(text)
                assert json.dumps(text) in _call_html_arg(mock_html)

    def test_special_chars_safely_encoded(self):
        """Quotes, backslashes, and accents must not break the JS string."""
        tts = _get_tts()
        text = 'She said "¡Hola!" and it\'s fine — \\backslash\\'
        with patch.object(tts, "ENABLE_TTS", True):
            with patch("tts_component.components.html") as mock_html:
                tts.render_tts_button(text)
                assert json.dumps(text) in _call_html_arg(mock_html)

    def test_custom_lang_in_output(self):
        tts = _get_tts()
        with patch.object(tts, "ENABLE_TTS", True):
            with patch("tts_component.components.html") as mock_html:
                tts.render_tts_button("Buenos días", lang="es-MX")
                assert "es-MX" in _call_html_arg(mock_html)

    def test_height_kwarg_is_38(self):
        tts = _get_tts()
        with patch.object(tts, "ENABLE_TTS", True):
            with patch("tts_component.components.html") as mock_html:
                tts.render_tts_button("Hola")
                _, kwargs = mock_html.call_args
                assert kwargs.get("height") == 38

    def test_scrolling_kwarg_is_false(self):
        tts = _get_tts()
        with patch.object(tts, "ENABLE_TTS", True):
            with patch("tts_component.components.html") as mock_html:
                tts.render_tts_button("Hola")
                _, kwargs = mock_html.call_args
                assert kwargs.get("scrolling") is False

    def test_read_aloud_button_text_present(self):
        tts = _get_tts()
        with patch.object(tts, "ENABLE_TTS", True):
            with patch("tts_component.components.html") as mock_html:
                tts.render_tts_button("Hola")
                assert "Read aloud" in _call_html_arg(mock_html)

    def test_stop_button_text_present(self):
        tts = _get_tts()
        with patch.object(tts, "ENABLE_TTS", True):
            with patch("tts_component.components.html") as mock_html:
                tts.render_tts_button("Hola")
                assert "Stop" in _call_html_arg(mock_html)

    def test_aria_label_present(self):
        """Accessibility: ARIA labels must be present on the buttons."""
        tts = _get_tts()
        with patch.object(tts, "ENABLE_TTS", True):
            with patch("tts_component.components.html") as mock_html:
                tts.render_tts_button("Hola")
                assert "aria-label" in _call_html_arg(mock_html)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_tts_language_default_is_es(self, monkeypatch):
        """TTS_LANGUAGE must default to 'es' when env var is absent."""
        monkeypatch.delenv("TTS_LANGUAGE", raising=False)
        import tts_component as tts
        importlib.reload(tts)
        assert tts.TTS_LANGUAGE == "es"

    def test_enable_tts_default_is_true(self, monkeypatch):
        monkeypatch.delenv("ENABLE_TTS", raising=False)
        import tts_component as tts
        importlib.reload(tts)
        assert tts.ENABLE_TTS is True

    def test_enable_tts_false_variants(self, monkeypatch):
        import tts_component as tts
        for val in ("false", "False", "FALSE", "0", "no", "off"):
            monkeypatch.setenv("ENABLE_TTS", val)
            importlib.reload(tts)
            assert tts.ENABLE_TTS is False, f"Expected False for ENABLE_TTS={val!r}"

    def test_enable_tts_true_variants(self, monkeypatch):
        import tts_component as tts
        for val in ("true", "True", "TRUE", "1", "yes"):
            monkeypatch.setenv("ENABLE_TTS", val)
            importlib.reload(tts)
            assert tts.ENABLE_TTS is True, f"Expected True for ENABLE_TTS={val!r}"
