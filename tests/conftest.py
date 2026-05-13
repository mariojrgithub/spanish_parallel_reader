"""
Shared pytest fixtures for the Spanish Parallel Reader test suite.
"""
from __future__ import annotations

import sys
import types
import unittest.mock as mock

import pytest

from checker import CheckerSettings


# ---------------------------------------------------------------------------
# Streamlit stub — runs at collection time, before any test module imports
# app.py. Provides enough of the Streamlit API for app.py to be importable
# in environments that don't have a running Streamlit server.
# ---------------------------------------------------------------------------

def _stub_streamlit() -> None:
    """Install a minimal Streamlit stub into sys.modules if not already present."""
    if "streamlit" in sys.modules:
        st = sys.modules["streamlit"]
        # Older stubs may lack cache_data — patch it in.
        if not hasattr(st, "cache_data"):
            def _cache_data(fn=None, **kwargs):
                if fn is not None:
                    return fn
                def _deco(f):
                    return f
                return _deco
            st.cache_data = _cache_data
        return

    stub = types.ModuleType("streamlit")

    # cache_data / cache_resource must support both:
    #   @st.cache_data          (fn is positional)
    #   @st.cache_data(ttl=120) (fn is None, kwargs present)
    def _cache_data(fn=None, **kwargs):
        if fn is not None:
            return fn
        def _deco(f):
            return f
        return _deco

    stub.cache_data = _cache_data
    stub.cache_resource = _cache_data

    # Standard UI components and utilities — all no-ops in tests.
    for attr in [
        "set_page_config", "markdown", "title", "caption", "write",
        "subheader", "header", "success", "error", "warning", "info",
        "stop", "rerun", "divider", "empty", "spinner", "status",
        "text_area", "text_input", "selectbox", "slider", "checkbox",
        "radio", "number_input", "file_uploader", "button", "progress",
        "dataframe", "download_button", "metric", "code", "image",
        "container", "expander",
    ]:
        setattr(stub, attr, mock.MagicMock())

    # columns(n) / columns([w1, w2, ...]) must be unpackable to the right length.
    def _mock_columns(spec, **kwargs):
        n = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
        return [mock.MagicMock() for _ in range(n)]

    stub.columns = _mock_columns

    # tabs(["Tab1", "Tab2", ...]) must be unpackable to the right length.
    def _mock_tabs(labels, **kwargs):
        return [mock.MagicMock() for _ in labels]

    stub.tabs = _mock_tabs

    # session_state supports both attribute access (st.session_state.key)
    # and item access (st.session_state["key"]).
    class _SessionState:
        def __init__(self) -> None:
            object.__setattr__(self, "_data", {})

        def __getattr__(self, name: str):
            data = object.__getattribute__(self, "_data")
            if name in data:
                return data[name]
            raise AttributeError(name)

        def __setattr__(self, name: str, value) -> None:
            object.__getattribute__(self, "_data")[name] = value

        def __getitem__(self, key):
            return object.__getattribute__(self, "_data")[key]

        def __setitem__(self, key, value) -> None:
            object.__getattribute__(self, "_data")[key] = value

        def __contains__(self, key) -> bool:
            return key in object.__getattribute__(self, "_data")

        def get(self, key, default=None):
            return object.__getattribute__(self, "_data").get(key, default)

        def pop(self, key, *args):
            return object.__getattribute__(self, "_data").pop(key, *args)

        def setdefault(self, key, default=None):
            return object.__getattribute__(self, "_data").setdefault(key, default)

        def keys(self):
            return object.__getattribute__(self, "_data").keys()

        def items(self):
            return object.__getattribute__(self, "_data").items()

        def values(self):
            return object.__getattribute__(self, "_data").values()

    stub.session_state = _SessionState()

    # sidebar is a context-manager no-op.
    sidebar = mock.MagicMock()
    sidebar.__enter__ = lambda s: s
    sidebar.__exit__ = mock.MagicMock(return_value=False)
    stub.sidebar = sidebar

    sys.modules["streamlit"] = stub

    # Stub streamlit.components and streamlit.components.v1 so that any module
    # that does `import streamlit.components.v1 as components` (e.g. tts_component)
    # can be imported in the test environment without a running Streamlit server.
    _components = types.ModuleType("streamlit.components")
    _components_v1 = types.ModuleType("streamlit.components.v1")
    _components_v1.html = mock.MagicMock()  # type: ignore[attr-defined]
    _components.v1 = _components_v1  # type: ignore[attr-defined]
    stub.components = _components  # type: ignore[attr-defined]
    sys.modules["streamlit.components"] = _components
    sys.modules["streamlit.components.v1"] = _components_v1

    # Stub heavy optional imports so app.py can be collected without them.
    for mod in ("fitz", "docx", "pandas"):
        sys.modules.setdefault(mod, mock.MagicMock())


_stub_streamlit()


# ---------------------------------------------------------------------------
# Checker settings factory — used across multiple test modules
# ---------------------------------------------------------------------------

@pytest.fixture
def checker_settings_factory():
    """Return a factory that builds CheckerSettings with sensible defaults."""
    def _make(
        mode: str = "smart",
        llm_enabled: bool = True,
        sample_rate: float = 1.0,
        require_pass: bool = False,
        cache_enabled: bool = True,
        timeout_seconds: float = 10.0,
        model: str = "qwen2.5:7b",
        max_chars: int = 2500,
        batch_size: int = 1,
        det_workers: int = 4,
        llm_concurrency: int = 1,
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
            batch_size=batch_size,
            det_workers=det_workers,
            llm_concurrency=llm_concurrency,
            ollama_host="http://localhost:11434",
        )
    return _make


@pytest.fixture
def default_settings(checker_settings_factory):
    return checker_settings_factory()


@pytest.fixture
def fast_settings(checker_settings_factory):
    return checker_settings_factory(mode="fast", llm_enabled=False)


@pytest.fixture
def strict_settings(checker_settings_factory):
    return checker_settings_factory(mode="strict")
