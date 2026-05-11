"""
Shared pytest fixtures for the Spanish Parallel Reader test suite.
"""
from __future__ import annotations

import pytest

from checker import CheckerSettings


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
        model: str = "aya-expanse:8b",
        max_chars: int = 2500,
        batch_size: int = 1,
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
