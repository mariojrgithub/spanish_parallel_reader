"""
Tests: model configuration defaults are correct (aya-expanse:8b).
"""
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_env_example_has_aya_expanse():
    content = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "OLLAMA_MODEL=aya-expanse:8b" in content, (
        ".env.example must set OLLAMA_MODEL=aya-expanse:8b"
    )


def test_env_example_has_no_qwen3_14b_default():
    """qwen3:14b must not appear as an uncommented default."""
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert "qwen3:14b" not in stripped, (
            f"Found unexpected uncommented qwen3:14b in .env.example: {line!r}"
        )


def test_docker_compose_has_aya_expanse_default():
    content = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "aya-expanse:8b" in content
    assert "qwen3:14b" not in content, (
        "docker-compose.yml still contains qwen3:14b default"
    )


def test_docker_compose_gpu_has_aya_expanse_default():
    content = (ROOT / "docker-compose.gpu.yml").read_text(encoding="utf-8")
    assert "aya-expanse:8b" in content
    assert "qwen3:14b" not in content, (
        "docker-compose.gpu.yml still contains qwen3:14b default"
    )


def test_app_py_default_model():
    """app.py fallback must be aya-expanse:8b, not qwen3:14b."""
    content = (ROOT / "app.py").read_text(encoding="utf-8")
    assert 'os.getenv("OLLAMA_MODEL", "aya-expanse:8b")' in content, (
        "app.py OLLAMA_MODEL fallback must be aya-expanse:8b"
    )


def test_keep_alive_default_is_minus_one():
    """OLLAMA_KEEP_ALIVE default must be '-1', not '-1m'."""
    content = (ROOT / "app.py").read_text(encoding="utf-8")
    assert 'os.getenv("OLLAMA_KEEP_ALIVE", "-1")' in content, (
        'app.py OLLAMA_KEEP_ALIVE default must be "-1"'
    )
    env_content = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "OLLAMA_KEEP_ALIVE=-1\n" in env_content or env_content.endswith("OLLAMA_KEEP_ALIVE=-1"), (
        ".env.example OLLAMA_KEEP_ALIVE must be -1"
    )
