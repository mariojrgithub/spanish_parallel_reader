"""
Tests: model configuration defaults are correct (qwen2.5:7b).
"""
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_env_example_has_qwen25_7b():
    content = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "OLLAMA_MODEL=qwen2.5:7b" in content, (
        ".env.example must set OLLAMA_MODEL=qwen2.5:7b"
    )


def test_env_example_has_available_models():
    content = (ROOT / ".env.example").read_text(encoding="utf-8")
    # Find the AVAILABLE_OLLAMA_MODELS line (may include extra models like 3b)
    for line in content.splitlines():
        if line.startswith("AVAILABLE_OLLAMA_MODELS="):
            value = line.split("=", 1)[1]
            models = [m.strip() for m in value.split(",")]
            assert models[0] == "qwen2.5:7b", ".env.example: qwen2.5:7b must be first"
            assert "qwen2.5:14b" in models, ".env.example: qwen2.5:14b must be listed"
            return
    raise AssertionError(".env.example must define AVAILABLE_OLLAMA_MODELS")


def test_env_example_14b_is_not_default():
    """qwen2.5:14b must not appear as an uncommented OLLAMA_MODEL default."""
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert not (stripped.startswith("OLLAMA_MODEL=") and "14b" in stripped), (
            f"qwen2.5:14b must not be the default OLLAMA_MODEL in .env.example: {line!r}"
        )


def test_docker_compose_has_qwen25_7b_default():
    content = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "qwen2.5:7b" in content
    assert "aya-expanse:8b" not in content, (
        "docker-compose.yml still contains aya-expanse:8b default"
    )


def test_docker_compose_gpu_has_qwen25_7b_default():
    content = (ROOT / "docker-compose.gpu.yml").read_text(encoding="utf-8")
    assert "qwen2.5:7b" in content
    assert "aya-expanse:8b" not in content, (
        "docker-compose.gpu.yml still contains aya-expanse:8b default"
    )


def test_docker_compose_has_available_models_env():
    content = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "AVAILABLE_OLLAMA_MODELS" in content


def test_app_py_default_model():
    """app.py fallback must be qwen2.5:7b."""
    content = (ROOT / "app.py").read_text(encoding="utf-8")
    assert 'os.getenv("OLLAMA_MODEL", "qwen2.5:7b")' in content, (
        'app.py OLLAMA_MODEL fallback must be "qwen2.5:7b"'
    )


def test_app_py_available_models():
    """app.py must parse AVAILABLE_OLLAMA_MODELS from env."""
    content = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "AVAILABLE_OLLAMA_MODELS" in content
    assert "qwen2.5:7b,qwen2.5:14b" in content


def test_checker_py_default_model():
    content = (ROOT / "checker.py").read_text(encoding="utf-8")
    assert '"qwen2.5:7b"' in content, (
        'checker.py OLLAMA_MODEL fallback must be "qwen2.5:7b"'
    )


def test_keep_alive_default_is_minus_one():
    """OLLAMA_KEEP_ALIVE default must be '-1'."""
    content = (ROOT / "app.py").read_text(encoding="utf-8")
    assert 'os.getenv("OLLAMA_KEEP_ALIVE", "-1")' in content, (
        'app.py OLLAMA_KEEP_ALIVE default must be "-1"'
    )
    env_content = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "OLLAMA_KEEP_ALIVE=-1" in env_content, (
        ".env.example OLLAMA_KEEP_ALIVE must be -1"
    )
