"""
Tests: model configuration defaults are correct (aya-expanse:8b).
"""
import os


def test_env_example_has_aya_expanse():
    with open(".env.example", "r") as f:
        content = f.read()
    assert "OLLAMA_MODEL=aya-expanse:8b" in content, (
        ".env.example must set OLLAMA_MODEL=aya-expanse:8b"
    )


def test_env_example_has_no_qwen3_14b_default():
    """qwen3:14b must not appear as an uncommented default."""
    with open(".env.example", "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "qwen3:14b" not in stripped, (
                f"Found unexpected uncommented qwen3:14b in .env.example: {line!r}"
            )


def test_docker_compose_has_aya_expanse_default():
    with open("docker-compose.yml", "r") as f:
        content = f.read()
    assert "aya-expanse:8b" in content
    assert "qwen3:14b" not in content, (
        "docker-compose.yml still contains qwen3:14b default"
    )


def test_docker_compose_gpu_has_aya_expanse_default():
    with open("docker-compose.gpu.yml", "r") as f:
        content = f.read()
    assert "aya-expanse:8b" in content
    assert "qwen3:14b" not in content, (
        "docker-compose.gpu.yml still contains qwen3:14b default"
    )


def test_app_py_default_model():
    """app.py fallback must be aya-expanse:8b, not qwen3:14b."""
    with open("app.py", "r", encoding="utf-8") as f:
        content = f.read()
    assert 'os.getenv("OLLAMA_MODEL", "aya-expanse:8b")' in content, (
        "app.py OLLAMA_MODEL fallback must be aya-expanse:8b"
    )
