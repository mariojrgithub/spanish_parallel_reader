"""
Shared Ollama HTTP session and thin call helpers.

Provides:
  session      — persistent requests.Session reused by all callers.
  chat()       — non-streaming /api/chat POST; returns message content string.
  stream_chat() — streaming /api/chat POST; returns (content, first_token_latency).
  load_model()  — /api/generate with empty prompt to pin model in GPU memory.

Both chat/stream_chat raise standard requests exceptions; callers handle them
according to their own semantics (fail-open in checker, re-raise in app).
"""
from __future__ import annotations

import json
import logging
import time
from typing import Callable, Optional

import requests

logger = logging.getLogger(__name__)

# Single TCP connection pool shared by all Ollama callers in this process.
session = requests.Session()


def chat(host: str, payload: dict, timeout: float) -> str:
    """Non-streaming /api/chat POST. Returns message content string.

    Raises requests.exceptions.HTTPError, ConnectionError, or Timeout on failure.
    """
    resp = session.post(f"{host}/api/chat", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "")


def stream_chat(
    host: str,
    payload: dict,
    timeout: float,
    on_token: Optional[Callable[[int], None]] = None,
) -> tuple[str, Optional[float]]:
    """Streaming /api/chat POST.

    Returns (full_content, first_token_latency_seconds | None).
    Calls on_token(total_chars_so_far) approximately every 150 characters.
    Raises HTTPError on non-2xx responses; raises on network errors.
    """
    t0 = time.monotonic()
    content = ""
    t_first_token: Optional[float] = None
    _last_on_token_len = 0

    with session.post(
        f"{host}/api/chat",
        json=payload,
        timeout=timeout,
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

    return content, t_first_token


def load_model(host: str, model: str, keep_alive: int = -1, timeout: float = 30.0) -> None:
    """Pin a model into Ollama's memory without generating any tokens.

    Uses POST /api/generate with an empty prompt, which loads model weights
    immediately without spending time on token generation.  keep_alive=-1
    means Ollama will never unload it automatically.

    Raises requests exceptions on network/HTTP failure; callers should swallow
    them (warmup is best-effort).
    """
    resp = session.post(
        f"{host}/api/generate",
        json={"model": model, "prompt": "", "keep_alive": keep_alive},
        timeout=timeout,
    )
    resp.raise_for_status()
