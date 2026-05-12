"""
Cold-start vs warm-model translation latency benchmark.

Usage (run from repo root):
    python benchmarks/bench_translation.py [--model aya-expanse:8b] [--host http://localhost:11434]

Requires:
    - A running Ollama instance with the model already available.
    - The app's Python environment (pip install -r requirements.txt).

Outputs a simple table of cold/warm latency per chunk size.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Allow running from repo root without installing the package.
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import requests

# Ensure app env vars are set so imports don't fail.
os.environ.setdefault("OLLAMA_MODEL", "aya-expanse:8b")
os.environ.setdefault("OLLAMA_HOST", "http://localhost:11434")

from app import OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_KEEP_ALIVE, translate_chunk

# ---------------------------------------------------------------------------
# Sample texts of varying sizes
# ---------------------------------------------------------------------------

_SAMPLES: dict[str, str] = {
    "tiny (100c)": (
        "The sun rises in the east and sets in the west. "
        "Every morning, birds sing in the trees."
    )[:100],
    "small (400c)": (
        "The history of Spain stretches back thousands of years, from prehistoric settlements "
        "to the Roman Empire, the Visigoths, and the Moorish caliphates. The Reconquista, "
        "spanning nearly eight centuries, culminated in 1492 with the fall of Granada and "
        "Christopher Columbus's voyage to the Americas. This convergence of events shaped "
        "the Spanish identity that still resonates today."
    ),
    "medium (1000c)": (
        "Language learning research consistently shows that extensive reading in the target "
        "language is one of the most effective strategies for acquiring vocabulary, grammar, "
        "and authentic usage patterns. The parallel reader format, placing the original text "
        "alongside a translation, allows learners to maintain comprehension while encountering "
        "unfamiliar structures. Over time, the brain internalises patterns implicitly, reducing "
        "cognitive load. This approach, sometimes called 'interlinear' or 'dual-language' "
        "reading, has roots in classical education — Latin students routinely studied texts with "
        "facing-page glosses. Modern research in second language acquisition (SLA) supports the "
        "combination of comprehensible input, deliberate vocabulary study, and spaced repetition, "
        "all of which can be embedded in a well-designed parallel reader."
    ),
}


def _ping_model(host: str, model: str) -> None:
    """Verify model is available; raise on failure."""
    r = requests.get(f"{host}/api/tags", timeout=10)
    r.raise_for_status()
    names = [m["name"] for m in r.json().get("models", [])]
    if model not in names:
        raise RuntimeError(f"Model {model!r} not found in Ollama. Available: {names}")


def _unload_model(host: str, model: str) -> None:
    """Force model eviction by setting keep_alive=0."""
    requests.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": " "}],
            "stream": False,
            "keep_alive": 0,
        },
        timeout=30,
    )
    time.sleep(2)  # Give Ollama time to evict


def _translate_one(chunk: str, label: str) -> float:
    """Translate chunk, return wall-clock seconds."""
    # Note: translate_chunk is a plain function, not @st.cache_data, so no .clear() needed.
    t0 = time.monotonic()
    translate_chunk(
        chunk=chunk,
        level="B1",
        style="Natural",
        region="Spain",
        fidelity="Closest meaning",
        include_literal=False,
        include_vocab=False,
        include_grammar=False,
        temperature=0.05,
    )
    return time.monotonic() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description="Translation latency benchmark")
    parser.add_argument("--model", default=OLLAMA_MODEL)
    parser.add_argument("--host", default=OLLAMA_HOST)
    parser.add_argument("--skip-cold", action="store_true", help="Skip cold-start measurement")
    args = parser.parse_args()

    os.environ["OLLAMA_MODEL"] = args.model
    os.environ["OLLAMA_HOST"] = args.host

    print(f"\nBenchmark: {args.model} at {args.host}\n")
    print(f"{'Chunk':<20} {'Cold (s)':>10} {'Warm (s)':>10}")
    print("-" * 44)

    _ping_model(args.host, args.model)

    for label, text in _SAMPLES.items():
        cold_s: float | None = None
        warm_s: float | None = None

        if not args.skip_cold:
            _unload_model(args.host, args.model)
            cold_s = _translate_one(text, label)

        # Warm run — model stays loaded from previous call
        warm_s = _translate_one(text, label)

        cold_str = f"{cold_s:.1f}" if cold_s is not None else "skipped"
        print(f"{label:<20} {cold_str:>10} {warm_s:>10.1f}")

    print()


if __name__ == "__main__":
    main()
