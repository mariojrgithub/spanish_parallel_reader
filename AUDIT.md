# Spanish Parallel Reader — Performance Audit Report

> **Scope:** Identify and implement optimizations that reduce translation completion time and perceived responsiveness.
> **Constraint:** Local-only (Ollama). No cloud APIs. No over-engineering.

---

## 1. Architecture Overview

| Layer | Technology | Notes |
|-------|-----------|-------|
| UI | Streamlit 1.45.0 | Script model — full re-run on every interaction |
| Translation engine | `app.py` | `translate_chunk()` streams from Ollama `/api/chat` |
| Quality checker | `checker.py` | 3-layer: deterministic → risk score → LLM |
| Model server | Ollama (Docker service) | `qwen2.5:7b` default, `keep_alive=-1` |
| Parallelism | `ThreadPoolExecutor` | Checker batch calls only; translation is sequential per chunk |
| Storage | `st.session_state` | In-memory; checker results cached by SHA-256 key |

The pipeline for a single document is:

```
split_into_chunks() → [translate_chunk() × N chunks] → batch_check_pairs() → render
```

Chunks are translated sequentially (streaming output rendered after each chunk). The checker runs concurrently across pairs using a `ThreadPoolExecutor`.

---

## 2. Bottleneck Analysis

### 2a. Time budget per translation session

For a ~1,000-word document split into ~2 chunks with `qwen2.5:7b` on CPU:

| Step | Estimated wall time | Notes |
|------|-------------------|-------|
| Model load (cold) | 5–15 s | Eliminated by `keep_alive=-1` after first call |
| Translation × 2 chunks | 60–180 s | Dominant cost; LLM inference |
| Checker (fast mode) | < 1 s | Deterministic only |
| Checker (smart mode) | 10–60 s | Adds 1–3 LLM calls for risky pairs |
| Checker (strict mode) | 30–120 s | Adds LLM call for every pair |

**Conclusion:** LLM inference dominates. Checker mode selection is the highest-impact configuration knob.

### 2b. Pre-existing optimizations (already correct)

The codebase already contained several good choices:

- `keep_alive=-1` — model stays loaded; no reload penalty between requests
- `requests.Session()` reused at module level in both `app.py` and `checker.py` — TCP connection pooled
- Streaming response with 150-char `on_token` callback — perceived latency reduced; user sees output immediately
- `@st.cache_data(ttl=120)` on `check_ollama()` — connectivity probe not repeated every rerender
- `@st.cache_data` on `split_into_chunks()` — chunking not recomputed on same text
- Adaptive `num_predict` budget (`_estimate_num_predict`) — avoids over-generating tokens
- `num_predict=512` cap on checker calls — checker never runs away
- Checker result cache keyed by SHA-256 — identical pairs not rechecked within session
- Regex patterns for text cleaning precompiled at module level
- `TranslationResponse` JSON schema precomputed at module level
- `--server.fileWatcherType=none` in Dockerfile — removes Streamlit's inotify overhead
- `OLLAMA_NUM_CTX=8192` — large enough for full chunks, not wastefully large

---

## 3. Bugs Found and Fixed

### Bug 1 — CRITICAL: Checker mode hardcoded to "smart", ignoring `CHECKER_MODE` env var

**File:** `app.py`, sidebar checker expander
**Impact:** In Docker deployments with `CHECKER_MODE=fast` in `.env`, the sidebar selectbox was initialized with `index=2` (hardcoded "smart"), completely ignoring the environment variable. Every translation session would run LLM-based checking on all pairs regardless of configuration. This added **30–120 seconds** of extra LLM calls per session on CPU hardware.

**Fix:** All five checker sidebar widgets now compute their initial values from the corresponding environment variables (`CHECKER_ENABLED`, `CHECKER_MODE`, `CHECKER_LLM_ENABLED`, `CHECKER_REQUIRE_PASS`, `CHECKER_DETAILED_DIAGNOSTICS`). The Docker default of `CHECKER_MODE=fast` now correctly produces a first-load UI in fast (deterministic-only) mode.

---

### Bug 2 — Test failures: wrong mock target in `test_checker.py`

**File:** `tests/test_checker.py`, `test_mocked_llm_checker_pass` and `test_mocked_llm_checker_fail`
**Impact:** Both tests patched `"requests.post"` but `checker.py` calls via `_http_session.post` (a `requests.Session` instance). The mock never intercepted the call; the real Ollama endpoint was hit and failed with `ConnectionError`, producing a "checker unavailable" result. Both tests reported false failures.

**Fix:** Changed both mock targets to `"checker._http_session.post"`.

---

### Bug 3 — Benchmark crash: `translate_chunk.clear()` on non-cached function

**File:** `benchmarks/bench_translation.py`
**Impact:** `translate_chunk` is not decorated with `@st.cache_data`, so it has no `.clear()` method. Any benchmark run would crash immediately with `AttributeError`.

**Fix:** Removed the `translate_chunk.clear()` call; added a comment clarifying that no cache-busting is needed.

---

### Bug 4 — Test collection failure: Streamlit stub missing `cache_data` and full session_state API

**File:** `tests/conftest.py`
**Impact:** `test_translate.py` failed to collect entirely because importing `app.py` triggered `@st.cache_data` at module level, and the existing partial stub (added by `setdefault` in `test_schemas.py`) lacked `cache_data`. Additionally, `app.py` uses `st.session_state.attr = value` (attribute-style), `st.session_state.pop()`, and `st.columns([w1, w2])` (must unpack to N items) — none of which worked with the old stub.

**Fix:** Replaced the per-test-module stub fragments with a single module-level stub in `conftest.py` that provides:
- `cache_data` / `cache_resource` supporting both `@st.cache_data` and `@st.cache_data(ttl=120)` call forms
- `_SessionState` — supports attribute get/set, item get/set, `in`, `get()`, `pop()`, `setdefault()`, `keys()`, `items()`, `values()`
- `columns(spec)` — returns a list of `MagicMock` of length `len(spec)` or `int(spec)`
- `tabs(labels)` — returns a list of `MagicMock` of length `len(labels)`
- All other UI components as `MagicMock()`
- Stubs for `fitz`, `docx`, `pandas` in `sys.modules`

---

## 4. Optimizations Implemented

### 4a. `OLLAMA_TEMPERATURE` / `OLLAMA_TOP_P` / `OLLAMA_REQUEST_TIMEOUT` — environment-configurable

**Files:** `app.py`, `.env`, `.env.example`, `docker-compose.yml`, `docker-compose.gpu.yml`, `README.md`

Previously, temperature was hardcoded at `0.1` in the slider default (not a bug, but not configurable), `top_p` was not set at all (Ollama server default used), and both call timeouts were hardcoded integers (`240` / `120`).

**Changes:**
- Added `OLLAMA_TEMPERATURE`, `OLLAMA_TOP_P`, `OLLAMA_REQUEST_TIMEOUT` module constants to `app.py`
- Both `translate_chunk` and `retranslate_pair` now pass `top_p` in their options dict
- Both functions use `OLLAMA_REQUEST_TIMEOUT` instead of literals
- Temperature slider default is `OLLAMA_TEMPERATURE` (respects `.env` on first load)
- All three variables documented in `.env`, `.env.example`, all compose files, and README

**Performance note:** `top_p=0.9` with `temperature=0.1` slightly tightens token sampling, which can reduce the chance of hallucinations requiring a retranslation pass.

### 4b. Removed dead code: `_TRANSLATION_SCHEMA_STR`

**File:** `app.py`

A multi-line JSON string of the Pydantic schema was computed at module startup and assigned to `_TRANSLATION_SCHEMA_STR`, but the variable was never referenced. Removed.

---

## 5. Optimizations Considered But Not Implemented

### 5a. Parallel chunk translation

Translating all chunks in parallel with `ThreadPoolExecutor` would reduce wall time proportionally to chunk count. However:
- Streaming per-chunk output (the current UX) requires sequential execution to render correctly
- Running 2+ concurrent Ollama LLM calls on a CPU system would cause resource contention and likely increase total time
- On GPU, parallelism would require either multiple model instances or careful batching

**Verdict:** Not implemented. Would degrade UX (no streaming) and likely increase total time on CPU. Revisit if GPU deployment with multiple GPU instances is targeted.

### 5b. Prompt caching / KV-cache prefix sharing

Ollama does not currently expose prompt prefix caching controls. The system prompt is repeated across every chunk call. There is no mechanism to share KV-cache across calls.

**Verdict:** Not implementable at the application layer.

### 5c. Smaller/faster models

`qwen2.5:3b` would be ~2× faster than `qwen2.5:7b` on CPU. Translation quality would be lower.

**Verdict:** User preference. Not changed. The model selector in the UI already supports switching.

### 5d. Replacing streaming with non-streaming for small chunks

For very short chunks, the overhead of streaming is negligible and streaming provides much better perceived responsiveness. No benefit to removing it.

**Verdict:** Not changed.

### 5e. Async HTTP client (httpx/aiohttp)

Replacing `requests` with `httpx` in async mode could reduce per-call overhead slightly. However, Streamlit's threading model makes true `asyncio` integration complex, and the current bottleneck is LLM inference time (seconds), not HTTP latency (milliseconds). The gain would be unmeasurable.

**Verdict:** Not implemented. Premature optimization.

---

## 6. LangGraph Evaluation

**Recommendation: Do not add LangGraph.**

The translation pipeline is:

```
split → translate_chunk × N → check_pair × N (parallel) → retranslate if needed
```

This is a linear DAG with one conditional branch (retranslate on check failure). It is fully implemented in ~150 lines of direct Python code with `ThreadPoolExecutor` for the parallel checker step.

LangGraph would provide:
- State machine with typed state between nodes
- Built-in graph visualization
- Retry/interrupt primitives
- Streaming at the graph level

LangGraph would cost:
- A new dependency (~50 MB)
- Significant refactor of the translation + checker orchestration
- More complex debugging (graph state vs. direct function calls)
- No measurable performance gain — the bottleneck is LLM inference, not orchestration overhead

**Conclusion:** LangGraph is designed for complex agentic workflows with branching, cycles, and human-in-the-loop steps. This pipeline has none of those. The existing orchestration code is simpler, faster to debug, and has no correctness gaps that a graph framework would fix. Do not add LangGraph.

---

## 7. Test Suite Results

### Before this audit

| Status | Count |
|--------|-------|
| Passed | 82 |
| Failed | 2 (`test_mocked_llm_checker_pass`, `test_mocked_llm_checker_fail`) |
| Collection errors | 1 (`test_translate.py` — 17 tests uncollectable) |
| **Total collectable** | **84** |

### After this audit

```
101 passed in 2.89s
```

All 101 tests pass. The 17 `test_translate.py` tests are now collected and passing.

---

## 8. Files Changed

| File | Change |
|------|--------|
| `app.py` | Added `OLLAMA_TEMPERATURE`, `OLLAMA_TOP_P`, `OLLAMA_REQUEST_TIMEOUT` constants; added `top_p` to both Ollama options dicts; changed hardcoded timeouts to use constant; fixed temperature slider default; fixed checker sidebar to respect all 5 env vars on first render; removed `_TRANSLATION_SCHEMA_STR` dead code; fixed options-dict indentation |
| `tests/conftest.py` | Replaced partial stub with complete module-level Streamlit stub supporting `cache_data`, attribute-style `session_state`, `columns`, `tabs`, and optional-import stubs |
| `tests/test_checker.py` | Fixed mock target from `"requests.post"` to `"checker._http_session.post"` in two tests |
| `benchmarks/bench_translation.py` | Removed `translate_chunk.clear()` crash |
| `.env` | Added `OLLAMA_TEMPERATURE=0.1`, `OLLAMA_TOP_P=0.9`, `OLLAMA_REQUEST_TIMEOUT=240` |
| `.env.example` | Same additions as `.env` |
| `docker-compose.yml` | Added `OLLAMA_NUM_CTX`, `OLLAMA_TEMPERATURE`, `OLLAMA_TOP_P`, `OLLAMA_REQUEST_TIMEOUT` to `app` service environment |
| `docker-compose.gpu.yml` | Same additions as `docker-compose.yml` |
| `README.md` | Added 3 env var rows to the table; clarified `CHECKER_MODE` sidebar behavior |

---

## 9. Running the Benchmark

Once Ollama is running with a model loaded:

```bash
# From the project root
python benchmarks/bench_translation.py --host http://localhost:11434

# Or inside Docker
docker compose exec app python benchmarks/bench_translation.py
```

The benchmark measures cold and warm latency for three text sizes (tiny / small / medium) and prints tokens-per-second throughput. Use this to baseline before and after model or hardware changes.

---

## 10. Recommended Configuration for Production Docker Deployment

```env
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_KEEP_ALIVE=-1
OLLAMA_NUM_CTX=8192
OLLAMA_TEMPERATURE=0.1
OLLAMA_TOP_P=0.9
OLLAMA_REQUEST_TIMEOUT=240
CHECKER_MODE=fast          # deterministic only — no extra LLM calls
CHECKER_LLM_ENABLED=false  # belt-and-suspenders: ensure LLM checker is off
CHECKER_SAMPLE_RATE=1.0
CHECKER_REQUIRE_PASS=false
MAX_CHARS_PER_CHUNK=2200
```

Switch to `CHECKER_MODE=smart` if translation quality validation is more important than speed.
