# Spanish Parallel Reader — Streamlit + Ollama

Turns English reading material into a side-by-side Spanish parallel reader with vocabulary, grammar notes, literal translations, exportable study notes, and an **output quality checker** that validates translations before they are shown or exported.

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- 8 GB RAM minimum for `qwen2.5:7b` (16 GB recommended); `qwen2.5:14b` requires ~12 GB
- Optional: NVIDIA GPU with 8 GB+ VRAM for faster inference

### Setup

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd spanish_parallel_reader
   ```

2. Create `.env` from `.env.example`:
   ```bash
   cp .env.example .env
   ```

3. Pull the default model (required before first run without Docker auto-pull):
   ```bash
   ollama pull qwen2.5:7b
   # Optional — higher quality, more RAM:
   ollama pull qwen2.5:14b
   ```

4. Start with Docker Compose (CPU):
   ```bash
   docker compose up --build
   ```

   For GPU support:
   ```bash
   docker compose -f docker-compose.gpu.yml up --build
   ```

   For macOS with Ollama running locally:
   ```bash
   docker compose -f docker-compose.mac.yml up --build
   ```

5. Open your browser:
   ```
   http://localhost:8502
   ```

> **First startup:** `qwen2.5:7b` (~4.7 GB) will be pulled automatically by the `ollama-pull` service. This takes 5–10 minutes on first run. Subsequent starts reuse the cached model.

### Run locally (without Docker)

```bash
pip install -r requirements.txt
OLLAMA_HOST=http://localhost:11434 streamlit run app.py
```

Requires Ollama running locally with `qwen2.5:7b` pulled.

---

## Models

| Model | Default | RAM | License |
|-------|---------|-----|---------|
| `qwen2.5:7b` | ✅ Yes | ~6–8 GB | Apache 2.0 |
| `qwen2.5:14b` | No | ~12 GB | Apache 2.0 |

Select the model in the **Ollama model** dropdown in the sidebar. The selection is applied per translation session.

To change the persisted default, set `OLLAMA_MODEL` in `.env`:
```env
OLLAMA_MODEL=qwen2.5:7b
AVAILABLE_OLLAMA_MODELS=qwen2.5:7b,qwen2.5:14b
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `qwen2.5:7b` | Default translation model |
| `AVAILABLE_OLLAMA_MODELS` | `qwen2.5:7b,qwen2.5:14b` | Comma-separated list shown in the UI model selector |
| `OLLAMA_HOST` | `http://ollama:11434` (Docker) / `http://localhost:11434` (local) | Ollama API endpoint |
| `OLLAMA_KEEP_ALIVE` | `-1` | Keep model in memory indefinitely |
| `OLLAMA_NUM_CTX` | `8192` | Context window in tokens |
| `MAX_CHARS_PER_CHUNK` | `2200` | Max characters per translation chunk |
| `APP_HOST_PORT` | `8502` | Host port for the Streamlit app |

---

## Troubleshooting

**Ollama not reachable**
- Docker: ensure the `ollama` service is healthy (`docker compose ps`).
- Local: ensure Ollama is running (`ollama serve`).
- Check `OLLAMA_HOST` in `.env` — use `http://ollama:11434` in Docker, `http://localhost:11434` locally.

**Model not found**
- Pull the model: `ollama pull qwen2.5:7b` or `ollama pull qwen2.5:14b`.
- In Docker, the `ollama-pull` service pulls automatically on startup.

**Insufficient memory**
- Use `qwen2.5:7b` instead of `qwen2.5:14b`.
- Reduce `MAX_CHARS_PER_CHUNK` in `.env` (e.g. `1200`).
- Lower `OLLAMA_NUM_CTX` (e.g. `4096`).

**Docker networking issues (macOS)**
- Use `docker compose -f docker-compose.mac.yml` which routes Ollama via `host.docker.internal`.

---

## Output Quality Checker

The checker automatically validates each generated reading pair after translation. It uses a layered approach to keep the UI fast while still catching meaningful translation errors.

### What the checker verifies

| Check | Description |
|-------|-------------|
| **Faithfulness** | Does the Spanish preserve the meaning of the English source? |
| **Hallucinations** | Did the model add facts, names, numbers, or dates not in the English? |
| **Omissions** | Is important meaning missing from the Spanish? |
| **Label correctness** | Are English/Spanish fields correctly assigned? |
| **Language quality** | Is the Spanish natural and grammatically appropriate? |
| **Completeness** | Is the Spanish translation suspiciously short or long? |

### Checker modes

| Mode | Behaviour |
|------|-----------|
| `off` | No checks run. |
| `fast` | Deterministic checks only. No extra model calls. Best performance. |
| `smart` | Deterministic checks for all pairs. LLM check only for risky pairs or sampled pairs. **Default.** |
| `strict` | Deterministic + LLM check for every pair. Slowest, highest confidence. |

Switch mode in the **🔍 Output Checker** expander in the sidebar, or set `CHECKER_MODE` in `.env`.

### How the checker appears in the UI

After translation, each reading pair in the **Parallel Reader** tab shows:

- A **status badge**: ✅ Passed / ℹ️ Info / ⚠️ Warning / ❌ Failed / Checker unavailable
- A short user-facing summary
- An expandable **Checker details** section with per-issue breakdown (when issues exist)

The checker badge appears after the comprehension question for each pair.

### Checker results in export

When you download the study notes Markdown, each pair includes a checker result block showing the status and summary. Detailed diagnostics are included only if **Show detailed diagnostics** is enabled.

### Blocking export on failure

By default (`CHECKER_REQUIRE_PASS=false`), export is always allowed and checker warnings are included in the exported Markdown.

If you enable **Require checker pass before export** in the sidebar (or set `CHECKER_REQUIRE_PASS=true`), Markdown export is blocked for sessions where any pair failed. You can still view all translations in the Reader tabs.

### Enabling and disabling the checker

- **Sidebar:** Use the **🔍 Output Checker** expander. Toggle "Enable output checker".
- **Environment:** Set `CHECKER_ENABLED=false` in `.env` to disable it for all sessions.

### Performance and caching

- Checker results are cached in Streamlit session state using a deterministic SHA-256 key based on the source text, Spanish output, model, mode, and prompt version.
- The checker only runs inside the translate button handler — never on normal Streamlit rerenders.
- Switching tabs or scrolling does **not** re-run the checker.
- Changing the checker model or mode will invalidate the cache so pairs are re-checked on the next translation.
- LLM calls are skipped entirely in `fast` mode. In `smart` mode they are skipped for low-risk pairs.
- Deterministic checks (number drift, language swap, empty fields, length ratio) run for every pair with no model call.
- Use `CHECKER_SAMPLE_RATE` to further reduce LLM call frequency in smart mode.

### Privacy

All checker calls go to the same local Ollama endpoint as translation. No text is sent to any external service.

---

## Model Information

### Default Model: Qwen2.5 7B

| Property | Value |
|----------|-------|
| **License** | Apache 2.0 (commercial use allowed) |
| **Size on disk** | ~4.7 GB |
| **Languages** | English, Spanish, and many others |
| **CPU speed** | 2–8 seconds per 200-word chunk |
| **GPU speed** | 1–2 seconds per 200-word chunk |

### Optional Model: Qwen2.5 14B

| Property | Value |
|----------|-------|
| **License** | Apache 2.0 |
| **Size on disk** | ~9 GB |
| **RAM required** | ~12 GB |
| **Quality** | Higher quality than 7B; slower |

> Select between models in the **Ollama model** dropdown in the sidebar. No restart required.

### Changing the default model

Edit `.env` and set `OLLAMA_MODEL`:

```env
OLLAMA_MODEL=qwen2.5:7b
```

Then restart:
```bash
docker compose down
docker compose up
```

### Using a separate checker model

By default the checker uses the same model as translation (`OLLAMA_MODEL`). To use a different model:

```env
CHECKER_MODEL=qwen2.5:7b
```

The checker model must already be pulled in Ollama.

---

## Checker Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHECKER_ENABLED` | `true` | Enable or disable the checker entirely. |
| `CHECKER_MODE` | `smart` | `off` / `fast` / `smart` / `strict` |
| `CHECKER_MODEL` | *(empty — uses OLLAMA_MODEL)* | Override the model used for LLM checking. |
| `CHECKER_SAMPLE_RATE` | `1.0` | Fraction of low-risk pairs to LLM-check in smart mode. |
| `CHECKER_MAX_CHARS` | `2500` | Max characters sent per field to the LLM checker. |
| `CHECKER_REQUIRE_PASS` | `false` | Block Markdown export when any pair fails the checker. |
| `CHECKER_TIMEOUT_SECONDS` | `45` | Timeout for each LLM checker call. |
| `CHECKER_CACHE_ENABLED` | `true` | Cache checker results in session state. |
| `CHECKER_DETAILED_DIAGNOSTICS` | `false` | Show per-issue breakdown in the UI. |
| `CHECKER_LLM_ENABLED` | `true` | Run LLM checks (when false, deterministic only). |
| `CHECKER_BATCH_SIZE` | `1` | Pairs checked per batch (informational). |

---

## Changing the App Port

Default port is `8502`. To use a different port, edit `.env`:

```env
APP_HOST_PORT=9000
```

Then restart: `docker compose restart app`

---

## Hardware Requirements

| Config | RAM | VRAM | Speed |
|--------|-----|------|-------|
| Minimal (CPU) | 8 GB | — | Slow (10+ sec/chunk) |
| Recommended (CPU) | 16 GB | — | Moderate (3–5 sec/chunk) |
| Recommended (GPU) | 8 GB | 8 GB+ | Fast (1–2 sec/chunk) |

---

## Privacy & Data

- **Local processing:** All text and translations run on your local machine via Ollama. No data is sent to external APIs.
- **Uploaded files:** Extracted into memory only; not written to disk.
- **Model storage:** Cached locally in the `ollama_models` Docker volume.
- **Checker:** Uses the same local Ollama endpoint. No user text leaves your machine.

---

## Supported File Formats

- PDF
- DOCX (Microsoft Word)
- TXT
- Markdown

---

## Language Levels and Styles

### CEFR Levels
- **A1** — Absolute beginner
- **A2** — Elementary
- **B1** — Intermediate (default)
- **B2** — Upper-intermediate
- **C1** — Advanced

### Spanish Regions
- **Neutral** — Internationally acceptable Spanish
- **Latin American** — Mexican/Colombian Spanish
- **European / Spain** — Spain Spanish

### Translation Styles
- Natural Spanish, Learner-friendly, Literal but readable, Literary, Journalistic

### Translation Fidelity
- Balanced, Closest meaning, Simpler learner wording, Preserve literary style

---

## Output Columns Explained

| Column | Content |
|--------|---------|
| **English** | Original source text, unchanged |
| **Español** | Natural Spanish translation at the selected level and region |
| **Literal Spanish** | Word-order-preserving translation showing English structure |
| **Vocabulary** | Spanish terms → English meanings (with usage notes) |

---

## Troubleshooting

### Ollama connection error
```
Could not reach Ollama at http://ollama:11434
```
1. Check containers are running: `docker compose ps`
2. Check Ollama logs: `docker compose logs ollama`
3. Wait for model pull: `docker compose logs ollama-pull`
4. Restart: `docker compose restart`

### Model pull timeout
Pull manually:
```bash
docker exec spanish-reader-ollama ollama pull aya-expanse:8b
```

### Out of memory
1. Reduce `MAX_CHARS_PER_CHUNK` in `.env` (default 2200)
2. Reduce the "Chunks to process" slider in the sidebar
3. Switch to a smaller model (e.g., `mistral:7b`)

### Slow translation
1. Lower the temperature slider in the sidebar
2. Reduce chunk size
3. Use GPU if available

### Checker timeout
- Increase `CHECKER_TIMEOUT_SECONDS` in `.env`
- Switch to `fast` mode for deterministic-only checking
- Set `CHECKER_MODE=off` to disable checking entirely

### Checker model not found
```
Checker model 'xyz' not found.
```
Either set `CHECKER_MODEL=` (blank, to use the translation model) or pull the model:
```bash
docker exec spanish-reader-ollama ollama pull xyz
```

### Checker is unavailable / checker badge shows "unavailable"
The checker fails open — your translation is always preserved. Check Ollama connectivity and model availability. The checker badge shows the reason.

---

## Architecture

- **Streamlit** — Web UI and session management
- **Ollama** — Local LLM inference with structured output
- **Pydantic** — Schema validation and JSON parsing
- **checker.py** — Output quality checker (layered: deterministic → risk score → LLM)
- **Docker Compose** — Containerized deployment (Ollama + app)
- **PyMuPDF / python-docx** — PDF and DOCX text extraction

---

## Known Checker Limitations

- Deterministic checks use heuristic word lists and regex — they are approximate and may produce false positives.
- The LLM checker relies on the model's own bilingual understanding and can make mistakes.
- Automated checking cannot guarantee translation accuracy — human review is always recommended for important content.
- The checker does not evaluate cultural adaptation, register subtleties, or dialect-specific vocabulary.
- Checker results are session-scoped and not persisted across browser sessions.

---

## License

This project is provided for educational and personal use.

**The default model (Aya Expanse 8B) is released under CC-BY-NC.** Users are responsible for verifying model license compatibility with their use case.


## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- 16 GB RAM recommended (8 GB minimum for `aya-expanse:8b` on CPU)
- Optional: NVIDIA GPU with 8 GB+ VRAM for faster inference

### Setup

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd spanish_translater
   ```

2. Create `.env` from `.env.example`:
   ```bash
   cp .env.example .env
   ```

3. Start with Docker Compose (CPU):
   ```bash
   docker compose up --build
   ```

   For GPU support:
   ```bash
   docker compose -f docker-compose.gpu.yml up --build
   ```

4. Open your browser:
   ```
   http://localhost:8502
   ```

> **First startup:** The `aya-expanse:8b` model (~5 GB) will be pulled automatically. This takes 5–10 minutes depending on your internet speed. Subsequent starts reuse the cached model.

---

## Model Information

### Default Model: Aya Expanse 8B

| Property | Value |
|----------|-------|
| **License** | CC-BY-NC (non-commercial use only) |
| **Size on disk** | ~5 GB |
| **Languages** | English, Spanish, and 100+ others |
| **CPU speed** | 2–10 seconds per 200-word chunk |
| **GPU speed** | 1–3 seconds per 200-word chunk |

> **Important:** Verify that your use case (personal study, educational, research) is compatible with the CC-BY-NC license before deploying this application commercially. [Learn more](https://huggingface.co/CohereForAI/aya-expanse-8b)

### Changing the Model

Edit `.env` and set `OLLAMA_MODEL`:

```env
OLLAMA_MODEL=qwen3:14b
```

Then restart:
```bash
docker compose down
docker compose up
```

---

## Changing the App Port

Default port is `8502`. To use a different port, edit `.env`:

```env
APP_HOST_PORT=9000
```

Then restart: `docker compose restart app`

---

## Hardware Requirements

| Config | RAM | VRAM | Speed |
|--------|-----|------|-------|
| Minimal (CPU) | 8 GB | — | Slow (10+ sec/chunk) |
| Recommended (CPU) | 16 GB | — | Moderate (3–5 sec/chunk) |
| Recommended (GPU) | 8 GB | 8 GB+ | Fast (1–2 sec/chunk) |

---

## Privacy & Data

- **Local processing:** All text and translations run on your local machine via Ollama. No data is sent to external APIs.
- **Uploaded files:** Extracted into memory only; not written to disk.
- **Model storage:** Cached locally in the `ollama_models` Docker volume.

---

## Supported File Formats

- PDF
- DOCX (Microsoft Word)
- TXT
- Markdown

---

## Language Levels and Styles

### CEFR Levels
- **A1** — Absolute beginner
- **A2** — Elementary
- **B1** — Intermediate (default)
- **B2** — Upper-intermediate
- **C1** — Advanced

### Spanish Regions
- **Neutral** — Internationally acceptable Spanish
- **Latin American** — Mexican/Colombian Spanish
- **European / Spain** — Spain Spanish

### Translation Styles
- Natural Spanish, Learner-friendly, Literal but readable, Literary, Journalistic

### Translation Fidelity
- Balanced, Closest meaning, Simpler learner wording, Preserve literary style

---

## Output Columns Explained

| Column | Content |
|--------|---------|
| **English** | Original source text, unchanged |
| **Español** | Natural Spanish translation at the selected level and region |
| **Literal Spanish** | Word-order-preserving translation showing English structure |
| **Vocabulary** | Spanish terms → English meanings (with usage notes) |

---

## Troubleshooting

### Ollama connection error
```
Could not reach Ollama at http://ollama:11434
```
1. Check containers are running: `docker compose ps`
2. Check Ollama logs: `docker compose logs ollama`
3. Wait for model pull: `docker compose logs ollama-pull`
4. Restart: `docker compose restart`

### Model pull timeout
Pull manually:
```bash
docker exec spanish-reader-ollama ollama pull aya-expanse:8b
```

### Out of memory
1. Reduce `MAX_CHARS_PER_CHUNK` in `.env` (default 2200)
2. Reduce the "Chunks to process" slider in the sidebar
3. Switch to a smaller model (e.g., `mistral:7b`)

### Slow translation
1. Lower the temperature slider in the sidebar
2. Reduce chunk size
3. Use GPU if available

---

## Architecture

- **Streamlit** — Web UI and session management
- **Ollama** — Local LLM inference with structured output
- **Pydantic** — Schema validation and JSON parsing
- **Docker Compose** — Containerized deployment (Ollama + app)
- **PyMuPDF / python-docx** — PDF and DOCX text extraction

---

## License

This project is provided for educational and personal use.

**The default model (Aya Expanse 8B) is released under CC-BY-NC.** Users are responsible for verifying model license compatibility with their use case.

