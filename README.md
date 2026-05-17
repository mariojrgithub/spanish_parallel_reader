
# Spanish Parallel Reader

## Overview
Spanish Parallel Reader is a Streamlit application that turns English reading material into a side-by-side Spanish parallel reader with vocabulary, grammar notes, literal translations, exportable study notes, and an output quality checker. All translation and checking runs locally using Ollama LLMs—no cloud APIs required.

## Features
- Translate and align English text into Spanish at various CEFR levels
- Vocabulary, grammar notes, literal translation, and comprehension questions
- Output quality checker (deterministic and LLM-based)
- Optional MongoDB-backed translation history with restore/delete/rename
- Export study notes as Markdown
- PDF, DOCX, TXT, and Markdown file support
- Text-to-speech (browser-based)

## Project Structure

```
app.py                # Main Streamlit app entry point
checker.py            # Output quality checker logic
text_processing.py    # Text extraction, cleaning, chunking
tts_component.py      # Browser-based TTS controls
infrastructure/       # Ollama client helpers
benchmarks/           # Benchmark scripts
tests/                # Pytest-based test suite
Dockerfile            # App container
docker-compose.yml    # Default (CPU) Docker Compose
docker-compose.gpu.yml# GPU-enabled Docker Compose
docker-compose.mac.yml# Mac-specific Docker Compose (host Ollama)
requirements.txt      # Python dependencies
.env.example          # Example environment config
.streamlit/config.toml# Streamlit config
```

## Requirements
- Python 3.12+ (for local development)
- pip (for dependency management)
- Docker & Docker Compose (for containerized usage)
- Ollama (for local LLM inference)

## Configuration
Copy `.env.example` to `.env` and edit as needed. All variables have safe defaults.

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `qwen2.5:7b` | Default translation model |
| `AVAILABLE_OLLAMA_MODELS` | `qwen2.5:7b,qwen2.5:3b,qwen2.5:14b` | Models shown in UI selector |
| `OLLAMA_HOST` | `http://ollama:11434` (Docker) / `http://localhost:11434` (local) | Ollama API endpoint |
| `APP_HOST_PORT` | `8502` | Host port for Streamlit app |
| `OLLAMA_KEEP_ALIVE` | `-1` | Keep model in memory |
| `OLLAMA_NUM_CTX` | `8192` | Context window size |
| `MAX_CHARS_PER_CHUNK` | `2200` | Max chars per translation chunk |
| `TRANSLATION_CACHE_MAX_ENTRIES` | `50` | Translation cache size |
| `TRANSLATION_INCLUDE_ENRICHMENTS` | `true` | Include vocab/grammar/literal |
| `MONGO_ENABLED` | `true` | Enable MongoDB-backed translation history |
| `MONGO_URI` | `mongodb://localhost:27017` locally / `mongodb://mongo:27017` in Docker | MongoDB connection string |
| `MONGO_DB` | `spanish_parallel_reader` | MongoDB database name |
| `MONGO_HISTORY_COLLECTION` | `translation_history` | MongoDB history collection |
| `MONGO_USER_ID` | `default-user` | Logical user key for history partitioning |
| `MONGO_HISTORY_LIMIT` | `25` | Max history entries shown in the UI |
| `MONGO_SAVE_SOURCE_TEXT` | `true` | Persist cleaned source text for full UI restore |
| `CHECKER_ENABLED` | `true` | Enable output checker |
| `CHECKER_MODE` | `smart` | Checker mode (off/fast/smart/strict) |
| `CHECKER_REQUIRE_PASS` | `false` | Block export on checker fail |
| `ENABLE_TTS` | `true` | Enable browser TTS |
| `TTS_LANGUAGE` | `es-MX` | TTS language |
| `TTS_RATE` | `0.9` | TTS speech rate |

See `.env.example` for all options and documentation.

## Local Development Setup

1. Install Python 3.12+ and pip.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start Ollama and pull the required model:
   ```bash
   ollama serve
   ollama pull qwen2.5:7b
   # Optionally: ollama pull qwen2.5:14b
   ```
5. Run the app:
   ```bash
   OLLAMA_HOST=http://localhost:11434 streamlit run app.py
   ```

## macOS Local Ollama Setup

1. [Install Ollama for macOS](https://ollama.com/download).
2. Start Ollama:
   ```bash
   ollama serve
   ```
3. Pull the required model(s):
   ```bash
   ollama pull qwen2.5:7b
   ollama pull qwen2.5:14b  # optional, more RAM required
   ```
4. To run the app locally (no Docker):
   ```bash
   pip install -r requirements.txt
   OLLAMA_HOST=http://localhost:11434 streamlit run app.py
   ```
5. To run the app in Docker but use your Mac's Ollama:
   ```bash
   docker compose -f docker-compose.mac.yml up --build
   # This uses host.docker.internal:11434 for OLLAMA_HOST and a local Mongo container by default
   ```
6. Open http://localhost:8502 in your browser.

**Troubleshooting (macOS):**
- If the Docker app cannot reach Ollama, ensure Ollama is running and the compose file is `docker-compose.mac.yml`.
- If you see connection errors, check that `OLLAMA_HOST` is set to `http://host.docker.internal:11434` in the Docker environment.
- If the model is missing, run `ollama pull qwen2.5:7b` on your Mac.
- For port conflicts, change `APP_HOST_PORT` in `.env` and restart Docker Compose.

## Running with Docker


**Default (CPU):**
```bash
docker compose up --build
```
*This profile runs Ollama plus MongoDB in Docker. The 7B and 14B models are also available as options.*

**With GPU:**
```bash
docker compose -f docker-compose.gpu.yml up --build
```
*This profile keeps the existing GPU Ollama behavior and adds MongoDB persistence.*

**On macOS with local Ollama:**
```bash
docker compose -f docker-compose.mac.yml up --build
```
*This profile still uses host Ollama and now adds MongoDB in Docker by default. Override `MONGO_URI` if you prefer a host Mongo instance.*

Open http://localhost:8502 in your browser.

## Usage

1. Upload a file or paste English text.
2. Select your desired CEFR level, region, and translation style.
3. Click "Translate".
4. Review the Spanish output, vocabulary, grammar notes, and literal translation.
5. Use the History section to reload prior translations without re-running the model.
6. Use the output checker to validate translations.
7. Export study notes as Markdown.

## Testing and Quality Checks

Run all tests with:
```bash
pytest
```

## Troubleshooting

- **Ollama not reachable:**
  - Ensure Ollama is running (`ollama serve`).
  - For Docker, check the `ollama` service is healthy (`docker compose ps`).
  - Check `OLLAMA_HOST` in `.env`.
- **Model not found:**
  - Run `ollama pull qwen2.5:7b` (or `qwen2.5:14b`).
  - In Docker, the `ollama-pull` service pulls automatically on startup.
- **Docker networking issues (macOS):**
  - Use `docker-compose.mac.yml` and ensure `OLLAMA_HOST` is `http://host.docker.internal:11434`.
- **Out of memory:**
  - Use `qwen2.5:7b` instead of `qwen2.5:14b`.
  - Lower `MAX_CHARS_PER_CHUNK` or `OLLAMA_NUM_CTX` in `.env`.
- **Port conflicts:**
  - Change `APP_HOST_PORT` in `.env` and restart Docker Compose.
- **Checker issues:**
  - See sidebar for checker status and logs.
- **Dependency install issues:**
  - Ensure Python 3.12+ and pip are installed.
  - Use a clean virtual environment.

## Maintenance Notes

- When changing models, ports, or environment variables, update `.env.example` and this README.
- Keep Docker Compose files and environment variable docs in sync.
- Test both local and Docker workflows after major changes.

---



## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `qwen2.5:7b` | Default translation model |
| `AVAILABLE_OLLAMA_MODELS` | `qwen2.5:7b,qwen2.5:14b` | Comma-separated list shown in the UI model selector |
| `OLLAMA_HOST` | `http://ollama:11434` (Docker) / `http://localhost:11434` (local) | Ollama API endpoint |
| `OLLAMA_KEEP_ALIVE` | `-1` | Keep model in memory indefinitely |
| `OLLAMA_NUM_CTX` | `8192` | Context window in tokens |
| `OLLAMA_TEMPERATURE` | `0.1` | Translation temperature (lower = more deterministic JSON output) |
| `OLLAMA_TOP_P` | `0.9` | Top-p nucleus sampling for translation |
| `OLLAMA_REQUEST_TIMEOUT` | `240` | Timeout in seconds for Ollama translation calls |
| `MAX_CHARS_PER_CHUNK` | `2200` | Max characters per translation chunk |
| `APP_HOST_PORT` | `8502` | Host port for the Streamlit app |

---


## Troubleshooting

- **Ollama not reachable:**
   - Ensure Ollama is running (`ollama serve`).
   - For Docker, check the `ollama` service is healthy (`docker compose ps`).
   - Check `OLLAMA_HOST` in `.env` — use `http://ollama:11434` in Docker, `http://localhost:11434` locally.
- **Model not found:**
   - Run `ollama pull qwen2.5:3b`, `ollama pull qwen2.5:7b`, or `ollama pull qwen2.5:14b` as needed.
   - In Docker, the `ollama-pull` service pulls automatically on startup.
- **Insufficient memory:**
   - Use `qwen2.5:3b` or `qwen2.5:7b` instead of `qwen2.5:14b`.
   - Reduce `MAX_CHARS_PER_CHUNK` in `.env` (e.g. `1200`).
   - Lower `OLLAMA_NUM_CTX` (e.g. `4096`).
- **Docker networking issues (macOS):**
   - Use `docker compose -f docker-compose.mac.yml` which routes Ollama via `host.docker.internal`.
- **Out of memory:**
   - Lower chunk size or use a smaller model.
- **Port conflicts:**
   - Change `APP_HOST_PORT` in `.env` and restart Docker Compose.
- **Checker/model errors:**
   - See sidebar for checker status and logs. Ensure the checker model is pulled.
- **Dependency install issues:**
   - Ensure Python 3.12+ and pip are installed. Use a clean virtual environment.

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


### Model Defaults by Docker Compose Profile

| Compose Profile         | Default Model   | Notes                                      |
|------------------------|-----------------|---------------------------------------------|
| `docker-compose.yml`   | `qwen2.5:3b`    | Best for CPU/low-memory systems             |
| `docker-compose.gpu.yml` | `qwen2.5:7b`  | Best for GPU, higher quality/speed          |
| `docker-compose.mac.yml` | `qwen2.5:7b`  | Best for Mac with local Ollama              |

**All profiles**: The 3B and 14B models are available as options in the UI. 14B requires more RAM.

#### Qwen2.5 3B (CPU default)
| Property         | Value                |
|------------------|---------------------|
| **License**      | Apache 2.0          |
| **Size on disk** | ~2.5 GB             |
| **RAM required** | ~4–6 GB             |
| **Quality**      | Fastest, lowest RAM |

#### Qwen2.5 7B (GPU/mac default)
| Property         | Value                |
|------------------|---------------------|
| **License**      | Apache 2.0          |
| **Size on disk** | ~4.7 GB             |
| **RAM required** | ~8 GB               |
| **Quality**      | Higher quality      |

#### Qwen2.5 14B (optional)
| Property         | Value                |
|------------------|---------------------|
| **License**      | Apache 2.0          |
| **Size on disk** | ~9 GB               |
| **RAM required** | ~12 GB              |
| **Quality**      | Highest, slowest    |

> Select between models in the **Ollama model** dropdown in the sidebar. No restart required.


### Changing the default model

Edit `.env` and set `OLLAMA_MODEL` to your preferred model (e.g. `qwen2.5:3b`, `qwen2.5:7b`, or `qwen2.5:14b`).

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
| `CHECKER_MODE` | `smart` | `off` / `fast` / `smart` / `strict`. The sidebar defaults to this value on first load, so Docker deployments respect `CHECKER_MODE=fast` without any user action. |
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

This project is provided for educational and personal use. The default models are Apache 2.0 licensed. Users are responsible for verifying model license compatibility with their use case.

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



