# Spanish Parallel Reader — Streamlit + Ollama

Turns English reading material into a side-by-side Spanish parallel reader with vocabulary, grammar notes, literal translations, and exportable study notes.

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

