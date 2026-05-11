import json
import os
import re
from typing import List, Literal

import fitz
import pandas as pd
import requests
import streamlit as st
from docx import Document
from pydantic import BaseModel, Field, ValidationError


# -----------------------------
# Configuration
# -----------------------------

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "aya-expanse:8b")
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "10m")
DEFAULT_MAX_CHARS = int(os.getenv("MAX_CHARS_PER_CHUNK", "2200"))

Difficulty = Literal["A1", "A2", "B1", "B2", "C1", "C2"]


# -----------------------------
# Structured output schemas
# -----------------------------

class VocabularyItem(BaseModel):
    spanish: str = Field(description="Spanish word or phrase")
    english: str = Field(description="English meaning")
    note: str = ""


class ReadingPair(BaseModel):
    english: str
    spanish: str
    literal_spanish: str = ""
    vocabulary: List[VocabularyItem] = Field(default_factory=list)
    grammar_notes: List[str] = Field(default_factory=list)
    comprehension_question_spanish: str = ""
    difficulty: Difficulty = "B1"


class TranslationResponse(BaseModel):
    title: str = ""
    summary_english: str = ""
    summary_spanish: str = ""
    pairs: List[ReadingPair]


# -----------------------------
# Page setup
# -----------------------------

st.set_page_config(
    page_title="Spanish Parallel Reader",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    .small-muted {
        font-size: 0.88rem;
        color: #6b7280;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📚 Spanish Parallel Reader")
st.caption("Local-first Spanish study with Streamlit + Ollama")


# -----------------------------
# Session state
# -----------------------------

if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""

if "results" not in st.session_state:
    st.session_state.results = []


# -----------------------------
# Text helpers
# -----------------------------

def set_source_text(text: str) -> None:
    cleaned = clean_text(text) if text else ""

    if cleaned != st.session_state.raw_text:
        st.session_state.raw_text = cleaned
        st.session_state.results = []

def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_text(uploaded_file) -> str:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    pages = []

    for page in doc:
        pages.append(page.get_text("text", sort=True))

    return clean_text("\n\n".join(pages))


def extract_docx_text(uploaded_file) -> str:
    doc = Document(uploaded_file)
    paragraphs = [
        paragraph.text.strip()
        for paragraph in doc.paragraphs
        if paragraph.text.strip()
    ]
    return clean_text("\n\n".join(paragraphs))


def extract_plain_text(uploaded_file) -> str:
    return clean_text(
        uploaded_file.read().decode("utf-8", errors="ignore")
    )


def split_into_chunks(text: str, max_chars: int) -> List[str]:
    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            parts = re.split(r"(?<=[.!?])\s+", paragraph)
        else:
            parts = [paragraph]

        for part in parts:
            if len(current) + len(part) + 2 <= max_chars:
                current = (
                    current + "\n\n" + part
                ).strip() if current else part
            else:
                if current:
                    chunks.append(current)
                current = part

    if current:
        chunks.append(current)

    return chunks


# -----------------------------
# Ollama helpers
# -----------------------------

def check_ollama():
    try:
        response = requests.get(
            f"{OLLAMA_HOST}/api/tags",
            timeout=5,
        )
        response.raise_for_status()

        models = [
            model.get("name", "")
            for model in response.json().get("models", [])
        ]

        if OLLAMA_MODEL in models:
            return True, f"Connected to Ollama using {OLLAMA_MODEL}."

        return (
            False,
            f"Ollama is reachable, but {OLLAMA_MODEL} is not listed yet. "
            f"Available: {', '.join(models) or 'none'}",
        )

    except Exception as exc:
        return False, f"Could not reach Ollama at {OLLAMA_HOST}: {exc}"


def translate_chunk(
    chunk: str,
    level: str,
    style: str,
    region: str,
    fidelity: str,
    include_literal: bool,
    include_vocab: bool,
    include_grammar: bool,
    temperature: float,
) -> TranslationResponse:
    schema = TranslationResponse.model_json_schema()

    system = (
        "You are a professional English-to-Spanish translator and Spanish language tutor. "
        "Prioritize accurate meaning transfer, natural Spanish, register preservation, and learner usefulness. "
        "Return only valid JSON matching the provided schema. "
        "Do not include Markdown, XML, chain-of-thought, or commentary outside JSON. "
        "Do not add facts not present in the source."
    )

    user = f"""
Create a Spanish parallel-reader lesson from the English text below.

Learner level: {level}
Spanish region preference: {region}
Translation style: {style}
Translation fidelity: {fidelity}
Include literal Spanish: {include_literal}
Include vocabulary: {include_vocab}
Include grammar notes: {include_grammar}

Rules:
- Preserve the original English in each pair.
- Translate into natural Spanish suitable for the selected level and region.
- If fidelity is "Closest meaning", preserve source meaning and nuance over simplification.
- If fidelity is "Simpler learner wording", simplify wording without changing facts.
- If fidelity is "Preserve literary style", preserve imagery, rhythm, and tone when possible.
- summary_english MUST be written in English. It is an English-language summary for the learner.
- summary_spanish MUST be written in Spanish. It is a Spanish-language summary for reading practice.
- If literal Spanish is disabled, set literal_spanish to an empty string.
- If literal Spanish is enabled, literal_spanish must be a word-for-word rendering of the English source into Spanish, preserving English word order even when it sounds awkward. It should differ visibly from the polished spanish field.
- If vocabulary is disabled, use an empty vocabulary list.
- If grammar notes are disabled, use an empty grammar_notes list.
- Use CEFR values only: A1, A2, B1, B2, C1, C2.
- Produce only JSON conforming to this schema: {json.dumps(schema, ensure_ascii=False)}

TEXT:
{chunk}
"""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": user,
            },
        ],
        "stream": False,
        "format": schema,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": temperature,
        },
    }

    response = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json=payload,
        timeout=240,
    )
    response.raise_for_status()

    content = response.json().get("message", {}).get("content", "")

    try:
        return TranslationResponse.model_validate_json(content)

    except ValidationError:
        start = content.find("{")
        end = content.rfind("}") + 1

        if start >= 0 and end > start:
            return TranslationResponse.model_validate_json(
                content[start:end]
            )

        raise


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:
    st.header("Settings")

    ok, status = check_ollama()

    if ok:
        st.success(status)
    else:
        st.warning(status)

    st.write("**Model**")
    st.code(OLLAMA_MODEL)

    with st.expander("Model guidance"):
        st.markdown(
            """
Default: `aya-expanse:8b` (multilingual, CC-BY-NC license).

To use a different model, set `OLLAMA_MODEL` in `.env` and restart containers.

**Alternatives:**
- `aya-expanse:32b` — larger, slower, higher quality
- `qwen3:14b` — if you need a different model
- `qwen3:8b` — smaller, faster

⚠️ **Hardware:** `aya-expanse:8b` requires 8–16 GB RAM. GPU with 8 GB+ VRAM recommended for fast inference.

🔒 **License:** Aya Expanse is released under CC-BY-NC (non-commercial use only). Verify compatibility before commercial deployment.
"""
        )

    with st.expander("ℹ️ Model License"):
        st.markdown(
            """
**Aya Expanse 8B** is released under **CC-BY-NC** (Creative Commons By-Attribution-NonCommercial).

This model is for **non-commercial use only**. Personal study, education, and research are typical non-commercial uses.

If you plan to use this application commercially, you must use a model with a compatible license.

[Model details](https://huggingface.co/CohereForAI/aya-expanse-8b)
"""
        )

    input_mode = st.radio(
        "Input type",
        [
            "Paste text",
            "Upload file",
        ],
    )

    level = st.selectbox(
        "Spanish level",
        [
            "A1 beginner",
            "A2 elementary",
            "B1 intermediate",
            "B2 upper-intermediate",
            "C1 advanced",
        ],
        index=2,
    )

    region = st.selectbox(
        "Spanish preference",
        [
            "Neutral",
            "Latin American",
            "European / Spain",
        ],
        index=0,
    )

    style = st.selectbox(
        "Translation style",
        [
            "Natural Spanish",
            "Learner-friendly Spanish",
            "Literal but readable Spanish",
            "Literary when appropriate",
            "Journalistic when appropriate",
        ],
        index=1,
    )

    fidelity = st.selectbox(
        "Translation fidelity",
        [
            "Balanced",
            "Closest meaning",
            "Simpler learner wording",
            "Preserve literary style",
        ],
        index=0,
    )

    max_chars = st.slider(
        "Max characters per chunk",
        800,
        4000,
        DEFAULT_MAX_CHARS,
        step=100,
    )

    chunks_to_process = st.slider(
        "Chunks to process",
        1,
        10,
        2,
    )

    temperature = st.slider(
        "Model temperature",
        0.0,
        0.3,
        0.1,
        step=0.05,
        help="Lower = more consistent JSON output. Keep below 0.3 for reliable structured translation.",
    )

    include_literal = st.checkbox(
        "Include literal Spanish",
        value=True,
    )

    include_vocab = st.checkbox(
        "Include vocabulary",
        value=True,
    )

    include_grammar = st.checkbox(
        "Include grammar notes",
        value=True,
    )

    if st.button("Clear session"):
        set_source_text("")
        st.rerun()


# -----------------------------
# Input
# -----------------------------

st.subheader("1. Add English source text")
st.markdown(
    '<p class="small-muted">Use pasted text or files you own / have permission to process. For news, paste an article excerpt or text you are allowed to use.</p>',
    unsafe_allow_html=True,
)

if input_mode == "Paste text":
    pasted = st.text_area(
        "Paste English text",
        height=280,
        placeholder="Paste a chapter excerpt, essay, article, or other English text...",
    )

    set_source_text(pasted)

else:
    uploaded = st.file_uploader(
        "Upload PDF, DOCX, TXT, or Markdown",
        type=[
            "pdf",
            "docx",
            "txt",
            "md",
        ],
    )

    if uploaded:
        suffix = uploaded.name.lower().split(".")[-1]

        try:
            size_mb = uploaded.size / 1024 / 1024
            with st.spinner(f"Extracting {suffix.upper()} ({size_mb:.1f} MB)…"):
                if suffix == "pdf":
                    set_source_text(extract_pdf_text(uploaded))
                elif suffix == "docx":
                    set_source_text(extract_docx_text(uploaded))
                else:
                    set_source_text(extract_plain_text(uploaded))
            st.success(f"Extracted {len(st.session_state.raw_text):,} characters")

        except Exception as exc:
            st.error(f"Could not extract text: {exc}")


chunks = (
    split_into_chunks(
        st.session_state.raw_text,
        max_chars,
    )
    if st.session_state.raw_text
    else []
)

if chunks:
    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Characters",
        f"{len(st.session_state.raw_text):,}",
    )

    c2.metric(
        "Chunks",
        len(chunks),
    )

    c3.metric(
        "Processed results",
        len(st.session_state.results),
    )

    with st.expander("Preview extracted text"):
        st.write(st.session_state.raw_text[:6000])


# -----------------------------
# Processing
# -----------------------------

st.subheader("2. Generate Spanish study view")

start_index = st.number_input(
    "Start at chunk",
    min_value=1,
    max_value=max(
        len(chunks),
        1,
    ),
    value=1,
)

if st.button(
    "Translate selected chunks",
    type="primary",
    disabled=not chunks or not ok,
):
    start = int(start_index) - 1
    selected_chunks = chunks[start : start + chunks_to_process]
    failed_chunks = []

    for idx, chunk in enumerate(
        selected_chunks,
        start=start + 1,
    ):
        with st.spinner(
            f"Translating chunk {idx} of {len(chunks)}..."
        ):
            try:
                result = translate_chunk(
                    chunk=chunk,
                    level=level,
                    style=style,
                    region=region,
                    fidelity=fidelity,
                    include_literal=include_literal,
                    include_vocab=include_vocab,
                    include_grammar=include_grammar,
                    temperature=temperature,
                )

                st.session_state.results.append(result)

            except Exception as exc:
                st.error(f"Chunk {idx} failed: {exc}")
                failed_chunks.append(idx)

    if failed_chunks:
        st.warning(
            f"{len(failed_chunks)} chunk(s) failed: {failed_chunks}. "
            "You can retry or continue viewing successful results."
        )


# -----------------------------
# Study tabs
# -----------------------------

if st.session_state.results:
    st.subheader("3. Study")

    tab_reader, tab_spanish, tab_vocab, tab_export = st.tabs(
        [
            "📖 Parallel Reader",
            "🇪🇸 Spanish First",
            "🧠 Vocabulary",
            "⬇️ Export",
        ]
    )

    with tab_reader:
        for result in st.session_state.results:
            if result.title:
                st.markdown(f"### {result.title}")

            if result.summary_english or result.summary_spanish:
                with st.expander("Summary"):
                    if result.summary_english:
                        st.markdown("**English summary**")
                        st.write(result.summary_english)

                    if result.summary_spanish:
                        st.markdown("**Spanish summary**")
                        st.write(result.summary_spanish)

            for pair in result.pairs:
                with st.container(border=True):
                    left, right = st.columns(2)

                    with left:
                        st.markdown("**English**")
                        st.write(pair.english)

                    with right:
                        st.markdown("**Español**")
                        st.write(pair.spanish)
                        st.markdown(
                            f'<span style="background:#e0f2fe;color:#0369a1;'
                            f'padding:2px 8px;border-radius:4px;'
                            f'font-size:0.82rem;font-weight:600;">'
                            f'{pair.difficulty}</span>',
                            unsafe_allow_html=True,
                        )

                    if pair.literal_spanish and pair.literal_spanish.strip() != pair.spanish.strip():
                        with st.expander("Literal Spanish"):
                            st.write(pair.literal_spanish)

                    if pair.grammar_notes:
                        with st.expander("Grammar notes"):
                            st.markdown(
                                "\n".join(f"- {note}" for note in pair.grammar_notes)
                            )

                    if pair.comprehension_question_spanish:
                        with st.expander("Comprehension question"):
                            st.write(pair.comprehension_question_spanish)

    with tab_spanish:
        st.caption(
            "Read Spanish first, then reveal English when you need it."
        )

        for result in st.session_state.results:
            for i, pair in enumerate(
                result.pairs,
                start=1,
            ):
                st.markdown(f"### Passage {i}")
                st.write(pair.spanish)

                with st.expander("Reveal English"):
                    st.write(pair.english)

    with tab_vocab:
        rows = [
            {
                "Spanish": vocab.spanish,
                "English": vocab.english,
                "Note": vocab.note,
            }
            for result in st.session_state.results
            for pair in result.pairs
            for vocab in pair.vocabulary
        ]

        if rows:
            df = pd.DataFrame(rows).drop_duplicates()

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download vocabulary CSV",
                df.to_csv(index=False).encode("utf-8"),
                "spanish_vocabulary.csv",
                "text/csv",
            )

        else:
            st.info("No vocabulary generated yet.")

    with tab_export:
        markdown = []

        for result in st.session_state.results:
            if result.title:
                markdown.append(f"# {result.title}\n")

            if result.summary_english:
                markdown.append(
                    f"**English summary:** {result.summary_english}\n"
                )

            if result.summary_spanish:
                markdown.append(
                    f"**Spanish summary:** {result.summary_spanish}\n"
                )

            for pair in result.pairs:
                markdown += [
                    "---\n",
                    "## English\n",
                    pair.english + "\n",
                    "## Español\n",
                    pair.spanish + "\n",
                ]

                if pair.literal_spanish:
                    markdown += [
                        "### Literal Spanish\n",
                        pair.literal_spanish + "\n",
                    ]

                if pair.grammar_notes:
                    markdown += ["### Grammar notes\n"]
                    markdown += [
                        f"- {note}"
                        for note in pair.grammar_notes
                    ]
                    markdown += [""]

                if pair.vocabulary:
                    markdown += ["### Vocabulary\n"]
                    markdown += [
                        f"- **{vocab.spanish}** = {vocab.english}. {vocab.note}"
                        for vocab in pair.vocabulary
                    ]
                    markdown += [""]

                if pair.comprehension_question_spanish:
                    markdown += [
                        "### Comprehension question\n",
                        pair.comprehension_question_spanish + "\n",
                    ]

        st.download_button(
            "Download study notes Markdown",
            "\n".join(markdown).encode("utf-8"),
            "spanish_parallel_reader.md",
            "text/markdown",
        )

else:
    st.info("Add text and translate a chunk to see the study interface.")