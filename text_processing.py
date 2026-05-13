"""
Pure text-processing helpers: extraction, cleaning, and chunking.

No Streamlit imports — safe to unit-test without a running Streamlit context.
"""
from __future__ import annotations

import re
from typing import List

import fitz
from docx import Document

# ---------------------------------------------------------------------------
# Pre-compiled patterns (shared with app.py via import)
# ---------------------------------------------------------------------------

RE_PAGE_NUM = re.compile(r"\n\s*\d+\s*\n")
RE_HSPACE = re.compile(r"[ \t]+")
RE_NEWLINES = re.compile(r"\n{3,}")
RE_SENTENCES = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = RE_PAGE_NUM.sub("\n", text)
    text = RE_HSPACE.sub(" ", text)
    text = RE_NEWLINES.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_pdf_text(uploaded_file) -> str:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    pages = [page.get_text("text", sort=True) for page in doc]
    return clean_text("\n\n".join(pages))


def extract_docx_text(uploaded_file) -> str:
    doc = Document(uploaded_file)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return clean_text("\n\n".join(paragraphs))


def extract_plain_text(uploaded_file) -> str:
    return clean_text(uploaded_file.read().decode("utf-8", errors="ignore"))


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _hard_wrap(text: str, max_chars: int) -> List[str]:
    """Split text at word boundaries when it exceeds max_chars.

    Used as a last-resort fallback inside split_into_chunks so that a single
    sentence with no punctuation can never produce an oversized chunk.
    Falls back to a hard character cut only when no space is found.
    """
    if len(text) <= max_chars:
        return [text]
    parts: List[str] = []
    while len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars  # no space — hard character cut
        parts.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        parts.append(text)
    return parts


def split_into_chunks(text: str, max_chars: int) -> List[str]:
    """Split *text* into chunks of at most *max_chars* characters.

    Respects paragraph boundaries first, sentence boundaries second, and
    word boundaries last (via _hard_wrap) so no chunk ever exceeds max_chars.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: List[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            parts = RE_SENTENCES.split(paragraph)
        else:
            parts = [paragraph]

        for part in parts:
            subparts = _hard_wrap(part, max_chars) if len(part) > max_chars else [part]
            for subpart in subparts:
                if len(current) + len(subpart) + 2 <= max_chars:
                    current = (current + "\n\n" + subpart).strip() if current else subpart
                else:
                    if current:
                        chunks.append(current)
                    current = subpart

    if current:
        chunks.append(current)

    return chunks
