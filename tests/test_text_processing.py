"""
Tests: text cleaning and chunking logic.
"""
import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List


# Reproduce helpers directly to avoid Streamlit import-time side effects
def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_chunks(text: str, max_chars: int) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            parts = re.split(r"(?<=[.!?])\s+", paragraph)
        else:
            parts = [paragraph]
        for part in parts:
            if len(current) + len(part) + 2 <= max_chars:
                current = (current + "\n\n" + part).strip() if current else part
            else:
                if current:
                    chunks.append(current)
                current = part
    if current:
        chunks.append(current)
    return chunks


# ── clean_text tests ──────────────────────────────────────────────────────────

def test_clean_text_removes_page_numbers():
    result = clean_text("Some text\n\n10\n\nMore text")
    assert "10" not in result

def test_clean_text_normalizes_horizontal_whitespace():
    result = clean_text("Text   with    spaces")
    assert result == "Text with spaces"

def test_clean_text_collapses_triple_newlines():
    result = clean_text("Para 1\n\n\n\nPara 2")
    assert "\n\n\n" not in result

def test_clean_text_strips_leading_trailing():
    result = clean_text("   hello   ")
    assert result == "hello"

def test_clean_text_normalizes_carriage_returns():
    result = clean_text("line1\r\nline2")
    assert "\r" not in result


# ── split_into_chunks tests ───────────────────────────────────────────────────

def test_chunks_respect_max_chars():
    # Build text with real sentence endings so the splitter can divide it
    sentence = "This is a long sentence with real words. "
    text = sentence * 120  # ~4800 chars, one big paragraph
    chunks = split_into_chunks(text, max_chars=500)
    # Every chunk must be at most max_chars (the splitter splits on sentence boundaries)
    assert all(len(c) <= 500 for c in chunks)

def test_chunks_preserves_all_content():
    text = "Para 1\n\nPara 2\n\nPara 3"
    chunks = split_into_chunks(text, max_chars=1000)
    joined = " ".join(chunks)
    assert "Para 1" in joined
    assert "Para 2" in joined
    assert "Para 3" in joined

def test_chunks_non_empty():
    text = "Hello world.\n\nAnother paragraph."
    chunks = split_into_chunks(text, max_chars=2200)
    assert all(c.strip() for c in chunks)

def test_chunks_single_long_paragraph_splits():
    sentence = "This is a sentence. "
    text = sentence * 200  # ~3800 chars
    chunks = split_into_chunks(text, max_chars=500)
    assert len(chunks) > 1
