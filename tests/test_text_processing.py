"""
Tests: text cleaning and chunking logic.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text_processing import clean_text, _hard_wrap, split_into_chunks


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


# ── _hard_wrap tests ──────────────────────────────────────────────────────────

def test_hard_wrap_short_text_unchanged():
    assert _hard_wrap("hello world", 100) == ["hello world"]

def test_hard_wrap_splits_at_word_boundary():
    # 10-char words separated by spaces; max_chars=15 → first cut at space before pos 15
    text = "abcdefghij klmnopqrst uvwxyzabcd"
    parts = _hard_wrap(text, 15)
    assert all(len(p) <= 15 for p in parts)
    assert " ".join(parts).replace("  ", " ") == text or "".join(parts) in text.replace(" ", "")

def test_hard_wrap_no_spaces_hard_cuts():
    # A single continuous string with no spaces must still be split
    text = "a" * 50
    parts = _hard_wrap(text, 20)
    assert all(len(p) <= 20 for p in parts)
    assert "".join(parts) == text

def test_hard_wrap_preserves_all_chars():
    text = "word " * 30  # 150 chars
    parts = _hard_wrap(text, 40)
    assert "".join(p.rstrip() + " " for p in parts).rstrip() == text.rstrip()


# ── hard-wrap integration tests ───────────────────────────────────────────────

def test_no_chunk_exceeds_max_chars_long_sentence():
    # Single sentence with no punctuation longer than max_chars
    long_word_run = ("longword " * 60).strip()  # ~540 chars, no sentence endings
    chunks = split_into_chunks(long_word_run, max_chars=100)
    assert all(len(c) <= 100 for c in chunks), \
        f"Oversized chunk: {max(len(c) for c in chunks)} > 100"

def test_no_chunk_exceeds_max_chars_mixed():
    # Mix of normal paragraphs and a run-on line
    run_on = "word " * 100  # 500 chars, no punctuation
    text = "Normal paragraph one.\n\n" + run_on.strip() + "\n\nNormal paragraph two."
    chunks = split_into_chunks(text, max_chars=200)
    assert all(len(c) <= 200 for c in chunks), \
        f"Oversized chunk: {max(len(c) for c in chunks)} > 200"

def test_empty_input_returns_empty_list():
    assert split_into_chunks("", max_chars=500) == []

def test_whitespace_only_input_returns_empty_list():
    assert split_into_chunks("   \n\n   ", max_chars=500) == []
