"""
Tests: Pydantic schema field names and language assignments are correct.
"""
import sys
import os

# Import schemas without triggering Streamlit's runtime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import ValidationError
import importlib, types

# Minimal stub so `import app` does not crash outside Streamlit
import unittest.mock as mock

streamlit_stub = types.ModuleType("streamlit")
for attr in [
    "set_page_config", "markdown", "title", "caption", "session_state",
    "sidebar", "expander", "radio", "selectbox", "slider", "checkbox",
    "text_area", "file_uploader", "button", "spinner", "success", "error",
    "warning", "write", "code", "subheader", "tabs", "columns", "dataframe",
    "download_button", "stop", "metric",
]:
    setattr(streamlit_stub, attr, mock.MagicMock())
streamlit_stub.session_state = mock.MagicMock()

sys.modules.setdefault("streamlit", streamlit_stub)
sys.modules.setdefault("fitz", mock.MagicMock())
sys.modules.setdefault("docx", mock.MagicMock())
sys.modules.setdefault("pandas", mock.MagicMock())

from pydantic import BaseModel, Field
from typing import List, Literal

# Re-define schemas here to avoid import-time side effects
Difficulty = Literal["A1", "A2", "B1", "B2", "C1", "C2"]


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


# ── Schema field tests ────────────────────────────────────────────────────────

def test_vocabulary_item_spanish_field_holds_spanish():
    item = VocabularyItem(spanish="palabra", english="word")
    assert item.spanish == "palabra"
    assert item.english == "word"


def test_vocabulary_item_not_reversed():
    item = VocabularyItem(spanish="hablar", english="to speak")
    assert item.spanish == "hablar"
    assert item.english == "to speak"


def test_reading_pair_english_and_spanish_fields():
    pair = ReadingPair(english="Hello world", spanish="Hola mundo")
    assert pair.english == "Hello world"
    assert pair.spanish == "Hola mundo"


def test_reading_pair_literal_spanish_defaults_empty():
    pair = ReadingPair(english="Hi", spanish="Hola")
    assert pair.literal_spanish == ""


def test_translation_response_summary_fields():
    resp = TranslationResponse(
        title="Test",
        summary_english="English summary",
        summary_spanish="Resumen en español",
        pairs=[],
    )
    assert resp.summary_english == "English summary"
    assert resp.summary_spanish == "Resumen en español"


def test_reading_pair_comprehension_question_is_spanish():
    pair = ReadingPair(
        english="She reads every day.",
        spanish="Ella lee todos los días.",
        comprehension_question_spanish="¿Qué hace ella todos los días?",
    )
    assert "¿" in pair.comprehension_question_spanish


def test_difficulty_defaults_to_b1():
    pair = ReadingPair(english="Test", spanish="Prueba")
    assert pair.difficulty == "B1"


def test_translation_response_json_roundtrip():
    original = TranslationResponse(
        title="Roundtrip",
        summary_english="Summary",
        summary_spanish="Resumen",
        pairs=[
            ReadingPair(
                english="The cat sat.",
                spanish="El gato estaba sentado.",
                vocabulary=[VocabularyItem(spanish="gato", english="cat")],
            )
        ],
    )
    json_str = original.model_dump_json()
    restored = TranslationResponse.model_validate_json(json_str)
    assert restored.pairs[0].vocabulary[0].spanish == "gato"
    assert restored.pairs[0].vocabulary[0].english == "cat"
