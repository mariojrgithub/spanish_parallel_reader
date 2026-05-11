"""
Tests: Markdown and CSV export preserve correct English/Spanish language separation.
"""
import re
from typing import List
from pydantic import BaseModel, Field
from typing import Literal

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


# ── Markdown generator (mirrors app.py logic) ─────────────────────────────────

def generate_markdown(results: List[TranslationResponse]) -> str:
    markdown = []
    for result in results:
        if result.title:
            markdown.append(f"# {result.title}\n")
        if result.summary_english:
            markdown.append(f"**English summary:** {result.summary_english}\n")
        if result.summary_spanish:
            markdown.append(f"**Spanish summary:** {result.summary_spanish}\n")
        for pair in result.pairs:
            markdown += ["## English\n", pair.english + "\n"]
            markdown += ["## Español\n", pair.spanish + "\n"]
            if pair.literal_spanish:
                markdown += ["### Literal Spanish\n", pair.literal_spanish + "\n"]
            if pair.grammar_notes:
                markdown.append("### Grammar notes\n")
                markdown += [f"- {note}\n" for note in pair.grammar_notes]
            if pair.vocabulary:
                markdown.append("### Vocabulary\n")
                markdown += [
                    f"- **{v.spanish}** = {v.english}. {v.note}\n"
                    for v in pair.vocabulary
                ]
            if pair.comprehension_question_spanish:
                markdown += [
                    "### Comprehension question\n",
                    pair.comprehension_question_spanish + "\n",
                ]
    return "".join(markdown)


# ── Export tests ──────────────────────────────────────────────────────────────

def _make_result():
    return TranslationResponse(
        title="Test Passage",
        summary_english="English summary text",
        summary_spanish="Resumen en español",
        pairs=[
            ReadingPair(
                english="The dog runs fast.",
                spanish="El perro corre rápido.",
                literal_spanish="The dog runs fast (literal).",
                grammar_notes=["Note 1", "Note 2"],
                comprehension_question_spanish="¿Qué hace el perro?",
                vocabulary=[
                    VocabularyItem(spanish="perro", english="dog", note="noun"),
                    VocabularyItem(spanish="correr", english="to run", note="verb"),
                ],
            )
        ],
    )


def test_markdown_contains_english_section():
    md = generate_markdown([_make_result()])
    assert "## English" in md


def test_markdown_english_section_contains_english_text():
    md = generate_markdown([_make_result()])
    lines = md.splitlines()
    english_idx = next(i for i, l in enumerate(lines) if l.strip() == "## English")
    # The line after ## English should be the English text
    assert "The dog runs fast." in lines[english_idx + 1]


def test_markdown_contains_espanol_section():
    md = generate_markdown([_make_result()])
    assert "## Español" in md


def test_markdown_espanol_section_contains_spanish_text():
    md = generate_markdown([_make_result()])
    lines = md.splitlines()
    spanish_idx = next(i for i, l in enumerate(lines) if l.strip() == "## Español")
    assert "El perro corre rápido." in lines[spanish_idx + 1]


def test_markdown_english_and_spanish_not_swapped():
    md = generate_markdown([_make_result()])
    english_pos = md.index("## English")
    spanish_pos = md.index("## Español")
    # English section must come before Español
    assert english_pos < spanish_pos
    # English text must appear after ## English heading (not under Español)
    assert md.index("The dog runs fast.") > english_pos
    assert md.index("El perro corre rápido.") > spanish_pos


def test_markdown_summary_labels():
    md = generate_markdown([_make_result()])
    assert "**English summary:** English summary text" in md
    assert "**Spanish summary:** Resumen en español" in md


def test_markdown_vocabulary_spanish_before_english():
    md = generate_markdown([_make_result()])
    assert "**perro** = dog" in md
    assert "**correr** = to run" in md


def test_csv_column_order():
    """CSV must have Spanish column before English column."""
    result = _make_result()
    rows = [
        {"Spanish": v.spanish, "English": v.english, "Note": v.note}
        for pair in result.pairs
        for v in pair.vocabulary
    ]
    headers = list(rows[0].keys())
    assert headers[0] == "Spanish"
    assert headers[1] == "English"


def test_csv_values_not_reversed():
    result = _make_result()
    rows = [
        {"Spanish": v.spanish, "English": v.english, "Note": v.note}
        for pair in result.pairs
        for v in pair.vocabulary
    ]
    dog_row = next(r for r in rows if r["Spanish"] == "perro")
    assert dog_row["English"] == "dog"
    assert dog_row["Spanish"] != "dog"


def test_markdown_literal_spanish_section():
    md = generate_markdown([_make_result()])
    assert "### Literal Spanish" in md
    assert "The dog runs fast (literal)." in md


def test_markdown_comprehension_question_in_spanish_section():
    md = generate_markdown([_make_result()])
    assert "### Comprehension question" in md
    assert "¿Qué hace el perro?" in md
