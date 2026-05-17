from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from checker import PairCheckResult
from infrastructure.translation_history import (
    MongoHistoryConfig,
    MongoTranslationHistoryStore,
    build_history_document,
    build_session_restore_state,
    deserialize_pair_check_result,
    deserialize_reading_pair,
    deserialize_translation_response,
    deserialize_vocabulary_item,
    serialize_pair_check_result,
    serialize_reading_pair,
    serialize_translation_response,
    serialize_vocabulary_item,
)


class VocabularyItem(BaseModel):
    spanish: str
    english: str
    note: str = ""


class ReadingPair(BaseModel):
    english: str
    spanish: str
    literal_spanish: str = ""
    vocabulary: list[VocabularyItem] = Field(default_factory=list)
    grammar_notes: list[str] = Field(default_factory=list)
    comprehension_question_spanish: str = ""
    difficulty: str = "B1"
    corrected_by_checker: bool = Field(default=False, exclude=True)
    correction_note: str = Field(default="", exclude=True)
    correction_reason: str = Field(default="", exclude=True)
    original_spanish_before_correction: str = Field(default="", exclude=True)


class TranslationResponse(BaseModel):
    title: str = ""
    summary_english: str = ""
    summary_spanish: str = ""
    pairs: list[ReadingPair]
    parse_warnings: list[str] = Field(default_factory=list, exclude=True)


def _sample_result() -> TranslationResponse:
    pair = ReadingPair(
        english="He ran quickly to the store.",
        spanish="Corrió rápidamente a la tienda.",
        literal_spanish="Él corrió rápidamente a la tienda.",
        vocabulary=[VocabularyItem(spanish="tienda", english="store", note="noun")],
        grammar_notes=["Simple preterite tense."],
        comprehension_question_spanish="¿Adónde corrió?",
        difficulty="B1",
        corrected_by_checker=True,
        correction_note="Fixed register mismatch",
        correction_reason="Preserved original tone",
        original_spanish_before_correction="Fue deprisa a la tienda.",
    )
    return TranslationResponse(
        title="Morning Errand",
        summary_english="A quick trip before closing.",
        summary_spanish="Un viaje rápido antes del cierre.",
        pairs=[pair],
        parse_warnings=["Dropped one malformed pair during parsing."],
    )


def test_vocabulary_item_roundtrip():
    item = VocabularyItem(spanish="libro", english="book", note="noun")
    serialized = serialize_vocabulary_item(item)
    restored = deserialize_vocabulary_item(serialized, VocabularyItem)

    assert restored == item


def test_reading_pair_roundtrip_preserves_runtime_fields():
    pair = _sample_result().pairs[0]
    serialized = serialize_reading_pair(pair)
    restored = deserialize_reading_pair(serialized, ReadingPair, VocabularyItem)

    assert restored.corrected_by_checker is True
    assert restored.correction_note == "Fixed register mismatch"
    assert restored.correction_reason == "Preserved original tone"
    assert restored.original_spanish_before_correction == "Fue deprisa a la tienda."


def test_translation_response_roundtrip_preserves_parse_warnings():
    result = _sample_result()
    serialized = serialize_translation_response(result)
    restored = deserialize_translation_response(
        serialized,
        TranslationResponse,
        ReadingPair,
        VocabularyItem,
    )

    assert restored.title == result.title
    assert restored.parse_warnings == result.parse_warnings
    assert restored.pairs[0].correction_note == "Fixed register mismatch"


def test_pair_check_result_roundtrip():
    original = PairCheckResult(
        passed=False,
        score=0.35,
        severity="fail",
        omission_issues=["Dropped the location."],
        corrected_spanish="Corrió rápidamente a la tienda.",
        user_facing_summary="Meaning was incomplete.",
        checked_with_llm=True,
        deterministic_only=False,
        truncated=True,
        cache_hit=False,
        checker_latency_ms=123.4,
    )

    serialized = serialize_pair_check_result(original)
    restored = deserialize_pair_check_result(serialized, PairCheckResult)

    assert restored == original


def test_build_history_document_counts_and_hashes():
    result = _sample_result()
    check_key = "check-1"
    check_result = PairCheckResult(
        passed=False,
        severity="warning",
        score=0.75,
        corrected_spanish="Corrió rápidamente a la tienda.",
        user_facing_summary="Minor wording issue.",
    )
    created = datetime(2026, 5, 16, 12, 0, tzinfo=timezone.utc)

    document = build_history_document(
        user_id="user-1",
        source_text="He ran quickly to the store.",
        input_mode="Upload file",
        uploaded_filename="story.txt",
        results=[result],
        checker_results={check_key: check_result},
        translation_settings={
            "selected_model": "qwen2.5:7b",
            "level": "B1 intermediate",
            "region": "Neutral",
            "style": "Learner-friendly Spanish",
            "fidelity": "Balanced",
            "temperature": 0.1,
            "max_chars": 2200,
            "start_chunk": 1,
            "chunks_processed": 1,
            "include_literal": True,
            "include_vocab": True,
            "include_grammar": True,
            "skip_enrichments": False,
            "show_audio_controls": True,
        },
        checker_settings={
            "enabled": True,
            "mode": "smart",
            "checker_model": "qwen2.5:7b",
            "require_pass_before_export": False,
            "detailed_diagnostics": False,
            "llm_checker_enabled": True,
        },
        created_at=created,
        updated_at=created,
    )

    assert document["pair_count"] == 1
    assert document["vocabulary_count"] == 1
    assert document["correction_count"] == 1
    assert document["label"] == "Morning Errand"
    assert document["session_hash"]
    assert document["source_hash"]
    assert document["checker_results"][check_key]["corrected_spanish"] == "Corrió rápidamente a la tienda."


def test_build_session_restore_state_returns_session_compatible_objects():
    result = _sample_result()
    document = build_history_document(
        user_id="user-1",
        source_text="He ran quickly to the store.",
        input_mode="Paste text",
        uploaded_filename="",
        results=[result],
        checker_results={
            "check-1": PairCheckResult(
                passed=True,
                severity="pass",
                score=1.0,
                user_facing_summary="Looks good.",
            )
        },
        translation_settings={"selected_model": "qwen2.5:7b", "input_mode": "Paste text"},
        checker_settings={"enabled": True, "mode": "smart"},
    )
    document["_id"] = "507f1f77bcf86cd799439011"

    restored = build_session_restore_state(
        document,
        translation_response_cls=TranslationResponse,
        reading_pair_cls=ReadingPair,
        vocabulary_item_cls=VocabularyItem,
        pair_check_result_cls=PairCheckResult,
    )

    assert restored["raw_text"] == "He ran quickly to the store."
    assert isinstance(restored["results"][0], TranslationResponse)
    assert restored["results"][0].pairs[0].corrected_by_checker is True
    assert isinstance(restored["checker_results"]["check-1"], PairCheckResult)
    assert restored["history_document_id"] == "507f1f77bcf86cd799439011"


def test_store_disabled_is_fail_open():
    store = MongoTranslationHistoryStore(
        MongoHistoryConfig(
            enabled=False,
            uri="mongodb://localhost:27017",
            db="db",
            collection="history",
            user_id="user-1",
            history_limit=10,
            save_source_text=True,
        )
    )

    assert store.list_history() == []
    assert store.load_history("507f1f77bcf86cd799439011") is None
    assert store.save_history({"session_hash": "abc"}) is None
    assert store.delete_history("507f1f77bcf86cd799439011") is False
    assert store.rename_history("507f1f77bcf86cd799439011", "Renamed") is False


def test_store_unavailable_mongo_is_fail_open(monkeypatch):
    class RaisingClient:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("cannot connect")

    monkeypatch.setattr(
        "infrastructure.translation_history.MongoClient",
        RaisingClient,
    )

    store = MongoTranslationHistoryStore(
        MongoHistoryConfig(
            enabled=True,
            uri="mongodb://localhost:27017",
            db="db",
            collection="history",
            user_id="user-1",
            history_limit=10,
            save_source_text=True,
        )
    )

    assert store.list_history() == []
    assert store.save_history({"session_hash": "abc"}) is None