from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
_SOURCE_PREVIEW_LIMIT = 160

try:
    from bson import ObjectId
    from pymongo import ASCENDING, DESCENDING, MongoClient
    from pymongo.collection import Collection
    from pymongo import ReturnDocument
except ImportError:  # pragma: no cover - exercised via config fail-open tests
    ObjectId = None  # type: ignore[assignment]
    MongoClient = None  # type: ignore[assignment]
    Collection = Any  # type: ignore[assignment,misc]
    ReturnDocument = None  # type: ignore[assignment]
    ASCENDING = 1
    DESCENDING = -1


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split())


def _preview_text(text: str) -> str:
    normalized = _normalize_whitespace(text)
    if len(normalized) <= _SOURCE_PREVIEW_LIMIT:
        return normalized
    return normalized[: _SOURCE_PREVIEW_LIMIT - 1].rstrip() + "…"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MongoHistoryConfig:
    enabled: bool
    uri: str
    db: str
    collection: str
    user_id: str
    history_limit: int
    save_source_text: bool


def get_mongo_history_config() -> MongoHistoryConfig:
    return MongoHistoryConfig(
        enabled=_bool_env("MONGO_ENABLED", False),
        uri=os.getenv("MONGO_URI", "mongodb://localhost:27017").strip() or "mongodb://localhost:27017",
        db=os.getenv("MONGO_DB", "spanish_parallel_reader").strip() or "spanish_parallel_reader",
        collection=os.getenv("MONGO_HISTORY_COLLECTION", "translation_history").strip() or "translation_history",
        user_id=os.getenv("MONGO_USER_ID", "default-user").strip() or "default-user",
        history_limit=max(1, _coerce_int(os.getenv("MONGO_HISTORY_LIMIT", "25"), 25)),
        save_source_text=_bool_env("MONGO_SAVE_SOURCE_TEXT", True),
    )


def serialize_vocabulary_item(item: Any) -> dict[str, Any]:
    return {
        "spanish": _coerce_str(getattr(item, "spanish", "")),
        "english": _coerce_str(getattr(item, "english", "")),
        "note": _coerce_str(getattr(item, "note", "")),
    }


def deserialize_vocabulary_item(data: Mapping[str, Any] | None, vocabulary_item_cls: type) -> Any:
    payload = {
        "spanish": _coerce_str((data or {}).get("spanish", "")),
        "english": _coerce_str((data or {}).get("english", "")),
        "note": _coerce_str((data or {}).get("note", "")),
    }
    return vocabulary_item_cls.model_validate(payload)


def serialize_reading_pair(pair: Any) -> dict[str, Any]:
    vocabulary = getattr(pair, "vocabulary", None) or []
    grammar_notes = getattr(pair, "grammar_notes", None) or []
    return {
        "english": _coerce_str(getattr(pair, "english", "")),
        "spanish": _coerce_str(getattr(pair, "spanish", "")),
        "literal_spanish": _coerce_str(getattr(pair, "literal_spanish", "")),
        "vocabulary": [serialize_vocabulary_item(item) for item in vocabulary],
        "grammar_notes": [_coerce_str(note) for note in grammar_notes],
        "comprehension_question_spanish": _coerce_str(
            getattr(pair, "comprehension_question_spanish", "")
        ),
        "difficulty": _coerce_str(getattr(pair, "difficulty", "B1"), "B1") or "B1",
        "corrected_by_checker": _coerce_bool(
            getattr(pair, "corrected_by_checker", False)
        ),
        "correction_note": _coerce_str(getattr(pair, "correction_note", "")),
        "correction_reason": _coerce_str(getattr(pair, "correction_reason", "")),
        "original_spanish_before_correction": _coerce_str(
            getattr(pair, "original_spanish_before_correction", "")
        ),
    }


def deserialize_reading_pair(
    data: Mapping[str, Any] | None,
    reading_pair_cls: type,
    vocabulary_item_cls: type,
) -> Any:
    raw = data or {}
    payload = {
        "english": _coerce_str(raw.get("english", "")),
        "spanish": _coerce_str(raw.get("spanish", "")),
        "literal_spanish": _coerce_str(raw.get("literal_spanish", "")),
        "vocabulary": [
            deserialize_vocabulary_item(item, vocabulary_item_cls)
            for item in raw.get("vocabulary", [])
            if isinstance(item, Mapping)
        ],
        "grammar_notes": [_coerce_str(note) for note in raw.get("grammar_notes", [])],
        "comprehension_question_spanish": _coerce_str(
            raw.get("comprehension_question_spanish", "")
        ),
        "difficulty": _coerce_str(raw.get("difficulty", "B1"), "B1") or "B1",
    }
    pair = reading_pair_cls.model_validate(payload)
    pair.corrected_by_checker = _coerce_bool(raw.get("corrected_by_checker", False))
    pair.correction_note = _coerce_str(raw.get("correction_note", ""))
    pair.correction_reason = _coerce_str(raw.get("correction_reason", ""))
    pair.original_spanish_before_correction = _coerce_str(
        raw.get("original_spanish_before_correction", "")
    )
    return pair


def serialize_translation_response(result: Any) -> dict[str, Any]:
    parse_warnings = getattr(result, "parse_warnings", None) or []
    pairs = getattr(result, "pairs", None) or []
    return {
        "title": _coerce_str(getattr(result, "title", "")),
        "summary_english": _coerce_str(getattr(result, "summary_english", "")),
        "summary_spanish": _coerce_str(getattr(result, "summary_spanish", "")),
        "pairs": [serialize_reading_pair(pair) for pair in pairs],
        "parse_warnings": [_coerce_str(item) for item in parse_warnings],
    }


def deserialize_translation_response(
    data: Mapping[str, Any] | None,
    translation_response_cls: type,
    reading_pair_cls: type,
    vocabulary_item_cls: type,
) -> Any:
    raw = data or {}
    pairs = [
        deserialize_reading_pair(item, reading_pair_cls, vocabulary_item_cls)
        for item in raw.get("pairs", [])
        if isinstance(item, Mapping)
    ]
    payload = {
        "title": _coerce_str(raw.get("title", "")),
        "summary_english": _coerce_str(raw.get("summary_english", "")),
        "summary_spanish": _coerce_str(raw.get("summary_spanish", "")),
        "pairs": pairs,
    }
    result = translation_response_cls.model_validate(payload)
    result.parse_warnings = [_coerce_str(item) for item in raw.get("parse_warnings", [])]
    return result


def serialize_pair_check_result(result: Any) -> dict[str, Any]:
    if isinstance(result, Mapping):
        raw = result
        getter = raw.get
    else:
        raw = result
        getter = lambda key, default=None: getattr(raw, key, default)

    return {
        "passed": _coerce_bool(getter("passed", True), True),
        "score": _coerce_float(getter("score", 1.0), 1.0),
        "severity": _coerce_str(getter("severity", "pass"), "pass") or "pass",
        "faithfulness_issues": [_coerce_str(item) for item in getter("faithfulness_issues", []) or []],
        "hallucination_issues": [_coerce_str(item) for item in getter("hallucination_issues", []) or []],
        "omission_issues": [_coerce_str(item) for item in getter("omission_issues", []) or []],
        "label_issues": [_coerce_str(item) for item in getter("label_issues", []) or []],
        "language_quality_issues": [_coerce_str(item) for item in getter("language_quality_issues", []) or []],
        "unsupported_claims": [_coerce_str(item) for item in getter("unsupported_claims", []) or []],
        "recommended_action": _coerce_str(getter("recommended_action", "")),
        "corrected_spanish": _coerce_str(getter("corrected_spanish", "")),
        "user_facing_summary": _coerce_str(getter("user_facing_summary", "")),
        "checked_with_llm": _coerce_bool(getter("checked_with_llm", False)),
        "deterministic_only": _coerce_bool(getter("deterministic_only", True), True),
        "truncated": _coerce_bool(getter("truncated", False)),
        "cache_hit": _coerce_bool(getter("cache_hit", False)),
        "checker_latency_ms": (
            _coerce_float(getter("checker_latency_ms", 0.0))
            if getter("checker_latency_ms", None) is not None
            else None
        ),
    }


def deserialize_pair_check_result(data: Mapping[str, Any] | None, pair_check_result_cls: type) -> Any:
    raw = serialize_pair_check_result(data or {})
    return pair_check_result_cls.model_validate(raw)


def build_history_document(
    *,
    user_id: str,
    source_text: str,
    input_mode: str,
    uploaded_filename: str,
    results: Sequence[Any],
    checker_results: Mapping[str, Any],
    translation_settings: Mapping[str, Any],
    checker_settings: Mapping[str, Any],
    label: str = "",
    title: str = "",
    schema_version: int = SCHEMA_VERSION,
    save_source_text: bool = True,
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> dict[str, Any]:
    serialized_results = [serialize_translation_response(result) for result in results]
    serialized_checker_results = {
        _coerce_str(key): serialize_pair_check_result(value)
        for key, value in checker_results.items()
    }
    effective_title = title or next(
        (
            item.get("title", "")
            for item in serialized_results
            if item.get("title", "")
        ),
        "",
    )
    source_preview = _preview_text(source_text)
    generated_label = (
        _coerce_str(label).strip()
        or effective_title.strip()
        or (source_preview[:80].rstrip() if source_preview else "Untitled translation")
    )
    pair_count = sum(len(item.get("pairs", [])) for item in serialized_results)
    vocabulary_count = sum(
        len(pair.get("vocabulary", []))
        for item in serialized_results
        for pair in item.get("pairs", [])
    )
    correction_count = sum(
        1
        for item in serialized_results
        for pair in item.get("pairs", [])
        if pair.get("corrected_by_checker")
    )

    source_hash = _hash_payload({"user_id": user_id, "source_text": source_text})
    session_hash = _hash_payload(
        {
            "user_id": user_id,
            "schema_version": schema_version,
            "source_text": source_text,
            "input_mode": input_mode,
            "uploaded_filename": uploaded_filename,
            "results": serialized_results,
            "checker_results": serialized_checker_results,
            "translation_settings": dict(translation_settings),
            "checker_settings": dict(checker_settings),
        }
    )

    now = updated_at or _utc_now()
    created = created_at or now
    document = {
        "schema_version": schema_version,
        "user_id": user_id,
        "label": generated_label,
        "title": effective_title,
        "source_preview": source_preview,
        "input": {
            "mode": _coerce_str(input_mode, "Paste text") or "Paste text",
            "uploaded_filename": _coerce_str(uploaded_filename),
        },
        "source_text": source_text if save_source_text else "",
        "results": serialized_results,
        "checker_results": serialized_checker_results,
        "translation_settings": dict(translation_settings),
        "checker_settings": dict(checker_settings),
        "pair_count": pair_count,
        "vocabulary_count": vocabulary_count,
        "correction_count": correction_count,
        "model": _coerce_str(translation_settings.get("selected_model", "")),
        "level": _coerce_str(translation_settings.get("level", "")),
        "region": _coerce_str(translation_settings.get("region", "")),
        "source_hash": source_hash,
        "session_hash": session_hash,
        "created_at": created,
        "updated_at": now,
    }
    return document


def build_session_restore_state(
    document: Mapping[str, Any],
    *,
    translation_response_cls: type,
    reading_pair_cls: type,
    vocabulary_item_cls: type,
    pair_check_result_cls: type,
) -> dict[str, Any]:
    translation_settings = dict(document.get("translation_settings", {}))
    checker_settings = dict(document.get("checker_settings", {}))
    results = [
        deserialize_translation_response(
            item,
            translation_response_cls,
            reading_pair_cls,
            vocabulary_item_cls,
        )
        for item in document.get("results", [])
        if isinstance(item, Mapping)
    ]
    checker_results = {
        _coerce_str(key): deserialize_pair_check_result(value, pair_check_result_cls)
        for key, value in dict(document.get("checker_results", {})).items()
        if isinstance(value, Mapping)
    }
    return {
        "raw_text": _coerce_str(document.get("source_text", "")),
        "results": results,
        "checker_results": checker_results,
        "history_label": _coerce_str(document.get("label", "")),
        "history_title": _coerce_str(document.get("title", "")),
        "history_uploaded_filename": _coerce_str(
            dict(document.get("input", {})).get("uploaded_filename", "")
        ),
        "input_mode": _coerce_str(
            dict(document.get("input", {})).get("mode", translation_settings.get("input_mode", "Paste text")),
            "Paste text",
        )
        or "Paste text",
        "translation_settings": translation_settings,
        "checker_settings": checker_settings,
        "schema_version": _coerce_int(document.get("schema_version", SCHEMA_VERSION), SCHEMA_VERSION),
        "history_document_id": _coerce_str(document.get("_id", "")),
        "history_session_hash": _coerce_str(document.get("session_hash", "")),
        "history_source_hash": _coerce_str(document.get("source_hash", "")),
    }


class MongoTranslationHistoryStore:
    def __init__(self, config: MongoHistoryConfig | None = None) -> None:
        self._config = config or get_mongo_history_config()
        self._client: Any = None
        self._collection: Any = None
        self._warned_unavailable = False
        self._indexes_ready = False

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def user_id(self) -> str:
        return self._config.user_id

    @property
    def save_source_text(self) -> bool:
        return self._config.save_source_text

    def status_message(self) -> str:
        if not self._config.enabled:
            return "History persistence is disabled."
        if MongoClient is None:
            return "pymongo is not installed."
        if self._get_collection() is None:
            return "MongoDB is unavailable."
        return "MongoDB history is available."

    def _get_collection(self) -> Collection | None:
        if not self._config.enabled:
            return None
        if MongoClient is None:
            if not self._warned_unavailable:
                logger.warning("MongoDB history requested but pymongo is not installed.")
                self._warned_unavailable = True
            return None
        if self._collection is not None:
            return self._collection
        try:
            self._client = MongoClient(
                self._config.uri,
                serverSelectionTimeoutMS=1500,
                connectTimeoutMS=1500,
                socketTimeoutMS=2000,
            )
            self._client.admin.command("ping")
            self._collection = self._client[self._config.db][self._config.collection]
            self._ensure_indexes(self._collection)
            return self._collection
        except Exception as exc:
            if not self._warned_unavailable:
                logger.warning("MongoDB history unavailable: %s", exc)
                self._warned_unavailable = True
            self._client = None
            self._collection = None
            return None

    def _ensure_indexes(self, collection: Collection) -> None:
        if self._indexes_ready:
            return
        collection.create_index(
            [("user_id", ASCENDING), ("updated_at", DESCENDING)],
            name="history_user_updated_at",
        )
        collection.create_index(
            [("user_id", ASCENDING), ("session_hash", ASCENDING)],
            unique=True,
            name="history_user_session_hash",
        )
        collection.create_index(
            [("user_id", ASCENDING), ("source_hash", ASCENDING)],
            name="history_user_source_hash",
        )
        self._indexes_ready = True

    def list_history(self) -> list[dict[str, Any]]:
        collection = self._get_collection()
        if collection is None:
            return []
        try:
            cursor = (
                collection.find(
                    {"user_id": self._config.user_id},
                    {
                        "label": 1,
                        "title": 1,
                        "source_preview": 1,
                        "created_at": 1,
                        "updated_at": 1,
                        "pair_count": 1,
                        "vocabulary_count": 1,
                        "correction_count": 1,
                        "model": 1,
                        "level": 1,
                        "region": 1,
                        "session_hash": 1,
                        "source_hash": 1,
                    },
                )
                .sort("updated_at", DESCENDING)
                .limit(self._config.history_limit)
            )
            return [self._normalize_document(item) for item in cursor]
        except Exception as exc:
            logger.warning("MongoDB history list failed: %s", exc)
            return []

    def load_history(self, history_id: str) -> dict[str, Any] | None:
        collection = self._get_collection()
        if collection is None:
            return None
        object_id = self._object_id(history_id)
        if object_id is None:
            return None
        try:
            document = collection.find_one({"_id": object_id, "user_id": self._config.user_id})
            return self._normalize_document(document) if document else None
        except Exception as exc:
            logger.warning("MongoDB history load failed: %s", exc)
            return None

    def save_history(
        self,
        document: Mapping[str, Any],
        *,
        preferred_history_id: str = "",
    ) -> dict[str, Any] | None:
        collection = self._get_collection()
        if collection is None:
            return None

        payload = dict(document)
        payload["user_id"] = self._config.user_id
        payload["updated_at"] = payload.get("updated_at") or _utc_now()
        payload["created_at"] = payload.get("created_at") or payload["updated_at"]

        try:
            preferred_id = self._object_id(preferred_history_id)
            if preferred_id is not None:
                result = collection.find_one_and_update(
                    {"_id": preferred_id, "user_id": self._config.user_id},
                    {"$set": payload},
                    return_document=ReturnDocument.AFTER,
                )
                if result is not None:
                    return self._normalize_document(result)

            # Exclude created_at from $set so it does not conflict with
            # $setOnInsert: MongoDB rejects an update where the same field
            # path appears in two operators (error code 40).
            # $setOnInsert ensures created_at is written only on first insert
            # and preserved unchanged on subsequent updates.
            set_payload = {k: v for k, v in payload.items() if k != "created_at"}
            updated = collection.find_one_and_update(
                {
                    "user_id": self._config.user_id,
                    "session_hash": payload.get("session_hash", ""),
                },
                {
                    "$set": set_payload,
                    "$setOnInsert": {"created_at": payload["created_at"]},
                },
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
            return self._normalize_document(updated) if updated is not None else None
        except Exception as exc:
            logger.warning("MongoDB history save failed: %s", exc)
            return None

    def delete_history(self, history_id: str) -> bool:
        collection = self._get_collection()
        if collection is None:
            return False
        object_id = self._object_id(history_id)
        if object_id is None:
            return False
        try:
            result = collection.delete_one({"_id": object_id, "user_id": self._config.user_id})
            return result.deleted_count > 0
        except Exception as exc:
            logger.warning("MongoDB history delete failed: %s", exc)
            return False

    def rename_history(self, history_id: str, label: str) -> bool:
        collection = self._get_collection()
        if collection is None:
            return False
        object_id = self._object_id(history_id)
        if object_id is None:
            return False
        clean_label = _coerce_str(label).strip()
        if not clean_label:
            return False
        try:
            result = collection.update_one(
                {"_id": object_id, "user_id": self._config.user_id},
                {"$set": {"label": clean_label, "updated_at": _utc_now()}},
            )
            return result.modified_count > 0
        except Exception as exc:
            logger.warning("MongoDB history rename failed: %s", exc)
            return False

    def _normalize_document(self, document: Mapping[str, Any] | None) -> dict[str, Any]:
        if not document:
            return {}
        normalized = dict(document)
        normalized["_id"] = _coerce_str(normalized.get("_id", ""))
        return normalized

    def _object_id(self, value: str) -> Any:
        if not value or ObjectId is None:
            return None
        try:
            return ObjectId(value)
        except Exception:
            return None