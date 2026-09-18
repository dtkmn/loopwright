from __future__ import annotations

import hashlib
import hmac
import json
import math
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional

try:
    from .answer_loop import SELF_CHECK_REFUSAL_ANSWER
    from .loop_engine import (
        ANSWER_CANDIDATE_SHA256_METADATA_KEY,
        EVIDENCE_SET_SHA256_METADATA_KEY,
        DEFAULT_LOOP_RECIPE_ID,
        LOOP_RECIPE_SCHEMA_VERSION,
        PUBLIC_REDACTION_TEXT,
        LoopReport,
        LoopRecipe,
        default_loop_recipe,
    )
    from .public_projection import (
        PUBLIC_REPORT_PROJECTION_SCHEMA_VERSION,
        project_public_report,
    )
except ImportError:
    from answer_loop import SELF_CHECK_REFUSAL_ANSWER
    from loop_engine import (
        ANSWER_CANDIDATE_SHA256_METADATA_KEY,
        EVIDENCE_SET_SHA256_METADATA_KEY,
        DEFAULT_LOOP_RECIPE_ID,
        LOOP_RECIPE_SCHEMA_VERSION,
        PUBLIC_REDACTION_TEXT,
        LoopReport,
        LoopRecipe,
        default_loop_recipe,
    )
    from public_projection import (
        PUBLIC_REPORT_PROJECTION_SCHEMA_VERSION,
        project_public_report,
    )


DEFAULT_THREAD_TITLE = "New thread"
MAX_THREAD_TITLE_LENGTH = 64
MAX_LOOP_RECIPE_LIST_LIMIT = 50
MAX_LOOP_RUN_LIST_LIMIT = 50
MAX_RECORD_ID_LENGTH = 96
ORDINARY_REFUSAL_TERMINAL_REASONS = frozenset(
    {"verification_failed", "retry_budget_exhausted"}
)
MODEL_THINKING_SHA256_METADATA_KEY = "model_thinking_sha256"
MODEL_THINKING_LABEL = "Model Thinking (unverified)"
MODEL_THINKING_NOTE = (
    "Model-emitted thinking is useful for debugging the loop, but it is not "
    "verified evidence."
)
MODEL_THINKING_REDACTION = "[redacted: terminal loop decision]"
THINKING_REDACTED_FINAL_DECISIONS = frozenset(
    {"block", "error", "refuse", "requires_review"}
)
STORED_TIMESTAMP_FALLBACK = "1970-01-01T00:00:00.000Z"
MAX_STORED_TIMESTAMP_BYTES = 64


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def canonical_stored_timestamp(value) -> str:
    """Return a bounded UTC timestamp without echoing malformed stored text."""

    if (
        type(value) is not str
        or not value
        or len(value.encode("utf-8")) > MAX_STORED_TIMESTAMP_BYTES
    ):
        return STORED_TIMESTAMP_FALLBACK
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (OverflowError, ValueError):
        return STORED_TIMESTAMP_FALLBACK
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return STORED_TIMESTAMP_FALLBACK
    try:
        return (
            parsed.astimezone(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
    except (OverflowError, ValueError):
        return STORED_TIMESTAMP_FALLBACK


def default_thread_id() -> str:
    return f"thread_{uuid.uuid4()}"


def default_thread_instance_id() -> str:
    return uuid.uuid4().hex


def default_recipe_id() -> str:
    return f"recipe_{uuid.uuid4().hex}"


def safe_title(value: str | None) -> str:
    title = " ".join(str(value or "").split()).strip()
    if not title:
        return DEFAULT_THREAD_TITLE
    return title[:MAX_THREAD_TITLE_LENGTH]


def safe_record_id(value: str | None, *, field_name: str) -> str:
    record_id = str(value or "")
    if not record_id:
        raise ValueError(f"{field_name} must not be empty.")
    if len(record_id.encode("utf-8")) > MAX_RECORD_ID_LENGTH:
        raise ValueError(f"{field_name} is too long.")
    if not (record_id[0].isascii() and record_id[0].isalnum()):
        raise ValueError(f"{field_name} must start with an ASCII letter or digit.")
    if not all(
        char.isascii() and (char.isalnum() or char in {"_", "-", ".", ":"})
        for char in record_id
    ):
        raise ValueError(f"{field_name} contains unsupported characters.")
    return record_id


def safe_quarantine_record_id(value: str, *, fallback: str) -> str:
    try:
        safe_value = safe_record_id(value, field_name="stored record id")
    except ValueError:
        return fallback
    return safe_value if safe_value == value else fallback


def json_dumps(value: dict | None) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_json_constant(value: str):
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number is not allowed")
    return parsed


def _strict_json_loads(value: str):
    return json.loads(
        value,
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_nonstandard_json_constant,
        parse_float=_parse_finite_json_float,
    )


def json_loads(value: str | None) -> Optional[dict]:
    if not value:
        return None
    data = _strict_json_loads(value)
    return data if isinstance(data, dict) else None


def required_json_loads(value: str | None) -> dict:
    data = json_loads(value)
    if data is None:
        raise ValueError("Stored JSON payload must be an object.")
    return data


def optional_json_loads(value: str | None) -> Optional[dict]:
    """Return optional historical JSON only when it is a strict object."""

    try:
        return json_loads(value)
    except (RecursionError, TypeError, ValueError):
        return None


def canonical_public_assistant_content(public_report: Mapping) -> str:
    """Return the one assistant string allowed for a canonical projection."""

    public_run = public_report.get("run")
    redaction = public_report.get("public_redaction")
    if not isinstance(public_run, Mapping) or not isinstance(redaction, Mapping):
        raise ValueError("Canonical public report is malformed.")
    if redaction.get("applied") is True:
        if (
            public_run.get("final_decision") == "refuse"
            and public_run.get("terminal_reason")
            in ORDINARY_REFUSAL_TERMINAL_REASONS
        ):
            return SELF_CHECK_REFUSAL_ANSWER
        return PUBLIC_REDACTION_TEXT
    final_answer = public_run.get("final_answer")
    if final_answer is None:
        return ""
    if type(final_answer) is not str:
        raise ValueError("Canonical public final answer must be a string or null.")
    return final_answer


def safe_stored_message_id(value) -> Optional[int]:
    """Return a persisted message identity only when SQLite kept it canonical."""

    return value if type(value) is int and value > 0 else None


def canonical_persisted_thinking(
    thinking: Optional[dict],
    report: Optional[LoopReport],
) -> Optional[dict]:
    """Bind durable model thinking to the raw run or redact it fail-closed."""

    if report is None or report.run.final_decision is None:
        return None
    if report.run.final_decision.value in THINKING_REDACTED_FINAL_DECISIONS:
        return {
            "available": False,
            "redacted": True,
            "label": MODEL_THINKING_LABEL,
            "content": MODEL_THINKING_REDACTION,
            "note": MODEL_THINKING_NOTE,
        }
    if not isinstance(thinking, Mapping):
        return None
    content = thinking.get("content")
    expected_digest = report.run.metadata.get(MODEL_THINKING_SHA256_METADATA_KEY)
    if (
        type(content) is not str
        or not content
        or type(expected_digest) is not str
        or len(expected_digest) != 64
        or any(char not in "0123456789abcdef" for char in expected_digest)
    ):
        return None
    actual_digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(expected_digest, actual_digest):
        return None
    return {
        "available": True,
        "redacted": False,
        "label": MODEL_THINKING_LABEL,
        "content": content,
        "note": MODEL_THINKING_NOTE,
    }


def validate_raw_loop_report_payload(payload: Mapping) -> None:
    """Reject non-canonical JSON shapes before permissive typed deserialization."""

    _validate_json_value(payload, label="loop report")
    _require_string_field(payload, "schema_version", label="loop report")
    run = _require_mapping_field(payload, "run", label="loop report")
    for field_name in (
        "run_id",
        "session_id",
        "user_input",
        "context_provider",
        "backend",
        "model_label",
        "started_at",
    ):
        _require_string_field(
            run,
            field_name,
            label="loop report run",
            nonempty=field_name in {"run_id", "session_id", "started_at"},
        )
    for field_name in (
        "completed_at",
        "final_decision",
        "terminal_reason",
        "final_answer",
        "error_message",
    ):
        _validate_optional_string_field(run, field_name, label="loop report run")
    _validate_optional_mapping_field(run, "metadata", label="loop report run")

    if "policy" in run:
        policy = _require_mapping_field(run, "policy", label="loop report run")
        if "max_retries" in policy:
            _require_nonnegative_int(
                policy["max_retries"],
                label="loop report policy max_retries",
            )
        for field_name in (
            "require_citations",
            "require_verifier_for_supported",
            "allow_mock_supported",
            "allow_tool_calls",
            "require_human_review_for_tools",
        ):
            if field_name in policy and type(policy[field_name]) is not bool:
                raise ValueError(f"loop report policy {field_name} must be a boolean")
        _validate_optional_mapping_field(policy, "metadata", label="loop report policy")

    steps = run.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError("loop report run steps must be an array")
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise ValueError(f"loop report step {index} must be an object")
        _validate_raw_loop_step(step, index=index)

    evidence = run.get("evidence", [])
    if not isinstance(evidence, list):
        raise ValueError("loop report run evidence must be an array")
    for index, reference in enumerate(evidence):
        if not isinstance(reference, Mapping):
            raise ValueError(f"loop report evidence {index} must be an object")
        _validate_raw_evidence_reference(reference, index=index)


def _validate_raw_loop_step(step: Mapping, *, index: int) -> None:
    label = f"loop report step {index}"
    for field_name in ("step_id", "phase", "started_at"):
        _require_string_field(
            step,
            field_name,
            label=label,
            nonempty=field_name in {"step_id", "started_at"},
        )
    _validate_optional_string_field(step, "decision", label=label)
    for field_name in (
        "name",
        "ended_at",
        "input_summary",
        "output_summary",
        "backend",
        "model_label",
        "error_message",
    ):
        _validate_optional_string_field(step, field_name, label=label)
    if "duration_ms" in step and step["duration_ms"] is not None:
        _require_nonnegative_int(step["duration_ms"], label=f"{label} duration_ms")
    if "retry_count" in step:
        _require_nonnegative_int(step["retry_count"], label=f"{label} retry_count")
    _validate_optional_mapping_field(step, "metadata", label=label)

    verification = step.get("verification")
    if verification is not None:
        if not isinstance(verification, Mapping):
            raise ValueError(f"{label} verification must be an object or null")
        _require_string_field(verification, "outcome", label=f"{label} verification")
        reasons = verification.get("reasons", [])
        if not isinstance(reasons, list) or not all(
            type(reason) is str for reason in reasons
        ):
            raise ValueError(f"{label} verification reasons must be an array of strings")
        for field_name in (
            "verifier",
            "verifier_backend",
            "verifier_model_label",
            "raw_response",
        ):
            _validate_optional_string_field(
                verification,
                field_name,
                label=f"{label} verification",
            )
        same_model = verification.get("same_model_as_drafter")
        if same_model is not None and type(same_model) is not bool:
            raise ValueError(
                f"{label} verification same_model_as_drafter must be a boolean or null"
            )
        _validate_optional_mapping_field(
            verification,
            "metadata",
            label=f"{label} verification",
        )

    human_review = step.get("human_review")
    if human_review is not None:
        if not isinstance(human_review, Mapping):
            raise ValueError(f"{label} human_review must be an object or null")
        for field_name in ("request_id", "reason", "instructions", "created_at"):
            _require_string_field(
                human_review,
                field_name,
                label=f"{label} human_review",
            )
        _validate_optional_string_field(
            human_review,
            "requested_by_step_id",
            label=f"{label} human_review",
        )
        _validate_optional_mapping_field(
            human_review,
            "metadata",
            label=f"{label} human_review",
        )


def _validate_raw_evidence_reference(reference: Mapping, *, index: int) -> None:
    label = f"loop report evidence {index}"
    _require_string_field(reference, "evidence_id", label=label, nonempty=True)
    _require_string_field(reference, "provider", label=label, nonempty=True)
    if "citation_id" not in reference:
        raise ValueError(f"{label} citation_id is required")
    _require_nonnegative_int(reference["citation_id"], label=f"{label} citation_id")
    if reference["citation_id"] < 1:
        raise ValueError(f"{label} citation_id must be positive")
    if "locator" in reference:
        locator = _require_mapping_field(reference, "locator", label=label)
        for field_name in ("page", "chunk_index"):
            if field_name in locator and locator[field_name] is not None:
                _require_nonnegative_int(
                    locator[field_name],
                    label=f"{label} locator {field_name}",
                )


def _validate_json_value(value, *, label: str) -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, label=f"{label}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"{label} contains a non-string object key")
            _validate_json_value(item, label=f"{label}.{key}")
        return
    raise ValueError(f"{label} contains a non-JSON value")


def _require_mapping_field(data: Mapping, field_name: str, *, label: str) -> Mapping:
    if field_name not in data or not isinstance(data[field_name], Mapping):
        raise ValueError(f"{label} {field_name} must be an object")
    return data[field_name]


def _require_string_field(
    data: Mapping,
    field_name: str,
    *,
    label: str,
    nonempty: bool = False,
) -> str:
    if field_name not in data or type(data[field_name]) is not str:
        raise ValueError(f"{label} {field_name} must be a string")
    value = data[field_name]
    if nonempty and (not value or value.strip() != value):
        raise ValueError(f"{label} {field_name} must be a nonempty canonical string")
    return value


def _validate_optional_string_field(
    data: Mapping,
    field_name: str,
    *,
    label: str,
) -> None:
    if field_name in data and data[field_name] is not None:
        if type(data[field_name]) is not str:
            raise ValueError(f"{label} {field_name} must be a string or null")


def _validate_optional_mapping_field(
    data: Mapping,
    field_name: str,
    *,
    label: str,
) -> None:
    if field_name in data and not isinstance(data[field_name], Mapping):
        raise ValueError(f"{label} {field_name} must be an object")


def _require_nonnegative_int(value, *, label: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")


def _payload_is_canonical_subset(provided, canonical) -> bool:
    if isinstance(provided, Mapping):
        return isinstance(canonical, Mapping) and all(
            key in canonical
            and _payload_is_canonical_subset(value, canonical[key])
            for key, value in provided.items()
        )
    if isinstance(provided, list):
        return (
            isinstance(canonical, list)
            and len(provided) == len(canonical)
            and all(
                _payload_is_canonical_subset(item, canonical[index])
                for index, item in enumerate(provided)
            )
        )
    return type(provided) is type(canonical) and provided == canonical


def _is_unbound_legacy_visible_answer_report(report: LoopReport) -> bool:
    """Identify v1 visible-answer rows written before binding contracts.

    There is no safe migration for these rows: the stored summaries cannot prove
    which terminal answer was emitted, and old supported rows also cannot prove
    which evidence set the verifier evaluated. Keep the raw SQLite value in
    place, but quarantine it from canonical/public serving with an explicit
    upgrade reason.
    """

    final_decision = report.run.final_decision
    if (
        report.schema_version != "loop-report/v1"
        or final_decision is None
        or final_decision.value not in {"final", "supported", "not_verified"}
    ):
        return False
    final_steps = [
        step for step in report.run.steps if step.phase.value == "final"
    ]
    if (
        len(final_steps) != 1
        or final_steps[0] is not report.run.steps[-1]
        or ANSWER_CANDIDATE_SHA256_METADATA_KEY not in final_steps[0].metadata
        or EVIDENCE_SET_SHA256_METADATA_KEY not in final_steps[0].metadata
    ):
        return True
    if final_decision.value != "supported":
        return False
    verify_steps = [
        step for step in report.run.steps if step.phase.value == "verify"
    ]
    draft_steps = [
        step for step in report.run.steps[:-1] if step.phase.value == "draft"
    ]
    candidate_steps = [
        step
        for step in report.run.steps[:-1]
        if step.phase.value == "draft"
        or (
            step.phase.value == "format_check"
            and step.metadata.get("sanitized_internal_labels") is True
        )
    ]
    bound_steps = [final_steps[0]]
    if verify_steps:
        bound_steps.append(verify_steps[-1])
    if candidate_steps:
        bound_steps.append(candidate_steps[-1])
    return (
        not verify_steps
        or not draft_steps
        or not candidate_steps
        or any(
            ANSWER_CANDIDATE_SHA256_METADATA_KEY not in step.metadata
            or EVIDENCE_SET_SHA256_METADATA_KEY not in step.metadata
            for step in bound_steps
        )
    )


def json_list_dumps(value: tuple[str, ...] | list[str] | None) -> str:
    return json.dumps(
        list(value or ()),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def json_list_loads(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    try:
        data = _strict_json_loads(value)
    except (RecursionError, TypeError, ValueError):
        return ()
    if not isinstance(data, list) or not all(type(item) is str for item in data):
        return ()
    return tuple(data)


def loop_run_metadata_from_report(report: LoopReport) -> dict:
    run = report.run
    run_id = safe_record_id(run.run_id, field_name="run_id")
    metadata = run.metadata
    return {
        "run_id": run_id,
        "schema_version": report.schema_version,
        "final_decision": (
            run.final_decision.value if run.final_decision is not None else None
        ),
        "context_provider": run.context_provider,
        "backend": run.backend,
        "model_label": run.model_label,
        "recipe_id": metadata.get("recipe_id"),
        "recipe_name": metadata.get("recipe_name"),
        "step_count": len(run.steps),
        "started_at": run.to_dict().get("started_at"),
        "completed_at": run.to_dict().get("completed_at"),
    }


class QuarantinedLoopRunError(RuntimeError):
    """Raised when a stored raw loop report cannot be safely projected."""


@dataclass(frozen=True)
class LoopRunRecord:
    run_id: str
    thread_id: str
    schema_version: str
    raw_report: dict
    public_report: dict
    final_decision: Optional[str]
    context_provider: Optional[str]
    backend: Optional[str]
    model_label: Optional[str]
    recipe_id: Optional[str]
    recipe_name: Optional[str]
    step_count: int
    started_at: Optional[str]
    completed_at: Optional[str]
    created_at: str
    user_message_id: Optional[int] = None
    assistant_message_id: Optional[int] = None
    quarantine_reason: Optional[str] = None

    def summary_dict(self) -> dict:
        public_run = self.public_report.get("run")
        if self.quarantine_reason or not isinstance(public_run, Mapping):
            return {
                "run_id": self.run_id,
                "thread_id": self.thread_id,
                "created_at": self.created_at,
                "projection_status": "quarantined",
                "quarantine_reason": (
                    self.quarantine_reason or "stored_loop_report_invalid"
                ),
            }
        steps = public_run.get("steps")
        return {
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "projection_status": "available",
            "projection_schema_version": (
                self.public_report.get("projection_schema_version")
            ),
            "final_decision": public_run.get("final_decision"),
            "terminal_reason": public_run.get("terminal_reason"),
            "context_provider": public_run.get("context_provider"),
            "backend": public_run.get("backend"),
            "model": public_run.get("model_label"),
            "step_count": len(steps) if isinstance(steps, list) else 0,
            "started_at": public_run.get("started_at"),
            "completed_at": public_run.get("completed_at"),
            "created_at": self.created_at,
        }

    def detail_dict(self, *, public: bool = True) -> dict:
        if self.quarantine_reason:
            raise QuarantinedLoopRunError(
                "Stored loop run is quarantined and cannot be inspected."
            )
        payload = self.summary_dict()
        payload["report"] = self.public_report if public else self.raw_report
        payload["public"] = public
        return payload


@dataclass(frozen=True)
class ThreadMessage:
    id: int
    thread_id: str
    role: str
    content: str
    created_at: str
    thinking: Optional[dict] = None
    loop_payload: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
            "thinking": self.thinking,
            "loop_payload": self.loop_payload,
        }


@dataclass(frozen=True)
class _MessageReadPolicy:
    """Canonical read bindings for messages claimed by durable loop runs."""

    claimed_message_ids: frozenset[int]
    reports_by_message_id: dict[int, LoopReport]
    contents_by_message_id: dict[int, str]

    def permits(self, message_id: int) -> bool:
        return (
            message_id not in self.claimed_message_ids
            or message_id in self.contents_by_message_id
        )


@dataclass(frozen=True)
class ThreadMemory:
    message_id: int
    thread_id: str
    role: str
    content: str
    created_at: str
    score: float

    def to_context_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "role": self.role,
            "content": self.content,
            "score": self.score,
        }


@dataclass(frozen=True)
class ThreadRecord:
    id: str
    title: str
    created_at: str
    updated_at: str
    instance_id: str
    generation: int = 0
    message_count: int = 0
    memory_count: int = 0
    loop_run_count: int = 0
    messages: tuple[ThreadMessage, ...] = ()
    loop_runs: tuple[LoopRunRecord, ...] = ()
    latest: Optional[dict] = None

    def summary_dict(self, *, include_latest: bool = False) -> dict:
        payload = {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": self.message_count,
            "memory_count": self.memory_count,
            "loop_run_count": self.loop_run_count,
        }
        if include_latest:
            payload["latest"] = self.latest
        return payload

    def detail_dict(self) -> dict:
        payload = self.summary_dict(include_latest=True)
        payload["messages"] = [message.to_dict() for message in self.messages]
        payload["loop_runs"] = [run.summary_dict() for run in self.loop_runs]
        return payload


class ThreadStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL")
            self._create_schema()

    @classmethod
    def in_memory(cls) -> "ThreadStore":
        return cls(":memory:")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _create_schema(self) -> None:
        binding_table_existed = (
            self._conn.execute(
                """
                SELECT 1
                FROM sqlite_master
                WHERE type = 'table' AND name = 'loop_run_message_bindings'
                """
            ).fetchone()
            is not None
        )
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS threads (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                instance_id TEXT NOT NULL DEFAULT '',
                generation INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                thinking_json TEXT,
                loop_payload_json TEXT,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_thread_id_id
                ON messages(thread_id, id);

            CREATE TABLE IF NOT EXISTS message_embeddings (
                message_id INTEGER PRIMARY KEY
                    REFERENCES messages(id) ON DELETE CASCADE,
                thread_id TEXT NOT NULL
                    REFERENCES threads(id) ON DELETE CASCADE,
                embedding_model TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_message_embeddings_thread_model
                ON message_embeddings(thread_id, embedding_model);

            CREATE TABLE IF NOT EXISTS loop_runs (
                run_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
                user_message_id INTEGER REFERENCES messages(id) ON DELETE SET NULL,
                assistant_message_id INTEGER REFERENCES messages(id) ON DELETE SET NULL,
                schema_version TEXT NOT NULL,
                raw_report_json TEXT NOT NULL,
                public_report_json TEXT NOT NULL,
                final_decision TEXT,
                context_provider TEXT,
                backend TEXT,
                model_label TEXT,
                recipe_id TEXT,
                recipe_name TEXT,
                step_count INTEGER NOT NULL DEFAULT 0,
                started_at TEXT,
                completed_at TEXT,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_loop_runs_thread_created
                ON loop_runs(thread_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS loop_run_message_bindings (
                run_id TEXT PRIMARY KEY
                    REFERENCES loop_runs(run_id) ON DELETE CASCADE,
                thread_id TEXT NOT NULL
                    REFERENCES threads(id) ON DELETE CASCADE,
                user_message_id INTEGER NOT NULL
                    REFERENCES messages(id) ON DELETE CASCADE,
                assistant_message_id INTEGER NOT NULL
                    REFERENCES messages(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS loop_recipes (
                recipe_id TEXT PRIMARY KEY,
                schema_version TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                goal TEXT NOT NULL,
                instructions TEXT NOT NULL DEFAULT '',
                success_criteria_json TEXT NOT NULL DEFAULT '[]',
                stop_condition TEXT NOT NULL,
                context_provider TEXT NOT NULL DEFAULT 'smart',
                model_profile TEXT NOT NULL DEFAULT 'quality',
                verifier TEXT NOT NULL DEFAULT 'default',
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_loop_recipes_updated
                ON loop_recipes(updated_at DESC);
            """
        )
        if not binding_table_existed:
            # Existing installations predate the independent binding witness.
            # Snapshot their current link once during migration; after the table
            # exists, missing or mismatched witnesses fail closed instead of
            # being silently regenerated from mutable loop_runs columns.
            self._conn.execute(
                """
                INSERT OR IGNORE INTO loop_run_message_bindings (
                    run_id, thread_id, user_message_id, assistant_message_id
                )
                SELECT
                    lr.run_id,
                    lr.thread_id,
                    lr.user_message_id,
                    lr.assistant_message_id
                FROM loop_runs lr
                JOIN threads t ON t.id = lr.thread_id
                JOIN messages user_message
                  ON user_message.id = lr.user_message_id
                 AND user_message.thread_id = lr.thread_id
                JOIN messages assistant_message
                  ON assistant_message.id = lr.assistant_message_id
                 AND assistant_message.thread_id = lr.thread_id
                """
            )
        columns = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(threads)").fetchall()
        }
        if "generation" not in columns:
            self._conn.execute(
                "ALTER TABLE threads ADD COLUMN generation INTEGER NOT NULL DEFAULT 0"
            )
        if "instance_id" not in columns:
            self._conn.execute(
                "ALTER TABLE threads ADD COLUMN instance_id TEXT NOT NULL DEFAULT ''"
            )
        for row in self._conn.execute(
            "SELECT id FROM threads WHERE instance_id = ''"
        ).fetchall():
            self._conn.execute(
                "UPDATE threads SET instance_id = ? WHERE id = ?",
                (default_thread_instance_id(), str(row["id"])),
            )
        self._conn.commit()

    def create_thread(
        self,
        *,
        title: str | None = None,
        thread_id: str | None = None,
    ) -> ThreadRecord:
        now = utc_now()
        record_id = thread_id or default_thread_id()
        instance_id = default_thread_instance_id()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO threads (
                    id, title, created_at, updated_at, instance_id
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (record_id, safe_title(title), now, now, instance_id),
            )
            self._conn.commit()
            record = self.get_thread(record_id)
            if record is None:
                raise RuntimeError("Thread was not created.")
            return record

    def ensure_thread(
        self,
        thread_id: str,
        *,
        title: str | None = None,
    ) -> ThreadRecord:
        record, _created = self.ensure_thread_with_created(
            thread_id,
            title=title,
        )
        return record

    def ensure_thread_with_created(
        self,
        thread_id: str,
        *,
        title: str | None = None,
    ) -> tuple[ThreadRecord, bool]:
        """Atomically return a thread and whether this call created it."""

        now = utc_now()
        instance_id = default_thread_instance_id()
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO threads (
                    id, title, created_at, updated_at, instance_id
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO NOTHING
                """,
                (
                    thread_id,
                    safe_title(title),
                    now,
                    now,
                    instance_id,
                ),
            )
            self._conn.commit()
            record = self.get_thread(thread_id)
            if record is None:
                raise RuntimeError("Thread was not ensured.")
            return record, cursor.rowcount > 0

    def list_threads(self, *, limit: int = 30) -> list[ThreadRecord]:
        with self._lock:
            snapshot_started = False
            try:
                # The summary row and its canonical visible counts must name the
                # same durable thread instance. Keep every SELECT in one read
                # snapshot so a cross-process delete/recreate cannot combine an
                # old title with replacement messages or memories.
                self._conn.execute("SAVEPOINT thread_list_snapshot")
                snapshot_started = True
                rows = self._conn.execute(
                    """
                    SELECT
                        t.id,
                        t.title,
                        t.created_at,
                        t.updated_at,
                        t.instance_id,
                        t.generation,
                        COUNT(m.id) AS message_count,
                        (
                            SELECT COUNT(*)
                            FROM message_embeddings me
                            WHERE me.thread_id = t.id
                        ) AS memory_count,
                        (
                            SELECT COUNT(*)
                            FROM loop_runs lr
                            WHERE lr.thread_id = t.id
                        ) AS loop_run_count,
                        (
                            SELECT loop_payload_json
                            FROM messages latest
                            WHERE latest.thread_id = t.id
                              AND latest.loop_payload_json IS NOT NULL
                            ORDER BY latest.id DESC
                            LIMIT 1
                        ) AS latest_json
                    FROM threads t
                    LEFT JOIN messages m ON m.thread_id = t.id
                    GROUP BY
                        t.id, t.title, t.created_at, t.updated_at,
                        t.instance_id, t.generation
                    ORDER BY t.updated_at DESC, t.created_at DESC
                    LIMIT ?
                    """,
                    (max(1, int(limit)),),
                ).fetchall()
                records = []
                for row in rows:
                    thread_id = str(row["id"])
                    policy = self._message_read_policy(thread_id)
                    usable_message_ids = self._usable_message_ids(
                        thread_id,
                        policy=policy,
                    )
                    records.append(
                        self._thread_from_row(
                            row,
                            messages=(),
                            message_count=len(usable_message_ids),
                            memory_count=self._usable_memory_count(
                                thread_id,
                                usable_message_ids=usable_message_ids,
                            ),
                        )
                    )
                self._conn.execute("RELEASE SAVEPOINT thread_list_snapshot")
                snapshot_started = False
                return records
            except Exception:
                if snapshot_started:
                    self._conn.execute(
                        "ROLLBACK TO SAVEPOINT thread_list_snapshot"
                    )
                    self._conn.execute(
                        "RELEASE SAVEPOINT thread_list_snapshot"
                    )
                raise

    def get_thread(self, thread_id: str) -> Optional[ThreadRecord]:
        with self._lock:
            snapshot_started = False
            try:
                # The thread row, messages, and loop reports form one durable
                # identity snapshot. Without an explicit read transaction, a
                # second connection can delete/recreate the same thread id after
                # the first SELECT and make this method combine the old instance
                # metadata with the replacement instance's messages. A savepoint
                # also preserves any transaction already owned by the caller.
                self._conn.execute("SAVEPOINT thread_read_snapshot")
                snapshot_started = True
                row = self._conn.execute(
                    """
                    SELECT
                        t.id,
                        t.title,
                        t.created_at,
                        t.updated_at,
                        t.instance_id,
                        t.generation,
                        COUNT(m.id) AS message_count,
                        (
                            SELECT COUNT(*)
                            FROM message_embeddings me
                            WHERE me.thread_id = t.id
                        ) AS memory_count,
                        (
                            SELECT COUNT(*)
                            FROM loop_runs lr
                            WHERE lr.thread_id = t.id
                        ) AS loop_run_count,
                        (
                            SELECT loop_payload_json
                            FROM messages latest
                            WHERE latest.thread_id = t.id
                              AND latest.loop_payload_json IS NOT NULL
                            ORDER BY latest.id DESC
                            LIMIT 1
                        ) AS latest_json
                    FROM threads t
                    LEFT JOIN messages m ON m.thread_id = t.id
                    WHERE t.id = ?
                    GROUP BY
                        t.id, t.title, t.created_at, t.updated_at,
                        t.instance_id, t.generation
                    """,
                    (thread_id,),
                ).fetchone()
                if row is None:
                    self._conn.execute("RELEASE SAVEPOINT thread_read_snapshot")
                    return None
                loop_runs = tuple(
                    self._loop_run_from_row(run_row)
                    for run_row in self._conn.execute(
                        """
                        SELECT
                            run_id, thread_id, user_message_id,
                            assistant_message_id, schema_version,
                            raw_report_json, public_report_json,
                            final_decision, context_provider, backend,
                            model_label, recipe_id, recipe_name, step_count,
                            started_at, completed_at, created_at
                        FROM loop_runs
                        WHERE thread_id = ?
                        ORDER BY
                            assistant_message_id DESC,
                            user_message_id DESC,
                            rowid DESC,
                            created_at DESC
                        LIMIT ?
                        """,
                        (thread_id, MAX_LOOP_RUN_LIST_LIMIT),
                    ).fetchall()
                )
                policy = self._message_read_policy(thread_id)
                messages = tuple(
                    message
                    for message_row in self._conn.execute(
                        """
                        SELECT id, thread_id, role, content, thinking_json,
                               loop_payload_json, created_at
                        FROM messages
                        WHERE thread_id = ?
                        ORDER BY id ASC
                        """,
                        (thread_id,),
                    ).fetchall()
                    if (
                        message := self._message_from_row(
                            message_row,
                            read_policy=policy,
                            bound_report=policy.reports_by_message_id.get(
                                safe_stored_message_id(message_row["id"])
                            ),
                            canonical_content=(
                                policy.contents_by_message_id.get(
                                    safe_stored_message_id(message_row["id"])
                                )
                            ),
                        )
                    )
                )
                usable_message_ids = frozenset(message.id for message in messages)
                usable_memory_count = self._usable_memory_count(
                    thread_id,
                    usable_message_ids=usable_message_ids,
                )
                record = self._thread_from_row(
                    row,
                    messages=messages,
                    loop_runs=loop_runs,
                    message_count=len(messages),
                    memory_count=usable_memory_count,
                )
                self._conn.execute("RELEASE SAVEPOINT thread_read_snapshot")
                return record
            except Exception:
                if snapshot_started:
                    self._conn.execute(
                        "ROLLBACK TO SAVEPOINT thread_read_snapshot"
                    )
                    self._conn.execute(
                        "RELEASE SAVEPOINT thread_read_snapshot"
                    )
                raise

    def recent_messages(self, thread_id: str, limit: int = 12) -> tuple[ThreadMessage, ...]:
        with self._lock:
            snapshot_started = False
            try:
                self._conn.execute("SAVEPOINT recent_message_snapshot")
                snapshot_started = True
                policy = self._message_read_policy(thread_id)
                rows = self._conn.execute(
                    """
                    SELECT id, thread_id, role, content, thinking_json,
                           loop_payload_json, created_at
                    FROM messages
                    WHERE thread_id = ?
                    ORDER BY id ASC
                    """,
                    (thread_id,),
                ).fetchall()
                messages = tuple(
                    message
                    for row in rows
                    if (
                        message := self._message_from_row(
                            row,
                            read_policy=policy,
                            bound_report=policy.reports_by_message_id.get(
                                safe_stored_message_id(row["id"])
                            ),
                            canonical_content=(
                                policy.contents_by_message_id.get(
                                    safe_stored_message_id(row["id"])
                                )
                            ),
                        )
                    )
                )
                self._conn.execute("RELEASE SAVEPOINT recent_message_snapshot")
                snapshot_started = False
                return messages[-max(1, int(limit)) :]
            except Exception:
                if snapshot_started:
                    self._conn.execute(
                        "ROLLBACK TO SAVEPOINT recent_message_snapshot"
                    )
                    self._conn.execute(
                        "RELEASE SAVEPOINT recent_message_snapshot"
                    )
                raise

    def has_message_embeddings(
        self,
        thread_id: str,
        embedding_model: str,
    ) -> bool:
        with self._lock:
            snapshot_started = False
            try:
                self._conn.execute("SAVEPOINT message_embedding_snapshot")
                snapshot_started = True
                policy = self._message_read_policy(thread_id)
                rows = self._conn.execute(
                    """
                    SELECT e.message_id
                    FROM message_embeddings e
                    JOIN messages m
                      ON m.id = e.message_id
                     AND m.thread_id = ?
                    WHERE e.thread_id = ? AND e.embedding_model = ?
                    """,
                    (thread_id, thread_id, str(embedding_model or "")),
                ).fetchall()
                has_embeddings = any(
                    policy.permits(message_id)
                    for row in rows
                    if (message_id := safe_stored_message_id(row["message_id"]))
                    is not None
                )
                self._conn.execute(
                    "RELEASE SAVEPOINT message_embedding_snapshot"
                )
                snapshot_started = False
                return has_embeddings
            except Exception:
                if snapshot_started:
                    self._conn.execute(
                        "ROLLBACK TO SAVEPOINT message_embedding_snapshot"
                    )
                    self._conn.execute(
                        "RELEASE SAVEPOINT message_embedding_snapshot"
                    )
                raise

    def upsert_message_embedding(
        self,
        message: ThreadMessage,
        *,
        embedding_model: str,
        vector: list[float] | tuple[float, ...],
    ) -> bool:
        vector_json = vector_to_json(vector)
        now = utc_now()
        with self._lock:
            row = self._conn.execute(
                """
                SELECT id
                FROM messages
                WHERE id = ? AND thread_id = ?
                """,
                (message.id, message.thread_id),
            ).fetchone()
            if row is None:
                return False
            self._conn.execute(
                """
                INSERT OR REPLACE INTO message_embeddings (
                    message_id, thread_id, embedding_model, vector_json, created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    message.thread_id,
                    str(embedding_model or ""),
                    vector_json,
                    now,
                ),
            )
            self._conn.commit()
            return True

    def semantic_memories(
        self,
        thread_id: str,
        *,
        embedding_model: str,
        query_vector: list[float] | tuple[float, ...],
        limit: int = 4,
        exclude_message_ids: tuple[int, ...] = (),
        min_score: float = 0.05,
    ) -> tuple[ThreadMemory, ...]:
        query = validate_vector(query_vector)
        excluded = {int(message_id) for message_id in exclude_message_ids}
        with self._lock:
            snapshot_started = False
            try:
                self._conn.execute("SAVEPOINT semantic_memory_snapshot")
                snapshot_started = True
                policy = self._message_read_policy(thread_id)
                rows = self._conn.execute(
                    """
                    SELECT
                        m.id,
                        m.thread_id,
                        m.role,
                        m.content,
                        m.created_at,
                        e.vector_json
                    FROM message_embeddings e
                    JOIN messages m
                      ON m.id = e.message_id
                     AND m.thread_id = ?
                    WHERE e.thread_id = ? AND e.embedding_model = ?
                    ORDER BY m.id ASC
                    """,
                    (thread_id, thread_id, str(embedding_model or "")),
                ).fetchall()
                self._conn.execute("RELEASE SAVEPOINT semantic_memory_snapshot")
                snapshot_started = False
            except Exception:
                if snapshot_started:
                    self._conn.execute(
                        "ROLLBACK TO SAVEPOINT semantic_memory_snapshot"
                    )
                    self._conn.execute(
                        "RELEASE SAVEPOINT semantic_memory_snapshot"
                    )
                raise

        scored = []
        for row in rows:
            message_id = int(row["id"])
            if message_id in excluded:
                continue
            if not policy.permits(message_id):
                continue
            role = str(row["role"])
            score = cosine_similarity(query, vector_from_json(row["vector_json"]))
            if score < float(min_score):
                continue
            scored.append(
                ThreadMemory(
                    message_id=message_id,
                    thread_id=str(row["thread_id"]),
                    role=role,
                    content=(
                        policy.contents_by_message_id[message_id]
                        if message_id in policy.claimed_message_ids
                        else str(row["content"])
                    ),
                    created_at=canonical_stored_timestamp(row["created_at"]),
                    score=round(score, 6),
                )
            )

        scored.sort(key=lambda memory: (-memory.score, memory.message_id))
        return tuple(scored[: max(1, int(limit))])

    def list_loop_runs(
        self,
        thread_id: str,
        *,
        limit: int = MAX_LOOP_RUN_LIST_LIMIT,
    ) -> tuple[LoopRunRecord, ...]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    run_id, thread_id, user_message_id, assistant_message_id,
                    schema_version, raw_report_json, public_report_json,
                    final_decision, context_provider, backend, model_label,
                    recipe_id, recipe_name, step_count, started_at,
                    completed_at, created_at
                FROM loop_runs
                WHERE thread_id = ?
                ORDER BY
                    assistant_message_id DESC,
                    user_message_id DESC,
                    rowid DESC,
                    created_at DESC
                LIMIT ?
                """,
                (thread_id, max(1, int(limit))),
            ).fetchall()
            return tuple(self._loop_run_from_row(row) for row in rows)

    def get_loop_run(
        self,
        thread_id: str,
        run_id: str,
    ) -> Optional[LoopRunRecord]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT
                    run_id, thread_id, user_message_id, assistant_message_id,
                    schema_version, raw_report_json, public_report_json,
                    final_decision, context_provider, backend, model_label,
                    recipe_id, recipe_name, step_count, started_at,
                    completed_at, created_at
                FROM loop_runs
                WHERE thread_id = ? AND run_id = ?
                """,
                (thread_id, run_id),
            ).fetchone()
            return self._loop_run_from_row(row) if row else None

    def ensure_default_recipe(self) -> LoopRecipe:
        with self._lock:
            existing = self.get_recipe(DEFAULT_LOOP_RECIPE_ID)
            if existing is not None:
                current_default = default_loop_recipe(created_at=existing.created_at)
                if (
                    existing.metadata.get("built_in") is True
                    and existing != current_default
                ):
                    self._conn.execute(
                        "DELETE FROM loop_recipes WHERE recipe_id = ?",
                        (DEFAULT_LOOP_RECIPE_ID,),
                    )
                    self._insert_recipe(current_default)
                    self._conn.commit()
                    return current_default
                return existing
            recipe = default_loop_recipe()
            self._insert_recipe(recipe)
            self._conn.commit()
            return recipe

    def list_recipes(
        self,
        *,
        limit: int = MAX_LOOP_RECIPE_LIST_LIMIT,
    ) -> tuple[LoopRecipe, ...]:
        with self._lock:
            self.ensure_default_recipe()
            rows = self._conn.execute(
                """
                SELECT
                    recipe_id, schema_version, name, description, goal,
                    instructions, success_criteria_json, stop_condition,
                    context_provider, model_profile, verifier, metadata_json,
                    created_at, updated_at
                FROM loop_recipes
                ORDER BY
                    CASE WHEN recipe_id = ? THEN 0 ELSE 1 END,
                    updated_at DESC
                LIMIT ?
                """,
                (DEFAULT_LOOP_RECIPE_ID, max(1, int(limit))),
            ).fetchall()
            return tuple(self._recipe_from_row(row) for row in rows)

    def get_recipe(self, recipe_id: str) -> Optional[LoopRecipe]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT
                    recipe_id, schema_version, name, description, goal,
                    instructions, success_criteria_json, stop_condition,
                    context_provider, model_profile, verifier, metadata_json,
                    created_at, updated_at
                FROM loop_recipes
                WHERE recipe_id = ?
                """,
                (recipe_id,),
            ).fetchone()
            return self._recipe_from_row(row) if row else None

    def create_recipe(
        self,
        *,
        name: str,
        goal: str,
        instructions: str = "",
        success_criteria: tuple[str, ...] | list[str] = (),
        stop_condition: str = "",
        context_provider: str = "smart",
        model_profile: str = "quality",
        verifier: str = "default",
        description: str = "",
        metadata: Optional[Mapping] = None,
        recipe_id: Optional[str] = None,
    ) -> LoopRecipe:
        now = datetime.now(timezone.utc)
        recipe = LoopRecipe(
            recipe_id=safe_record_id(
                recipe_id or default_recipe_id(),
                field_name="recipe_id",
            ),
            name=name,
            description=description,
            goal=goal,
            instructions=instructions,
            success_criteria=tuple(success_criteria or ()),
            stop_condition=stop_condition or default_loop_recipe().stop_condition,
            context_provider=context_provider,
            model_profile=model_profile,
            verifier=verifier,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )
        with self._lock:
            if self.get_recipe(recipe.recipe_id) is not None:
                raise ValueError("Loop recipe already exists.")
            self._insert_recipe(recipe)
            self._conn.commit()
            return recipe

    def update_recipe(
        self,
        recipe_id: str,
        *,
        name: Optional[str] = None,
        goal: Optional[str] = None,
        instructions: Optional[str] = None,
        success_criteria: Optional[tuple[str, ...] | list[str]] = None,
        stop_condition: Optional[str] = None,
        context_provider: Optional[str] = None,
        model_profile: Optional[str] = None,
        verifier: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Mapping] = None,
    ) -> LoopRecipe:
        now = datetime.now(timezone.utc)
        with self._lock:
            existing = self.get_recipe(recipe_id)
            if existing is None:
                raise KeyError(recipe_id)
            recipe = LoopRecipe(
                recipe_id=existing.recipe_id,
                name=existing.name if name is None else name,
                description=(
                    existing.description if description is None else description
                ),
                goal=existing.goal if goal is None else goal,
                instructions=(
                    existing.instructions
                    if instructions is None
                    else instructions
                ),
                success_criteria=(
                    existing.success_criteria
                    if success_criteria is None
                    else tuple(success_criteria)
                ),
                stop_condition=(
                    existing.stop_condition
                    if stop_condition is None
                    else stop_condition
                ),
                context_provider=(
                    existing.context_provider
                    if context_provider is None
                    else context_provider
                ),
                model_profile=(
                    existing.model_profile
                    if model_profile is None
                    else model_profile
                ),
                verifier=existing.verifier if verifier is None else verifier,
                created_at=existing.created_at,
                updated_at=now,
                metadata=existing.metadata if metadata is None else metadata,
            )
            self._conn.execute("DELETE FROM loop_recipes WHERE recipe_id = ?", (recipe_id,))
            self._insert_recipe(recipe)
            self._conn.commit()
            return recipe

    def delete_recipe(self, recipe_id: str) -> bool:
        if recipe_id == DEFAULT_LOOP_RECIPE_ID:
            return False
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM loop_recipes WHERE recipe_id = ?",
                (recipe_id,),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def rename_thread(self, thread_id: str, title: str) -> ThreadRecord:
        now = utc_now()
        with self._lock:
            cursor = self._conn.execute(
                """
                UPDATE threads
                SET title = ?, updated_at = ?
                WHERE id = ?
                """,
                (safe_title(title), now, thread_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(thread_id)
            self._conn.commit()
            record = self.get_thread(thread_id)
            if record is None:
                raise KeyError(thread_id)
            return record

    def append_message(
        self,
        thread_id: str,
        *,
        role: str,
        content: str,
        thinking: Optional[dict] = None,
        loop_payload: Optional[dict] = None,
    ) -> ThreadMessage:
        if role not in {"user", "assistant"}:
            raise ValueError("Thread message role must be 'user' or 'assistant'.")
        now = utc_now()
        with self._lock:
            self.ensure_thread(thread_id)
            cursor = self._conn.execute(
                """
                INSERT INTO messages (
                    thread_id, role, content, thinking_json,
                    loop_payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    role,
                    str(content or ""),
                    json_dumps(thinking),
                    json_dumps(loop_payload),
                    now,
                ),
            )
            self._conn.execute(
                "UPDATE threads SET updated_at = ? WHERE id = ?",
                (now, thread_id),
            )
            self._conn.commit()
            message_id = int(cursor.lastrowid)
            row = self._conn.execute(
                """
                SELECT id, thread_id, role, content, thinking_json,
                       loop_payload_json, created_at
                FROM messages
                WHERE id = ?
                """,
                (message_id,),
            ).fetchone()
            return self._message_from_row(row)

    def append_turn(
        self,
        thread_id: str,
        *,
        user_content: str,
        assistant_content: str,
        thinking: Optional[dict] = None,
        loop_payload: Optional[dict] = None,
        raw_loop_report: Optional[dict] = None,
        public_loop_report: Optional[dict] = None,
        expected_generation: Optional[int] = None,
        expected_instance_id: Optional[str] = None,
        title_if_empty: Optional[str] = None,
    ) -> Optional[tuple[ThreadMessage, ThreadMessage]]:
        if (expected_generation is None) != (expected_instance_id is None):
            raise ValueError(
                "expected_generation and expected_instance_id must be supplied together."
            )
        if type(user_content) is not str:
            raise TypeError("user_content must be a string.")
        if type(assistant_content) is not str:
            raise TypeError("assistant_content must be a string.")
        # Validate caller data even when it will be dropped for lacking a raw
        # report binding. This preserves transactional rejection of non-JSON
        # values instead of silently accepting malformed persistence input.
        json_dumps(thinking)
        now = utc_now()
        user_text = user_content
        assistant_text = assistant_content
        if (raw_loop_report is None) != (public_loop_report is None):
            raise ValueError(
                "raw_loop_report and public_loop_report must be supplied together."
            )
        typed_loop_report = None
        loop_run_metadata = None
        raw_loop_report_json = None
        public_loop_report_json = None
        if raw_loop_report is not None and public_loop_report is not None:
            if not isinstance(raw_loop_report, Mapping):
                raise ValueError("raw_loop_report must be an object.")
            if not isinstance(public_loop_report, Mapping):
                raise ValueError("public_loop_report must be an object.")
            supplied_raw_json = json_dumps(dict(raw_loop_report))
            normalized_raw_report = required_json_loads(supplied_raw_json)
            supplied_public_json = json_dumps(dict(public_loop_report))
            normalized_public_report = required_json_loads(supplied_public_json)
            validate_raw_loop_report_payload(normalized_raw_report)
            typed_loop_report = LoopReport.from_dict(normalized_raw_report)
            canonical_raw_report = typed_loop_report.to_dict()
            canonical_raw_json = json_dumps(canonical_raw_report)
            if supplied_raw_json != canonical_raw_json:
                raise ValueError("raw_loop_report is not canonical.")
            if typed_loop_report.run.session_id != thread_id:
                raise ValueError("Loop report session identity does not match thread_id.")
            if typed_loop_report.run.user_input != user_text:
                raise ValueError(
                    "user_content does not match the loop report user input."
                )
            if safe_record_id(thread_id, field_name="thread_id") != thread_id:
                raise ValueError("thread_id must be canonical.")
            if (
                safe_record_id(
                    typed_loop_report.run.session_id,
                    field_name="session_id",
                )
                != typed_loop_report.run.session_id
            ):
                raise ValueError("Loop report session identity must be canonical.")
            canonical_public_report = project_public_report(
                typed_loop_report,
                expected_run_id=typed_loop_report.run.run_id,
                expected_session_id=thread_id,
            )
            if (
                canonical_public_report.get("projection_schema_version")
                != PUBLIC_REPORT_PROJECTION_SCHEMA_VERSION
            ):
                raise ValueError("Unsupported public report projection schema.")
            canonical_public_json = json_dumps(canonical_public_report)
            if supplied_public_json != canonical_public_json:
                raise ValueError(
                    "public_loop_report does not match the canonical public projection."
                )
            expected_assistant_content = canonical_public_assistant_content(
                canonical_public_report
            )
            if assistant_text != expected_assistant_content:
                raise ValueError(
                    "assistant_content does not match the canonical public answer."
                )
            loop_run_metadata = loop_run_metadata_from_report(typed_loop_report)
            if loop_run_metadata["run_id"] != typed_loop_report.run.run_id:
                raise ValueError("Loop report run identity must be canonical.")
            raw_loop_report_json = canonical_raw_json
            public_loop_report_json = canonical_public_json
        thinking_json = json_dumps(
            canonical_persisted_thinking(thinking, typed_loop_report)
        )
        loop_payload_json = json_dumps(loop_payload)
        with self._lock:
            try:
                # The instance/generation check is a cross-connection compare-and-
                # swap boundary, not merely an in-process mutex concern. Acquire
                # SQLite's write reservation before reading the guard and keep it
                # through both messages and the optional loop run. Otherwise a
                # second app process can clear or ABA-recreate the thread after the
                # SELECT and receive stale messages from this turn.
                self._conn.execute("BEGIN IMMEDIATE")
                thread_row = self._conn.execute(
                    """
                    SELECT
                        title,
                        instance_id,
                        generation,
                        (
                            SELECT COUNT(*)
                            FROM messages
                            WHERE thread_id = threads.id
                        ) AS message_count
                    FROM threads
                    WHERE id = ?
                    """,
                    (thread_id,),
                ).fetchone()
                create_thread_in_transaction = thread_row is None
                if thread_row is None:
                    if (
                        expected_generation is not None
                        or expected_instance_id is not None
                    ):
                        self._conn.rollback()
                        return None
                    thread_row = {
                        "title": DEFAULT_THREAD_TITLE,
                        "instance_id": default_thread_instance_id(),
                        "generation": 0,
                        "message_count": 0,
                    }
                if (
                    expected_instance_id is not None
                    and str(thread_row["instance_id"]) != expected_instance_id
                ):
                    self._conn.rollback()
                    return None
                if (
                    expected_generation is not None
                    and int(thread_row["generation"]) != int(expected_generation)
                ):
                    self._conn.rollback()
                    return None
                if create_thread_in_transaction:
                    self._conn.execute(
                        """
                        INSERT INTO threads (
                            id, title, created_at, updated_at, instance_id
                        )
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            thread_id,
                            DEFAULT_THREAD_TITLE,
                            now,
                            now,
                            str(thread_row["instance_id"]),
                        ),
                    )
                if (
                    title_if_empty
                    and int(thread_row["message_count"] or 0) == 0
                    and str(thread_row["title"]) == DEFAULT_THREAD_TITLE
                ):
                    self._conn.execute(
                        """
                        UPDATE threads
                        SET title = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (safe_title(title_if_empty), now, thread_id),
                    )
                user_cursor = self._conn.execute(
                    """
                    INSERT INTO messages (
                        thread_id, role, content, thinking_json,
                        loop_payload_json, created_at
                    )
                    VALUES (?, 'user', ?, NULL, NULL, ?)
                    """,
                    (thread_id, user_text, now),
                )
                assistant_cursor = self._conn.execute(
                    """
                    INSERT INTO messages (
                        thread_id, role, content, thinking_json,
                        loop_payload_json, created_at
                    )
                    VALUES (?, 'assistant', ?, ?, ?, ?)
                    """,
                    (
                        thread_id,
                        assistant_text,
                        thinking_json,
                        loop_payload_json,
                        now,
                    ),
                )
                if loop_run_metadata and raw_loop_report_json and public_loop_report_json:
                    self._conn.execute(
                        """
                        INSERT INTO loop_runs (
                            run_id, thread_id, user_message_id,
                            assistant_message_id, schema_version,
                            raw_report_json, public_report_json, final_decision,
                            context_provider, backend, model_label, recipe_id,
                            recipe_name, step_count, started_at, completed_at,
                            created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            loop_run_metadata["run_id"],
                            thread_id,
                            int(user_cursor.lastrowid),
                            int(assistant_cursor.lastrowid),
                            loop_run_metadata["schema_version"],
                            raw_loop_report_json,
                            public_loop_report_json,
                            loop_run_metadata["final_decision"],
                            loop_run_metadata["context_provider"],
                            loop_run_metadata["backend"],
                            loop_run_metadata["model_label"],
                            loop_run_metadata["recipe_id"],
                            loop_run_metadata["recipe_name"],
                            int(loop_run_metadata["step_count"]),
                            loop_run_metadata["started_at"],
                            loop_run_metadata["completed_at"],
                            now,
                        ),
                    )
                    self._conn.execute(
                        """
                        INSERT INTO loop_run_message_bindings (
                            run_id, thread_id, user_message_id,
                            assistant_message_id
                        )
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            loop_run_metadata["run_id"],
                            thread_id,
                            int(user_cursor.lastrowid),
                            int(assistant_cursor.lastrowid),
                        ),
                    )
                self._conn.execute(
                    "UPDATE threads SET updated_at = ? WHERE id = ?",
                    (now, thread_id),
                )
                rows = self._conn.execute(
                    """
                    SELECT id, thread_id, role, content, thinking_json,
                           loop_payload_json, created_at
                    FROM messages
                    WHERE id IN (?, ?)
                    ORDER BY id ASC
                    """,
                    (int(user_cursor.lastrowid), int(assistant_cursor.lastrowid)),
                ).fetchall()
                if len(rows) != 2:
                    raise RuntimeError("Thread turn was not persisted.")
                persisted_turn = tuple(
                    self._message_from_row(
                        row,
                        read_policy=(
                            _MessageReadPolicy(
                                claimed_message_ids=frozenset(
                                    {
                                        int(user_cursor.lastrowid),
                                        int(assistant_cursor.lastrowid),
                                    }
                                ),
                                reports_by_message_id={
                                    int(user_cursor.lastrowid): typed_loop_report,
                                    int(assistant_cursor.lastrowid): typed_loop_report,
                                },
                                contents_by_message_id={
                                    int(user_cursor.lastrowid): user_text,
                                    int(assistant_cursor.lastrowid): assistant_text,
                                },
                            )
                            if typed_loop_report is not None
                            else None
                        ),
                        bound_report=(
                            typed_loop_report
                            if typed_loop_report is not None
                            else None
                        ),
                        canonical_content=(
                            user_text
                            if row["role"] == "user"
                            else assistant_text
                        ),
                    )
                    for row in rows
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            return persisted_turn

    def clear_thread(
        self,
        thread_id: str,
        *,
        expected_instance_id: Optional[str] = None,
        expected_generation: Optional[int] = None,
    ) -> Optional[ThreadRecord]:
        if (expected_instance_id is None) != (expected_generation is None):
            raise ValueError(
                "expected_instance_id and expected_generation must be supplied "
                "together."
            )
        if expected_instance_id is not None:
            if type(expected_instance_id) is not str or not expected_instance_id:
                raise ValueError("expected_instance_id must be a nonempty string.")
            if type(expected_generation) is not int or expected_generation < 0:
                raise ValueError(
                    "expected_generation must be a non-negative integer."
                )
        now = utc_now()
        with self._lock:
            try:
                if expected_instance_id is None:
                    observed_row = self._conn.execute(
                        """
                        SELECT instance_id, generation
                        FROM threads
                        WHERE id = ?
                        """,
                        (thread_id,),
                    ).fetchone()
                    if observed_row is None:
                        return None
                    observed_instance_id = str(observed_row["instance_id"])
                    observed_generation = int(observed_row["generation"])
                else:
                    observed_instance_id = expected_instance_id
                    observed_generation = expected_generation

                # The caller-supplied identity, or the first read for legacy
                # callers, names the exact thread instance to clear. Reserve
                # SQLite's write lock, then compare before any destructive write.
                # A clear that waited behind a concurrent delete/recreate must not
                # silently retarget the ABA replacement.
                self._conn.execute("BEGIN IMMEDIATE")
                current_row = self._conn.execute(
                    """
                    SELECT instance_id, generation
                    FROM threads
                    WHERE id = ?
                    """,
                    (thread_id,),
                ).fetchone()
                if (
                    current_row is None
                    or str(current_row["instance_id"]) != observed_instance_id
                    or int(current_row["generation"]) != observed_generation
                ):
                    self._conn.rollback()
                    return None
                self._conn.execute(
                    """
                    DELETE FROM loop_runs
                    WHERE thread_id = ?
                      AND EXISTS (
                          SELECT 1
                          FROM threads
                          WHERE id = ? AND instance_id = ?
                      )
                    """,
                    (thread_id, thread_id, observed_instance_id),
                )
                self._conn.execute(
                    """
                    DELETE FROM messages
                    WHERE thread_id = ?
                      AND EXISTS (
                          SELECT 1
                          FROM threads
                          WHERE id = ? AND instance_id = ?
                      )
                    """,
                    (thread_id, thread_id, observed_instance_id),
                )
                cursor = self._conn.execute(
                    """
                    UPDATE threads
                    SET updated_at = ?, generation = generation + 1
                    WHERE id = ? AND instance_id = ? AND generation = ?
                    """,
                    (
                        now,
                        thread_id,
                        observed_instance_id,
                        observed_generation,
                    ),
                )
                if cursor.rowcount != 1:
                    self._conn.rollback()
                    return None
                cleared = self.get_thread(thread_id)
                self._conn.commit()
                return cleared
            except Exception:
                self._conn.rollback()
                raise

    def delete_thread(
        self,
        thread_id: str,
        *,
        expected_instance_id: Optional[str] = None,
        expected_generation: Optional[int] = None,
    ) -> bool:
        if (expected_instance_id is None) != (expected_generation is None):
            raise ValueError(
                "expected_instance_id and expected_generation must be supplied "
                "together."
            )
        if expected_instance_id is not None:
            if type(expected_instance_id) is not str or not expected_instance_id:
                raise ValueError("expected_instance_id must be a nonempty string.")
            if type(expected_generation) is not int or expected_generation < 0:
                raise ValueError(
                    "expected_generation must be a non-negative integer."
                )
        with self._lock:
            if expected_instance_id is None:
                cursor = self._conn.execute(
                    "DELETE FROM threads WHERE id = ?",
                    (thread_id,),
                )
            else:
                cursor = self._conn.execute(
                    """
                    DELETE FROM threads
                    WHERE id = ? AND instance_id = ? AND generation = ?
                    """,
                    (
                        thread_id,
                        expected_instance_id,
                        expected_generation,
                    ),
                )
            self._conn.commit()
            return cursor.rowcount > 0

    def delete_empty_thread_if_current(
        self,
        thread_id: str,
        *,
        expected_instance_id: str,
        expected_generation: int,
        expected_updated_at: str,
        expected_title: str,
    ) -> bool:
        """Atomically remove only the unchanged empty thread named by the caller."""

        with self._lock:
            cursor = self._conn.execute(
                """
                DELETE FROM threads
                WHERE id = ?
                  AND instance_id = ?
                  AND generation = ?
                  AND updated_at = ?
                  AND title = ?
                  AND NOT EXISTS (
                      SELECT 1 FROM messages WHERE messages.thread_id = threads.id
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM loop_runs WHERE loop_runs.thread_id = threads.id
                  )
                """,
                (
                    thread_id,
                    expected_instance_id,
                    int(expected_generation),
                    expected_updated_at,
                    expected_title,
                ),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def _thread_from_row(
        self,
        row: sqlite3.Row,
        *,
        messages: tuple[ThreadMessage, ...],
        loop_runs: tuple[LoopRunRecord, ...] = (),
        message_count: Optional[int] = None,
        memory_count: Optional[int] = None,
    ) -> ThreadRecord:
        return ThreadRecord(
            id=str(row["id"]),
            title=str(row["title"]),
            created_at=canonical_stored_timestamp(row["created_at"]),
            updated_at=canonical_stored_timestamp(row["updated_at"]),
            instance_id=str(row["instance_id"]),
            generation=int(row["generation"] or 0),
            message_count=(
                int(row["message_count"] or len(messages))
                if message_count is None
                else int(message_count)
            ),
            memory_count=(
                int(row["memory_count"] or 0)
                if memory_count is None
                else int(memory_count)
            ),
            loop_run_count=int(row["loop_run_count"] or len(loop_runs)),
            messages=messages,
            loop_runs=loop_runs,
            latest=None,
        )

    def _message_from_row(
        self,
        row: sqlite3.Row,
        *,
        read_policy: Optional[_MessageReadPolicy] = None,
        bound_report: Optional[LoopReport] = None,
        canonical_content: Optional[str] = None,
    ) -> Optional[ThreadMessage]:
        role = str(row["role"])
        message_id = int(row["id"])
        claimed = (
            read_policy is not None
            and message_id in read_policy.claimed_message_ids
        )
        if claimed and (bound_report is None or canonical_content is None):
            return None
        stored_thinking = optional_json_loads(row["thinking_json"])
        return ThreadMessage(
            id=message_id,
            thread_id=str(row["thread_id"]),
            role=role,
            content=(
                canonical_content if claimed else str(row["content"])
            ),
            created_at=canonical_stored_timestamp(row["created_at"]),
            thinking=(
                canonical_persisted_thinking(stored_thinking, bound_report)
                if role == "assistant"
                else None
            ),
            loop_payload=None,
        )

    def _usable_message_ids(
        self,
        thread_id: str,
        *,
        policy: Optional[_MessageReadPolicy] = None,
    ) -> frozenset[int]:
        current_policy = policy or self._message_read_policy(thread_id)
        return frozenset(
            message_id
            for row in self._conn.execute(
                """
                SELECT id
                FROM messages
                WHERE thread_id = ?
                """,
                (thread_id,),
            ).fetchall()
            if (message_id := safe_stored_message_id(row["id"])) is not None
            and current_policy.permits(message_id)
        )

    def _usable_memory_count(
        self,
        thread_id: str,
        *,
        usable_message_ids: frozenset[int],
    ) -> int:
        return sum(
            1
            for row in self._conn.execute(
                """
                SELECT e.message_id
                FROM message_embeddings e
                JOIN messages m
                  ON m.id = e.message_id
                 AND m.thread_id = ?
                WHERE e.thread_id = ?
                """,
                (thread_id, thread_id),
            ).fetchall()
            if safe_stored_message_id(row["message_id"])
            in usable_message_ids
        )

    def _message_read_policy(
        self,
        thread_id: str,
    ) -> _MessageReadPolicy:
        """Reconstruct valid run pairs canonically and omit invalid claimed pairs."""

        run_rows = self._conn.execute(
            """
            WITH target_messages AS (
                SELECT id
                FROM messages
                WHERE thread_id = ?
            )
            SELECT
                lr.run_id, lr.thread_id, lr.user_message_id,
                lr.assistant_message_id, lr.schema_version,
                lr.raw_report_json, lr.public_report_json,
                lr.final_decision, lr.context_provider, lr.backend,
                lr.model_label, lr.recipe_id, lr.recipe_name,
                lr.step_count, lr.started_at, lr.completed_at,
                lr.created_at
            FROM loop_runs lr
            WHERE lr.thread_id = ?
               OR lr.user_message_id IN target_messages
               OR lr.assistant_message_id IN target_messages
               OR EXISTS (
                    SELECT 1
                    FROM loop_run_message_bindings binding
                    WHERE binding.run_id = lr.run_id
                      AND (
                           binding.thread_id = ?
                           OR binding.user_message_id IN target_messages
                           OR binding.assistant_message_id IN target_messages
                      )
               )
            """,
            (thread_id, thread_id, thread_id),
        ).fetchall()
        claimed_ids = {
            message_id
            for row in run_rows
            for field_name in ("user_message_id", "assistant_message_id")
            if (message_id := safe_stored_message_id(row[field_name])) is not None
        }
        binding_rows = self._conn.execute(
            """
            WITH target_messages AS (
                SELECT id
                FROM messages
                WHERE thread_id = ?
            )
            SELECT
                binding.run_id,
                binding.thread_id,
                binding.user_message_id,
                binding.assistant_message_id
            FROM loop_run_message_bindings binding
            LEFT JOIN loop_runs lr ON lr.run_id = binding.run_id
            WHERE binding.thread_id = ?
               OR binding.user_message_id IN target_messages
               OR binding.assistant_message_id IN target_messages
               OR lr.thread_id = ?
               OR lr.user_message_id IN target_messages
               OR lr.assistant_message_id IN target_messages
            """,
            (thread_id, thread_id, thread_id),
        ).fetchall()
        claimed_ids.update(
            message_id
            for binding_row in binding_rows
            for field_name in ("user_message_id", "assistant_message_id")
            if (message_id := safe_stored_message_id(binding_row[field_name]))
            is not None
        )

        valid_pairs: list[tuple[int, int, LoopReport, str, str]] = []
        claim_counts: dict[int, int] = {}
        for row in run_rows:
            record = self._loop_run_from_row(row)
            user_message_id = record.user_message_id
            assistant_message_id = record.assistant_message_id
            if (
                record.quarantine_reason is not None
                or user_message_id is None
                or assistant_message_id is None
            ):
                continue
            try:
                report = LoopReport.from_dict(record.raw_report)
                assistant_content = canonical_public_assistant_content(
                    record.public_report
                )
            except (KeyError, TypeError, ValueError):
                continue
            valid_pairs.append(
                (
                    user_message_id,
                    assistant_message_id,
                    report,
                    report.run.user_input,
                    assistant_content,
                )
            )
            claim_counts[user_message_id] = claim_counts.get(user_message_id, 0) + 1
            claim_counts[assistant_message_id] = (
                claim_counts.get(assistant_message_id, 0) + 1
            )

        reports_by_id: dict[int, LoopReport] = {}
        contents_by_id: dict[int, str] = {}
        for (
            user_message_id,
            assistant_message_id,
            report,
            user_content,
            assistant_content,
        ) in valid_pairs:
            if (
                claim_counts[user_message_id] != 1
                or claim_counts[assistant_message_id] != 1
            ):
                continue
            reports_by_id[user_message_id] = report
            reports_by_id[assistant_message_id] = report
            contents_by_id[user_message_id] = user_content
            contents_by_id[assistant_message_id] = assistant_content
        return _MessageReadPolicy(
            claimed_message_ids=frozenset(claimed_ids),
            reports_by_message_id=reports_by_id,
            contents_by_message_id=contents_by_id,
        )

    def _validate_loop_run_message_binding(
        self,
        row: sqlite3.Row,
        typed_report: LoopReport,
        public_report: Mapping,
    ) -> None:
        """Require stored message links to match the canonical persisted turn."""

        user_message_id = safe_stored_message_id(row["user_message_id"])
        assistant_message_id = safe_stored_message_id(row["assistant_message_id"])
        if user_message_id is None or assistant_message_id is None:
            raise ValueError("Stored loop run message identity is missing or malformed.")
        if user_message_id == assistant_message_id:
            raise ValueError("Stored loop run message identities must be distinct.")

        binding_row = self._conn.execute(
            """
            SELECT thread_id, user_message_id, assistant_message_id
            FROM loop_run_message_bindings
            WHERE run_id = ?
            """,
            (typed_report.run.run_id,),
        ).fetchone()
        if (
            binding_row is None
            or binding_row["thread_id"] != typed_report.run.session_id
            or safe_stored_message_id(binding_row["user_message_id"])
            != user_message_id
            or safe_stored_message_id(binding_row["assistant_message_id"])
            != assistant_message_id
        ):
            raise ValueError(
                "Stored loop run message identities do not match their durable binding."
            )

        binding_ownership_rows = self._conn.execute(
            """
            SELECT run_id, user_message_id, assistant_message_id
            FROM loop_run_message_bindings
            WHERE user_message_id IN (?, ?)
               OR assistant_message_id IN (?, ?)
            """,
            (
                user_message_id,
                assistant_message_id,
                user_message_id,
                assistant_message_id,
            ),
        ).fetchall()
        if (
            len(binding_ownership_rows) != 1
            or binding_ownership_rows[0]["run_id"] != typed_report.run.run_id
            or safe_stored_message_id(
                binding_ownership_rows[0]["user_message_id"]
            )
            != user_message_id
            or safe_stored_message_id(
                binding_ownership_rows[0]["assistant_message_id"]
            )
            != assistant_message_id
        ):
            raise ValueError(
                "Stored loop run messages are not owned by exactly one durable "
                "binding."
            )

        ownership_rows = self._conn.execute(
            """
            SELECT run_id, user_message_id, assistant_message_id
            FROM loop_runs
            WHERE user_message_id IN (?, ?)
               OR assistant_message_id IN (?, ?)
            """,
            (
                user_message_id,
                assistant_message_id,
                user_message_id,
                assistant_message_id,
            ),
        ).fetchall()
        if (
            len(ownership_rows) != 1
            or ownership_rows[0]["run_id"] != typed_report.run.run_id
            or safe_stored_message_id(ownership_rows[0]["user_message_id"])
            != user_message_id
            or safe_stored_message_id(ownership_rows[0]["assistant_message_id"])
            != assistant_message_id
        ):
            raise ValueError(
                "Stored loop run message identities are not owned by exactly one run."
            )

        message_rows = self._conn.execute(
            """
            SELECT id, thread_id, role, content
            FROM messages
            WHERE id IN (?, ?)
            """,
            (user_message_id, assistant_message_id),
        ).fetchall()
        if len(message_rows) != 2:
            raise ValueError("Stored loop run message link is missing.")
        messages_by_id = {
            message_row["id"]: message_row for message_row in message_rows
        }
        expected_bindings = (
            (
                user_message_id,
                "user",
                typed_report.run.user_input,
            ),
            (
                assistant_message_id,
                "assistant",
                canonical_public_assistant_content(public_report),
            ),
        )
        for message_id, expected_role, expected_content in expected_bindings:
            message_row = messages_by_id.get(message_id)
            if message_row is None:
                raise ValueError("Stored loop run message link is missing.")
            if message_row["thread_id"] != typed_report.run.session_id:
                raise ValueError("Stored loop run message thread does not match.")
            if message_row["role"] != expected_role:
                raise ValueError("Stored loop run message role does not match.")
            if message_row["content"] != expected_content:
                raise ValueError("Stored loop run message content does not match.")

    def _loop_run_from_row(self, row: sqlite3.Row) -> LoopRunRecord:
        run_id = str(row["run_id"])
        thread_id = str(row["thread_id"])
        created_at = canonical_stored_timestamp(row["created_at"])
        user_message_id = safe_stored_message_id(row["user_message_id"])
        assistant_message_id = safe_stored_message_id(row["assistant_message_id"])
        quarantine_reason = "stored_loop_report_invalid"
        try:
            if safe_record_id(run_id, field_name="run_id") != run_id:
                raise ValueError("Stored run identity is not canonical.")
            if safe_record_id(thread_id, field_name="thread_id") != thread_id:
                raise ValueError("Stored thread identity is not canonical.")
            raw_payload = required_json_loads(row["raw_report_json"])
            validate_raw_loop_report_payload(raw_payload)
            typed_report = LoopReport.from_dict(raw_payload)
            canonical_raw_report = typed_report.to_dict()
            if not _payload_is_canonical_subset(raw_payload, canonical_raw_report):
                raise ValueError("Stored loop report is not canonical.")
            if str(row["schema_version"]) != typed_report.schema_version:
                raise ValueError("Stored loop report schema identity mismatch.")
            if _is_unbound_legacy_visible_answer_report(typed_report):
                quarantine_reason = "legacy_visible_answer_unbound"
                raise ValueError(
                    "Legacy visible-answer loop report lacks required bindings."
                )
            public_report = project_public_report(
                typed_report,
                expected_run_id=run_id,
                expected_session_id=thread_id,
            )
            if (
                public_report.get("projection_schema_version")
                != PUBLIC_REPORT_PROJECTION_SCHEMA_VERSION
            ):
                raise ValueError("Unsupported public report projection schema.")
            try:
                self._validate_loop_run_message_binding(
                    row,
                    typed_report,
                    public_report,
                )
            except (KeyError, TypeError, ValueError) as exc:
                quarantine_reason = "stored_message_binding_invalid"
                raise ValueError(
                    "Stored loop run message binding is invalid."
                ) from exc
            public_run = public_report["run"]
            steps = public_run.get("steps")
            raw_metadata = typed_report.run.metadata
        except (
            AttributeError,
            KeyError,
            OverflowError,
            RecursionError,
            TypeError,
            ValueError,
        ):
            return LoopRunRecord(
                run_id=safe_quarantine_record_id(
                    run_id,
                    fallback="run_invalid",
                ),
                thread_id=safe_quarantine_record_id(
                    thread_id,
                    fallback="thread_invalid",
                ),
                user_message_id=user_message_id,
                assistant_message_id=assistant_message_id,
                schema_version=str(row["schema_version"]),
                raw_report={},
                public_report={},
                final_decision=None,
                context_provider=None,
                backend=None,
                model_label=None,
                recipe_id=None,
                recipe_name=None,
                step_count=0,
                started_at=None,
                completed_at=None,
                created_at=created_at,
                quarantine_reason=quarantine_reason,
            )
        return LoopRunRecord(
            run_id=run_id,
            thread_id=thread_id,
            user_message_id=user_message_id,
            assistant_message_id=assistant_message_id,
            schema_version=typed_report.schema_version,
            raw_report=canonical_raw_report,
            public_report=public_report,
            final_decision=public_run.get("final_decision"),
            context_provider=public_run.get("context_provider"),
            backend=public_run.get("backend"),
            model_label=public_run.get("model_label"),
            recipe_id=raw_metadata.get("recipe_id"),
            recipe_name=raw_metadata.get("recipe_name"),
            step_count=len(steps) if isinstance(steps, list) else 0,
            started_at=public_run.get("started_at"),
            completed_at=public_run.get("completed_at"),
            created_at=created_at,
        )

    def _recipe_from_row(self, row: sqlite3.Row) -> LoopRecipe:
        return LoopRecipe(
            recipe_id=str(row["recipe_id"]),
            schema_version=str(row["schema_version"] or LOOP_RECIPE_SCHEMA_VERSION),
            name=str(row["name"]),
            description=str(row["description"] or ""),
            goal=str(row["goal"]),
            instructions=str(row["instructions"] or ""),
            success_criteria=json_list_loads(row["success_criteria_json"]),
            stop_condition=str(row["stop_condition"] or ""),
            context_provider=str(row["context_provider"] or "smart"),
            model_profile=str(row["model_profile"] or "quality"),
            verifier=str(row["verifier"] or "default"),
            metadata=optional_json_loads(row["metadata_json"]) or {},
            created_at=datetime.fromisoformat(
                canonical_stored_timestamp(row["created_at"]).replace(
                    "Z", "+00:00"
                )
            ),
            updated_at=datetime.fromisoformat(
                canonical_stored_timestamp(row["updated_at"]).replace(
                    "Z", "+00:00"
                )
            ),
        )

    def _insert_recipe(self, recipe: LoopRecipe) -> None:
        self._conn.execute(
            """
            INSERT INTO loop_recipes (
                recipe_id, schema_version, name, description, goal,
                instructions, success_criteria_json, stop_condition,
                context_provider, model_profile, verifier, metadata_json,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                recipe.recipe_id,
                recipe.schema_version,
                recipe.name,
                recipe.description,
                recipe.goal,
                recipe.instructions,
                json_list_dumps(recipe.success_criteria),
                recipe.stop_condition,
                recipe.context_provider,
                recipe.model_profile,
                recipe.verifier,
                json_dumps(dict(recipe.metadata)),
                recipe.created_at.astimezone(timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z"),
                recipe.updated_at.astimezone(timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z"),
            ),
        )


def validate_vector(vector: list[float] | tuple[float, ...]) -> list[float]:
    values = [float(value) for value in vector]
    if not values:
        raise ValueError("Embedding vector must not be empty.")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Embedding vector must contain only finite values.")
    return values


def vector_to_json(vector: list[float] | tuple[float, ...]) -> str:
    return json.dumps(
        validate_vector(vector),
        separators=(",", ":"),
        allow_nan=False,
    )


def vector_from_json(value: str) -> list[float]:
    data = _strict_json_loads(value)
    if not isinstance(data, list):
        raise ValueError("Stored embedding vector must be a list.")
    return validate_vector(data)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
