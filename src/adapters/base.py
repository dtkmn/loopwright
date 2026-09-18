import json
import re
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple

from src.loop_engine import (
    LOOP_SESSION_SCHEMA_VERSION,
    SCHEMA_VERSION,
    LoopReport,
    LoopSession,
)


_ADAPTER_IDENTITY_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}")
_MAX_SAFE_JSON_INTEGER = (1 << 53) - 1


class LoopReportAdapter(Protocol):
    adapter_name: str
    adapter_schema_version: str

    def export_report(
        self,
        report: LoopReport,
        *,
        public: bool = True,
        session_id: Optional[str] = None,
        source_jsonl_line: Optional[int] = None,
    ) -> Dict[str, Any]:
        ...

    def export_session(
        self,
        session: LoopSession,
        *,
        public: bool = True,
        source_jsonl_lines: Optional[Sequence[Optional[int]]] = None,
    ) -> Dict[str, Any]:
        ...


def require_run_id(run_id: Any) -> str:
    if (
        type(run_id) is not str
        or _ADAPTER_IDENTITY_PATTERN.fullmatch(run_id) is None
    ):
        raise ValueError("run_id must be a canonical identity")
    return run_id


def require_report_identities(report: LoopReport) -> None:
    """Validate adapter-visible identities before projection or raw detachment."""

    require_run_id(report.run.run_id)
    require_optional_session_id(
        report.run.session_id,
        field_name="report session_id",
    )
    seen = set()
    for step in report.run.steps:
        step_id = require_step_id(step.step_id)
        if step_id in seen:
            raise ValueError(
                f"LoopRun contains duplicate step_id {step_id!r}; "
                "adapter step identities would be ambiguous."
            )
        seen.add(step_id)


def require_step_id(step_id: Any) -> str:
    if (
        type(step_id) is not str
        or _ADAPTER_IDENTITY_PATTERN.fullmatch(step_id) is None
    ):
        raise ValueError("step_id must be a canonical identity")
    return step_id


def require_supported_report_schema(report: LoopReport) -> None:
    if type(report) is not LoopReport:
        raise ValueError("report must be an exact LoopReport record")
    if (
        type(report.schema_version) is not str
        or report.schema_version != SCHEMA_VERSION
    ):
        raise ValueError(f"Unsupported loop report schema: {report.schema_version!r}")


def canonical_raw_report_payload(report: LoopReport) -> Dict[str, Any]:
    """Detach and reconstruct the exact strict-JSON raw adapter input."""

    require_supported_report_schema(report)
    try:
        source_payload = report.to_dict()
        detached_payload = json.loads(
            json.dumps(
                source_payload,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        if detached_payload != source_payload:
            raise ValueError("loop report changed during strict JSON detachment")
        reconstructed = LoopReport.from_dict(detached_payload)
        if reconstructed.to_dict() != detached_payload:
            raise ValueError("loop report did not reconstruct canonically")
    except Exception as exc:
        raise ValueError(
            "raw loop report must contain canonical JSON values"
        ) from exc
    return detached_payload


def resolve_session_id(
    report_session_id: Any,
    fallback_session_id: Any,
) -> Optional[str]:
    report_session_id = require_optional_session_id(
        report_session_id,
        field_name="report session_id",
    )
    fallback_session_id = require_optional_session_id(
        fallback_session_id,
        field_name="fallback session_id",
    )
    if (
        report_session_id is not None
        and fallback_session_id is not None
        and report_session_id != fallback_session_id
    ):
        raise ValueError("fallback session_id conflicts with report session_id")
    return report_session_id or fallback_session_id


def require_optional_session_id(
    session_id: Any,
    *,
    field_name: str = "session_id",
) -> Optional[str]:
    if session_id is None:
        return None
    if (
        type(session_id) is not str
        or _ADAPTER_IDENTITY_PATTERN.fullmatch(session_id) is None
    ):
        raise ValueError(f"{field_name} must be a canonical identity or None")
    return session_id


def require_unique_run_ids(session: LoopSession) -> None:
    seen = set()
    for report in session.reports:
        require_supported_report_schema(report)
        run_id = require_run_id(report.run.run_id)
        if run_id in seen:
            raise ValueError(
                f"LoopSession contains duplicate run_id {run_id!r}; "
                "adapter identities would be ambiguous."
            )
        seen.add(run_id)


def require_supported_session_schema(session: LoopSession) -> None:
    if type(session) is not LoopSession:
        raise ValueError("session must be an exact LoopSession record")
    if (
        type(session.schema_version) is not str
        or session.schema_version != LOOP_SESSION_SCHEMA_VERSION
    ):
        raise ValueError(
            f"Unsupported loop session schema: {session.schema_version!r}"
        )
    if require_optional_session_id(
        session.session_id,
        field_name="session session_id",
    ) is None:
        raise ValueError("session session_id must be a canonical identity")


def require_unique_step_ids(run: Mapping[str, Any]) -> None:
    steps = run.get("steps", ())
    if not isinstance(steps, (list, tuple)):
        raise ValueError("loop steps must be an array")

    seen = set()
    for step in steps:
        if not isinstance(step, Mapping):
            raise ValueError("loop step must be an object")
        step_id = require_step_id(step.get("step_id"))
        if step_id in seen:
            raise ValueError(
                f"LoopRun contains duplicate step_id {step_id!r}; "
                "adapter step identities would be ambiguous."
            )
        seen.add(step_id)


def require_source_jsonl_line(source_jsonl_line: Optional[int]) -> Optional[int]:
    if source_jsonl_line is None:
        return None
    if (
        type(source_jsonl_line) is not int
        or source_jsonl_line < 1
        or source_jsonl_line > _MAX_SAFE_JSON_INTEGER
    ):
        raise ValueError(
            "source_jsonl_line must be a positive JSON-safe integer or None"
        )
    return source_jsonl_line


def source_jsonl_lines_for_session(
    report_count: int,
    source_jsonl_lines: Optional[Sequence[Optional[int]]],
) -> Tuple[Optional[int], ...]:
    if source_jsonl_lines is None:
        return (None,) * report_count

    lines = tuple(require_source_jsonl_line(line) for line in source_jsonl_lines)
    if len(lines) != report_count:
        raise ValueError(
            "source_jsonl_lines must contain exactly one entry per report"
        )

    known_lines = tuple(line for line in lines if line is not None)
    if len(set(known_lines)) != len(known_lines):
        raise ValueError("known source_jsonl_lines must be unique")
    return lines
