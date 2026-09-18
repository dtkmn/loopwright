from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Optional, Sequence

from src.adapters.base import (
    require_run_id,
    require_report_identities,
    require_source_jsonl_line,
    require_supported_report_schema,
    require_supported_session_schema,
    require_unique_run_ids,
    require_unique_step_ids,
    resolve_session_id,
    source_jsonl_lines_for_session,
)
from src.adapters.redaction import (
    report_payload,
    require_public_bool,
    run_with_session_fallback,
)
from src.loop_engine import LoopReport, LoopSession
from src.public_projection import PUBLIC_REPORT_PROJECTION_SCHEMA_VERSION


ADAPTER_NAME = "openai_trace"
ADAPTER_SCHEMA_VERSION = "openai-trace-export/v2"
WORKFLOW_NAME = "Loopwright"


class OpenAITraceAdapter:
    adapter_name = ADAPTER_NAME
    adapter_schema_version = ADAPTER_SCHEMA_VERSION

    def export_report(
        self,
        report: LoopReport,
        *,
        public: bool = True,
        session_id: Optional[str] = None,
        source_jsonl_line: Optional[int] = None,
    ) -> Dict[str, Any]:
        require_supported_report_schema(report)
        require_report_identities(report)
        source_jsonl_line = require_source_jsonl_line(source_jsonl_line)
        payload = report_payload(report, public=public)
        resolved_session_id = resolve_session_id(
            payload["run"].get("session_id"),
            session_id,
        )
        run = run_with_session_fallback(payload["run"], resolved_session_id)
        run_id = require_run_id(run["run_id"])
        require_unique_step_ids(run)
        trace_id = _trace_id(run_id, session_id=run.get("session_id"))
        return {
            "adapter_name": self.adapter_name,
            "adapter_schema_version": self.adapter_schema_version,
            "source_schema_version": payload["schema_version"],
            "source_projection_schema_version": payload.get(
                "projection_schema_version"
            ),
            "public": public,
            "trace": _trace_from_run(
                run,
                trace_id=trace_id,
                public=public,
                source_jsonl_line=source_jsonl_line,
            ),
            "source_report": payload,
        }

    def export_session(
        self,
        session: LoopSession,
        *,
        public: bool = True,
        source_jsonl_lines: Optional[Sequence[Optional[int]]] = None,
    ) -> Dict[str, Any]:
        require_public_bool(public)
        require_supported_session_schema(session)
        require_unique_run_ids(session)
        source_lines = source_jsonl_lines_for_session(
            session.report_count,
            source_jsonl_lines,
        )
        report_exports = [
            self.export_report(
                report,
                public=public,
                session_id=session.session_id,
                source_jsonl_line=source_line,
            )
            for report, source_line in zip(session.reports, source_lines)
        ]
        return {
            "adapter_name": self.adapter_name,
            "adapter_schema_version": self.adapter_schema_version,
            "source_schema_version": session.schema_version,
            "source_projection_schema_version": (
                PUBLIC_REPORT_PROJECTION_SCHEMA_VERSION if public else None
            ),
            "public": public,
            "session_id": session.session_id,
            "trace_count": len(report_exports),
            "traces": [report_export["trace"] for report_export in report_exports],
        }


def export_report(
    report: LoopReport,
    *,
    public: bool = True,
    session_id: Optional[str] = None,
    source_jsonl_line: Optional[int] = None,
) -> Dict[str, Any]:
    return OpenAITraceAdapter().export_report(
        report,
        public=public,
        session_id=session_id,
        source_jsonl_line=source_jsonl_line,
    )


def export_session(
    session: LoopSession,
    *,
    public: bool = True,
    source_jsonl_lines: Optional[Sequence[Optional[int]]] = None,
) -> Dict[str, Any]:
    return OpenAITraceAdapter().export_session(
        session,
        public=public,
        source_jsonl_lines=source_jsonl_lines,
    )


def _trace_from_run(
    run: Mapping[str, Any],
    *,
    trace_id: str,
    public: bool,
    source_jsonl_line: Optional[int],
) -> Dict[str, Any]:
    metadata = {
        "loopwright_run_id": run["run_id"],
        "loopwright_schema": "loop-report/v1",
        "context_provider": run["context_provider"],
        "backend": run["backend"],
        "model_label": run["model_label"],
        "final_decision": run["final_decision"],
        "terminal_reason": run.get("terminal_reason"),
        "public": public,
    }
    if run.get("metadata"):
        metadata["loop_metadata"] = run["metadata"]

    return {
        "id": trace_id,
        "trace_id": trace_id,
        "workflow_name": WORKFLOW_NAME,
        "group_id": run.get("session_id"),
        "source_jsonl_line": source_jsonl_line,
        "started_at": run.get("started_at"),
        "ended_at": run.get("completed_at"),
        "final_answer": run.get("final_answer"),
        "evidence": list(run.get("evidence") or ()),
        "metadata": metadata,
        "spans": [
            _span_from_step(step, trace_id=trace_id) for step in run.get("steps", ())
        ],
    }


def _span_from_step(
    step: Mapping[str, Any],
    *,
    trace_id: str,
) -> Dict[str, Any]:
    phase = step["phase"]
    step_id = step["step_id"]
    span_id = _span_id(trace_id, step_id)
    custom_data = {
        "loopwright_step_id": step_id,
        "phase": phase,
        "decision": step.get("decision"),
        "input": step.get("input_summary"),
        "output": step.get("output_summary"),
        "backend": step.get("backend"),
        "model": step.get("model_label"),
        "retry_count": step.get("retry_count", 0),
        "error": step.get("error_message"),
        "error_present": bool(
            step.get("error_present") or step.get("error_message")
        ),
        "human_review_required": bool(
            step.get("human_review_required") or step.get("human_review")
        ),
        "metadata": step.get("metadata") or {},
    }
    if step.get("verification"):
        custom_data["verification"] = step["verification"]
    if step.get("human_review"):
        custom_data["human_review"] = step["human_review"]

    span_data = {
        "type": "custom",
        "name": step.get("name") or phase,
        "data": custom_data,
    }

    return {
        "id": span_id,
        "span_id": span_id,
        "trace_id": trace_id,
        "parent_id": None,
        "started_at": step.get("started_at"),
        "ended_at": step.get("ended_at"),
        "duration_ms": step.get("duration_ms"),
        "status": _status_for_step(step),
        "span_data": span_data,
    }


def _status_for_step(step: Mapping[str, Any]) -> str:
    decision = step.get("decision")
    if (
        step.get("error_present")
        or step.get("error_message")
        or decision in {"block", "error"}
    ):
        return "error"
    if decision in {"refuse", "requires_review", "retry"}:
        return decision
    return "ok"


def _trace_id(run_id: str, *, session_id: Optional[str]) -> str:
    return _hashed_id("trace", session_id or "", run_id, hex_length=32)


def _span_id(trace_id: str, step_id: str) -> str:
    return _hashed_id("span", trace_id, step_id, hex_length=24)


def _hashed_id(prefix: str, *identity_parts: str, hex_length: int) -> str:
    identity = json.dumps(
        identity_parts,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()[:hex_length]
    return f"{prefix}_{digest}"
