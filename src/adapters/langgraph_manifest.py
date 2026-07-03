from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from src.adapters.redaction import (
    report_payload,
    require_public_bool,
    run_with_session_fallback,
)
from src.loop_engine import LoopReport, LoopSession


ADAPTER_NAME = "langgraph_manifest"
ADAPTER_SCHEMA_VERSION = "langgraph-manifest-export/v1"
CHECKPOINT_NAMESPACE = "loopwright"


class LangGraphManifestAdapter:
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
        payload = report_payload(report, public=public)
        run = run_with_session_fallback(payload["run"], session_id)
        manifest = _manifest_from_run(
            run,
            public=public,
            source_jsonl_line=source_jsonl_line,
        )
        return {
            "adapter_name": self.adapter_name,
            "adapter_schema_version": self.adapter_schema_version,
            "source_schema_version": payload["schema_version"],
            "public": public,
            "manifest": manifest,
            "source_report": payload,
        }

    def export_session(
        self, session: LoopSession, *, public: bool = True
    ) -> Dict[str, Any]:
        require_public_bool(public)
        manifests = []
        for index, report in enumerate(session.reports, start=1):
            payload = report_payload(report, public=public)
            run = run_with_session_fallback(payload["run"], session.session_id)
            manifests.append(
                _manifest_from_run(
                    run,
                    public=public,
                    source_jsonl_line=index,
                )
            )
        return {
            "adapter_name": self.adapter_name,
            "adapter_schema_version": self.adapter_schema_version,
            "source_schema_version": session.schema_version,
            "public": public,
            "session_id": session.session_id,
            "thread_id": session.session_id,
            "manifest_count": len(manifests),
            "manifests": manifests,
        }


def export_report(
    report: LoopReport,
    *,
    public: bool = True,
    session_id: Optional[str] = None,
    source_jsonl_line: Optional[int] = None,
) -> Dict[str, Any]:
    return LangGraphManifestAdapter().export_report(
        report,
        public=public,
        session_id=session_id,
        source_jsonl_line=source_jsonl_line,
    )


def export_session(session: LoopSession, *, public: bool = True) -> Dict[str, Any]:
    return LangGraphManifestAdapter().export_session(session, public=public)


def _manifest_from_run(
    run: Mapping[str, Any],
    *,
    public: bool,
    source_jsonl_line: Optional[int],
) -> Dict[str, Any]:
    thread_id = run.get("session_id") or "default"
    checkpoints = [
        _checkpoint_from_step(
            step,
            run=run,
            thread_id=thread_id,
            step_index=index,
            source_jsonl_line=source_jsonl_line,
        )
        for index, step in enumerate(run.get("steps", ()))
    ]
    return {
        "thread_id": thread_id,
        "run_id": run["run_id"],
        "source_jsonl_line": source_jsonl_line,
        "context_provider": run["context_provider"],
        "backend": run["backend"],
        "model_label": run["model_label"],
        "started_at": run.get("started_at"),
        "completed_at": run.get("completed_at"),
        "final_decision": run.get("final_decision"),
        "phase_order": [checkpoint["phase"] for checkpoint in checkpoints],
        "nodes": [
            {
                "node_id": checkpoint["node"],
                "phase": checkpoint["phase"],
                "decision": checkpoint["decision"],
            }
            for checkpoint in checkpoints
        ],
        "checkpoints": checkpoints,
        "terminal_state": {
            "user_input": run.get("user_input"),
            "final_answer": run.get("final_answer"),
            "error_message": run.get("error_message"),
            "final_decision": run.get("final_decision"),
        },
        "metadata": {
            "public": public,
            "policy": run.get("policy"),
            "run_metadata": run.get("metadata") or {},
        },
    }


def _checkpoint_from_step(
    step: Mapping[str, Any],
    *,
    run: Mapping[str, Any],
    thread_id: str,
    step_index: int,
    source_jsonl_line: Optional[int],
) -> Dict[str, Any]:
    phase = step["phase"]
    return {
        "checkpoint_id": f"checkpoint_{run['run_id']}_{step_index}",
        "checkpoint_ns": CHECKPOINT_NAMESPACE,
        "thread_id": thread_id,
        "run_id": run["run_id"],
        "source_jsonl_line": source_jsonl_line,
        "step_index": step_index,
        "step_id": step["step_id"],
        "node": phase,
        "phase": phase,
        "decision": step.get("decision"),
        "started_at": step.get("started_at"),
        "ended_at": step.get("ended_at"),
        "duration_ms": step.get("duration_ms"),
        "state": {
            "input_summary": step.get("input_summary"),
            "output_summary": step.get("output_summary"),
            "error_message": step.get("error_message"),
            "retry_count": step.get("retry_count", 0),
            "verification": step.get("verification"),
            "human_review": step.get("human_review"),
        },
        "metadata": step.get("metadata") or {},
    }
