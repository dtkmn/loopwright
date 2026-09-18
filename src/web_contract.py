from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from dataclasses import replace
from typing import Mapping, Optional

try:
    from .ai_loop_engine import (
        MAX_DOCUMENT_CHUNKS,
        DocumentProcessingReport,
        DocumentQAStatus,
        QueryResult,
    )
    from .answer_loop import SELF_CHECK_REFUSAL_ANSWER
    from .loop_engine import PUBLIC_REDACTION_REASON, PUBLIC_REDACTION_TEXT
    from .public_projection import PUBLIC_REPORT_PROJECTION_SCHEMA_VERSION
except ImportError:
    from ai_loop_engine import (
        MAX_DOCUMENT_CHUNKS,
        DocumentProcessingReport,
        DocumentQAStatus,
        QueryResult,
    )
    from answer_loop import SELF_CHECK_REFUSAL_ANSWER
    from loop_engine import PUBLIC_REDACTION_REASON, PUBLIC_REDACTION_TEXT
    from public_projection import PUBLIC_REPORT_PROJECTION_SCHEMA_VERSION


APP_TITLE = "Loopwright"
TERMINAL_PUBLIC_REDACTION = PUBLIC_REDACTION_TEXT
MODEL_THINKING_REDACTION = "[redacted: terminal loop decision]"
MODEL_THINKING_LABEL = "Model Thinking (unverified)"
MODEL_THINKING_NOTE = (
    "Model-emitted thinking is useful for debugging the loop, but it is not "
    "verified evidence."
)
MODEL_THINKING_SHA256_METADATA_KEY = "model_thinking_sha256"
MODEL_THINKING_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
ORDINARY_REFUSAL_TERMINAL_REASONS = frozenset(
    {"verification_failed", "retry_budget_exhausted"}
)
TEXT_ENCODING_OPTIONS = {
    "Auto": "auto",
    "UTF-8 / Western": "utf-8-or-western",
    "UTF-8": "utf-8",
    "Western (Windows-1252)": "cp1252",
    "Latin-1 (ISO-8859-1)": "latin-1",
    "Central European (Windows-1250)": "cp1250",
    "Cyrillic (Windows-1251)": "cp1251",
    "Turkish (Windows-1254)": "cp1254",
    "Baltic (Windows-1257)": "cp1257",
}


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def normalize_text_encoding(value: str | None) -> Optional[str]:
    if value is None:
        return "auto"
    selected = value.strip()
    if selected in TEXT_ENCODING_OPTIONS:
        return TEXT_ENCODING_OPTIONS[selected]
    if selected in set(TEXT_ENCODING_OPTIONS.values()):
        return selected
    return None


def upload_status_message(uploaded_name: str, qa_status: DocumentQAStatus) -> str:
    report = qa_status.processing_report
    document_name = (
        report.attempted_document_name
        if report and report.attempted_document_name
        else uploaded_name
    )

    if report and not report.success:
        active_message = (
            f"Active file remains {report.active_document_name}."
            if report.active_document_name
            else "No active file is loaded."
        )
        return (
            f"File context {document_name} failed during {report.phase}. "
            f"{active_message} Error: {report.error_message}"
        )

    chunk_message = ""
    if report:
        chunk_message = f" Chunks: {report.chunk_count}"
        if report.truncated:
            chunk_message += f" (truncated at {report.max_chunk_limit})."
        else:
            chunk_message += "."

    if qa_status.mock_mode:
        return (
            f"File context {document_name} processed in mock mode. "
            f"Profile: {qa_status.profile_label}. "
            f"Max output: {qa_status.max_output_tokens} tokens. "
            f"Active model: {qa_status.active_model_label}. "
            f"{chunk_message} "
            "Answers will be demonstration responses until a real LLM backend "
            "is configured."
        )
    return (
        f"File context {document_name} indexed. "
        f"Profile: {qa_status.profile_label}. "
        f"Max output: {qa_status.max_output_tokens} tokens. "
        f"Backend: {qa_status.active_backend}. "
        f"Active model: {qa_status.active_model_label}. "
        f"{chunk_message} "
        "Inference will be validated on the first question."
    )


def runtime_status_dict(qa_status: DocumentQAStatus) -> dict:
    report = qa_status.processing_report
    return {
        "active_document": qa_status.document_name,
        "last_attempted_document": (
            report.attempted_document_name if report else None
        ),
        "backend": qa_status.active_backend,
        "model": qa_status.active_model_label,
        "profile": qa_status.profile_label,
        "max_output_tokens": qa_status.max_output_tokens,
        "app_device": qa_status.device,
        "embeddings_model": qa_status.embeddings_model,
        "embeddings_device": qa_status.embeddings_device,
        "ready_for_queries": qa_status.ready_for_queries,
        "readiness_scope": "retrieval_pipeline",
        "direct_query_available": True,
        "query_mode": "contextual" if qa_status.ready_for_queries else "direct",
        "context_optional": True,
        "inference_validated": False,
        "last_success": report.success if report else None,
        "phase": report.phase if report else None,
        "file_extension": report.file_extension if report else None,
        "chunk_count": report.chunk_count if report else 0,
        "truncated": report.truncated if report else False,
        "max_chunk_limit": report.max_chunk_limit if report else None,
        "text_encoding_mode": report.text_encoding_mode if report else None,
        "last_error": report.error_message if report else None,
    }


def status_with_unexpected_upload_error(
    qa_status: DocumentQAStatus,
    uploaded_name: str,
    selected_encoding: str,
    exc: Exception,
) -> DocumentQAStatus:
    previous_report = qa_status.processing_report
    max_chunk_limit = (
        previous_report.max_chunk_limit if previous_report else MAX_DOCUMENT_CHUNKS
    )
    failure_report = DocumentProcessingReport(
        attempted_document_name=uploaded_name,
        active_document_name=qa_status.document_name,
        success=False,
        phase="unexpected",
        file_extension=os.path.splitext(uploaded_name)[1].lower() or None,
        chunk_count=0,
        truncated=False,
        max_chunk_limit=max_chunk_limit,
        text_encoding_mode=selected_encoding or "auto",
        backend=qa_status.active_backend,
        model_label=qa_status.active_model_label,
        error_message=str(exc),
    )
    return replace(qa_status, processing_report=failure_report)


def public_trace_error(
    query_result: QueryResult,
    public_loop_report: Optional[dict],
) -> Optional[str]:
    if public_loop_report is None:
        return None
    return public_trace_error_from_report(public_loop_report)


def public_loop_report_dict(query_result: QueryResult) -> Optional[dict]:
    return (
        query_result.loop_report.to_public_dict()
        if query_result.loop_report
        else None
    )


def terminal_public_redaction_applied(
    public_loop_report: Optional[Mapping],
) -> bool:
    redaction = (public_loop_report or {}).get("public_redaction") or {}
    return bool(redaction.get("applied"))


def public_answer_text(
    query_result: QueryResult,
    public_loop_report: Optional[dict],
) -> Optional[str]:
    if public_loop_report is None:
        return None
    return public_answer_text_from_report(public_loop_report)


def _ordinary_refusal_applied(public_loop_report: Mapping) -> bool:
    run = public_loop_report.get("run")
    return bool(
        isinstance(run, Mapping)
        and run.get("final_decision") == "refuse"
        and run.get("terminal_reason") in ORDINARY_REFUSAL_TERMINAL_REASONS
    )


def _live_query_result_matches_public_report(
    query_result: QueryResult,
    public_loop_report: Optional[dict],
) -> bool:
    """Bind optional live-only trace fields to the projected run identity."""

    loop_report = query_result.loop_report
    if loop_report is None or public_loop_report is None:
        return False

    projected_run = public_loop_report.get("run")
    if not isinstance(projected_run, Mapping):
        return False
    raw_run = loop_report.run
    trace = query_result.trace
    if (
        projected_run.get("run_id") != raw_run.run_id
        or projected_run.get("session_id") != raw_run.session_id
        or query_result.answer != raw_run.final_answer
        or trace.question != raw_run.user_input
        or trace.backend != raw_run.backend
        or trace.model_label != raw_run.model_label
    ):
        return False

    projected_answer = projected_run.get("final_answer")
    if projected_answer is not None and projected_answer != query_result.answer:
        return False

    return [citation.citation_id for citation in trace.citations] == [
        reference.citation_id for reference in raw_run.evidence
    ]


def visible_terminal_redaction_applied(
    query_result: QueryResult,
    public_loop_report: Optional[dict],
) -> bool:
    return terminal_public_redaction_applied(public_loop_report)


def model_thinking_dict(
    model_thinking: Optional[str], *, terminal_redaction: bool = False
) -> dict:
    thinking = model_thinking.strip() if isinstance(model_thinking, str) else ""
    if terminal_redaction:
        return {
            "available": False,
            "redacted": True,
            "label": MODEL_THINKING_LABEL,
            "content": MODEL_THINKING_REDACTION,
            "note": MODEL_THINKING_NOTE,
        }
    return {
        "available": bool(thinking),
        "redacted": False,
        "label": MODEL_THINKING_LABEL,
        "content": thinking or None,
        "note": MODEL_THINKING_NOTE,
    }


def _bound_model_thinking(
    query_result: QueryResult,
    *,
    live_trace_matches: bool,
) -> Optional[str]:
    """Return thinking only when the raw run binds its exact live value."""

    loop_report = query_result.loop_report
    thinking = query_result.trace.model_thinking
    if (
        not live_trace_matches
        or loop_report is None
        or not isinstance(thinking, str)
        or not thinking.strip()
    ):
        return None
    expected_digest = loop_report.run.metadata.get(
        MODEL_THINKING_SHA256_METADATA_KEY
    )
    if (
        type(expected_digest) is not str
        or MODEL_THINKING_SHA256_PATTERN.fullmatch(expected_digest) is None
    ):
        return None
    actual_digest = hashlib.sha256(thinking.encode("utf-8")).hexdigest()
    return thinking if hmac.compare_digest(expected_digest, actual_digest) else None


def loop_phase_label(phase: Optional[str]) -> str:
    labels = {
        "input": "Input",
        "context_select": "Context",
        "retrieve": "Retrieve",
        "draft": "Draft",
        "format_check": "Format",
        "mechanical_check": "Check",
        "verify": "Verify",
        "retry": "Retry",
        "refuse": "Refuse",
        "final": "Final",
        "error": "Error",
    }
    return labels.get(
        str(phase or ""),
        str(phase or "step").replace("_", " ").title(),
    )


def loop_step_detail(step: dict) -> str:
    parts = []
    verification = step.get("verification") or {}
    if verification.get("outcome"):
        parts.append(f"verifier: {verification.get('outcome')}")

    if step.get("error_present"):
        parts.append("error present")
    if step.get("human_review_required"):
        parts.append("human review required")

    retry_count = step.get("retry_count") or 0
    if retry_count:
        parts.append(f"retry #{retry_count}")

    return "; ".join(parts) or "-"


def _public_check_outcome(step: dict) -> Optional[str]:
    if step.get("error_present") or step.get("decision") == "error":
        return "failed"
    if step.get("decision") == "retry":
        return "retry"
    return "passed"


def loop_summary_dict(query_result: Optional[QueryResult]) -> dict:
    if query_result is None:
        return {
            "context_provider": None,
            "attempted_context_provider": None,
            "evidence_fallback": False,
            "evidence_fallback_reason": None,
            "document": None,
            "backend": None,
            "model": None,
            "retrieved_chunk_count": 0,
            "conversation_context_count": 0,
            "semantic_memory_count": 0,
            "semantic_memory_status": None,
            "recipe_id": None,
            "recipe_name": None,
            "draft_attempt_count": 0,
            "format_check": None,
            "mechanical_check": None,
            "verifier": None,
            "retry_attempted": False,
            "refused": False,
            "final_decision": None,
            "terminal_reason": None,
            "last_error": None,
        }

    public_loop_report = public_loop_report_dict(query_result)
    run = (public_loop_report or {}).get("run") or {}
    steps = run.get("steps") or []
    format_steps = [
        step for step in steps if step.get("phase") == "format_check"
    ]
    mechanical_steps = [
        step for step in steps if step.get("phase") == "mechanical_check"
    ]
    verify_steps = [step for step in steps if step.get("phase") == "verify"]
    retry_attempted = any(step.get("phase") == "retry" for step in steps)
    final_decision = run.get("final_decision")

    verifier = None
    if verify_steps:
        verify_step = verify_steps[-1]
        verification = verify_step.get("verification") or {}
        verifier = {
            "decision": verify_step.get("decision"),
            "outcome": verification.get("outcome"),
            "reasons": [],
        }

    terminal_redaction = terminal_public_redaction_applied(public_loop_report)
    evidence = run.get("evidence") or []

    return {
        "context_provider": run.get("context_provider"),
        "attempted_context_provider": None,
        "evidence_fallback": False,
        "evidence_fallback_reason": None,
        "document": None,
        "backend": run.get("backend"),
        "model": run.get("model_label"),
        "retrieved_chunk_count": 0 if terminal_redaction else len(evidence),
        "conversation_context_count": run.get("conversation_context_count"),
        "semantic_memory_count": run.get("semantic_memory_count"),
        "semantic_memory_status": run.get("semantic_memory_status"),
        "recipe_id": None,
        "recipe_name": None,
        "draft_attempt_count": sum(
            1 for step in steps if step.get("phase") == "draft"
        ),
        "format_check": _public_check_outcome(format_steps[-1]) if format_steps else None,
        "mechanical_check": (
            _public_check_outcome(mechanical_steps[-1]) if mechanical_steps else None
        ),
        "verifier": verifier,
        "retry_attempted": retry_attempted,
        "refused": final_decision == "refuse"
        or any(step.get("phase") == "refuse" for step in steps),
        "final_decision": final_decision,
        "terminal_reason": run.get("terminal_reason"),
        "last_error": public_trace_error(query_result, public_loop_report),
    }


def loop_timeline_dict(query_result: Optional[QueryResult]) -> dict:
    if query_result is None:
        return {
            "rows": [],
            "final_decision": None,
            "terminal_reason": None,
            "last_error": None,
            "empty": True,
        }

    public_loop_report = public_loop_report_dict(query_result)
    run = (public_loop_report or {}).get("run") or {}
    steps = run.get("steps") or []
    rows = []

    if steps:
        for index, step in enumerate(steps, start=1):
            phase = step.get("phase")
            rows.append(
                {
                    "index": index,
                    "phase": loop_phase_label(phase),
                    "phase_key": phase,
                    "decision": step.get("decision"),
                    "step": loop_phase_label(phase),
                    "signals": loop_step_detail(step),
                }
            )
    else:
        rows.append(
            {
                "index": 1,
                "phase": "Error",
                "phase_key": "error",
                "decision": "error",
                "step": "Trace unavailable",
                "signals": "No public loop report",
            }
        )

    return {
        "rows": rows,
        "final_decision": run.get("final_decision"),
        "terminal_reason": run.get("terminal_reason"),
        "last_error": public_trace_error(query_result, public_loop_report),
        "empty": False,
    }


def answer_trace_dict(query_result: Optional[QueryResult]) -> dict:
    if query_result is None:
        return {
            "question": None,
            "answer": None,
            "document": None,
            "backend": None,
            "model": None,
            "retrieved_chunk_count": 0,
            "citations": [],
            "self_check": None,
            "model_thinking": model_thinking_dict(None),
            "loop_report": None,
            "error": None,
        }

    trace = query_result.trace
    public_loop_report = public_loop_report_dict(query_result)
    terminal_redaction = visible_terminal_redaction_applied(
        query_result,
        public_loop_report,
    )
    final_decision = ((public_loop_report or {}).get("run") or {}).get(
        "final_decision"
    )
    model_thinking_redaction = terminal_redaction or final_decision in {
        "block",
        "error",
        "refuse",
        "requires_review",
    }
    run = (public_loop_report or {}).get("run") or {}
    live_trace_matches = _live_query_result_matches_public_report(
        query_result,
        public_loop_report,
    )
    bound_model_thinking = _bound_model_thinking(
        query_result,
        live_trace_matches=live_trace_matches,
    )
    question = (
        TERMINAL_PUBLIC_REDACTION
        if terminal_redaction
        else (trace.question if live_trace_matches else None)
    )
    answer = public_answer_text(query_result, public_loop_report)
    evidence = [] if terminal_redaction else run.get("evidence") or []
    verify_steps = [
        step for step in run.get("steps") or [] if step.get("phase") == "verify"
    ]
    verification = (verify_steps[-1].get("verification") or {}) if verify_steps else {}
    return {
        "question": question,
        "answer": answer,
        "document": None,
        "backend": run.get("backend"),
        "model": run.get("model_label"),
        "retrieved_chunk_count": len(evidence),
        "citations": [
            {
                "id": reference.get("citation_id"),
                "evidence_id": reference.get("evidence_id"),
                "provider": reference.get("provider"),
                "source": reference.get("evidence_id"),
                "page": (reference.get("locator") or {}).get("page"),
                "chunk": (
                    (reference.get("locator") or {}).get("chunk_index") + 1
                    if (reference.get("locator") or {}).get("chunk_index") is not None
                    else None
                ),
                "excerpt": None,
            }
            for reference in evidence
        ],
        "self_check": (
            {
                "outcome": verification.get("outcome"),
                "reasons": [],
                "retry_attempted": any(
                    step.get("phase") == "retry" for step in run.get("steps") or []
                ),
            }
            if verification
            else None
        ),
        "model_thinking": model_thinking_dict(
            bound_model_thinking,
            terminal_redaction=model_thinking_redaction,
        ),
        "loop_report": public_loop_report,
        "error": public_trace_error(query_result, public_loop_report),
    }


def query_response_dict(query_result: QueryResult) -> dict:
    public_loop_report = public_loop_report_dict(query_result)
    answer = public_answer_text(query_result, public_loop_report)
    return {
        "answer": answer,
        "timeline": loop_timeline_dict(query_result),
        "summary": loop_summary_dict(query_result),
        "trace": answer_trace_dict(query_result),
    }


def public_loop_payload_from_report(public_report: Mapping) -> dict:
    """Rebuild restart UI state solely from a canonical public projection."""

    if not isinstance(public_report, Mapping):
        raise ValueError("public loop report must be an object")
    if (
        public_report.get("projection_schema_version")
        != PUBLIC_REPORT_PROJECTION_SCHEMA_VERSION
        or public_report.get("public") is not True
    ):
        raise ValueError("unsupported public loop report projection")
    run = public_report.get("run")
    if not isinstance(run, Mapping):
        raise ValueError("public loop report run must be an object")
    steps = run.get("steps")
    evidence = run.get("evidence")
    if not isinstance(steps, list) or not isinstance(evidence, list):
        raise ValueError("public loop report collections are malformed")
    if not all(isinstance(step, Mapping) for step in steps):
        raise ValueError("public loop report step must be an object")

    public_redaction = public_report.get("public_redaction")
    if not isinstance(public_redaction, Mapping):
        raise ValueError("public loop report redaction state is malformed")
    terminal_redaction = bool(public_redaction.get("applied"))
    answer = public_answer_text_from_report(public_report)
    verify_steps = [step for step in steps if step.get("phase") == "verify"]
    verification = (
        (verify_steps[-1].get("verification") or {}) if verify_steps else {}
    )
    format_steps = [step for step in steps if step.get("phase") == "format_check"]
    mechanical_steps = [
        step for step in steps if step.get("phase") == "mechanical_check"
    ]
    citations = _public_evidence_citations(evidence)
    public_error = public_trace_error_from_report(public_report)
    timeline = {
        "rows": [
            {
                "index": index,
                "phase": loop_phase_label(step.get("phase")),
                "phase_key": step.get("phase"),
                "decision": step.get("decision"),
                "step": loop_phase_label(step.get("phase")),
                "signals": loop_step_detail(dict(step)),
            }
            for index, step in enumerate(steps, start=1)
        ],
        "final_decision": run.get("final_decision"),
        "terminal_reason": run.get("terminal_reason"),
        "last_error": public_error,
        "empty": False,
    }
    summary = {
        "context_provider": run.get("context_provider"),
        "attempted_context_provider": None,
        "evidence_fallback": False,
        "evidence_fallback_reason": None,
        "document": None,
        "backend": run.get("backend"),
        "model": run.get("model_label"),
        "retrieved_chunk_count": len(citations),
        "conversation_context_count": run.get("conversation_context_count"),
        "semantic_memory_count": run.get("semantic_memory_count"),
        "semantic_memory_status": run.get("semantic_memory_status"),
        "recipe_id": None,
        "recipe_name": None,
        "draft_attempt_count": sum(
            1 for step in steps if step.get("phase") == "draft"
        ),
        "format_check": (
            _public_check_outcome(format_steps[-1]) if format_steps else None
        ),
        "mechanical_check": (
            _public_check_outcome(mechanical_steps[-1])
            if mechanical_steps
            else None
        ),
        "verifier": (
            {
                "decision": verify_steps[-1].get("decision"),
                "outcome": verification.get("outcome"),
                "reasons": [],
            }
            if verification
            else None
        ),
        "retry_attempted": any(step.get("phase") == "retry" for step in steps),
        "refused": run.get("final_decision") == "refuse",
        "final_decision": run.get("final_decision"),
        "terminal_reason": run.get("terminal_reason"),
        "last_error": public_error,
    }
    trace = {
        "source": "durable_public_report",
        "public": True,
        "question": None,
        "answer": answer,
        "document": None,
        "backend": run.get("backend"),
        "model": run.get("model_label"),
        "retrieved_chunk_count": len(citations),
        "citations": citations,
        "self_check": (
            {
                "outcome": verification.get("outcome"),
                "reasons": [],
                "retry_attempted": summary["retry_attempted"],
            }
            if verification
            else None
        ),
        "model_thinking": model_thinking_dict(
            None,
            terminal_redaction=terminal_redaction,
        ),
        "loop_report": dict(public_report),
        "error": public_error,
    }
    return {
        "answer": answer,
        "timeline": timeline,
        "summary": summary,
        "trace": trace,
    }


def public_answer_text_from_report(public_loop_report: Mapping) -> Optional[str]:
    if terminal_public_redaction_applied(public_loop_report):
        if _ordinary_refusal_applied(public_loop_report):
            return SELF_CHECK_REFUSAL_ANSWER
        return TERMINAL_PUBLIC_REDACTION
    run = public_loop_report.get("run")
    if not isinstance(run, Mapping):
        return None
    projected_answer = run.get("final_answer")
    return projected_answer if isinstance(projected_answer, str) else None


def public_trace_error_from_report(public_loop_report: Mapping) -> Optional[str]:
    if terminal_public_redaction_applied(public_loop_report):
        if _ordinary_refusal_applied(public_loop_report):
            return None
        return PUBLIC_REDACTION_REASON
    run = public_loop_report.get("run")
    return (
        "loop_error"
        if isinstance(run, Mapping) and run.get("error_present")
        else None
    )


def _public_evidence_citations(evidence: list) -> list[dict]:
    citations = []
    for reference in evidence:
        if not isinstance(reference, Mapping):
            raise ValueError("public evidence reference must be an object")
        locator = reference.get("locator")
        if not isinstance(locator, Mapping):
            raise ValueError("public evidence locator must be an object")
        chunk_index = locator.get("chunk_index")
        citations.append(
            {
                "id": reference.get("citation_id"),
                "evidence_id": reference.get("evidence_id"),
                "provider": reference.get("provider"),
                "source": reference.get("evidence_id"),
                "page": locator.get("page"),
                "chunk": chunk_index + 1 if chunk_index is not None else None,
                "excerpt": None,
            }
        )
    return citations


def empty_query_response_dict() -> dict:
    return {
        "answer": None,
        "timeline": loop_timeline_dict(None),
        "summary": loop_summary_dict(None),
        "trace": answer_trace_dict(None),
    }


def pretty_json(data: dict) -> str:
    return json.dumps(data, indent=2)
