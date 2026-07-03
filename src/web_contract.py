from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Optional

try:
    from .ai_loop_engine import (
        MAX_DOCUMENT_CHUNKS,
        DocumentProcessingReport,
        DocumentQAStatus,
        QueryResult,
    )
    from .loop_engine import PUBLIC_REDACTION_REASON, PUBLIC_REDACTION_TEXT
except ImportError:
    from ai_loop_engine import (
        MAX_DOCUMENT_CHUNKS,
        DocumentProcessingReport,
        DocumentQAStatus,
        QueryResult,
    )
    from loop_engine import PUBLIC_REDACTION_REASON, PUBLIC_REDACTION_TEXT


APP_TITLE = "Loopwright"
TERMINAL_PUBLIC_REDACTION = PUBLIC_REDACTION_TEXT
MODEL_THINKING_REDACTION = "[redacted: terminal loop decision]"
MODEL_THINKING_LABEL = "Model Thinking (unverified)"
MODEL_THINKING_NOTE = (
    "Model-emitted thinking is useful for debugging the loop, but it is not "
    "verified evidence."
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
    if visible_terminal_redaction_applied(query_result, public_loop_report):
        return PUBLIC_REDACTION_REASON
    return query_result.trace.error_message


def public_loop_report_dict(query_result: QueryResult) -> Optional[dict]:
    return (
        query_result.loop_report.to_public_dict()
        if query_result.loop_report
        else None
    )


def terminal_public_redaction_applied(public_loop_report: Optional[dict]) -> bool:
    redaction = (public_loop_report or {}).get("public_redaction") or {}
    return bool(redaction.get("applied"))


def visible_terminal_redaction_applied(
    query_result: QueryResult,
    public_loop_report: Optional[dict],
) -> bool:
    if not terminal_public_redaction_applied(public_loop_report):
        return False
    loop_report = query_result.loop_report
    if loop_report is None:
        return True
    run = loop_report.run
    final_decision = getattr(run.final_decision, "value", run.final_decision)
    if final_decision in {"block", "requires_review"}:
        return True
    if final_decision != "refuse":
        return True
    if run.metadata.get("after_run_guardrail") or run.metadata.get("guardrail_decision"):
        return True
    for step in run.steps:
        if step.human_review is not None:
            return True
        if step.metadata.get("guardrail_decision"):
            return True
    return False


def model_thinking_dict(
    model_thinking: Optional[str], *, terminal_redaction: bool = False
) -> dict:
    thinking = model_thinking.strip() if isinstance(model_thinking, str) else ""
    if terminal_redaction and thinking:
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
    output_summary = step.get("output_summary")
    if output_summary:
        parts.append(str(output_summary))
    error_message = step.get("error_message")
    if error_message and error_message != output_summary:
        parts.append(f"error: {error_message}")

    metadata = step.get("metadata") or {}
    reasons = metadata.get("reasons") or []
    if reasons:
        parts.append(f"reasons: {', '.join(str(reason) for reason in reasons)}")
    if metadata.get("retrieved_chunk_count") is not None:
        parts.append(f"chunks: {metadata.get('retrieved_chunk_count')}")
    if metadata.get("semantic_memory_count") is not None:
        parts.append(f"memory: {metadata.get('semantic_memory_count')}")
    if metadata.get("semantic_memory_status"):
        parts.append(f"memory status: {metadata.get('semantic_memory_status')}")
    citation_ids = metadata.get("citation_ids") or []
    if citation_ids:
        parts.append(f"citations: {', '.join(str(value) for value in citation_ids)}")
    inline_ids = metadata.get("inline_citation_ids") or []
    if inline_ids:
        parts.append(
            f"inline citations: {', '.join(str(value) for value in inline_ids)}"
        )

    verification = step.get("verification") or {}
    if verification.get("outcome"):
        parts.append(f"verifier: {verification.get('outcome')}")
    verification_reasons = verification.get("reasons") or []
    if verification_reasons and not reasons:
        parts.append(
            "reasons: "
            + ", ".join(str(reason) for reason in verification_reasons)
        )

    retry_count = step.get("retry_count") or 0
    if retry_count:
        parts.append(f"retry #{retry_count}")

    return "; ".join(parts) or "-"


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
            "last_error": None,
        }

    public_loop_report = public_loop_report_dict(query_result)
    run = (public_loop_report or {}).get("run") or {}
    steps = run.get("steps") or []
    trace = query_result.trace
    format_steps = [
        step for step in steps if step.get("phase") == "format_check"
    ]
    mechanical_steps = [
        step for step in steps if step.get("phase") == "mechanical_check"
    ]
    verify_steps = [step for step in steps if step.get("phase") == "verify"]
    retry_attempted = any(step.get("phase") == "retry" for step in steps)
    if trace.self_check:
        retry_attempted = retry_attempted or trace.self_check.retry_attempted
    final_decision = run.get("final_decision")

    verifier = None
    if verify_steps:
        verify_step = verify_steps[-1]
        verification = verify_step.get("verification") or {}
        verifier = {
            "decision": verify_step.get("decision"),
            "outcome": verification.get("outcome")
            or verify_step.get("output_summary"),
            "reasons": verification.get("reasons")
            or verify_step.get("metadata", {}).get("reasons", []),
        }

    return {
        "context_provider": run.get("context_provider"),
        "attempted_context_provider": run.get("metadata", {}).get(
            "attempted_context_provider"
        ),
        "evidence_fallback": bool(
            run.get("metadata", {}).get("evidence_fallback")
        ),
        "evidence_fallback_reason": run.get("metadata", {}).get(
            "evidence_fallback_reason"
        ),
        "document": trace.document_name,
        "backend": trace.backend,
        "model": trace.model_label,
        "retrieved_chunk_count": trace.retrieved_chunk_count,
        "conversation_context_count": run.get("metadata", {}).get(
            "conversation_context_turns",
            0,
        ),
        "semantic_memory_count": run.get("metadata", {}).get(
            "semantic_memory_turns",
            0,
        ),
        "semantic_memory_status": run.get("metadata", {}).get(
            "semantic_memory_status"
        ),
        "recipe_id": run.get("metadata", {}).get("recipe_id"),
        "recipe_name": run.get("metadata", {}).get("recipe_name"),
        "draft_attempt_count": sum(
            1 for step in steps if step.get("phase") == "draft"
        ),
        "format_check": (
            format_steps[-1].get("output_summary") if format_steps else None
        ),
        "mechanical_check": (
            mechanical_steps[-1].get("output_summary") if mechanical_steps else None
        ),
        "verifier": verifier,
        "retry_attempted": retry_attempted,
        "refused": final_decision == "refuse"
        or any(step.get("phase") == "refuse" for step in steps),
        "final_decision": final_decision,
        "last_error": public_trace_error(query_result, public_loop_report),
    }


def loop_timeline_dict(query_result: Optional[QueryResult]) -> dict:
    if query_result is None:
        return {
            "rows": [],
            "final_decision": None,
            "last_error": None,
            "empty": True,
        }

    public_loop_report = public_loop_report_dict(query_result)
    run = (public_loop_report or {}).get("run") or {}
    steps = run.get("steps") or []
    trace = query_result.trace
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
                    "step": step.get("name") or loop_phase_label(phase),
                    "signals": loop_step_detail(step),
                }
            )
    else:
        fallback_row = 1
        rows.append(
            {
                "index": fallback_row,
                "phase": "Context",
                "phase_key": "context_select",
                "decision": "continue",
                "step": trace.document_name or "No active context",
                "signals": trace.backend,
            }
        )
        fallback_row += 1
        rows.append(
            {
                "index": fallback_row,
                "phase": "Retrieve",
                "phase_key": "retrieve",
                "decision": "continue",
                "step": "Prompt evidence",
                "signals": f"{trace.retrieved_chunk_count} chunks",
            }
        )
        fallback_row += 1
        if trace.self_check:
            rows.append(
                {
                    "index": fallback_row,
                    "phase": "Check",
                    "phase_key": "mechanical_check",
                    "decision": trace.self_check.outcome,
                    "step": "Self-check",
                    "signals": ", ".join(trace.self_check.reasons),
                }
            )
            fallback_row += 1
        if trace.error_message:
            rows.append(
                {
                    "index": fallback_row,
                    "phase": "Error",
                    "phase_key": "error",
                    "decision": "error",
                    "step": "Query error",
                    "signals": trace.error_message,
                }
            )

    return {
        "rows": rows,
        "final_decision": run.get("final_decision"),
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
    self_check = trace.self_check
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
    question = (
        TERMINAL_PUBLIC_REDACTION
        if terminal_redaction
        else trace.question
    )
    answer = TERMINAL_PUBLIC_REDACTION if terminal_redaction else query_result.answer
    return {
        "question": question,
        "answer": answer,
        "document": trace.document_name,
        "backend": trace.backend,
        "model": trace.model_label,
        "retrieved_chunk_count": trace.retrieved_chunk_count,
        "citations": [
            {
                "id": citation.citation_id,
                "source": citation.source_name,
                "page": citation.page,
                "chunk": (
                    citation.chunk_index + 1
                    if citation.chunk_index is not None
                    else None
                ),
                "excerpt": citation.excerpt,
            }
            for citation in trace.citations
        ],
        "self_check": (
            {
                "outcome": self_check.outcome,
                "reasons": self_check.reasons,
                "retry_attempted": self_check.retry_attempted,
            }
            if self_check
            else None
        ),
        "model_thinking": model_thinking_dict(
            trace.model_thinking,
            terminal_redaction=model_thinking_redaction,
        ),
        "loop_report": public_loop_report,
        "error": public_trace_error(query_result, public_loop_report),
    }


def query_response_dict(query_result: QueryResult) -> dict:
    public_loop_report = public_loop_report_dict(query_result)
    answer = (
        TERMINAL_PUBLIC_REDACTION
        if visible_terminal_redaction_applied(query_result, public_loop_report)
        else query_result.answer
    )
    return {
        "answer": answer,
        "timeline": loop_timeline_dict(query_result),
        "summary": loop_summary_dict(query_result),
        "trace": answer_trace_dict(query_result),
    }


def empty_query_response_dict() -> dict:
    return {
        "answer": None,
        "timeline": loop_timeline_dict(None),
        "summary": loop_summary_dict(None),
        "trace": answer_trace_dict(None),
    }


def pretty_json(data: dict) -> str:
    return json.dumps(data, indent=2)
