import json
from datetime import datetime, timezone

import pytest

from src.adapters.openai_trace import (
    ADAPTER_SCHEMA_VERSION,
    OpenAITraceAdapter,
    export_report,
    export_session,
)
from src.loop_engine import (
    GuardrailDecision,
    LoopDecision,
    LoopPhase,
    LoopPolicy,
    LoopReport,
    LoopRun,
    LoopSession,
    LoopStep,
    PUBLIC_REDACTION_REASON,
    PUBLIC_REDACTION_TEXT,
    VerificationOutcome,
    VerificationResult,
)


def utc(value):
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def sample_report() -> LoopReport:
    started_at = utc("2026-06-24T01:00:00")
    retrieved_at = utc("2026-06-24T01:00:01")
    drafted_at = utc("2026-06-24T01:00:03")
    verified_at = utc("2026-06-24T01:00:04")
    return LoopReport(
        run=LoopRun(
            run_id="run_phoenix",
            session_id="session_local",
            user_input="When does Project Phoenix launch?",
            context_provider="document",
            backend="ollama",
            model_label="Ollama (nemotron-3-nano:4b)",
            started_at=started_at,
            completed_at=verified_at,
            steps=(
                LoopStep(
                    step_id="step_retrieve",
                    phase=LoopPhase.RETRIEVE,
                    decision=LoopDecision.CONTINUE,
                    name="Retrieve prompt evidence",
                    started_at=started_at,
                    ended_at=retrieved_at,
                    output_summary="Retrieved one cited chunk.",
                    metadata={"citation_count": 1},
                ),
                LoopStep(
                    step_id="step_draft",
                    phase=LoopPhase.DRAFT,
                    decision=LoopDecision.CONTINUE,
                    name="Draft answer",
                    started_at=retrieved_at,
                    ended_at=drafted_at,
                    input_summary="question plus prompt evidence",
                    output_summary="Project Phoenix launches in June 2026 [1].",
                    backend="ollama",
                    model_label="Ollama (nemotron-3-nano:4b)",
                ),
                LoopStep(
                    step_id="step_verify",
                    phase=LoopPhase.VERIFY,
                    decision=LoopDecision.SUPPORTED,
                    name="LLM verifier",
                    started_at=drafted_at,
                    ended_at=verified_at,
                    input_summary="answer plus cited excerpt",
                    output_summary="answer supported",
                    backend="ollama",
                    model_label="Ollama (nemotron-3-nano:4b)",
                    verification=VerificationResult(
                        outcome=VerificationOutcome.SUPPORTED,
                        reasons=("llm_verifier_supported",),
                        verifier="ollama",
                        raw_response='{"outcome":"supported"}',
                    ),
                ),
            ),
            final_decision=LoopDecision.SUPPORTED,
            final_answer="Project Phoenix launches in June 2026 [1].",
            metadata={"document": "project_phoenix.md"},
        )
    )


def guardrail_blocked_report(secret: str) -> LoopReport:
    started_at = utc("2026-06-24T02:00:00")
    decision = GuardrailDecision(
        decision=LoopDecision.BLOCK,
        reason=secret,
        metadata={"blocked_text": secret},
    )
    return LoopReport(
        run=LoopRun(
            run_id="run_blocked",
            session_id="session_guardrail",
            user_input=secret,
            context_provider="document",
            backend="mock",
            model_label="MockLLM (explicit demo)",
            started_at=started_at,
            completed_at=started_at,
            steps=(
                LoopStep(
                    step_id="step_draft",
                    phase=LoopPhase.DRAFT,
                    decision=LoopDecision.CONTINUE,
                    started_at=started_at,
                    ended_at=started_at,
                    output_summary=secret,
                    metadata={"draft": secret},
                ),
                LoopStep(
                    step_id="step_guardrail",
                    phase=LoopPhase.ERROR,
                    decision=LoopDecision.BLOCK,
                    name="Guardrail decision",
                    started_at=started_at,
                    ended_at=started_at,
                    error_message=secret,
                    metadata={"guardrail_decision": decision.to_dict()},
                ),
            ),
            final_decision=LoopDecision.BLOCK,
            final_answer=secret,
            error_message=secret,
            metadata={"unsafe": secret},
        )
    )


def unmarked_terminal_block_report(secret: str) -> LoopReport:
    started_at = utc("2026-06-24T02:30:00")
    return LoopReport(
        run=LoopRun(
            run_id="run_policy_block",
            session_id="session_guardrail",
            user_input=secret,
            context_provider="document",
            backend="mock",
            model_label="MockLLM (explicit demo)",
            started_at=started_at,
            completed_at=started_at,
            steps=(
                LoopStep(
                    step_id="step_policy_block",
                    phase=LoopPhase.ERROR,
                    decision=LoopDecision.BLOCK,
                    name="Policy block",
                    started_at=started_at,
                    ended_at=started_at,
                    input_summary=secret,
                    output_summary=secret,
                    error_message=secret,
                    metadata={"policy_reason": secret},
                ),
            ),
            final_decision=LoopDecision.BLOCK,
            final_answer=secret,
            error_message=secret,
            metadata={"policy_reason": secret},
        )
    )


def terminal_refusal_report_with_secret_verifier_reason(secret: str) -> LoopReport:
    started_at = utc("2026-06-24T02:45:00")
    return LoopReport(
        run=LoopRun(
            run_id="run_refuse",
            session_id="session_guardrail",
            user_input="Should this answer be returned?",
            context_provider="document",
            backend="ollama",
            model_label="Ollama (nemotron-3-nano:4b)",
            started_at=started_at,
            completed_at=started_at,
            steps=(
                LoopStep(
                    step_id="step_verify_refuse",
                    phase=LoopPhase.VERIFY,
                    decision=LoopDecision.REFUSE,
                    name="LLM verifier",
                    started_at=started_at,
                    ended_at=started_at,
                    verification=VerificationResult(
                        outcome=VerificationOutcome.UNSUPPORTED,
                        reasons=(secret,),
                        verifier=secret,
                        raw_response=secret,
                        metadata={"reason": secret},
                    ),
                ),
            ),
            final_decision=LoopDecision.REFUSE,
            final_answer=None,
            error_message="verification_refused",
        )
    )


def terminal_block_report_with_secret_policy_metadata(secret: str) -> LoopReport:
    started_at = utc("2026-06-24T02:50:00")
    return LoopReport(
        run=LoopRun(
            run_id="run_policy_metadata",
            session_id="session_guardrail",
            user_input="Should policy metadata leak?",
            context_provider="document",
            backend="mock",
            model_label="MockLLM (explicit demo)",
            policy=LoopPolicy(metadata={"secret": secret}),
            started_at=started_at,
            completed_at=started_at,
            steps=(
                LoopStep(
                    step_id="step_policy_block",
                    phase=LoopPhase.ERROR,
                    decision=LoopDecision.BLOCK,
                    name="Policy block",
                    started_at=started_at,
                    ended_at=started_at,
                ),
            ),
            final_decision=LoopDecision.BLOCK,
            final_answer=None,
            error_message="policy_blocked",
        )
    )


def test_export_report_maps_loop_report_to_openai_trace_shape():
    payload = export_report(sample_report())

    assert payload["adapter_name"] == "openai_trace"
    assert payload["adapter_schema_version"] == ADAPTER_SCHEMA_VERSION
    assert payload["source_schema_version"] == "loop-report/v1"
    assert payload["public"] is True

    trace = payload["trace"]
    assert trace["trace_id"] == "trace_run_phoenix"
    assert trace["workflow_name"] == "Loopwright"
    assert trace["group_id"] == "session_local"
    assert trace["metadata"]["loopwright_run_id"] == "run_phoenix"
    assert trace["metadata"]["final_decision"] == "supported"
    assert [span["span_data"]["phase"] for span in trace["spans"]] == [
        "retrieve",
        "draft",
        "verify",
    ]
    assert trace["spans"][1]["span_data"]["type"] == "generation"
    assert trace["spans"][2]["span_data"]["type"] == "guardrail"
    assert trace["spans"][2]["span_data"]["verification"]["outcome"] == "supported"
    assert trace["spans"][0]["duration_ms"] == 1000


def test_export_report_is_json_serializable_and_does_not_mutate_report():
    report = sample_report()
    before = report.to_dict()

    payload = export_report(report)

    assert report.to_dict() == before
    assert json.loads(json.dumps(payload))["trace"]["trace_id"] == "trace_run_phoenix"


def test_public_export_redacts_terminal_decision_content_by_default():
    secret = "blocked draft should not leak"
    payload = export_report(guardrail_blocked_report(secret))
    serialized = json.dumps(payload)

    assert payload["public"] is True
    assert secret not in serialized
    assert payload["source_report"]["public_redaction"]["applied"] is True
    assert payload["trace"]["metadata"]["loop_metadata"]["redacted"] is True
    assert payload["trace"]["spans"][0]["span_data"]["output"] == PUBLIC_REDACTION_TEXT


def test_public_export_redacts_unmarked_terminal_block_content():
    secret = "unmarked policy block should not leak"
    payload = export_report(unmarked_terminal_block_report(secret))
    serialized = json.dumps(payload)

    assert payload["public"] is True
    assert secret not in serialized
    assert payload["source_report"]["public_redaction"]["applied"] is True
    assert payload["trace"]["metadata"]["loop_metadata"]["redacted"] is True
    assert payload["trace"]["spans"][0]["span_data"]["input"] == PUBLIC_REDACTION_TEXT
    assert payload["trace"]["spans"][0]["span_data"]["output"] == PUBLIC_REDACTION_TEXT
    assert payload["trace"]["spans"][0]["span_data"]["error"] == PUBLIC_REDACTION_REASON


def test_public_export_redacts_terminal_verifier_reasons():
    secret = "secret verifier reason should not leak"
    payload = export_report(terminal_refusal_report_with_secret_verifier_reason(secret))
    serialized = json.dumps(payload)
    verification = payload["trace"]["spans"][0]["span_data"]["verification"]

    assert payload["public"] is True
    assert secret not in serialized
    assert verification["reasons"] == [PUBLIC_REDACTION_TEXT]
    assert verification["verifier"] is None
    assert verification["raw_response"] is None
    assert verification["metadata"]["redacted"] is True


def test_public_export_redacts_terminal_policy_metadata():
    secret = "secret policy metadata should not leak"
    payload = export_report(terminal_block_report_with_secret_policy_metadata(secret))
    serialized = json.dumps(payload)
    policy_metadata = payload["source_report"]["run"]["policy"]["metadata"]

    assert payload["public"] is True
    assert secret not in serialized
    assert policy_metadata["redacted"] is True


def test_raw_export_requires_explicit_public_false():
    secret = "raw blocked draft"
    payload = OpenAITraceAdapter().export_report(
        guardrail_blocked_report(secret),
        public=False,
    )

    assert payload["public"] is False
    assert secret in json.dumps(payload)


@pytest.mark.parametrize("bad_public", [None, 0, "", "false"])
def test_export_report_rejects_non_bool_public_flags(bad_public):
    with pytest.raises(ValueError, match="public must be a boolean"):
        OpenAITraceAdapter().export_report(sample_report(), public=bad_public)


@pytest.mark.parametrize("bad_public", [None, 0, "", "false"])
def test_export_session_rejects_non_bool_public_flags(bad_public):
    session = LoopSession(session_id="session_local").add_report(sample_report())

    with pytest.raises(ValueError, match="public must be a boolean"):
        OpenAITraceAdapter().export_session(session, public=bad_public)


def test_export_session_maps_reports_to_trace_collection():
    first = sample_report()
    second = LoopReport(
        run=LoopRun(
            run_id="run_second",
            session_id="session_local",
            user_input="What is the budget?",
            context_provider="document",
            backend="mock",
            model_label="MockLLM (explicit demo)",
            started_at=utc("2026-06-24T03:00:00"),
            completed_at=utc("2026-06-24T03:00:00"),
            final_decision=LoopDecision.NOT_VERIFIED,
            final_answer="The budget is $42 million [1].",
        )
    )
    session = LoopSession(session_id="session_local").add_report(first).add_report(
        second
    )

    payload = export_session(session)

    assert payload["adapter_name"] == "openai_trace"
    assert payload["source_schema_version"] == "loop-session/v1"
    assert payload["session_id"] == "session_local"
    assert payload["trace_count"] == 2
    assert [trace["trace_id"] for trace in payload["traces"]] == [
        "trace_run_phoenix",
        "trace_run_second",
    ]
    assert {trace["group_id"] for trace in payload["traces"]} == {"session_local"}
