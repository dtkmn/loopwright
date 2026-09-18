import json
import re
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from src.adapters.openai_trace import (
    ADAPTER_SCHEMA_VERSION,
    OpenAITraceAdapter,
    export_report,
    export_session,
)
from src.loop_engine import (
    ANSWER_CANDIDATE_SHA256_METADATA_KEY,
    EVIDENCE_SET_SHA256_METADATA_KEY,
    EvidenceReference,
    GuardrailDecision,
    LoopDecision,
    LoopPhase,
    LoopPolicy,
    LoopReport,
    LoopRun,
    LoopSession,
    LoopStep,
    LoopTerminalReason,
    VerificationOutcome,
    VerificationResult,
    answer_candidate_sha256,
    evidence_set_sha256,
)


def utc(value):
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def sample_report(run_id: str = "run_phoenix") -> LoopReport:
    final_answer = "Project Phoenix launches in June 2026 [1]."
    candidate_digest = answer_candidate_sha256(final_answer)
    evidence = EvidenceReference.from_source(
        citation_id=1,
        provider="document",
        source_identity="project_phoenix.md",
        page=None,
        chunk_index=0,
        excerpt="Project Phoenix launches in June 2026.",
    )
    evidence_digest = evidence_set_sha256((evidence,))
    started_at = utc("2026-06-24T01:00:00")
    retrieved_at = utc("2026-06-24T01:00:01")
    drafted_at = utc("2026-06-24T01:00:03")
    verified_at = utc("2026-06-24T01:00:04")
    finalized_at = utc("2026-06-24T01:00:05")
    return LoopReport(
        run=LoopRun(
            run_id=run_id,
            session_id="session_local",
            user_input="When does Project Phoenix launch?",
            context_provider="document",
            backend="ollama",
            model_label="Ollama (nemotron-3-nano:4b)",
            started_at=started_at,
            completed_at=finalized_at,
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
                    retry_count=0,
                    metadata={
                        ANSWER_CANDIDATE_SHA256_METADATA_KEY: candidate_digest,
                        EVIDENCE_SET_SHA256_METADATA_KEY: evidence_digest,
                    },
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
                    retry_count=0,
                    metadata={
                        ANSWER_CANDIDATE_SHA256_METADATA_KEY: candidate_digest,
                        EVIDENCE_SET_SHA256_METADATA_KEY: evidence_digest,
                    },
                    verification=VerificationResult(
                        outcome=VerificationOutcome.SUPPORTED,
                        reasons=("llm_verifier_supported",),
                        verifier="ollama",
                        verifier_backend="ollama",
                        verifier_model_label="Ollama (nemotron-3-nano:4b)",
                        same_model_as_drafter=True,
                        raw_response='{"outcome":"supported"}',
                    ),
                ),
                LoopStep(
                    step_id="step_final",
                    phase=LoopPhase.FINAL,
                    decision=LoopDecision.SUPPORTED,
                    name="Finalize answer",
                    started_at=verified_at,
                    ended_at=finalized_at,
                    output_summary=final_answer,
                    backend="ollama",
                    model_label="Ollama (nemotron-3-nano:4b)",
                    retry_count=0,
                    metadata={
                        ANSWER_CANDIDATE_SHA256_METADATA_KEY: candidate_digest,
                        EVIDENCE_SET_SHA256_METADATA_KEY: evidence_digest,
                    },
                ),
            ),
            evidence=(evidence,),
            final_decision=LoopDecision.SUPPORTED,
            terminal_reason=LoopTerminalReason.COMPLETED,
            final_answer=final_answer,
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
                    metadata={"guardrail_decision": decision.decision.value},
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
                    decision=LoopDecision.NOT_VERIFIED,
                    name="LLM verifier",
                    started_at=started_at,
                    ended_at=started_at,
                    verification=VerificationResult(
                        outcome=VerificationOutcome.UNSUPPORTED,
                        reasons=(secret,),
                        verifier=secret,
                        verifier_backend=secret,
                        verifier_model_label=secret,
                        raw_response=secret,
                        metadata={"reason": secret},
                    ),
                ),
                LoopStep(
                    step_id="step_refuse",
                    phase=LoopPhase.REFUSE,
                    decision=LoopDecision.REFUSE,
                    started_at=started_at,
                    ended_at=started_at,
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


def error_report_with_secret_diagnostics(secret: str) -> LoopReport:
    started_at = utc("2026-06-24T02:55:00")
    return LoopReport(
        run=LoopRun(
            run_id="run_error",
            session_id="session_error",
            user_input=secret,
            context_provider="document",
            backend="mock",
            model_label="MockLLM (explicit demo)",
            started_at=started_at,
            completed_at=started_at,
            steps=(
                LoopStep(
                    step_id="step_error",
                    phase=LoopPhase.ERROR,
                    decision=LoopDecision.ERROR,
                    name=secret,
                    started_at=started_at,
                    ended_at=started_at,
                    input_summary=secret,
                    output_summary=secret,
                    error_message=secret,
                    metadata={"secret": secret},
                ),
            ),
            final_decision=LoopDecision.ERROR,
            terminal_reason=LoopTerminalReason.ERROR,
            error_message=secret,
            metadata={"secret": secret},
        )
    )


def test_export_report_maps_loop_report_to_openai_trace_shape():
    payload = export_report(sample_report())

    assert payload["adapter_name"] == "openai_trace"
    assert ADAPTER_SCHEMA_VERSION == "openai-trace-export/v2"
    assert payload["adapter_schema_version"] == ADAPTER_SCHEMA_VERSION
    assert payload["source_schema_version"] == "loop-report/v1"
    assert payload["source_projection_schema_version"] == "loop-public-report/v1"
    assert payload["public"] is True

    trace = payload["trace"]
    assert re.fullmatch(r"trace_[0-9a-f]{32}", trace["trace_id"])
    assert trace["trace_id"] == export_report(sample_report())["trace"]["trace_id"]
    assert trace["workflow_name"] == "Loopwright"
    assert trace["group_id"] == "session_local"
    assert trace["source_jsonl_line"] is None
    assert trace["metadata"]["loopwright_run_id"] == "run_phoenix"
    assert trace["metadata"]["final_decision"] == "supported"
    assert trace["metadata"]["terminal_reason"] == "completed"
    assert trace["final_answer"] == "Project Phoenix launches in June 2026 [1]."
    assert [item["evidence_id"] for item in trace["evidence"]] == [
        payload["source_report"]["run"]["evidence"][0]["evidence_id"]
    ]
    assert [span["span_data"]["data"]["phase"] for span in trace["spans"]] == [
        "retrieve",
        "draft",
        "verify",
        "final",
    ]
    assert trace["spans"][1]["span_data"]["type"] == "custom"
    assert all(
        set(span["span_data"]) == {"type", "name", "data"}
        for span in trace["spans"]
    )
    assert (
        trace["spans"][1]["span_data"]["data"]["loopwright_step_id"]
        == "step_draft"
    )
    assert re.fullmatch(r"span_[0-9a-f]{24}", trace["spans"][1]["span_id"])
    assert trace["spans"][2]["span_data"]["type"] == "custom"
    verification = trace["spans"][2]["span_data"]["data"]["verification"]
    assert verification["outcome"] == "supported"
    assert verification["same_model_as_drafter"] is True
    assert trace["spans"][0]["duration_ms"] == 1000


def test_export_session_does_not_invent_source_jsonl_line_references():
    session = (
        LoopSession(session_id="session_local")
        .add_report(sample_report("run_one"))
        .add_report(sample_report("run_two"))
    )

    payload = export_session(session)

    assert [trace["source_jsonl_line"] for trace in payload["traces"]] == [None, None]
    assert [trace["final_answer"] for trace in payload["traces"]] == [
        "Project Phoenix launches in June 2026 [1].",
        "Project Phoenix launches in June 2026 [1].",
    ]
    assert payload["traces"][0]["evidence"] == export_report(
        sample_report("run_one")
    )["trace"]["evidence"]
    assert payload["traces"][1]["evidence"] == export_report(
        sample_report("run_two")
    )["trace"]["evidence"]


def test_export_session_preserves_explicit_source_jsonl_line_references():
    session = (
        LoopSession(session_id="session_local")
        .add_report(sample_report("run_one"))
        .add_report(sample_report("run_two"))
    )

    payload = export_session(session, source_jsonl_lines=(4, 9))

    assert [trace["source_jsonl_line"] for trace in payload["traces"]] == [4, 9]


def test_export_session_rejects_mutated_unsupported_session_schema():
    session = LoopSession(session_id="session_local")
    object.__setattr__(session, "schema_version", "bogus-session/v999")

    with pytest.raises(ValueError, match="Unsupported loop session schema"):
        export_session(session)


@pytest.mark.parametrize("mutation", ["schema", "subclass"])
def test_export_session_rejects_unsupported_report_shape(mutation):
    report = sample_report()
    if mutation == "schema":
        object.__setattr__(report, "schema_version", "bogus-report/v999")
    else:
        class LoopReportSubclass(LoopReport):
            pass

        report = LoopReportSubclass(run=report.run)
    session = LoopSession(session_id="session_local", reports=(report,))

    with pytest.raises(ValueError, match="(loop report schema|exact LoopReport)"):
        export_session(session, public=False)


@pytest.mark.parametrize(
    "invalid_session_id",
    ["../foreign", "https://secret.example/x", "alpha beta", "\x00x"],
)
def test_public_empty_session_export_rejects_noncanonical_identity(
    invalid_session_id,
):
    session = LoopSession(session_id="session_local")
    object.__setattr__(session, "session_id", invalid_session_id)

    with pytest.raises(ValueError, match="canonical identity"):
        export_session(session)


@pytest.mark.parametrize("mutation", ["schema", "subclass"])
def test_raw_export_rejects_unsupported_report_shape(mutation):
    report = sample_report()
    if mutation == "schema":
        object.__setattr__(report, "schema_version", "bogus-report/v999")
    else:
        class LoopReportSubclass(LoopReport):
            pass

        report = LoopReportSubclass(run=report.run)

    with pytest.raises(ValueError, match="(loop report schema|exact LoopReport)"):
        export_report(report, public=False)


@pytest.mark.parametrize("hostile_metadata", [{"value": float("nan")}, {"value": b"x"}])
def test_raw_export_rejects_noncanonical_json_metadata(hostile_metadata):
    report = sample_report()
    report = replace(report, run=replace(report.run, metadata=hostile_metadata))

    with pytest.raises(ValueError, match="canonical JSON"):
        export_report(report, public=False)


def test_export_report_is_json_serializable_and_does_not_mutate_report():
    report = sample_report()
    before = report.to_dict()

    payload = export_report(report)

    assert report.to_dict() == before
    assert json.loads(json.dumps(payload))["trace"]["trace_id"] == payload["trace"][
        "trace_id"
    ]


def test_public_export_redacts_terminal_decision_content_by_default():
    secret = "blocked draft should not leak"
    payload = export_report(guardrail_blocked_report(secret))
    serialized = json.dumps(payload)

    assert payload["public"] is True
    assert secret not in serialized
    assert payload["source_report"]["public_redaction"]["applied"] is True
    assert "loop_metadata" not in payload["trace"]["metadata"]
    assert payload["trace"]["spans"][0]["span_data"]["data"]["output"] is None


def test_public_export_redacts_unmarked_terminal_block_content():
    secret = "unmarked policy block should not leak"
    payload = export_report(unmarked_terminal_block_report(secret))
    serialized = json.dumps(payload)

    assert payload["public"] is True
    assert secret not in serialized
    assert payload["source_report"]["public_redaction"]["applied"] is True
    assert "loop_metadata" not in payload["trace"]["metadata"]
    span_data = payload["trace"]["spans"][0]["span_data"]["data"]
    assert span_data["input"] is None
    assert span_data["output"] is None
    assert span_data["error"] is None


def test_public_export_redacts_terminal_verifier_reasons():
    secret = "secret verifier reason should not leak"
    payload = export_report(terminal_refusal_report_with_secret_verifier_reason(secret))
    serialized = json.dumps(payload)
    verification = payload["trace"]["spans"][0]["span_data"]["data"][
        "verification"
    ]

    assert payload["public"] is True
    assert secret not in serialized
    assert "reasons" not in verification
    assert "verifier" not in verification
    assert verification["verifier_backend"] is None
    assert verification["verifier_model_label"] is None
    assert verification["same_model_as_drafter"] is None
    assert "raw_response" not in verification
    assert "metadata" not in verification


def test_public_export_redacts_terminal_policy_metadata():
    secret = "secret policy metadata should not leak"
    payload = export_report(terminal_block_report_with_secret_policy_metadata(secret))
    serialized = json.dumps(payload)

    assert payload["public"] is True
    assert secret not in serialized
    assert "metadata" not in payload["source_report"]["run"]["policy"]


def test_public_export_preserves_error_presence_without_error_text():
    secret = "secret error diagnostic"

    payload = export_report(error_report_with_secret_diagnostics(secret))
    span = payload["trace"]["spans"][0]

    assert secret not in json.dumps(payload)
    assert span["status"] == "error"
    assert span["span_data"]["data"]["error"] is None
    assert span["span_data"]["data"]["error_present"] is True


def test_raw_export_requires_explicit_public_false():
    secret = "raw blocked draft"
    payload = OpenAITraceAdapter().export_report(
        guardrail_blocked_report(secret),
        public=False,
    )

    assert payload["public"] is False
    assert payload["source_projection_schema_version"] is None
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


@pytest.mark.parametrize("bad_session_id", [0, False, "", [], {}])
def test_export_report_rejects_invalid_session_fallback(bad_session_id):
    with pytest.raises(ValueError, match="fallback session_id"):
        export_report(sample_report(), session_id=bad_session_id)


def test_export_report_rejects_conflicting_session_fallback():
    with pytest.raises(ValueError, match="conflicts with report session_id"):
        export_report(sample_report(), session_id="session_other")


def test_export_report_rejects_malformed_report_session_identity():
    report = sample_report()
    object.__setattr__(report.run, "session_id", 0)

    with pytest.raises(ValueError, match="report session_id"):
        export_report(report, public=False)


@pytest.mark.parametrize("identity_field", ["run_id", "session_id", "step_id"])
@pytest.mark.parametrize(
    "bad_identity",
    [" x ", "bad/run", "bad\nrun", "url://run", "\x00x"],
)
def test_raw_export_rejects_noncanonical_identity(identity_field, bad_identity):
    report = sample_report()
    target = report.run.steps[0] if identity_field == "step_id" else report.run
    object.__setattr__(target, identity_field, bad_identity)

    with pytest.raises(ValueError, match=identity_field):
        export_report(report, public=False)


@pytest.mark.parametrize(
    "source_lines",
    [
        (1,),
        (1, 0),
        (1, True),
        (1, 1),
        (2**53, 2**53 + 1),
    ],
)
def test_export_session_rejects_invalid_source_jsonl_lines(source_lines):
    session = (
        LoopSession(session_id="session_local")
        .add_report(sample_report("run_one"))
        .add_report(sample_report("run_two"))
    )

    with pytest.raises(ValueError, match="source_jsonl"):
        export_session(session, source_jsonl_lines=source_lines)


@pytest.mark.parametrize("source_line", [0, -1, True, 1.5, "1", 2**53, 10**100])
def test_export_report_rejects_invalid_source_jsonl_line(source_line):
    with pytest.raises(ValueError, match="source_jsonl_line"):
        export_report(sample_report(), source_jsonl_line=source_line)


def test_trace_ids_do_not_collapse_distinct_run_id_characters():
    session = (
        LoopSession(session_id="session_local")
        .add_report(sample_report("run-a"))
        .add_report(sample_report("run.a"))
    )

    payload = export_session(session)
    traces = payload["traces"]

    assert traces[0]["trace_id"] != traces[1]["trace_id"]
    assert [trace["metadata"]["loopwright_run_id"] for trace in traces] == [
        "run-a",
        "run.a",
    ]


def test_trace_id_includes_session_identity():
    report = sample_report()
    other_session_report = LoopReport(
        run=replace(report.run, session_id="session_other")
    )

    first_trace = export_report(report)["trace"]
    second_trace = export_report(other_session_report)["trace"]

    assert first_trace["metadata"]["loopwright_run_id"] == "run_phoenix"
    assert second_trace["metadata"]["loopwright_run_id"] == "run_phoenix"
    assert first_trace["group_id"] != second_trace["group_id"]
    assert first_trace["trace_id"] != second_trace["trace_id"]


def test_span_ids_do_not_collapse_valid_identity_part_boundaries():
    first = sample_report("run-part")
    first = LoopReport(
        run=replace(
            first.run,
            steps=(replace(first.run.steps[0], step_id="step"),),
        )
    )
    second = sample_report("run")
    second = LoopReport(
        run=replace(
            second.run,
            steps=(replace(second.run.steps[0], step_id="part-step"),),
        )
    )

    traces = export_session(
        LoopSession(session_id="session_local")
        .add_report(first)
        .add_report(second),
        public=False,
    )["traces"]

    assert traces[0]["spans"][0]["span_id"] != traces[1]["spans"][0]["span_id"]


def test_export_report_rejects_duplicate_step_identity():
    report = sample_report()
    object.__setattr__(
        report.run,
        "steps",
        (report.run.steps[0], report.run.steps[0]),
    )

    with pytest.raises(ValueError, match="duplicate step"):
        export_report(report, public=False)


def test_phase_names_without_real_provider_provenance_stay_custom():
    report = sample_report()
    synthetic_draft = replace(
        report.run.steps[1],
        backend="mock",
        model_label="MockLLM (explicit demo)",
    )
    synthetic_verify = replace(
        report.run.steps[2],
        backend="mock",
        model_label="MockLLM (explicit demo)",
        verification=replace(
            report.run.steps[2].verification,
            verifier_backend="mock",
            verifier_model_label="MockLLM (explicit demo)",
        ),
    )
    synthetic_report = LoopReport(
        run=replace(report.run, steps=(synthetic_draft, synthetic_verify))
    )

    spans = export_report(synthetic_report, public=False)["trace"]["spans"]

    assert [span["span_data"]["type"] for span in spans] == ["custom", "custom"]


def test_export_session_maps_reports_to_trace_collection():
    first = sample_report()
    second_answer = "The budget is $42 million."
    second_answer_digest = answer_candidate_sha256(second_answer)
    second_evidence_digest = evidence_set_sha256(())
    second_started_at = utc("2026-06-24T03:00:00")
    second_binding = {
        ANSWER_CANDIDATE_SHA256_METADATA_KEY: second_answer_digest,
        EVIDENCE_SET_SHA256_METADATA_KEY: second_evidence_digest,
    }
    second = LoopReport(
        run=LoopRun(
            run_id="run_second",
            session_id="session_local",
            user_input="What is the budget?",
            context_provider="document",
            backend="mock",
            model_label="MockLLM (explicit demo)",
            started_at=second_started_at,
            completed_at=second_started_at,
            steps=(
                LoopStep(
                    step_id="step_second_draft",
                    phase=LoopPhase.DRAFT,
                    decision=LoopDecision.CONTINUE,
                    started_at=second_started_at,
                    ended_at=second_started_at,
                    output_summary=second_answer,
                    backend="mock",
                    model_label="MockLLM (explicit demo)",
                    retry_count=0,
                    metadata=second_binding,
                ),
                LoopStep(
                    step_id="step_second_verify",
                    phase=LoopPhase.VERIFY,
                    decision=LoopDecision.NOT_VERIFIED,
                    started_at=second_started_at,
                    ended_at=second_started_at,
                    backend="mock",
                    model_label="MockLLM (explicit demo)",
                    retry_count=0,
                    verification=VerificationResult(
                        outcome=VerificationOutcome.NOT_VERIFIED,
                    ),
                    metadata=second_binding,
                ),
                LoopStep(
                    step_id="step_second_final",
                    phase=LoopPhase.FINAL,
                    decision=LoopDecision.NOT_VERIFIED,
                    started_at=second_started_at,
                    ended_at=second_started_at,
                    output_summary=second_answer,
                    backend="mock",
                    model_label="MockLLM (explicit demo)",
                    retry_count=0,
                    metadata=second_binding,
                ),
            ),
            final_decision=LoopDecision.NOT_VERIFIED,
            terminal_reason=LoopTerminalReason.NOT_VERIFIED,
            final_answer=second_answer,
        )
    )
    session = LoopSession(session_id="session_local").add_report(first).add_report(
        second
    )

    payload = export_session(session)

    assert payload["adapter_name"] == "openai_trace"
    assert payload["source_schema_version"] == "loop-session/v1"
    assert payload["source_projection_schema_version"] == "loop-public-report/v1"
    assert payload["session_id"] == "session_local"
    assert payload["trace_count"] == 2
    trace_ids = [trace["trace_id"] for trace in payload["traces"]]
    assert len(set(trace_ids)) == 2
    assert all(re.fullmatch(r"trace_[0-9a-f]{32}", trace_id) for trace_id in trace_ids)
    assert {trace["group_id"] for trace in payload["traces"]} == {"session_local"}
