import json
from datetime import datetime, timezone

import pytest

from src.adapters.langgraph_manifest import (
    ADAPTER_SCHEMA_VERSION,
    LangGraphManifestAdapter,
    export_report,
    export_session,
)
from src.loop_engine import (
    HumanReviewRequest,
    LoopDecision,
    LoopPhase,
    LoopPolicy,
    LoopReport,
    LoopRun,
    LoopSession,
    LoopStep,
    PUBLIC_REDACTION_TEXT,
    VerificationOutcome,
    VerificationResult,
)


def utc(value):
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def sample_report(run_id="run_phoenix") -> LoopReport:
    started_at = utc("2026-06-24T04:00:00")
    retrieved_at = utc("2026-06-24T04:00:01")
    drafted_at = utc("2026-06-24T04:00:02")
    verified_at = utc("2026-06-24T04:00:03")
    return LoopReport(
        run=LoopRun(
            run_id=run_id,
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
                    started_at=started_at,
                    ended_at=retrieved_at,
                    output_summary="Retrieved one chunk.",
                    metadata={"citation_count": 1},
                ),
                LoopStep(
                    step_id="step_draft",
                    phase=LoopPhase.DRAFT,
                    decision=LoopDecision.CONTINUE,
                    started_at=retrieved_at,
                    ended_at=drafted_at,
                    input_summary="question plus evidence",
                    output_summary="Project Phoenix launches in June 2026 [1].",
                ),
                LoopStep(
                    step_id="step_verify",
                    phase=LoopPhase.VERIFY,
                    decision=LoopDecision.SUPPORTED,
                    started_at=drafted_at,
                    ended_at=verified_at,
                    verification=VerificationResult(
                        outcome=VerificationOutcome.SUPPORTED,
                        reasons=("llm_verifier_supported",),
                        verifier="ollama",
                    ),
                ),
            ),
            final_decision=LoopDecision.SUPPORTED,
            final_answer="Project Phoenix launches in June 2026 [1].",
            metadata={"document": "project_phoenix.md"},
        )
    )


def terminal_report_with_secret_everywhere(secret: str) -> LoopReport:
    started_at = utc("2026-06-24T04:30:00")
    review = HumanReviewRequest(
        reason=secret,
        instructions=secret,
        requested_by_step_id="step_review",
        metadata={"secret": secret},
    )
    return LoopReport(
        run=LoopRun(
            run_id="run_terminal",
            session_id="session_local",
            user_input=secret,
            context_provider="document",
            backend="mock",
            model_label="MockLLM (explicit demo)",
            policy=LoopPolicy(metadata={"secret": secret}),
            started_at=started_at,
            completed_at=started_at,
            steps=(
                LoopStep(
                    step_id="step_review",
                    phase=LoopPhase.VERIFY,
                    decision=LoopDecision.REQUIRES_REVIEW,
                    name="Policy review",
                    started_at=started_at,
                    ended_at=started_at,
                    input_summary=secret,
                    output_summary=secret,
                    error_message=secret,
                    verification=VerificationResult(
                        outcome=VerificationOutcome.UNSUPPORTED,
                        reasons=(secret,),
                        verifier=secret,
                        raw_response=secret,
                        metadata={"secret": secret},
                    ),
                    human_review=review,
                    metadata={"secret": secret},
                ),
            ),
            final_decision=LoopDecision.REQUIRES_REVIEW,
            final_answer=secret,
            error_message=secret,
            metadata={"secret": secret},
        )
    )


def test_export_report_maps_loop_report_to_langgraph_manifest_shape():
    payload = export_report(sample_report())

    assert payload["adapter_name"] == "langgraph_manifest"
    assert payload["adapter_schema_version"] == ADAPTER_SCHEMA_VERSION
    assert payload["source_schema_version"] == "loop-report/v1"
    assert payload["public"] is True

    manifest = payload["manifest"]
    assert manifest["thread_id"] == "session_local"
    assert manifest["run_id"] == "run_phoenix"
    assert manifest["phase_order"] == ["retrieve", "draft", "verify"]
    assert manifest["final_decision"] == "supported"
    assert manifest["terminal_state"]["final_answer"] == (
        "Project Phoenix launches in June 2026 [1]."
    )
    assert [checkpoint["checkpoint_ns"] for checkpoint in manifest["checkpoints"]] == [
        "loopwright",
        "loopwright",
        "loopwright",
    ]
    assert manifest["checkpoints"][1]["node"] == "draft"
    assert manifest["checkpoints"][1]["state"]["output_summary"].startswith(
        "Project Phoenix"
    )


def test_export_report_is_json_serializable_and_does_not_mutate_report():
    report = sample_report()
    before = report.to_dict()

    payload = export_report(report)

    assert report.to_dict() == before
    assert json.loads(json.dumps(payload))["manifest"]["run_id"] == "run_phoenix"


def test_export_session_adds_source_jsonl_line_references():
    session = (
        LoopSession(session_id="session_local")
        .add_report(sample_report("run_one"))
        .add_report(sample_report("run_two"))
    )

    payload = export_session(session)

    assert payload["source_schema_version"] == "loop-session/v1"
    assert payload["thread_id"] == "session_local"
    assert payload["manifest_count"] == 2
    assert [manifest["source_jsonl_line"] for manifest in payload["manifests"]] == [
        1,
        2,
    ]
    assert [manifest["run_id"] for manifest in payload["manifests"]] == [
        "run_one",
        "run_two",
    ]


def test_public_export_redacts_terminal_content():
    secret = "terminal manifest secret should not leak"
    report = terminal_report_with_secret_everywhere(secret)
    payload = export_report(report)
    serialized = json.dumps(payload)

    assert payload["public"] is True
    assert secret not in serialized
    assert payload["manifest"]["terminal_state"]["user_input"] == PUBLIC_REDACTION_TEXT
    assert payload["manifest"]["metadata"]["policy"]["metadata"]["redacted"] is True
    checkpoint_state = payload["manifest"]["checkpoints"][0]["state"]
    assert checkpoint_state["verification"]["reasons"] == [PUBLIC_REDACTION_TEXT]
    assert checkpoint_state["human_review"] is None


def test_raw_export_requires_explicit_public_false():
    secret = "raw manifest secret"
    report = terminal_report_with_secret_everywhere(secret)
    payload = LangGraphManifestAdapter().export_report(report, public=False)

    assert payload["public"] is False
    assert secret in json.dumps(payload)


@pytest.mark.parametrize("bad_public", [None, 0, "", "false"])
def test_export_report_rejects_non_bool_public_flags(bad_public):
    with pytest.raises(ValueError, match="public must be a boolean"):
        LangGraphManifestAdapter().export_report(sample_report(), public=bad_public)


@pytest.mark.parametrize("bad_public", [None, 0, "", "false"])
def test_export_session_rejects_non_bool_public_flags(bad_public):
    session = LoopSession(session_id="session_local").add_report(sample_report())

    with pytest.raises(ValueError, match="public must be a boolean"):
        LangGraphManifestAdapter().export_session(session, public=bad_public)
