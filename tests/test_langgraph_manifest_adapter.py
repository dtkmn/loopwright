import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from src.adapters.langgraph_manifest import (
    ADAPTER_SCHEMA_VERSION,
    LangGraphManifestAdapter,
    export_report,
    export_session,
)
from src.loop_engine import (
    ANSWER_CANDIDATE_SHA256_METADATA_KEY,
    EVIDENCE_SET_SHA256_METADATA_KEY,
    EvidenceReference,
    HumanReviewRequest,
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


def sample_report(run_id="run_phoenix") -> LoopReport:
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
    started_at = utc("2026-06-24T04:00:00")
    retrieved_at = utc("2026-06-24T04:00:01")
    drafted_at = utc("2026-06-24T04:00:02")
    verified_at = utc("2026-06-24T04:00:03")
    finalized_at = utc("2026-06-24T04:00:04")
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
                    started_at=drafted_at,
                    ended_at=verified_at,
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
                    ),
                ),
                LoopStep(
                    step_id="step_final",
                    phase=LoopPhase.FINAL,
                    decision=LoopDecision.SUPPORTED,
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


def terminal_report_with_secret_everywhere(secret: str) -> LoopReport:
    started_at = utc("2026-06-24T04:30:00")
    review = HumanReviewRequest(
        reason=secret,
        instructions=secret,
        requested_by_step_id="step_review",
        created_at=started_at,
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
                    step_id="step_verify_review",
                    phase=LoopPhase.VERIFY,
                    decision=LoopDecision.NOT_VERIFIED,
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
                        verifier_backend=secret,
                        verifier_model_label=secret,
                        raw_response=secret,
                        metadata={"secret": secret},
                    ),
                    metadata={"secret": secret},
                ),
                LoopStep(
                    step_id="step_review",
                    phase=LoopPhase.ERROR,
                    decision=LoopDecision.REQUIRES_REVIEW,
                    started_at=started_at,
                    ended_at=started_at,
                    error_message=secret,
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
    assert ADAPTER_SCHEMA_VERSION == "langgraph-manifest-export/v2"
    assert payload["adapter_schema_version"] == ADAPTER_SCHEMA_VERSION
    assert payload["source_schema_version"] == "loop-report/v1"
    assert payload["source_projection_schema_version"] == "loop-public-report/v1"
    assert payload["public"] is True

    manifest = payload["manifest"]
    assert manifest["thread_id"] == "session_local"
    assert manifest["run_id"] == "run_phoenix"
    assert manifest["phase_order"] == ["retrieve", "draft", "verify", "final"]
    assert manifest["final_decision"] == "supported"
    assert manifest["terminal_reason"] == "completed"
    assert manifest["terminal_state"]["terminal_reason"] == "completed"
    assert manifest["terminal_state"]["final_answer"] == (
        "Project Phoenix launches in June 2026 [1]."
    )
    assert [item["evidence_id"] for item in manifest["evidence"]] == [
        payload["source_report"]["run"]["evidence"][0]["evidence_id"]
    ]
    assert [checkpoint["checkpoint_ns"] for checkpoint in manifest["checkpoints"]] == [
        "loopwright",
        "loopwright",
        "loopwright",
        "loopwright",
    ]
    assert manifest["checkpoints"][1]["node"] == "draft"
    assert manifest["checkpoints"][1]["state"]["output_summary"] is None


def test_export_report_is_json_serializable_and_does_not_mutate_report():
    report = sample_report()
    before = report.to_dict()

    payload = export_report(report)

    assert report.to_dict() == before
    assert json.loads(json.dumps(payload))["manifest"]["run_id"] == "run_phoenix"


def test_standalone_report_does_not_invent_langgraph_thread_identity():
    report = sample_report()
    report = LoopReport(run=replace(report.run, session_id=None))

    manifest = export_report(report)["manifest"]

    assert manifest["thread_id"] is None
    assert {
        checkpoint["thread_id"] for checkpoint in manifest["checkpoints"]
    } == {None}


def test_export_session_does_not_invent_source_jsonl_line_references():
    session = (
        LoopSession(session_id="session_local")
        .add_report(sample_report("run_one"))
        .add_report(sample_report("run_two"))
    )

    payload = export_session(session)

    assert payload["source_schema_version"] == "loop-session/v1"
    assert payload["source_projection_schema_version"] == "loop-public-report/v1"
    assert payload["thread_id"] == "session_local"
    assert payload["manifest_count"] == 2
    assert [manifest["source_jsonl_line"] for manifest in payload["manifests"]] == [
        None,
        None,
    ]
    assert [manifest["run_id"] for manifest in payload["manifests"]] == [
        "run_one",
        "run_two",
    ]
    assert payload["manifests"][0]["evidence"] == export_report(
        sample_report("run_one")
    )["manifest"]["evidence"]
    assert payload["manifests"][1]["evidence"] == export_report(
        sample_report("run_two")
    )["manifest"]["evidence"]


def test_export_session_preserves_explicit_source_jsonl_line_references():
    session = (
        LoopSession(session_id="session_local")
        .add_report(sample_report("run_one"))
        .add_report(sample_report("run_two"))
    )

    payload = export_session(session, source_jsonl_lines=(4, 9))

    assert [manifest["source_jsonl_line"] for manifest in payload["manifests"]] == [
        4,
        9,
    ]
    assert {
        checkpoint["source_jsonl_line"]
        for manifest in payload["manifests"]
        for checkpoint in manifest["checkpoints"]
    } == {4, 9}


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


def test_public_export_redacts_terminal_content():
    secret = "terminal manifest secret should not leak"
    report = terminal_report_with_secret_everywhere(secret)
    payload = export_report(report)
    serialized = json.dumps(payload)

    assert payload["public"] is True
    assert secret not in serialized
    assert payload["manifest"]["terminal_state"]["user_input"] is None
    assert "metadata" not in payload["manifest"]["metadata"]["policy"]
    checkpoint_state = payload["manifest"]["checkpoints"][-1]["state"]
    verification_state = payload["manifest"]["checkpoints"][0]["state"]
    assert payload["manifest"]["error_present"] is True
    assert payload["manifest"]["terminal_state"]["error_present"] is True
    assert checkpoint_state["error_present"] is True
    assert checkpoint_state["human_review_required"] is True
    assert "reasons" not in verification_state["verification"]
    assert verification_state["verification"]["verifier_backend"] is None
    assert verification_state["verification"]["verifier_model_label"] is None
    assert verification_state["verification"]["same_model_as_drafter"] is None
    assert checkpoint_state["human_review"] is None


def test_raw_export_requires_explicit_public_false():
    secret = "raw manifest secret"
    report = terminal_report_with_secret_everywhere(secret)
    payload = LangGraphManifestAdapter().export_report(report, public=False)

    assert payload["public"] is False
    assert payload["source_projection_schema_version"] is None
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


def test_export_report_rejects_conflicting_session_fallback():
    with pytest.raises(ValueError, match="conflicts with report session_id"):
        export_report(sample_report(), session_id="session_other")


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


@pytest.mark.parametrize("bad_step_id", [None, 7, True, "", "   "])
def test_raw_export_rejects_invalid_step_identity(bad_step_id):
    report = sample_report()
    object.__setattr__(report.run.steps[0], "step_id", bad_step_id)

    with pytest.raises(ValueError, match="step_id must be a canonical identity"):
        export_report(report, public=False)


def test_raw_export_rejects_duplicate_step_identity():
    report = sample_report()
    object.__setattr__(
        report.run,
        "steps",
        (report.run.steps[0], report.run.steps[0]),
    )

    with pytest.raises(ValueError, match="duplicate step"):
        export_report(report, public=False)


@pytest.mark.parametrize(
    ("bad_session_id", "message"),
    [
        (0, "report session_id"),
        ("", "report session_id"),
        ("session_other", "conflicts with report session_id"),
    ],
)
def test_raw_session_export_rejects_corrupted_report_session_identity(
    bad_session_id,
    message,
):
    report = sample_report()
    session = LoopSession(session_id="session_local").add_report(report)
    object.__setattr__(report.run, "session_id", bad_session_id)

    with pytest.raises(ValueError, match=message):
        export_session(session, public=False)


@pytest.mark.parametrize(
    "source_lines",
    [(1,), (1, 1), (1, True), (2**53, 2**53 + 1)],
)
def test_export_session_rejects_invalid_source_jsonl_lines(source_lines):
    session = (
        LoopSession(session_id="session_local")
        .add_report(sample_report("run_one"))
        .add_report(sample_report("run_two"))
    )

    with pytest.raises(ValueError, match="source_jsonl"):
        export_session(session, source_jsonl_lines=source_lines)
