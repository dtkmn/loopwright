import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from src import public_projection as public_projection_module
from src.loop_engine import (
    ANSWER_CANDIDATE_SHA256_METADATA_KEY,
    DEFAULT_LOOP_RECIPE_ID,
    EVIDENCE_SET_SHA256_METADATA_KEY,
    EvidenceLocator,
    EvidenceReference,
    GuardrailDecision,
    HumanReviewRequest,
    LoopDecision,
    LoopPhase,
    LoopPolicy,
    LoopReport,
    LoopRecipe,
    LoopRun,
    LoopSession,
    LoopStep,
    LoopTerminalReason,
    PUBLIC_REDACTION_REASON,
    VerificationOutcome,
    VerificationResult,
    answer_candidate_sha256,
    evidence_set_sha256,
)
from src.public_projection import (
    PUBLIC_REPORT_PROJECTION_SCHEMA_VERSION,
    PublicProjectionError,
)


def utc(value):
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


PUBLIC_PROJECTION_AT = utc("2026-06-23T14:00:00")


def test_evidence_reference_is_stable_typed_and_content_sensitive():
    reference = EvidenceReference.from_source(
        citation_id=1,
        provider="document",
        source_identity="confidential-roadmap.txt",
        page=7,
        chunk_index=2,
        excerpt="Project Phoenix launches in June 2026.",
    )
    same_reference = EvidenceReference.from_source(
        citation_id=9,
        provider="document",
        source_identity="confidential-roadmap.txt",
        page=7,
        chunk_index=2,
        excerpt="Project Phoenix launches in June 2026.",
    )
    changed_content = EvidenceReference.from_source(
        citation_id=1,
        provider="document",
        source_identity="confidential-roadmap.txt",
        page=7,
        chunk_index=2,
        excerpt="Project Phoenix launches in July 2026.",
    )

    payload = reference.to_dict()
    serialized = json.dumps(payload)

    assert reference.evidence_id == same_reference.evidence_id
    assert reference.evidence_id != changed_content.evidence_id
    assert reference.evidence_id.startswith("evidence_")
    assert len(reference.evidence_id) == len("evidence_") + 64
    assert payload == {
        "evidence_id": reference.evidence_id,
        "citation_id": 1,
        "provider": "document",
        "locator": {"page": 7, "chunk_index": 2},
    }
    assert "confidential-roadmap" not in serialized
    assert "Project Phoenix" not in serialized
    assert EvidenceReference.from_dict(payload) == reference


def test_loop_report_evidence_round_trips_and_legacy_reports_default_empty():
    evidence = EvidenceReference.from_source(
        citation_id=1,
        provider="web",
        source_identity="Example — https://example.test/source",
        page=None,
        chunk_index=None,
        excerpt="A cited fact.",
    )
    report = LoopReport(
        run=LoopRun(
            run_id="run_evidence",
            user_input="What is the cited fact?",
            context_provider="web",
            backend="mock",
            model_label="MockLLM (explicit demo)",
            evidence=(evidence,),
        )
    )

    payload = report.to_dict()
    restored = LoopReport.from_dict(json.loads(json.dumps(payload)))
    legacy_payload = json.loads(json.dumps(payload))
    legacy_payload["run"].pop("evidence")

    assert restored.run.evidence == (evidence,)
    assert LoopReport.from_dict(legacy_payload).run.evidence == ()


def test_loop_report_round_trips_through_json():
    started_at = utc("2026-06-23T10:00:00")
    ended_at = utc("2026-06-23T10:00:01")
    evidence = EvidenceReference.from_source(
        citation_id=1,
        provider="document",
        source_identity="project_phoenix_brief.md",
        page=None,
        chunk_index=0,
        excerpt="Project Phoenix launches in June 2026.",
    )
    verification = VerificationResult(
        outcome=VerificationOutcome.SUPPORTED,
        reasons=("mechanical_checks_passed", "llm_verifier_supported"),
        verifier="ollama",
        verifier_backend="ollama",
        verifier_model_label="Ollama (nemotron-3-nano:4b)",
        same_model_as_drafter=True,
        raw_response='{"outcome":"supported"}',
        metadata={"citation_count": 1},
    )
    step = LoopStep(
        step_id="step_verify",
        phase=LoopPhase.VERIFY,
        decision=LoopDecision.SUPPORTED,
        name="LLM verifier",
        started_at=started_at,
        ended_at=ended_at,
        input_summary="answer plus cited excerpts",
        output_summary="answer supported",
        backend="ollama",
        model_label="Ollama (nemotron-3-nano:4b)",
        retry_count=0,
        verification=verification,
        metadata={"prompt_chunks": 1},
    )
    run = LoopRun(
        run_id="run_phoenix",
        session_id="session_local",
        user_input="When does Project Phoenix launch?",
        context_provider="document",
        backend="ollama",
        model_label="Ollama (nemotron-3-nano:4b)",
        policy=LoopPolicy(max_retries=1),
        started_at=started_at,
        completed_at=ended_at,
        steps=(step,),
        evidence=(evidence,),
        final_decision=LoopDecision.SUPPORTED,
        terminal_reason=LoopTerminalReason.COMPLETED,
        final_answer="Project Phoenix launches in June 2026 [1].",
        metadata={"document": "project_phoenix_brief.md"},
    )
    report = LoopReport(run=run)

    payload = json.loads(json.dumps(report.to_dict()))
    restored = LoopReport.from_dict(payload)

    assert restored == report
    assert payload["schema_version"] == "loop-report/v1"
    assert payload["run"]["steps"][0]["phase"] == "verify"
    assert payload["run"]["steps"][0]["duration_ms"] == 1000
    assert payload["run"]["final_decision"] == "supported"
    assert payload["run"]["terminal_reason"] == "completed"
    assert payload["run"]["steps"][0]["verification"] == {
        "outcome": "supported",
        "reasons": ["mechanical_checks_passed", "llm_verifier_supported"],
        "verifier": "ollama",
        "verifier_backend": "ollama",
        "verifier_model_label": "Ollama (nemotron-3-nano:4b)",
        "same_model_as_drafter": True,
        "raw_response": '{"outcome":"supported"}',
        "metadata": {"citation_count": 1},
    }


@pytest.mark.parametrize(
    "record_factory",
    [
        lambda naive: LoopStep(
            step_id="step_naive",
            phase=LoopPhase.DRAFT,
            started_at=naive,
        ),
        lambda naive: LoopRun(
            run_id="run_naive",
            user_input="question",
            context_provider="none",
            backend="mock",
            model_label="MockLLM",
            started_at=naive,
        ),
    ],
)
def test_loop_records_reject_timezone_naive_timestamps(record_factory):
    with pytest.raises(ValueError, match="must be timezone-aware"):
        record_factory(datetime(2026, 6, 23, 10, 0, 0))


def test_loop_step_rejects_reversed_timestamp_interval():
    with pytest.raises(ValueError, match="ended_at must not be before started_at"):
        LoopStep(
            step_id="step_reversed",
            phase=LoopPhase.DRAFT,
            started_at=utc("2026-06-23T10:00:01"),
            ended_at=utc("2026-06-23T10:00:00"),
        )


def test_loop_run_rejects_reversed_timestamp_interval():
    with pytest.raises(
        ValueError,
        match="completed_at must not be before started_at",
    ):
        LoopRun(
            run_id="run_reversed",
            user_input="question",
            context_provider="none",
            backend="mock",
            model_label="MockLLM",
            started_at=utc("2026-06-23T10:00:01"),
            completed_at=utc("2026-06-23T10:00:00"),
        )


def test_loop_recipe_round_trips_and_summarizes():
    recipe = LoopRecipe(
        recipe_id=DEFAULT_LOOP_RECIPE_ID,
        name="General assistant loop",
        description="Default recipe",
        goal="Answer clearly.",
        instructions="Be direct.",
        success_criteria=("Addresses the request.", "Names uncertainty."),
        stop_condition="Stop after a safe final answer.",
        context_provider="smart",
        model_profile="quality",
        verifier="default",
        created_at=utc("2026-06-23T10:00:00"),
        updated_at=utc("2026-06-23T10:01:00"),
    )

    restored = LoopRecipe.from_dict(json.loads(json.dumps(recipe.to_dict())))

    assert restored == recipe
    assert restored.summary_dict()["is_default"] is True
    assert restored.runtime_dict()["success_criteria"] == [
        "Addresses the request.",
        "Names uncertainty.",
    ]


def test_loop_recipe_rejects_missing_goal():
    with pytest.raises(ValueError, match="goal"):
        LoopRecipe(recipe_id="recipe_bad", name="Bad", goal="")


def test_loop_session_keeps_reports_and_exports_jsonl(tmp_path):
    started_at = utc("2026-06-23T10:00:00")
    report = LoopReport(
        run=LoopRun(
            run_id="run_export",
            session_id="session_local",
            user_input="When does Project Phoenix launch?",
            context_provider="document",
            backend="mock",
            model_label="MockLLM (explicit demo)",
            started_at=started_at,
            completed_at=started_at,
            final_decision=LoopDecision.NOT_VERIFIED,
            final_answer="Project Phoenix launches in June 2026 [1].",
        )
    )

    session = LoopSession(session_id="session_local").add_report(report)
    artifact_path = session.write_jsonl(tmp_path / "session.jsonl")
    restored = LoopSession.from_jsonl(artifact_path.read_text(encoding="utf-8"))

    assert session.report_count == 1
    assert session.to_dict()["schema_version"] == "loop-session/v1"
    assert artifact_path.exists()
    assert restored == session
    assert json.loads(artifact_path.read_text(encoding="utf-8"))["run"]["run_id"] == (
        "run_export"
    )


def test_loop_session_rejects_public_projection_jsonl_as_non_rehydratable():
    answer = "answer"
    report = _visible_report(answer=answer)
    report = replace(
        report,
        run=replace(
            report.run,
            run_id="run_public_only",
            session_id="session_public_only",
        ),
    )
    session = LoopSession(
        session_id="session_public_only",
        reports=(report,),
    )

    with pytest.raises(ValueError, match="non-rehydratable"):
        LoopSession.from_jsonl(session.to_jsonl(public=True))


def test_loop_session_rejects_cross_session_reports():
    report = LoopReport(
        run=LoopRun(
            run_id="run_other",
            session_id="other_session",
            user_input="What happened?",
            context_provider="document",
            backend="mock",
            model_label="MockLLM (explicit demo)",
        )
    )

    with pytest.raises(ValueError, match="another session"):
        LoopSession(session_id="session_local").add_report(report)


def test_verification_result_rejects_unknown_outcome():
    with pytest.raises(ValueError):
        VerificationResult.from_dict(
            {
                "outcome": "suported",
                "reasons": ["typo_should_not_survive"],
            }
        )


def test_verification_result_rejects_non_boolean_same_model_flag():
    with pytest.raises(ValueError, match="same_model_as_drafter must be a boolean"):
        VerificationResult(
            outcome=VerificationOutcome.SUPPORTED,
            same_model_as_drafter="true",
        )


def test_legacy_completed_report_without_terminal_reason_is_unspecified():
    payload = LoopReport(
        run=LoopRun(
            run_id="run_legacy",
            user_input="What happened?",
            context_provider="none",
            backend="mock",
            model_label="MockLLM (explicit demo)",
            final_decision=LoopDecision.NOT_VERIFIED,
        )
    ).to_dict()
    payload["run"].pop("terminal_reason")

    restored = LoopReport.from_dict(payload)

    assert restored.run.terminal_reason == LoopTerminalReason.UNSPECIFIED


def test_loop_report_rejects_unknown_schema_version():
    report = LoopReport(
        run=LoopRun(
            run_id="run_unsupported_schema",
            user_input="What happened?",
            context_provider="document",
            backend="mock",
            model_label="MockLLM (explicit demo)",
        )
    ).to_dict()
    report["schema_version"] = "loop-report/v99"

    with pytest.raises(ValueError, match="Unsupported loop report schema"):
        LoopReport.from_dict(report)


@pytest.mark.parametrize(
    ("field_path", "expected_error"),
    [
        (("run", "backend"), "backend must be a string"),
        (("run", "model_label"), "model_label must be a string"),
        (("run", "final_answer"), "final_answer must be a string"),
        (("run", "steps", 0, "model_label"), "step.model_label must be a string"),
    ],
)
def test_loop_report_rejects_non_string_content_fields_before_projection(
    field_path,
    expected_error,
):
    report = LoopReport(
        run=LoopRun(
            run_id="run_hostile_type",
            user_input="question",
            context_provider="none",
            backend="mock",
            model_label="MockLLM",
            steps=(LoopStep(phase=LoopPhase.DRAFT),),
            final_decision=LoopDecision.NOT_VERIFIED,
            terminal_reason=LoopTerminalReason.NOT_VERIFIED,
            final_answer="answer",
        )
    ).to_dict()
    target = report
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = {"secret": "SECRET_COERCED_VALUE"}

    with pytest.raises(ValueError, match=expected_error):
        LoopReport.from_dict(report)


def test_public_report_redacts_terminal_verification_reasons():
    secret_reason = "SECRET_VERIFIER_REASON"
    secret_verifier = "SECRET_GATEWAY_VERIFIER"
    secret_policy = "SECRET_POLICY_METADATA"
    report = LoopReport(
        run=LoopRun(
            run_id="run_terminal_verifier",
            user_input="Blocked prompt",
            context_provider="document",
            backend="mock",
            model_label="MockLLM (explicit demo)",
            policy=LoopPolicy(metadata={"secret": secret_policy}),
            steps=(
                LoopStep(
                    phase=LoopPhase.VERIFY,
                    decision=LoopDecision.NOT_VERIFIED,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                    name="LLM verifier",
                    output_summary="unsupported",
                    verification=VerificationResult(
                        outcome=VerificationOutcome.UNSUPPORTED,
                        reasons=(secret_reason,),
                        verifier=secret_verifier,
                        verifier_backend=secret_verifier,
                        verifier_model_label=secret_verifier,
                        raw_response=secret_reason,
                        metadata={"debug": secret_reason},
                    ),
                ),
                LoopStep(
                    phase=LoopPhase.ERROR,
                    decision=LoopDecision.BLOCK,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                    name="Guardrail decision",
                    output_summary="blocked",
                    metadata={"guardrail_decision": "block"},
                ),
            ),
            started_at=PUBLIC_PROJECTION_AT,
            completed_at=PUBLIC_PROJECTION_AT,
            final_decision=LoopDecision.BLOCK,
            final_answer="Blocked.",
            error_message=secret_reason,
        )
    )

    raw_json = json.dumps(report.to_dict())
    public_payload = report.to_public_dict()
    public_json = json.dumps(public_payload)
    verification = public_payload["run"]["steps"][0]["verification"]

    assert secret_verifier in raw_json
    assert secret_policy in raw_json
    assert secret_reason not in public_json
    assert secret_verifier not in public_json
    assert secret_policy not in public_json
    assert public_payload["public_redaction"]["reason"] == PUBLIC_REDACTION_REASON
    assert public_payload["projection_schema_version"] == (
        PUBLIC_REPORT_PROJECTION_SCHEMA_VERSION
    )
    assert public_payload["run"]["error_present"] is True
    assert public_payload["run"]["final_answer"] is None
    assert public_payload["run"]["backend"] is None
    assert public_payload["run"]["model_label"] is None
    assert verification["verifier_backend"] is None
    assert verification["verifier_model_label"] is None
    assert verification["same_model_as_drafter"] is None
    assert set(verification) == {
        "outcome",
        "verifier_backend",
        "verifier_model_label",
        "same_model_as_drafter",
    }
    assert "metadata" not in public_payload["run"]["policy"]
    assert "name" not in public_payload["run"]["steps"][0]


def test_public_report_redacts_unmarked_terminal_verifier_identity():
    secret_verifier = "SECRET_OPERATOR_VERIFIER"
    report = LoopReport(
        run=LoopRun(
            run_id="run_terminal_without_guardrail_step",
            user_input="Sensitive prompt",
            context_provider="none",
            backend="openai-compatible",
            model_label="gateway-model",
            steps=(
                LoopStep(
                    phase=LoopPhase.VERIFY,
                    decision=LoopDecision.NOT_VERIFIED,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                    name="LLM verifier",
                    output_summary="refused by verifier",
                    verification=VerificationResult(
                        outcome=VerificationOutcome.UNSUPPORTED,
                        reasons=("unsupported_claim",),
                        verifier=secret_verifier,
                    ),
                ),
                LoopStep(
                    phase=LoopPhase.REFUSE,
                    decision=LoopDecision.REFUSE,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                ),
            ),
            started_at=PUBLIC_PROJECTION_AT,
            completed_at=PUBLIC_PROJECTION_AT,
            final_decision=LoopDecision.REFUSE,
            final_answer="I cannot verify that.",
        )
    )

    raw_json = json.dumps(report.to_dict())
    public_payload = report.to_public_dict()
    public_json = json.dumps(public_payload)
    verification = public_payload["run"]["steps"][0]["verification"]

    assert secret_verifier in raw_json
    assert secret_verifier not in public_json
    assert public_payload["public_redaction"]["applied"] is True
    assert public_payload["public_redaction"]["reason"] == PUBLIC_REDACTION_REASON
    assert "verifier" not in verification


def test_public_projection_is_versioned_and_recursively_allowlisted():
    omitted_secret = "SECRET_OMITTED_FIELD"
    evidence = EvidenceReference.from_source(
        citation_id=1,
        provider="document",
        source_identity=f"{omitted_secret}.txt",
        page=2,
        chunk_index=4,
        excerpt=f"Evidence {omitted_secret}",
    )
    answer = "The public answer claim [1]."
    answer_digest = answer_candidate_sha256(answer)
    evidence_digest = evidence_set_sha256((evidence,))
    report = LoopReport(
        run=LoopRun(
            run_id="run_allowlist",
            session_id="thread_allowlist",
            user_input=f"Prompt {omitted_secret}",
            context_provider="document",
            backend="ollama",
            model_label="gpt-oss:20b",
            policy=LoopPolicy(metadata={"secret": omitted_secret}),
            steps=(
                LoopStep(
                    step_id="step_allowlist_draft",
                    phase=LoopPhase.DRAFT,
                    decision=LoopDecision.CONTINUE,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                    backend="ollama",
                    model_label="gpt-oss:20b",
                    metadata={
                        ANSWER_CANDIDATE_SHA256_METADATA_KEY: answer_digest,
                        EVIDENCE_SET_SHA256_METADATA_KEY: evidence_digest,
                        "secret": omitted_secret,
                    },
                ),
                LoopStep(
                    step_id="step_allowlist",
                    phase=LoopPhase.VERIFY,
                    decision=LoopDecision.SUPPORTED,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                    name=f"Name {omitted_secret}",
                    input_summary=f"Input {omitted_secret}",
                    output_summary=f"Output {omitted_secret}",
                    backend="ollama",
                    model_label="gpt-oss:20b",
                    error_message=f"Error {omitted_secret}",
                    verification=VerificationResult(
                        outcome=VerificationOutcome.SUPPORTED,
                        reasons=(omitted_secret,),
                        verifier="ollama",
                        verifier_backend="ollama",
                        verifier_model_label="gpt-oss:20b",
                        same_model_as_drafter=True,
                        raw_response=omitted_secret,
                        metadata={"secret": omitted_secret},
                    ),
                    metadata={
                        ANSWER_CANDIDATE_SHA256_METADATA_KEY: answer_digest,
                        EVIDENCE_SET_SHA256_METADATA_KEY: evidence_digest,
                        "secret": omitted_secret,
                    },
                ),
                LoopStep(
                    step_id="step_allowlist_final",
                    phase=LoopPhase.FINAL,
                    decision=LoopDecision.SUPPORTED,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                    backend="ollama",
                    model_label="gpt-oss:20b",
                    metadata={
                        ANSWER_CANDIDATE_SHA256_METADATA_KEY: answer_digest,
                        EVIDENCE_SET_SHA256_METADATA_KEY: evidence_digest,
                        "secret": omitted_secret,
                    },
                ),
            ),
            started_at=PUBLIC_PROJECTION_AT,
            completed_at=PUBLIC_PROJECTION_AT,
            evidence=(evidence,),
            final_decision=LoopDecision.SUPPORTED,
            terminal_reason=LoopTerminalReason.COMPLETED,
            final_answer=answer,
            error_message=f"Error {omitted_secret}",
            metadata={"recipe_name": omitted_secret},
        )
    )

    payload = report.to_public_dict()
    run = payload["run"]
    step = run["steps"][1]
    verification = step["verification"]

    assert set(payload) == {
        "schema_version",
        "projection_schema_version",
        "public",
        "public_redaction",
        "run",
    }
    assert set(run) == {
        "run_id",
        "session_id",
        "context_provider",
        "conversation_context_count",
        "semantic_memory_count",
        "semantic_memory_status",
        "backend",
        "model_label",
        "policy",
        "started_at",
        "completed_at",
        "steps",
        "evidence",
        "final_decision",
        "terminal_reason",
        "final_answer",
        "error_present",
    }
    assert set(step) == {
        "step_id",
        "phase",
        "decision",
        "started_at",
        "ended_at",
        "duration_ms",
        "backend",
        "model_label",
        "retry_count",
        "error_present",
        "verification",
        "human_review_required",
    }
    assert set(verification) == {
        "outcome",
        "verifier_backend",
        "verifier_model_label",
        "same_model_as_drafter",
    }
    assert payload["projection_schema_version"] == (
        PUBLIC_REPORT_PROJECTION_SCHEMA_VERSION
    )
    assert payload["public"] is True
    assert run["final_answer"] == answer
    assert omitted_secret not in json.dumps(payload)


@pytest.mark.parametrize(
    ("final_decision", "terminal_reason", "step_decision", "guardrail_marker"),
    [
        (LoopDecision.ERROR, LoopTerminalReason.BLOCKED, LoopDecision.ERROR, None),
        (LoopDecision.ERROR, LoopTerminalReason.ERROR, LoopDecision.BLOCK, None),
        (LoopDecision.ERROR, LoopTerminalReason.ERROR, LoopDecision.ERROR, "block"),
    ],
)
def test_public_projection_rejects_inconsistent_terminal_contracts(
    final_decision,
    terminal_reason,
    step_decision,
    guardrail_marker,
):
    report = LoopReport(
        run=LoopRun(
            run_id="run_inconsistent",
            user_input="SECRET_PROMPT",
            context_provider="none",
            backend="mock",
            model_label="MockLLM",
            started_at=PUBLIC_PROJECTION_AT,
            steps=(
                LoopStep(
                    phase=LoopPhase.ERROR,
                    decision=step_decision,
                    started_at=PUBLIC_PROJECTION_AT,
                    output_summary="SECRET_DRAFT",
                    metadata=(
                        {"guardrail_decision": guardrail_marker}
                        if guardrail_marker is not None
                        else {}
                    ),
                ),
            ),
            final_decision=final_decision,
            terminal_reason=terminal_reason,
            final_answer="SECRET_ANSWER",
        )
    )

    with pytest.raises(PublicProjectionError, match="inconsistent terminal contract"):
        report.to_public_dict()


@pytest.mark.parametrize(
    ("final_decision", "terminal_reason", "error_match"),
    [
        (
            LoopDecision.REFUSE,
            LoopTerminalReason.VERIFICATION_FAILED,
            "verification_failed",
        ),
        (
            LoopDecision.REFUSE,
            LoopTerminalReason.POLICY_REFUSED,
            "policy_refused",
        ),
        (LoopDecision.BLOCK, LoopTerminalReason.BLOCKED, "blocked"),
        (LoopDecision.ERROR, LoopTerminalReason.ERROR, "error lacks"),
    ],
)
def test_public_projection_rejects_terminal_reason_without_causal_step(
    final_decision,
    terminal_reason,
    error_match,
):
    report = LoopReport(
        run=LoopRun(
            run_id="run_uncausal_terminal_reason",
            user_input="question",
            context_provider="none",
            backend="mock",
            model_label="MockLLM",
            started_at=PUBLIC_PROJECTION_AT,
            completed_at=PUBLIC_PROJECTION_AT,
            steps=(
                LoopStep(
                    step_id="step_uncausal_terminal_final",
                    phase=LoopPhase.FINAL,
                    decision=final_decision,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                ),
            ),
            final_decision=final_decision,
            terminal_reason=terminal_reason,
            final_answer="Terminal answer.",
        )
    )

    with pytest.raises(PublicProjectionError, match=error_match):
        report.to_public_dict()


def test_public_projection_rejects_trace_unavailable_without_trace_loss_boundary():
    report = _visible_report()
    report = replace(
        report,
        run=replace(
            report.run,
            terminal_reason=LoopTerminalReason.TRACE_UNAVAILABLE,
        ),
    )

    with pytest.raises(PublicProjectionError, match="trace_unavailable"):
        report.to_public_dict()


@pytest.mark.parametrize("metadata_location", ["step", "run"])
def test_public_projection_rejects_serialized_guardrail_decision_mismatch(
    metadata_location,
):
    guardrail_metadata = {
        "guardrail_decision": GuardrailDecision(
            decision=LoopDecision.BLOCK,
            reason="SECRET_GUARDRAIL_REASON",
        ).to_dict()
    }
    report = LoopReport(
        run=LoopRun(
            run_id="run_serialized_guardrail",
            user_input="SECRET_PROMPT",
            context_provider="none",
            backend="SECRET_BACKEND",
            model_label="SECRET_MODEL",
            steps=(
                LoopStep(
                    step_id="step_serialized_guardrail",
                    phase=LoopPhase.ERROR,
                    decision=LoopDecision.ERROR,
                    started_at=PUBLIC_PROJECTION_AT,
                    metadata=(
                        guardrail_metadata if metadata_location == "step" else {}
                    ),
                ),
            ),
            started_at=PUBLIC_PROJECTION_AT,
            final_decision=LoopDecision.ERROR,
            terminal_reason=LoopTerminalReason.ERROR,
            final_answer="SECRET_ANSWER",
            metadata=(guardrail_metadata if metadata_location == "run" else {}),
        )
    )

    with pytest.raises(PublicProjectionError, match="malformed guardrail decision"):
        report.to_public_dict()


@pytest.mark.parametrize("metadata_location", ["step", "run"])
@pytest.mark.parametrize("guardrail_value", [None, ""])
def test_public_projection_rejects_empty_guardrail_decision_marker(
    metadata_location,
    guardrail_value,
):
    guardrail_metadata = {"guardrail_decision": guardrail_value}
    report = LoopReport(
        run=LoopRun(
            run_id="run_empty_guardrail",
            user_input="SECRET_PROMPT",
            context_provider="none",
            backend="SECRET_BACKEND",
            model_label="SECRET_MODEL",
            steps=(
                LoopStep(
                    step_id="step_empty_guardrail",
                    phase=LoopPhase.ERROR,
                    decision=LoopDecision.ERROR,
                    started_at=PUBLIC_PROJECTION_AT,
                    metadata=(
                        guardrail_metadata if metadata_location == "step" else {}
                    ),
                ),
            ),
            started_at=PUBLIC_PROJECTION_AT,
            final_decision=LoopDecision.NOT_VERIFIED,
            terminal_reason=LoopTerminalReason.NOT_VERIFIED,
            final_answer="SECRET_ANSWER",
            metadata=(guardrail_metadata if metadata_location == "run" else {}),
        )
    )

    with pytest.raises(PublicProjectionError, match="malformed guardrail decision"):
        report.to_public_dict()


def test_public_projection_rejects_after_run_guardrail_mismatch():
    report = LoopReport(
        run=LoopRun(
            run_id="run_after_guardrail",
            user_input="SECRET_PROMPT",
            context_provider="none",
            backend="SECRET_BACKEND",
            model_label="SECRET_MODEL",
            final_decision=LoopDecision.ERROR,
            terminal_reason=LoopTerminalReason.ERROR,
            final_answer="SECRET_ANSWER",
            metadata={"after_run_guardrail": True},
        )
    )

    with pytest.raises(PublicProjectionError, match="after-run guardrail"):
        report.to_public_dict()


def test_public_projection_rejects_after_run_marker_without_post_final_guardrail():
    report = LoopReport(
        run=LoopRun(
            run_id="run_missing_after_guardrail_step",
            user_input="question",
            context_provider="none",
            backend="mock",
            model_label="MockLLM",
            started_at=PUBLIC_PROJECTION_AT,
            completed_at=PUBLIC_PROJECTION_AT,
            steps=(
                LoopStep(
                    step_id="step_before_missing_after_guardrail",
                    phase=LoopPhase.INPUT,
                    decision=LoopDecision.BLOCK,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                    error_message="blocked",
                ),
                LoopStep(
                    step_id="step_missing_after_guardrail_final",
                    phase=LoopPhase.FINAL,
                    decision=LoopDecision.BLOCK,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                ),
            ),
            final_decision=LoopDecision.BLOCK,
            terminal_reason=LoopTerminalReason.BLOCKED,
            final_answer="Blocked.",
            metadata={"after_run_guardrail": True},
        )
    )

    with pytest.raises(PublicProjectionError, match="post-final causal boundary"):
        report.to_public_dict()


def test_public_projection_rejects_guardrail_marker_laundered_on_continue_step():
    report = LoopReport(
        run=LoopRun(
            run_id="run_laundered_guardrail_marker",
            user_input="question",
            context_provider="none",
            backend="mock",
            model_label="MockLLM",
            started_at=PUBLIC_PROJECTION_AT,
            completed_at=PUBLIC_PROJECTION_AT,
            steps=(
                LoopStep(
                    step_id="step_laundered_block_marker",
                    phase=LoopPhase.DRAFT,
                    decision=LoopDecision.CONTINUE,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                    metadata={"guardrail_decision": "block"},
                ),
                LoopStep(
                    step_id="step_laundered_block_final",
                    phase=LoopPhase.FINAL,
                    decision=LoopDecision.BLOCK,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                ),
            ),
            final_decision=LoopDecision.BLOCK,
            terminal_reason=LoopTerminalReason.BLOCKED,
            final_answer="Blocked.",
        )
    )

    with pytest.raises(PublicProjectionError, match="guardrail marker"):
        report.to_public_dict()


def test_public_projection_rejects_mismatched_canonical_guardrail_reason():
    report = LoopReport(
        run=LoopRun(
            run_id="run_mismatched_guardrail_reason",
            user_input="question",
            context_provider="none",
            backend="mock",
            model_label="MockLLM",
            started_at=PUBLIC_PROJECTION_AT,
            completed_at=PUBLIC_PROJECTION_AT,
            steps=(
                LoopStep(
                    step_id="step_mismatched_guardrail_reason",
                    phase=LoopPhase.ERROR,
                    decision=LoopDecision.BLOCK,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                    output_summary="different output",
                    error_message="different error",
                    metadata={
                        "guardrail_decision": "block",
                        "guardrail_reason": "canonical reason",
                    },
                ),
                LoopStep(
                    step_id="step_mismatched_guardrail_reason_final",
                    phase=LoopPhase.FINAL,
                    decision=LoopDecision.BLOCK,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                ),
            ),
            final_decision=LoopDecision.BLOCK,
            terminal_reason=LoopTerminalReason.BLOCKED,
            final_answer="Blocked.",
        )
    )

    with pytest.raises(PublicProjectionError, match="guardrail reason"):
        report.to_public_dict()


def _supported_report(
    *,
    backend="ollama",
    policy=None,
    evidence=True,
    verify_decision=LoopDecision.SUPPORTED,
    verification_outcome=VerificationOutcome.SUPPORTED,
    verifier_backend="ollama",
    verifier_model_label="Model",
):
    answer = "Supported answer [1]." if evidence else "Supported answer."
    candidate_digest = answer_candidate_sha256(answer)
    evidence_reference = EvidenceReference.from_source(
        citation_id=1,
        provider="document",
        source_identity="source.txt",
        page=None,
        chunk_index=0,
        excerpt="Supported evidence.",
    )
    evidence_digest = evidence_set_sha256(
        (evidence_reference,) if evidence else ()
    )
    verification = (
        VerificationResult(
            outcome=verification_outcome,
            verifier_backend=verifier_backend,
            verifier_model_label=verifier_model_label,
        )
        if verification_outcome is not None
        else None
    )
    steps = [
        LoopStep(
            step_id="step_supported_draft",
            phase=LoopPhase.DRAFT,
            decision=LoopDecision.CONTINUE,
            started_at=PUBLIC_PROJECTION_AT,
            ended_at=PUBLIC_PROJECTION_AT,
            backend=backend,
            model_label="Model",
            metadata={
                ANSWER_CANDIDATE_SHA256_METADATA_KEY: candidate_digest,
                EVIDENCE_SET_SHA256_METADATA_KEY: evidence_digest,
            },
        )
    ]
    if verify_decision is not None:
        steps.append(
            LoopStep(
                step_id="step_supported_verify",
                phase=LoopPhase.VERIFY,
                decision=verify_decision,
                started_at=PUBLIC_PROJECTION_AT,
                ended_at=PUBLIC_PROJECTION_AT,
                backend=backend,
                model_label="Model",
                verification=verification,
                metadata={
                    ANSWER_CANDIDATE_SHA256_METADATA_KEY: candidate_digest,
                    EVIDENCE_SET_SHA256_METADATA_KEY: evidence_digest,
                },
            ),
        )
    steps.append(
        LoopStep(
            step_id="step_supported_final",
            phase=LoopPhase.FINAL,
            decision=LoopDecision.SUPPORTED,
            started_at=PUBLIC_PROJECTION_AT,
            ended_at=PUBLIC_PROJECTION_AT,
            backend=backend,
            model_label="Model",
            metadata={
                ANSWER_CANDIDATE_SHA256_METADATA_KEY: candidate_digest,
                EVIDENCE_SET_SHA256_METADATA_KEY: evidence_digest,
            },
        )
    )
    return LoopReport(
        run=LoopRun(
            run_id="run_supported_contract",
            user_input="question",
            context_provider="document",
            backend=backend,
            model_label="Model",
            policy=policy or LoopPolicy(),
            started_at=PUBLIC_PROJECTION_AT,
            completed_at=PUBLIC_PROJECTION_AT,
            steps=tuple(steps),
            evidence=(evidence_reference,) if evidence else (),
            final_decision=LoopDecision.SUPPORTED,
            terminal_reason=LoopTerminalReason.COMPLETED,
            final_answer=answer,
        )
    )


def _visible_report(
    *,
    answer="Answer",
    decision=LoopDecision.NOT_VERIFIED,
    evidence=(),
    metadata=None,
):
    started_at = PUBLIC_PROJECTION_AT
    answer_digest = answer_candidate_sha256(answer)
    evidence_digest = evidence_set_sha256(tuple(evidence))
    steps = [
        LoopStep(
            step_id="step_visible_draft",
            phase=LoopPhase.DRAFT,
            decision=LoopDecision.CONTINUE,
            started_at=started_at,
            ended_at=started_at,
            backend="mock",
            model_label="MockLLM",
            metadata={
                ANSWER_CANDIDATE_SHA256_METADATA_KEY: answer_digest,
                EVIDENCE_SET_SHA256_METADATA_KEY: evidence_digest,
            },
        )
    ]
    if decision == LoopDecision.NOT_VERIFIED:
        steps.append(
            LoopStep(
                step_id="step_visible_verify",
                phase=LoopPhase.VERIFY,
                decision=LoopDecision.NOT_VERIFIED,
                started_at=started_at,
                ended_at=started_at,
                backend="mock",
                model_label="MockLLM",
                verification=VerificationResult(
                    outcome=VerificationOutcome.NOT_VERIFIED,
                ),
                metadata={
                    ANSWER_CANDIDATE_SHA256_METADATA_KEY: answer_digest,
                    EVIDENCE_SET_SHA256_METADATA_KEY: evidence_digest,
                },
            )
        )
    steps.append(
        LoopStep(
            step_id="step_visible_final",
            phase=LoopPhase.FINAL,
            decision=decision,
            started_at=started_at,
            ended_at=started_at,
            backend="mock",
            model_label="MockLLM",
            metadata={
                ANSWER_CANDIDATE_SHA256_METADATA_KEY: answer_digest,
                EVIDENCE_SET_SHA256_METADATA_KEY: evidence_digest,
            },
        )
    )
    return LoopReport(
        run=LoopRun(
            run_id="run_visible_contract",
            user_input="question",
            context_provider="document" if evidence else "none",
            backend="mock",
            model_label="MockLLM",
            started_at=started_at,
            completed_at=started_at,
            steps=tuple(steps),
            evidence=tuple(evidence),
            final_decision=decision,
            terminal_reason=(
                LoopTerminalReason.COMPLETED
                if decision == LoopDecision.FINAL
                else LoopTerminalReason.NOT_VERIFIED
            ),
            final_answer=answer,
            metadata=dict(metadata or {}),
        )
    )


def test_public_projection_rejects_stale_failed_verifier_terminal_cause():
    old_answer = "Old answer"
    final_answer = "Supported retry answer"
    empty_evidence_digest = evidence_set_sha256(())
    old_digest = answer_candidate_sha256(old_answer)
    final_digest = answer_candidate_sha256(final_answer)
    failed_verify = LoopStep(
        step_id="step_stale_failure_verify",
        phase=LoopPhase.VERIFY,
        decision=LoopDecision.NOT_VERIFIED,
        started_at=PUBLIC_PROJECTION_AT,
        ended_at=PUBLIC_PROJECTION_AT,
        retry_count=0,
        verification=VerificationResult(outcome=VerificationOutcome.UNSUPPORTED),
        metadata={
            ANSWER_CANDIDATE_SHA256_METADATA_KEY: old_digest,
            EVIDENCE_SET_SHA256_METADATA_KEY: empty_evidence_digest,
            "reasons": ["llm_verifier_unsupported"],
        },
    )
    steps = (
        LoopStep(
            step_id="step_stale_failure_draft",
            phase=LoopPhase.DRAFT,
            decision=LoopDecision.CONTINUE,
            started_at=PUBLIC_PROJECTION_AT,
            ended_at=PUBLIC_PROJECTION_AT,
            retry_count=0,
            metadata={
                ANSWER_CANDIDATE_SHA256_METADATA_KEY: old_digest,
                EVIDENCE_SET_SHA256_METADATA_KEY: empty_evidence_digest,
            },
        ),
        failed_verify,
        LoopStep(
            step_id="step_stale_failure_retry",
            phase=LoopPhase.RETRY,
            decision=LoopDecision.RETRY,
            started_at=PUBLIC_PROJECTION_AT,
            ended_at=PUBLIC_PROJECTION_AT,
            retry_count=1,
            metadata={
                "retry_reason": "web_verifier",
                "reasons": ["llm_verifier_unsupported"],
                "retry_count": 1,
                "max_retries": 1,
                "retry_trigger_step_id": failed_verify.step_id,
            },
        ),
        LoopStep(
            step_id="step_current_supported_draft",
            phase=LoopPhase.DRAFT,
            decision=LoopDecision.CONTINUE,
            started_at=PUBLIC_PROJECTION_AT,
            ended_at=PUBLIC_PROJECTION_AT,
            retry_count=1,
            metadata={
                ANSWER_CANDIDATE_SHA256_METADATA_KEY: final_digest,
                EVIDENCE_SET_SHA256_METADATA_KEY: empty_evidence_digest,
            },
        ),
        LoopStep(
            step_id="step_current_supported_verify",
            phase=LoopPhase.VERIFY,
            decision=LoopDecision.SUPPORTED,
            started_at=PUBLIC_PROJECTION_AT,
            ended_at=PUBLIC_PROJECTION_AT,
            retry_count=1,
            verification=VerificationResult(outcome=VerificationOutcome.SUPPORTED),
            metadata={
                ANSWER_CANDIDATE_SHA256_METADATA_KEY: final_digest,
                EVIDENCE_SET_SHA256_METADATA_KEY: empty_evidence_digest,
            },
        ),
        LoopStep(
            step_id="step_stale_failure_refuse",
            phase=LoopPhase.REFUSE,
            decision=LoopDecision.REFUSE,
            started_at=PUBLIC_PROJECTION_AT,
            ended_at=PUBLIC_PROJECTION_AT,
            retry_count=1,
        ),
        LoopStep(
            step_id="step_stale_failure_final",
            phase=LoopPhase.FINAL,
            decision=LoopDecision.REFUSE,
            started_at=PUBLIC_PROJECTION_AT,
            ended_at=PUBLIC_PROJECTION_AT,
            retry_count=1,
        ),
    )
    report = LoopReport(
        run=LoopRun(
            run_id="run_stale_failure_cause",
            user_input="question",
            context_provider="none",
            backend="mock",
            model_label="MockLLM",
            policy=LoopPolicy(max_retries=1),
            started_at=PUBLIC_PROJECTION_AT,
            completed_at=PUBLIC_PROJECTION_AT,
            steps=steps,
            final_decision=LoopDecision.REFUSE,
            terminal_reason=LoopTerminalReason.VERIFICATION_FAILED,
            final_answer="Refused.",
        )
    )

    with pytest.raises(PublicProjectionError, match="current failed verifier"):
        report.to_public_dict()


def test_public_projection_rejects_stale_trace_unavailable_terminal_cause():
    old_answer = "Old untraced answer"
    final_answer = "Current traced answer"
    empty_evidence_digest = evidence_set_sha256(())
    old_digest = answer_candidate_sha256(old_answer)
    final_digest = answer_candidate_sha256(final_answer)
    failed_verify = LoopStep(
        step_id="step_old_untraced_verify",
        phase=LoopPhase.VERIFY,
        decision=LoopDecision.NOT_VERIFIED,
        started_at=PUBLIC_PROJECTION_AT,
        ended_at=PUBLIC_PROJECTION_AT,
        retry_count=0,
        verification=VerificationResult(outcome=VerificationOutcome.UNSUPPORTED),
        metadata={
            ANSWER_CANDIDATE_SHA256_METADATA_KEY: old_digest,
            EVIDENCE_SET_SHA256_METADATA_KEY: empty_evidence_digest,
            "trace_available": False,
            "verifier_skipped": True,
            "reasons": ["llm_verifier_unsupported"],
        },
    )
    report = LoopReport(
        run=LoopRun(
            run_id="run_stale_trace_cause",
            user_input="question",
            context_provider="none",
            backend="mock",
            model_label="MockLLM",
            policy=LoopPolicy(max_retries=1),
            started_at=PUBLIC_PROJECTION_AT,
            completed_at=PUBLIC_PROJECTION_AT,
            steps=(
                LoopStep(
                    step_id="step_old_untraced_draft",
                    phase=LoopPhase.DRAFT,
                    decision=LoopDecision.CONTINUE,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                    retry_count=0,
                    metadata={
                        ANSWER_CANDIDATE_SHA256_METADATA_KEY: old_digest,
                        EVIDENCE_SET_SHA256_METADATA_KEY: empty_evidence_digest,
                        "trace_available": False,
                    },
                ),
                failed_verify,
                LoopStep(
                    step_id="step_trace_retry",
                    phase=LoopPhase.RETRY,
                    decision=LoopDecision.RETRY,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                    retry_count=1,
                    metadata={
                        "retry_reason": "web_verifier",
                        "reasons": ["llm_verifier_unsupported"],
                        "retry_count": 1,
                        "max_retries": 1,
                        "retry_trigger_step_id": failed_verify.step_id,
                    },
                ),
                LoopStep(
                    step_id="step_current_traced_draft",
                    phase=LoopPhase.DRAFT,
                    decision=LoopDecision.CONTINUE,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                    backend="mock",
                    model_label="MockLLM",
                    retry_count=1,
                    metadata={
                        ANSWER_CANDIDATE_SHA256_METADATA_KEY: final_digest,
                        EVIDENCE_SET_SHA256_METADATA_KEY: empty_evidence_digest,
                    },
                ),
                LoopStep(
                    step_id="step_current_traced_verify",
                    phase=LoopPhase.VERIFY,
                    decision=LoopDecision.NOT_VERIFIED,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                    backend="mock",
                    model_label="MockLLM",
                    retry_count=1,
                    verification=VerificationResult(
                        outcome=VerificationOutcome.NOT_VERIFIED
                    ),
                    metadata={
                        ANSWER_CANDIDATE_SHA256_METADATA_KEY: final_digest,
                        EVIDENCE_SET_SHA256_METADATA_KEY: empty_evidence_digest,
                    },
                ),
                LoopStep(
                    step_id="step_stale_trace_final",
                    phase=LoopPhase.FINAL,
                    decision=LoopDecision.NOT_VERIFIED,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                    backend="mock",
                    model_label="MockLLM",
                    retry_count=1,
                    metadata={
                        ANSWER_CANDIDATE_SHA256_METADATA_KEY: final_digest,
                        EVIDENCE_SET_SHA256_METADATA_KEY: empty_evidence_digest,
                    },
                ),
            ),
            final_decision=LoopDecision.NOT_VERIFIED,
            terminal_reason=LoopTerminalReason.TRACE_UNAVAILABLE,
            final_answer=final_answer,
        )
    )

    with pytest.raises(PublicProjectionError, match="trace_unavailable"):
        report.to_public_dict()


def test_public_projection_rejects_stale_error_terminal_cause():
    report = _visible_report(decision=LoopDecision.NOT_VERIFIED)
    current_steps = tuple(
        replace(
            step,
            decision=(
                LoopDecision.SUPPORTED
                if step.phase == LoopPhase.VERIFY
                else LoopDecision.ERROR
                if step.phase == LoopPhase.FINAL
                else step.decision
            ),
            verification=(
                VerificationResult(outcome=VerificationOutcome.SUPPORTED)
                if step.phase == LoopPhase.VERIFY
                else step.verification
            ),
        )
        for step in report.run.steps
    )
    stale_error = LoopStep(
        step_id="step_stale_recovered_error",
        phase=LoopPhase.ERROR,
        decision=LoopDecision.ERROR,
        started_at=PUBLIC_PROJECTION_AT,
        ended_at=PUBLIC_PROJECTION_AT,
        error_message="recovered_error",
    )
    report = replace(
        report,
        run=replace(
            report.run,
            steps=(stale_error, *current_steps),
            final_decision=LoopDecision.ERROR,
            terminal_reason=LoopTerminalReason.ERROR,
            error_message="terminal_error",
        ),
    )

    with pytest.raises(PublicProjectionError, match="current causal failed step"):
        report.to_public_dict()


def test_public_projection_rejects_resolved_retry_denial_terminal_cause():
    answer = "Recovered answer"
    answer_digest = answer_candidate_sha256(answer)
    evidence_digest = evidence_set_sha256(())
    report = LoopReport(
        run=LoopRun(
            run_id="run_resolved_retry_denial",
            user_input="question",
            context_provider="none",
            backend="mock",
            model_label="MockLLM",
            policy=LoopPolicy(max_retries=0),
            started_at=PUBLIC_PROJECTION_AT,
            completed_at=PUBLIC_PROJECTION_AT,
            steps=(
                LoopStep(
                    step_id="step_resolved_denial_draft",
                    phase=LoopPhase.DRAFT,
                    decision=LoopDecision.CONTINUE,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                    metadata={
                        ANSWER_CANDIDATE_SHA256_METADATA_KEY: answer_digest,
                        EVIDENCE_SET_SHA256_METADATA_KEY: evidence_digest,
                    },
                ),
                LoopStep(
                    step_id="step_resolved_retry_denial",
                    phase=LoopPhase.MECHANICAL_CHECK,
                    decision=LoopDecision.NOT_VERIFIED,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                    output_summary="needs_retry",
                    metadata={
                        "reasons": ["missing_inline_citations"],
                        "retry_denied": True,
                        "retry_unavailable": False,
                        "retry_budget_exhausted": True,
                    },
                ),
                LoopStep(
                    step_id="step_after_resolved_denial_verify",
                    phase=LoopPhase.VERIFY,
                    decision=LoopDecision.SUPPORTED,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                    verification=VerificationResult(
                        outcome=VerificationOutcome.SUPPORTED
                    ),
                    metadata={
                        ANSWER_CANDIDATE_SHA256_METADATA_KEY: answer_digest,
                        EVIDENCE_SET_SHA256_METADATA_KEY: evidence_digest,
                    },
                ),
                LoopStep(
                    step_id="step_resolved_denial_final",
                    phase=LoopPhase.FINAL,
                    decision=LoopDecision.ERROR,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                ),
            ),
            final_decision=LoopDecision.ERROR,
            terminal_reason=LoopTerminalReason.RETRY_BUDGET_EXHAUSTED,
            final_answer="Error.",
        )
    )

    with pytest.raises(PublicProjectionError, match="stale or resolved retry denial"):
        report.to_public_dict()


@pytest.mark.parametrize(
    ("verify_decision", "verification_outcome"),
    [
        (None, None),
        (LoopDecision.CONTINUE, VerificationOutcome.SUPPORTED),
        (LoopDecision.SUPPORTED, VerificationOutcome.UNSUPPORTED),
    ],
)
def test_public_projection_rejects_unsupported_verifier_provenance(
    verify_decision,
    verification_outcome,
):
    with pytest.raises(
        PublicProjectionError,
        match="(supported|step|visible answer) contract",
    ):
        _supported_report(
            verify_decision=verify_decision,
            verification_outcome=verification_outcome,
        ).to_public_dict()


def test_public_projection_rejects_supported_without_required_evidence():
    with pytest.raises(PublicProjectionError, match="context evidence is required"):
        _supported_report(evidence=False).to_public_dict()


def test_public_projection_rejects_no_context_final_completion():
    with pytest.raises(PublicProjectionError, match="no-context visible completion"):
        _visible_report(decision=LoopDecision.FINAL).to_public_dict()


def test_public_projection_rejects_no_context_evidence():
    evidence = EvidenceReference.from_source(
        citation_id=1,
        provider="document",
        source_identity="source.txt",
        page=None,
        chunk_index=0,
        excerpt="Evidence that cannot belong to a no-context run.",
    )
    report = _visible_report(
        answer="Unverified answer [1].",
        evidence=(evidence,),
    )
    report = replace(
        report,
        run=replace(report.run, context_provider="none"),
    )

    with pytest.raises(PublicProjectionError, match="no-context run contains evidence"):
        report.to_public_dict()


def test_public_projection_allows_no_context_not_verified_completion():
    public_run = _visible_report().to_public_dict()["run"]

    assert public_run["context_provider"] == "none"
    assert public_run["final_decision"] == "not_verified"
    assert public_run["evidence"] == []


@pytest.mark.parametrize(
    "model_label",
    [
        "\u0080",
        "\u009f",
        "\u0301",
        "\u0488",
        "\u0903",
        "\ufe0f",
        "\U000e0100",
        "Model\u0080",
    ],
)
def test_public_projection_rejects_unsafe_or_baseless_model_label(
    model_label,
):
    report = _visible_report()
    report = replace(
        report,
        run=replace(
            report.run,
            model_label=model_label,
            steps=tuple(
                replace(step, model_label=model_label)
                for step in report.run.steps
            ),
        ),
    )

    with pytest.raises(PublicProjectionError, match="display label"):
        report.to_public_dict()


def test_public_projection_allows_model_label_with_base_and_unicode_marks():
    model_label = "Model\u0301\ufe0f\U000e0100"
    report = _visible_report()
    report = replace(
        report,
        run=replace(
            report.run,
            model_label=model_label,
            steps=tuple(
                replace(step, model_label=model_label)
                for step in report.run.steps
            ),
        ),
    )

    assert report.to_public_dict()["run"]["model_label"] == model_label


def test_public_projection_rejects_mock_supported_regardless_of_policy_flag():
    with pytest.raises(PublicProjectionError, match="concrete real drafter"):
        _supported_report(
            backend="mock",
            verifier_backend="mock",
        ).to_public_dict()


@pytest.mark.parametrize("verifier_backend", [None, "", "   "])
def test_public_projection_requires_real_verifier_backend_provenance(
    verifier_backend,
):
    with pytest.raises(
        PublicProjectionError,
        match="(verifier provenance|real verifier backend provenance)",
    ):
        _supported_report(verifier_backend=verifier_backend).to_public_dict()


def test_public_projection_does_not_honor_supported_policy_exceptions():
    report = _supported_report(
        backend="mock",
        verifier_backend="mock",
        policy=LoopPolicy(
            require_citations=False,
            require_verifier_for_supported=False,
            allow_mock_supported=True,
        ),
    )

    with pytest.raises(PublicProjectionError, match="concrete real drafter"):
        report.to_public_dict()


def test_public_projection_rejects_supported_without_terminal_final_step():
    report = _supported_report()
    report = replace(report, run=replace(report.run, steps=report.run.steps[:-1]))

    with pytest.raises(PublicProjectionError, match="terminal final step"):
        report.to_public_dict()


def test_public_projection_rejects_draft_after_supported_verification():
    report = _supported_report()
    replacement = LoopStep(
        step_id="step_unverified_replacement",
        phase=LoopPhase.DRAFT,
        decision=LoopDecision.CONTINUE,
        started_at=PUBLIC_PROJECTION_AT,
        metadata={
            ANSWER_CANDIDATE_SHA256_METADATA_KEY: answer_candidate_sha256(
                report.run.final_answer
            ),
            EVIDENCE_SET_SHA256_METADATA_KEY: evidence_set_sha256(
                report.run.evidence
            ),
        },
    )
    report = replace(
        report,
        run=replace(
            report.run,
            steps=(*report.run.steps[:-1], replacement, report.run.steps[-1]),
        ),
    )

    with pytest.raises(
        PublicProjectionError,
        match="(final candidate was not verified|new draft lacks a recorded retry)",
    ):
        report.to_public_dict()


def test_public_projection_rejects_final_answer_with_unknown_citation():
    report = _supported_report()
    hostile_answer = "Unsupported citation [999]."
    hostile_digest = answer_candidate_sha256(hostile_answer)
    bound_steps = tuple(
        replace(
            step,
            metadata={
                **dict(step.metadata),
                ANSWER_CANDIDATE_SHA256_METADATA_KEY: hostile_digest,
            },
        )
        for step in report.run.steps
    )
    report = replace(
        report,
        run=replace(
            report.run,
            steps=bound_steps,
            final_answer=hostile_answer,
        ),
    )

    with pytest.raises(PublicProjectionError, match="cites unknown evidence"):
        report.to_public_dict()


def test_public_projection_rejects_swapped_supported_evidence_set():
    report = _supported_report()
    swapped = EvidenceReference.from_source(
        citation_id=1,
        provider="document",
        source_identity="different.txt",
        page=None,
        chunk_index=0,
        excerpt="Different evidence.",
    )
    report = replace(report, run=replace(report.run, evidence=(swapped,)))

    with pytest.raises(PublicProjectionError, match="final evidence set"):
        report.to_public_dict()


@pytest.mark.parametrize("decision", [LoopDecision.FINAL, LoopDecision.NOT_VERIFIED])
def test_public_projection_rejects_swapped_visible_evidence_set(decision):
    original = EvidenceReference.from_source(
        citation_id=1,
        provider="document",
        source_identity="original.txt",
        page=None,
        chunk_index=0,
        excerpt="Original evidence.",
    )
    swapped = EvidenceReference.from_source(
        citation_id=1,
        provider="document",
        source_identity="swapped.txt",
        page=None,
        chunk_index=0,
        excerpt="Swapped evidence.",
    )
    report = _visible_report(
        answer="Visible answer [1].",
        decision=decision,
        evidence=(original,),
    )
    report = replace(report, run=replace(report.run, evidence=(swapped,)))

    with pytest.raises(PublicProjectionError, match="final evidence set"):
        report.to_public_dict()


def test_public_projection_rejects_not_verified_evidence_rebound_only_at_final():
    original = EvidenceReference.from_source(
        citation_id=1,
        provider="document",
        source_identity="original.txt",
        page=None,
        chunk_index=0,
        excerpt="Original evidence.",
    )
    swapped = EvidenceReference.from_source(
        citation_id=1,
        provider="document",
        source_identity="swapped.txt",
        page=None,
        chunk_index=0,
        excerpt="Swapped evidence.",
    )
    report = _visible_report(answer="Visible answer [1].", evidence=(original,))
    final_step = replace(
        report.run.steps[-1],
        metadata={
            **dict(report.run.steps[-1].metadata),
            EVIDENCE_SET_SHA256_METADATA_KEY: evidence_set_sha256((swapped,)),
        },
    )
    report = replace(
        report,
        run=replace(
            report.run,
            steps=(*report.run.steps[:-1], final_step),
            evidence=(swapped,),
        ),
    )

    with pytest.raises(PublicProjectionError, match="final evidence set"):
        report.to_public_dict()


def test_public_projection_rejects_not_verified_answer_rebound_only_at_final():
    report = _visible_report(answer="Original answer")
    hostile_answer = "Hostile replacement"
    final_step = replace(
        report.run.steps[-1],
        metadata={
            **dict(report.run.steps[-1].metadata),
            ANSWER_CANDIDATE_SHA256_METADATA_KEY: answer_candidate_sha256(
                hostile_answer
            ),
        },
    )
    report = replace(
        report,
        run=replace(
            report.run,
            steps=(*report.run.steps[:-1], final_step),
            final_answer=hostile_answer,
        ),
    )

    with pytest.raises(PublicProjectionError, match="answer binding contract"):
        report.to_public_dict()


@pytest.mark.parametrize("step_index", [0, 1, 2])
def test_public_projection_requires_evidence_binding_on_supported_path(step_index):
    report = _supported_report()
    steps = list(report.run.steps)
    step = steps[step_index]
    metadata = dict(step.metadata)
    metadata.pop(EVIDENCE_SET_SHA256_METADATA_KEY)
    steps[step_index] = replace(step, metadata=metadata)
    report = replace(report, run=replace(report.run, steps=tuple(steps)))

    with pytest.raises(PublicProjectionError, match="final evidence set"):
        report.to_public_dict()


def test_public_projection_rejects_retry_budget_bypass():
    report = _visible_report()
    retries = tuple(
        LoopStep(
            step_id=f"step_retry_{number}",
            phase=LoopPhase.RETRY,
            decision=LoopDecision.RETRY,
            started_at=PUBLIC_PROJECTION_AT,
            retry_count=number,
        )
        for number in (1, 2)
    )
    report = replace(
        report,
        run=replace(
            report.run,
            policy=LoopPolicy(max_retries=1),
            steps=(*retries, *report.run.steps),
        ),
    )

    with pytest.raises(PublicProjectionError, match="retry budget exceeded"):
        report.to_public_dict()


def test_public_projection_rejects_new_supported_draft_without_retry():
    report = _supported_report(policy=LoopPolicy(max_retries=0))
    old_draft = replace(
        report.run.steps[0],
        step_id="step_old_draft",
    )
    old_verify = LoopStep(
        step_id="step_old_verify",
        phase=LoopPhase.VERIFY,
        decision=LoopDecision.NOT_VERIFIED,
        started_at=PUBLIC_PROJECTION_AT,
        backend="ollama",
        model_label="Model",
        verification=VerificationResult(
            outcome=VerificationOutcome.UNSUPPORTED,
            verifier_backend="ollama",
            verifier_model_label="Model",
        ),
        metadata=dict(report.run.steps[1].metadata),
    )
    report = replace(
        report,
        run=replace(
            report.run,
            steps=(old_draft, old_verify, *report.run.steps),
        ),
    )

    with pytest.raises(PublicProjectionError, match="new draft lacks a recorded retry"):
        report.to_public_dict()


def test_public_projection_rejects_retry_without_new_draft_candidate():
    report = _visible_report()
    trigger = LoopStep(
        step_id="step_unconsumed_retry_trigger",
        phase=LoopPhase.FORMAT_CHECK,
        decision=LoopDecision.RETRY,
        started_at=PUBLIC_PROJECTION_AT,
        metadata={"reasons": ["compact_ordered_list"]},
        output_summary="needs_retry",
    )
    retry = LoopStep(
        step_id="step_unconsumed_retry",
        phase=LoopPhase.RETRY,
        decision=LoopDecision.RETRY,
        started_at=PUBLIC_PROJECTION_AT,
        retry_count=1,
        metadata={
            "retry_reason": "format_check",
            "reasons": ["compact_ordered_list"],
            "retry_count": 1,
            "max_retries": 1,
            "retry_trigger_step_id": trigger.step_id,
        },
    )
    report = replace(
        report,
        run=replace(
            report.run,
            policy=LoopPolicy(max_retries=1),
            steps=(
                trigger,
                retry,
                *(
                    replace(step, retry_count=1)
                    for step in report.run.steps[1:]
                ),
            ),
        ),
    )

    with pytest.raises(PublicProjectionError, match="retry has no new draft candidate"):
        report.to_public_dict()


def test_public_projection_rejects_two_requests_spending_one_retry_record():
    report = _visible_report()
    format_request = LoopStep(
        step_id="step_double_format_request",
        phase=LoopPhase.FORMAT_CHECK,
        decision=LoopDecision.RETRY,
        started_at=PUBLIC_PROJECTION_AT,
        output_summary="needs_retry",
        metadata={"reasons": ["compact_ordered_list"]},
    )
    mechanical_request = LoopStep(
        step_id="step_double_mechanical_request",
        phase=LoopPhase.MECHANICAL_CHECK,
        decision=LoopDecision.RETRY,
        started_at=PUBLIC_PROJECTION_AT,
        output_summary="needs_retry",
        metadata={"reasons": ["missing_inline_citations"]},
    )
    retry = LoopStep(
        step_id="step_double_retry",
        phase=LoopPhase.RETRY,
        decision=LoopDecision.RETRY,
        started_at=PUBLIC_PROJECTION_AT,
        retry_count=1,
        metadata={
            "retry_reason": "self_check",
            "reasons": ["missing_inline_citations"],
            "retry_count": 1,
            "max_retries": 1,
            "retry_trigger_step_id": mechanical_request.step_id,
        },
    )
    retry_steps = tuple(replace(step, retry_count=1) for step in report.run.steps)
    report = replace(
        report,
        run=replace(
            report.run,
            policy=LoopPolicy(max_retries=1),
            steps=(format_request, mechanical_request, retry, *retry_steps),
        ),
    )

    with pytest.raises(PublicProjectionError, match="multiple retry requests"):
        report.to_public_dict()


def test_public_projection_rejects_retry_token_theft_from_pending_request():
    report = _visible_report()
    pending_request = LoopStep(
        step_id="step_pending_format_request",
        phase=LoopPhase.FORMAT_CHECK,
        decision=LoopDecision.RETRY,
        started_at=PUBLIC_PROJECTION_AT,
        output_summary="needs_retry",
        metadata={"reasons": ["compact_ordered_list"]},
    )
    search_error = LoopStep(
        step_id="step_interposed_search_error",
        phase=LoopPhase.ERROR,
        decision=LoopDecision.ERROR,
        started_at=PUBLIC_PROJECTION_AT,
        output_summary="web_search_failed",
        metadata={"reasons": ["web_search_failed"]},
    )
    retry = LoopStep(
        step_id="step_stolen_retry_token",
        phase=LoopPhase.RETRY,
        decision=LoopDecision.RETRY,
        started_at=PUBLIC_PROJECTION_AT,
        retry_count=1,
        metadata={
            "retry_reason": "smart_web_direct_fallback",
            "reasons": ["web_search_failed"],
            "retry_count": 1,
            "max_retries": 1,
            "retry_trigger_step_id": search_error.step_id,
        },
    )
    retry_steps = tuple(replace(step, retry_count=1) for step in report.run.steps)
    report = replace(
        report,
        run=replace(
            report.run,
            policy=LoopPolicy(max_retries=1),
            steps=(pending_request, search_error, retry, *retry_steps),
        ),
    )

    with pytest.raises(PublicProjectionError, match="pending request"):
        report.to_public_dict()


def _retry_aborted_by_terminal_guardrail_report():
    return LoopReport(
        run=LoopRun(
            run_id="run_retry_guardrail_abort",
            user_input="question",
            context_provider="none",
            backend="mock",
            model_label="MockLLM",
            policy=LoopPolicy(max_retries=1),
            started_at=PUBLIC_PROJECTION_AT,
            steps=(
                LoopStep(
                    step_id="step_guarded_retry_trigger",
                    phase=LoopPhase.FORMAT_CHECK,
                    decision=LoopDecision.RETRY,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                    output_summary="needs_retry",
                    metadata={"reasons": ["compact_ordered_list"]},
                ),
                LoopStep(
                    step_id="step_guarded_retry",
                    phase=LoopPhase.RETRY,
                    decision=LoopDecision.RETRY,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                    retry_count=1,
                    metadata={
                        "retry_reason": "format_check",
                        "reasons": ["compact_ordered_list"],
                        "retry_count": 1,
                        "max_retries": 1,
                        "retry_trigger_step_id": "step_guarded_retry_trigger",
                    },
                ),
                LoopStep(
                    step_id="step_retry_blocked",
                    phase=LoopPhase.ERROR,
                    decision=LoopDecision.BLOCK,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                    retry_count=1,
                    metadata={"guardrail_decision": "block"},
                ),
                LoopStep(
                    step_id="step_retry_blocked_final",
                    phase=LoopPhase.FINAL,
                    decision=LoopDecision.BLOCK,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                    retry_count=1,
                ),
            ),
            final_decision=LoopDecision.BLOCK,
            terminal_reason=LoopTerminalReason.BLOCKED,
            final_answer="Blocked.",
            completed_at=PUBLIC_PROJECTION_AT,
        )
    )



def test_public_projection_allows_retry_aborted_by_terminal_guardrail():
    payload = _retry_aborted_by_terminal_guardrail_report().to_public_dict()

    assert payload["public_redaction"]["applied"] is True
    assert payload["run"]["final_answer"] is None


@pytest.mark.parametrize(
    "stale_step_id",
    ["step_retry_blocked", "step_retry_blocked_final"],
)
def test_public_projection_rejects_stale_post_retry_terminal_steps(stale_step_id):
    report = _retry_aborted_by_terminal_guardrail_report()
    hostile_steps = tuple(
        replace(step, retry_count=0) if step.step_id == stale_step_id else step
        for step in report.run.steps
    )
    report = replace(report, run=replace(report.run, steps=hostile_steps))

    with pytest.raises(PublicProjectionError, match="current retry epoch"):
        report.to_public_dict()


def test_public_projection_rejects_execution_after_guardrail_without_final_step():
    report = LoopReport(
        run=LoopRun(
            run_id="run_guardrail_continuation_without_final",
            user_input="question",
            context_provider="none",
            backend="mock",
            model_label="MockLLM",
            started_at=PUBLIC_PROJECTION_AT,
            steps=(
                LoopStep(
                    step_id="step_terminal_guardrail",
                    phase=LoopPhase.ERROR,
                    decision=LoopDecision.BLOCK,
                    started_at=PUBLIC_PROJECTION_AT,
                    metadata={"guardrail_decision": "block"},
                ),
                LoopStep(
                    step_id="step_illegal_post_guardrail_draft",
                    phase=LoopPhase.DRAFT,
                    decision=LoopDecision.CONTINUE,
                    started_at=PUBLIC_PROJECTION_AT,
                ),
            ),
            final_decision=LoopDecision.BLOCK,
            terminal_reason=LoopTerminalReason.BLOCKED,
            final_answer="Blocked.",
        )
    )

    with pytest.raises(PublicProjectionError, match="execution continued"):
        report.to_public_dict()


def test_public_projection_identity_shortcut_rejects_interposed_check_step():
    answer = "The indexed file is `roadmap.txt`."
    report = LoopReport(
        run=LoopRun(
            run_id="run_forged_identity_check",
            user_input="Which file is indexed?",
            context_provider="document",
            backend="mock",
            model_label="MockLLM",
            started_at=PUBLIC_PROJECTION_AT,
            steps=(
                LoopStep(
                    step_id="step_forged_identity_check",
                    phase=LoopPhase.MECHANICAL_CHECK,
                    decision=LoopDecision.CONTINUE,
                    started_at=PUBLIC_PROJECTION_AT,
                ),
                LoopStep(
                    step_id="step_forged_identity_final",
                    phase=LoopPhase.FINAL,
                    decision=LoopDecision.FINAL,
                    started_at=PUBLIC_PROJECTION_AT,
                    backend="mock",
                    model_label="MockLLM",
                    metadata={
                        "identity_answer": True,
                        ANSWER_CANDIDATE_SHA256_METADATA_KEY: (
                            answer_candidate_sha256(answer)
                        ),
                        EVIDENCE_SET_SHA256_METADATA_KEY: evidence_set_sha256(()),
                    },
                ),
            ),
            final_decision=LoopDecision.FINAL,
            terminal_reason=LoopTerminalReason.COMPLETED,
            final_answer=answer,
            metadata={
                "document_name": "roadmap.txt",
                "identity_answer": True,
            },
        )
    )

    with pytest.raises(PublicProjectionError, match="lacks a bound candidate"):
        report.to_public_dict()


def test_public_projection_rejects_retry_after_final_supported_verifier():
    report = _supported_report()
    retry = LoopStep(
        step_id="step_retry_after_verify",
        phase=LoopPhase.RETRY,
        decision=LoopDecision.RETRY,
        started_at=PUBLIC_PROJECTION_AT,
        retry_count=1,
    )
    report = replace(
        report,
        run=replace(
            report.run,
            steps=(*report.run.steps[:-1], retry, report.run.steps[-1]),
        ),
    )

    with pytest.raises(
        PublicProjectionError,
        match="(unresolved retry|retry has no new draft candidate|retry trigger)",
    ):
        report.to_public_dict()


def test_public_projection_rejects_auto_supported_backend():
    report = _supported_report(
        backend="auto",
        verifier_backend="auto",
        verifier_model_label="Model",
    )

    with pytest.raises(PublicProjectionError, match="concrete real drafter"):
        report.to_public_dict()


def test_public_projection_rejects_supported_candidate_provenance_mismatch():
    report = _supported_report()
    hostile_draft = replace(
        report.run.steps[0],
        backend="mock",
        model_label="MockLLM",
    )
    report = replace(
        report,
        run=replace(
            report.run,
            steps=(hostile_draft, *report.run.steps[1:]),
        ),
    )

    with pytest.raises(PublicProjectionError, match="provenance does not match"):
        report.to_public_dict()


def test_public_projection_rejects_supported_final_provenance_mismatch():
    report = _supported_report()
    hostile_final = replace(
        report.run.steps[-1],
        backend="mock",
        model_label="MockLLM",
    )
    report = replace(
        report,
        run=replace(
            report.run,
            steps=(*report.run.steps[:-1], hostile_final),
        ),
    )

    with pytest.raises(PublicProjectionError, match="final provenance"):
        report.to_public_dict()


@pytest.mark.parametrize("report_factory", [_visible_report, _supported_report])
def test_public_projection_rejects_unresolved_failed_check_after_candidate(
    report_factory,
):
    report = report_factory()
    failed_check = LoopStep(
        step_id="step_unresolved_candidate_check",
        phase=LoopPhase.FORMAT_CHECK,
        decision=LoopDecision.ERROR,
        started_at=PUBLIC_PROJECTION_AT,
        ended_at=PUBLIC_PROJECTION_AT,
        output_summary="needs_retry",
        metadata={
            "reasons": ["compact_ordered_list"],
            "retry_denied": True,
            "retry_unavailable": True,
            "retry_budget_exhausted": False,
        },
    )
    report = replace(
        report,
        run=replace(
            report.run,
            steps=(report.run.steps[0], failed_check, *report.run.steps[1:]),
        ),
    )

    with pytest.raises(PublicProjectionError, match="unresolved step"):
        report.to_public_dict()


def test_public_projection_rejects_supported_without_real_draft_phase():
    report = _supported_report()
    hostile_candidate = replace(
        report.run.steps[0],
        phase=LoopPhase.FORMAT_CHECK,
        metadata={
            **dict(report.run.steps[0].metadata),
            "sanitized_internal_labels": True,
        },
    )
    report = replace(
        report,
        run=replace(
            report.run,
            steps=(hostile_candidate, *report.run.steps[1:]),
        ),
    )

    with pytest.raises(
        PublicProjectionError,
        match="(draft candidate is required|sanitized candidate lacks)",
    ):
        report.to_public_dict()


def _sanitized_not_verified_report():
    raw_answer = "Answer marked not_verified"
    sanitized_answer = "Answer marked not verified"
    raw_digest = answer_candidate_sha256(raw_answer)
    report = _visible_report(answer=sanitized_answer)
    source_draft = replace(
        report.run.steps[0],
        step_id="step_sanitizer_source_draft",
        metadata={
            ANSWER_CANDIDATE_SHA256_METADATA_KEY: raw_digest,
            EVIDENCE_SET_SHA256_METADATA_KEY: evidence_set_sha256(()),
        },
    )
    format_step = LoopStep(
        step_id="step_sanitizer_source_format",
        phase=LoopPhase.FORMAT_CHECK,
        decision=LoopDecision.ERROR,
        started_at=PUBLIC_PROJECTION_AT,
        ended_at=PUBLIC_PROJECTION_AT,
        backend="mock",
        model_label="MockLLM",
        output_summary="needs_retry",
        metadata={
            "reasons": ["internal_verification_label"],
            "retry_denied": True,
            "retry_unavailable": True,
            "retry_budget_exhausted": False,
        },
    )
    sanitizer_step = LoopStep(
        step_id="step_sanitizer_candidate",
        phase=LoopPhase.FORMAT_CHECK,
        decision=LoopDecision.CONTINUE,
        started_at=PUBLIC_PROJECTION_AT,
        ended_at=PUBLIC_PROJECTION_AT,
        backend="mock",
        model_label="MockLLM",
        output_summary="format_sanitized",
        metadata={
            "reasons": ["internal_verification_label"],
            "sanitized_internal_labels": True,
            "sanitized_from_answer_sha256": raw_digest,
            "resolved_format_step_id": format_step.step_id,
            "format_resolution": "deterministic_format_sanitizer",
            ANSWER_CANDIDATE_SHA256_METADATA_KEY: answer_candidate_sha256(
                sanitized_answer
            ),
            EVIDENCE_SET_SHA256_METADATA_KEY: evidence_set_sha256(()),
        },
    )
    return replace(
        report,
        run=replace(
            report.run,
            steps=(
                source_draft,
                format_step,
                sanitizer_step,
                *report.run.steps[1:],
            ),
        ),
    )


def test_public_projection_accepts_exact_deterministic_sanitizer_lineage():
    report = _sanitized_not_verified_report()

    assert report.to_public_dict()["run"]["final_answer"] == report.run.final_answer


def test_public_projection_rejects_sanitizer_after_passing_format_check():
    report = _sanitized_not_verified_report()
    source_format = replace(report.run.steps[1], output_summary="format_passed")
    report = replace(
        report,
        run=replace(
            report.run,
            steps=(report.run.steps[0], source_format, *report.run.steps[2:]),
        ),
    )

    with pytest.raises(PublicProjectionError, match="sanitized candidate lacks"):
        report.to_public_dict()


def test_public_projection_rejects_forged_sanitizer_budget_exhaustion():
    report = _sanitized_not_verified_report()
    source_format = report.run.steps[1]
    source_format = replace(
        source_format,
        metadata={
            **dict(source_format.metadata),
            "retry_unavailable": False,
            "retry_budget_exhausted": True,
        },
    )
    report = replace(
        report,
        run=replace(
            report.run,
            steps=(report.run.steps[0], source_format, *report.run.steps[2:]),
        ),
    )

    with pytest.raises(PublicProjectionError, match="sanitized candidate lacks"):
        report.to_public_dict()


def test_public_projection_rejects_noop_sanitizer_candidate():
    report = _sanitized_not_verified_report()
    final_digest = answer_candidate_sha256(report.run.final_answer)
    source_draft = replace(
        report.run.steps[0],
        metadata={
            **dict(report.run.steps[0].metadata),
            ANSWER_CANDIDATE_SHA256_METADATA_KEY: final_digest,
        },
    )
    sanitizer = replace(
        report.run.steps[2],
        metadata={
            **dict(report.run.steps[2].metadata),
            "sanitized_from_answer_sha256": final_digest,
        },
    )
    report = replace(
        report,
        run=replace(
            report.run,
            steps=(
                source_draft,
                report.run.steps[1],
                sanitizer,
                *report.run.steps[3:],
            ),
        ),
    )

    with pytest.raises(PublicProjectionError, match="sanitized candidate lacks"):
        report.to_public_dict()


def test_public_projection_requires_sanitizer_output_digest_under_redaction():
    report = _sanitized_not_verified_report()
    sanitizer = report.run.steps[2]
    sanitizer_metadata = dict(sanitizer.metadata)
    sanitizer_metadata.pop(ANSWER_CANDIDATE_SHA256_METADATA_KEY)
    sanitizer = replace(sanitizer, metadata=sanitizer_metadata)
    guardrail = LoopStep(
        step_id="step_sanitizer_guardrail",
        phase=LoopPhase.ERROR,
        decision=LoopDecision.BLOCK,
        started_at=PUBLIC_PROJECTION_AT,
        metadata={"guardrail_decision": "block"},
    )
    terminal = LoopStep(
        step_id="step_sanitizer_guardrail_final",
        phase=LoopPhase.FINAL,
        decision=LoopDecision.BLOCK,
        started_at=PUBLIC_PROJECTION_AT,
    )
    report = replace(
        report,
        run=replace(
            report.run,
            steps=(
                report.run.steps[0],
                report.run.steps[1],
                sanitizer,
                guardrail,
                terminal,
            ),
            final_decision=LoopDecision.BLOCK,
            terminal_reason=LoopTerminalReason.BLOCKED,
            final_answer="Blocked.",
        ),
    )

    with pytest.raises(PublicProjectionError, match="sanitized candidate lacks"):
        report.to_public_dict()


def test_public_projection_rejects_sanitized_candidate_after_failed_verifier():
    report = _supported_report(policy=LoopPolicy(max_retries=0))
    evidence_digest = evidence_set_sha256(report.run.evidence)
    old_answer_digest = answer_candidate_sha256("Old unsupported answer [1].")
    old_draft = replace(
        report.run.steps[0],
        metadata={
            ANSWER_CANDIDATE_SHA256_METADATA_KEY: old_answer_digest,
            EVIDENCE_SET_SHA256_METADATA_KEY: evidence_digest,
        },
    )
    old_verify = LoopStep(
        step_id="step_old_sanitizer_verify",
        phase=LoopPhase.VERIFY,
        decision=LoopDecision.NOT_VERIFIED,
        started_at=PUBLIC_PROJECTION_AT,
        backend="ollama",
        model_label="Model",
        verification=VerificationResult(
            outcome=VerificationOutcome.UNSUPPORTED,
            verifier_backend="ollama",
            verifier_model_label="Model",
        ),
        metadata={
            ANSWER_CANDIDATE_SHA256_METADATA_KEY: old_answer_digest,
            EVIDENCE_SET_SHA256_METADATA_KEY: evidence_digest,
        },
    )
    forged_resolved_format = LoopStep(
        step_id="step_forged_resolved_format",
        phase=LoopPhase.FORMAT_CHECK,
        decision=LoopDecision.CONTINUE,
        started_at=PUBLIC_PROJECTION_AT,
        backend="ollama",
        model_label="Model",
        metadata={
            "retry_resolved_without_retry": True,
            "retry_resolution": "deterministic_format_sanitizer",
        },
    )
    hostile_sanitized = replace(
        report.run.steps[0],
        step_id="step_hostile_sanitized_candidate",
        phase=LoopPhase.FORMAT_CHECK,
        metadata={
            **dict(report.run.steps[0].metadata),
            "sanitized_internal_labels": True,
            "sanitized_from_answer_sha256": old_answer_digest,
        },
    )
    report = replace(
        report,
        run=replace(
            report.run,
            steps=(
                old_draft,
                old_verify,
                forged_resolved_format,
                hostile_sanitized,
                report.run.steps[1],
                report.run.steps[2],
            ),
        ),
    )

    with pytest.raises(PublicProjectionError, match="sanitized candidate lacks"):
        report.to_public_dict()


def test_public_projection_rejects_failed_supported_draft_candidate():
    report = _supported_report()
    hostile_draft = replace(
        report.run.steps[0],
        decision=LoopDecision.ERROR,
        error_message="draft_failed",
    )
    report = replace(
        report,
        run=replace(
            report.run,
            steps=(hostile_draft, *report.run.steps[1:]),
        ),
    )

    with pytest.raises(
        PublicProjectionError,
        match="(did not complete successfully|lacks a bound candidate)",
    ):
        report.to_public_dict()


def test_public_projection_rejects_contradictory_verifier_identity():
    report = _supported_report()
    verify_step = report.run.steps[1]
    hostile_verification = replace(verify_step.verification, verifier="mock")
    verify_step = replace(verify_step, verification=hostile_verification)
    report = replace(
        report,
        run=replace(
            report.run,
            steps=(report.run.steps[0], verify_step, report.run.steps[2]),
        ),
    )

    with pytest.raises(PublicProjectionError, match="verifier identity"):
        report.to_public_dict()


def test_public_projection_rejects_contradictory_verify_step_provenance():
    report = _supported_report()
    verify_step = replace(
        report.run.steps[1],
        backend="openai-compatible",
        model_label="OtherModel",
    )
    report = replace(
        report,
        run=replace(
            report.run,
            steps=(report.run.steps[0], verify_step, report.run.steps[2]),
        ),
    )

    with pytest.raises(PublicProjectionError, match="verify-step provenance"):
        report.to_public_dict()


def test_public_projection_rejects_verifier_overlapping_bound_candidate():
    report = _supported_report()
    draft = replace(
        report.run.steps[0],
        started_at=PUBLIC_PROJECTION_AT,
        ended_at=PUBLIC_PROJECTION_AT + timedelta(seconds=10),
    )
    verify = replace(
        report.run.steps[1],
        started_at=PUBLIC_PROJECTION_AT + timedelta(seconds=1),
        ended_at=PUBLIC_PROJECTION_AT + timedelta(seconds=2),
    )
    final = replace(
        report.run.steps[2],
        started_at=PUBLIC_PROJECTION_AT + timedelta(seconds=11),
        ended_at=PUBLIC_PROJECTION_AT + timedelta(seconds=12),
    )
    report = replace(
        report,
        run=replace(
            report.run,
            steps=(draft, verify, final),
            completed_at=PUBLIC_PROJECTION_AT + timedelta(seconds=12),
        ),
    )

    with pytest.raises(PublicProjectionError, match="(candidate completed|overlap)"):
        report.to_public_dict()


def test_public_projection_rejects_final_before_verifier_completion():
    report = _supported_report()
    draft = replace(
        report.run.steps[0],
        started_at=PUBLIC_PROJECTION_AT,
        ended_at=PUBLIC_PROJECTION_AT + timedelta(seconds=1),
    )
    verify = replace(
        report.run.steps[1],
        started_at=PUBLIC_PROJECTION_AT + timedelta(seconds=2),
        ended_at=PUBLIC_PROJECTION_AT + timedelta(seconds=10),
    )
    final = replace(
        report.run.steps[2],
        started_at=PUBLIC_PROJECTION_AT + timedelta(seconds=3),
        ended_at=PUBLIC_PROJECTION_AT + timedelta(seconds=4),
    )
    report = replace(
        report,
        run=replace(
            report.run,
            steps=(draft, verify, final),
            completed_at=PUBLIC_PROJECTION_AT + timedelta(seconds=10),
        ),
    )

    with pytest.raises(PublicProjectionError, match="overlap"):
        report.to_public_dict()


def test_public_projection_rejects_retry_before_trigger_completion():
    report = _visible_report()
    initial_draft = replace(
        report.run.steps[0],
        step_id="step_temporal_initial_draft",
        started_at=PUBLIC_PROJECTION_AT,
        ended_at=PUBLIC_PROJECTION_AT,
    )
    trigger = LoopStep(
        step_id="step_temporal_retry_trigger",
        phase=LoopPhase.FORMAT_CHECK,
        decision=LoopDecision.RETRY,
        started_at=PUBLIC_PROJECTION_AT,
        ended_at=PUBLIC_PROJECTION_AT + timedelta(seconds=10),
        output_summary="needs_retry",
        metadata={"reasons": ["compact_ordered_list"]},
    )
    retry = LoopStep(
        step_id="step_temporal_retry",
        phase=LoopPhase.RETRY,
        decision=LoopDecision.RETRY,
        started_at=PUBLIC_PROJECTION_AT + timedelta(seconds=1),
        ended_at=PUBLIC_PROJECTION_AT + timedelta(seconds=2),
        retry_count=1,
        metadata={
            "retry_reason": "format_check",
            "reasons": ["compact_ordered_list"],
            "retry_count": 1,
            "max_retries": 1,
            "retry_trigger_step_id": trigger.step_id,
        },
    )
    retry_tail = tuple(
        replace(
            step,
            retry_count=1,
            started_at=PUBLIC_PROJECTION_AT + timedelta(seconds=11),
            ended_at=PUBLIC_PROJECTION_AT + timedelta(seconds=11),
        )
        for step in report.run.steps
    )
    report = replace(
        report,
        run=replace(
            report.run,
            policy=LoopPolicy(max_retries=1),
            steps=(initial_draft, trigger, retry, *retry_tail),
            completed_at=PUBLIC_PROJECTION_AT + timedelta(seconds=12),
        ),
    )

    with pytest.raises(PublicProjectionError, match="overlap"):
        report.to_public_dict()


@pytest.mark.parametrize("decision", [LoopDecision.FINAL, LoopDecision.NOT_VERIFIED])
def test_public_projection_binds_every_visible_final_answer(decision):
    report = _visible_report(decision=decision)
    if decision == LoopDecision.FINAL:
        report = replace(
            report,
            run=replace(report.run, context_provider="document"),
        )
    final_step = replace(report.run.steps[-1], metadata={})
    report = replace(report, run=replace(report.run, steps=(final_step,)))

    with pytest.raises(PublicProjectionError, match="answer binding contract"):
        report.to_public_dict()


def test_public_projection_rejects_unknown_citation_on_not_verified_answer():
    report = _visible_report(answer="Unverified claim [1].")

    with pytest.raises(PublicProjectionError, match="cites unknown evidence"):
        report.to_public_dict()


def test_public_projection_rejects_unbounded_inline_citation_integer():
    report = _visible_report(answer=f"Claim [{10 ** 40}].")

    with pytest.raises(PublicProjectionError, match="inline citation"):
        report.to_public_dict()


def test_answer_candidate_digest_hashes_exact_utf8_bytes():
    assert answer_candidate_sha256("answer") != answer_candidate_sha256(" answer")
    assert answer_candidate_sha256("answer") != answer_candidate_sha256("answer ")


def test_public_projection_rejects_unavailable_middleware_retry_marker():
    report = LoopReport(
        run=LoopRun(
            run_id="run_middleware_retry",
            user_input="question",
            context_provider="none",
            backend="mock",
            model_label="MockLLM",
            started_at=PUBLIC_PROJECTION_AT,
            steps=(
                LoopStep(
                    step_id="step_middleware_retry",
                    phase=LoopPhase.ERROR,
                    decision=LoopDecision.RETRY,
                    started_at=PUBLIC_PROJECTION_AT,
                    metadata={"guardrail_decision": "retry"},
                ),
            ),
            final_decision=LoopDecision.NOT_VERIFIED,
            terminal_reason=LoopTerminalReason.NOT_VERIFIED,
            final_answer="Answer",
        )
    )

    with pytest.raises(PublicProjectionError, match="terminal step"):
        report.to_public_dict()


def test_public_projection_allows_causally_bound_budgeted_retry_step():
    answer = "Answer"
    report = _visible_report(answer=answer)
    trigger = LoopStep(
        step_id="step_budgeted_retry_trigger",
        phase=LoopPhase.MECHANICAL_CHECK,
        decision=LoopDecision.RETRY,
        started_at=PUBLIC_PROJECTION_AT,
        ended_at=PUBLIC_PROJECTION_AT,
        output_summary="needs_retry",
        metadata={"reasons": ["missing_inline_citations"]},
    )
    retry = LoopStep(
        step_id="step_budgeted_retry",
        phase=LoopPhase.RETRY,
        decision=LoopDecision.RETRY,
        started_at=PUBLIC_PROJECTION_AT,
        ended_at=PUBLIC_PROJECTION_AT,
        retry_count=1,
        metadata={
            "retry_reason": "self_check",
            "reasons": ["missing_inline_citations"],
            "retry_count": 1,
            "max_retries": 1,
            "retry_trigger_step_id": trigger.step_id,
        },
    )
    retry_steps = tuple(
        replace(step, retry_count=1)
        for step in report.run.steps
    )
    initial_draft = replace(
        report.run.steps[0],
        step_id="step_budgeted_initial_draft",
    )
    report = replace(
        report,
        run=replace(
            report.run,
            policy=LoopPolicy(max_retries=1),
            steps=(initial_draft, trigger, retry, *retry_steps),
        ),
    )

    assert report.to_public_dict()["run"]["final_answer"] == "Answer"


def test_public_projection_rejects_uncausal_retry_budget_terminal_reason():
    refuse_step = LoopStep(
        step_id="step_uncausal_budget_refuse",
        phase=LoopPhase.REFUSE,
        decision=LoopDecision.REFUSE,
        started_at=PUBLIC_PROJECTION_AT,
        ended_at=PUBLIC_PROJECTION_AT,
    )
    report = LoopReport(
        run=LoopRun(
            run_id="run_uncausal_budget",
            user_input="question",
            context_provider="none",
            backend="mock",
            model_label="MockLLM",
            policy=LoopPolicy(max_retries=1),
            started_at=PUBLIC_PROJECTION_AT,
            completed_at=PUBLIC_PROJECTION_AT,
            steps=(refuse_step,),
            final_decision=LoopDecision.REFUSE,
            terminal_reason=LoopTerminalReason.RETRY_BUDGET_EXHAUSTED,
            final_answer="Refused.",
        )
    )

    with pytest.raises(PublicProjectionError, match="causal denied retry"):
        report.to_public_dict()


def test_public_projection_rejects_retry_denial_flags_on_non_requesting_step():
    report = LoopReport(
        run=LoopRun(
            run_id="run_false_retry_denial_flags",
            user_input="question",
            context_provider="none",
            backend="mock",
            model_label="MockLLM",
            policy=LoopPolicy(max_retries=0),
            started_at=PUBLIC_PROJECTION_AT,
            completed_at=PUBLIC_PROJECTION_AT,
            steps=(
                LoopStep(
                    step_id="step_false_retry_denial_input",
                    phase=LoopPhase.INPUT,
                    decision=LoopDecision.CONTINUE,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                    metadata={
                        "reasons": ["forged_retry_request"],
                        "retry_denied": True,
                        "retry_unavailable": False,
                        "retry_budget_exhausted": True,
                    },
                ),
                LoopStep(
                    step_id="step_false_retry_denial_refuse",
                    phase=LoopPhase.REFUSE,
                    decision=LoopDecision.REFUSE,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                ),
                LoopStep(
                    step_id="step_false_retry_denial_final",
                    phase=LoopPhase.FINAL,
                    decision=LoopDecision.REFUSE,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                ),
            ),
            final_decision=LoopDecision.REFUSE,
            terminal_reason=LoopTerminalReason.RETRY_BUDGET_EXHAUSTED,
            final_answer="Refused.",
        )
    )

    with pytest.raises(PublicProjectionError, match="causal denied retry"):
        report.to_public_dict()


def test_public_projection_rejects_orphan_retry_requesting_check():
    report = LoopReport(
        run=LoopRun(
            run_id="run_orphan_retry_request",
            user_input="question",
            context_provider="none",
            backend="mock",
            model_label="MockLLM",
            policy=LoopPolicy(max_retries=0),
            started_at=PUBLIC_PROJECTION_AT,
            completed_at=PUBLIC_PROJECTION_AT,
            steps=(
                LoopStep(
                    step_id="step_orphan_retry_request",
                    phase=LoopPhase.FORMAT_CHECK,
                    decision=LoopDecision.ERROR,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                    output_summary="needs_retry",
                    metadata={
                        "reasons": ["compact_ordered_list"],
                        "retry_denied": True,
                        "retry_unavailable": False,
                        "retry_budget_exhausted": True,
                    },
                ),
                LoopStep(
                    step_id="step_orphan_retry_refuse",
                    phase=LoopPhase.REFUSE,
                    decision=LoopDecision.REFUSE,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                ),
                LoopStep(
                    step_id="step_orphan_retry_final",
                    phase=LoopPhase.FINAL,
                    decision=LoopDecision.REFUSE,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                ),
            ),
            final_decision=LoopDecision.REFUSE,
            terminal_reason=LoopTerminalReason.RETRY_BUDGET_EXHAUSTED,
            final_answer="Refused.",
        )
    )

    with pytest.raises(PublicProjectionError, match="causal denied retry"):
        report.to_public_dict()


def test_public_projection_rejects_failed_attempt_detached_from_retry_token():
    initial_answer = "Initial answer"
    recovered_answer = "Recovered answer"
    empty_evidence_digest = evidence_set_sha256(())
    trigger = LoopStep(
        step_id="step_detached_failure_trigger",
        phase=LoopPhase.FORMAT_CHECK,
        decision=LoopDecision.RETRY,
        started_at=PUBLIC_PROJECTION_AT,
        ended_at=PUBLIC_PROJECTION_AT,
        output_summary="needs_retry",
        metadata={"reasons": ["compact_ordered_list"]},
    )
    retry = LoopStep(
        step_id="step_detached_failure_retry",
        phase=LoopPhase.RETRY,
        decision=LoopDecision.RETRY,
        started_at=PUBLIC_PROJECTION_AT,
        ended_at=PUBLIC_PROJECTION_AT,
        retry_count=1,
        metadata={
            "retry_reason": "format_check",
            "reasons": ["compact_ordered_list"],
            "retry_count": 1,
            "max_retries": 1,
            "retry_trigger_step_id": trigger.step_id,
        },
    )
    report = LoopReport(
        run=LoopRun(
            run_id="run_detached_failed_retry_attempt",
            user_input="question",
            context_provider="none",
            backend="mock",
            model_label="MockLLM",
            policy=LoopPolicy(max_retries=1),
            started_at=PUBLIC_PROJECTION_AT,
            completed_at=PUBLIC_PROJECTION_AT,
            steps=(
                LoopStep(
                    step_id="step_detached_failure_initial_draft",
                    phase=LoopPhase.DRAFT,
                    decision=LoopDecision.CONTINUE,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                    metadata={
                        ANSWER_CANDIDATE_SHA256_METADATA_KEY: (
                            answer_candidate_sha256(initial_answer)
                        ),
                        EVIDENCE_SET_SHA256_METADATA_KEY: empty_evidence_digest,
                    },
                ),
                trigger,
                retry,
                LoopStep(
                    step_id="step_detached_failure_recovered_draft",
                    phase=LoopPhase.DRAFT,
                    decision=LoopDecision.CONTINUE,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                    retry_count=1,
                    metadata={
                        ANSWER_CANDIDATE_SHA256_METADATA_KEY: (
                            answer_candidate_sha256(recovered_answer)
                        ),
                        EVIDENCE_SET_SHA256_METADATA_KEY: empty_evidence_digest,
                    },
                ),
                LoopStep(
                    step_id="step_detached_failure_late_error",
                    phase=LoopPhase.ERROR,
                    decision=LoopDecision.ERROR,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                    retry_count=1,
                    output_summary="web_search_failed",
                    error_message="web_search_failed",
                    metadata={
                        "planned_phase": LoopPhase.DRAFT.value,
                        "reasons": ["web_search_failed"],
                    },
                ),
                LoopStep(
                    step_id="step_detached_failure_final",
                    phase=LoopPhase.FINAL,
                    decision=LoopDecision.ERROR,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                    retry_count=1,
                    metadata={
                        "retry_denied": True,
                        "retry_unavailable": False,
                        "retry_budget_exhausted": True,
                    },
                ),
            ),
            final_decision=LoopDecision.ERROR,
            terminal_reason=LoopTerminalReason.RETRY_BUDGET_EXHAUSTED,
            final_answer="Error.",
            error_message="web_search_failed",
        )
    )

    with pytest.raises(PublicProjectionError, match="causal denied retry"):
        report.to_public_dict()


def test_public_projection_rejects_terminal_verifier_with_stale_retry_count():
    answer = "Answer"
    report = _visible_report(answer=answer)
    initial_draft = replace(
        report.run.steps[0],
        step_id="step_stale_retry_initial_draft",
    )
    trigger = LoopStep(
        step_id="step_stale_retry_trigger",
        phase=LoopPhase.MECHANICAL_CHECK,
        decision=LoopDecision.RETRY,
        started_at=PUBLIC_PROJECTION_AT,
        output_summary="needs_retry",
        metadata={"reasons": ["missing_inline_citations"]},
    )
    retry = LoopStep(
        step_id="step_stale_retry",
        phase=LoopPhase.RETRY,
        decision=LoopDecision.RETRY,
        started_at=PUBLIC_PROJECTION_AT,
        retry_count=1,
        metadata={
            "retry_reason": "self_check",
            "reasons": ["missing_inline_citations"],
            "retry_count": 1,
            "max_retries": 1,
            "retry_trigger_step_id": trigger.step_id,
        },
    )
    retry_draft = replace(report.run.steps[0], retry_count=1)
    stale_verify = replace(report.run.steps[1], retry_count=0)
    retry_final = replace(report.run.steps[2], retry_count=1)
    report = replace(
        report,
        run=replace(
            report.run,
            policy=LoopPolicy(max_retries=1),
            steps=(
                initial_draft,
                trigger,
                retry,
                retry_draft,
                stale_verify,
                retry_final,
            ),
        ),
    )

    with pytest.raises(PublicProjectionError, match="verifier retry count"):
        report.to_public_dict()


def test_public_projection_rejects_not_verified_candidate_from_foreign_drafter():
    report = _visible_report(answer="Answer")
    foreign_draft = replace(
        report.run.steps[0],
        backend="ollama",
        model_label="Foreign drafter",
    )
    report = replace(
        report,
        run=replace(
            report.run,
            steps=(foreign_draft, *report.run.steps[1:]),
        ),
    )

    with pytest.raises(PublicProjectionError, match="provenance"):
        report.to_public_dict()


def test_public_projection_rejects_sanitized_candidate_from_foreign_source_draft():
    report = _sanitized_not_verified_report()
    foreign_source_draft = replace(
        report.run.steps[0],
        backend="ollama",
        model_label="Foreign drafter",
    )
    report = replace(
        report,
        run=replace(
            report.run,
            steps=(foreign_source_draft, *report.run.steps[1:]),
        ),
    )

    with pytest.raises(PublicProjectionError, match="source draft provenance"):
        report.to_public_dict()


@pytest.mark.parametrize("field_name", ["retry_count", "max_retries"])
def test_public_projection_rejects_boolean_retry_metadata(field_name):
    report = _visible_report(answer="Answer")
    trigger = LoopStep(
        step_id="step_strict_retry_trigger",
        phase=LoopPhase.MECHANICAL_CHECK,
        decision=LoopDecision.RETRY,
        started_at=PUBLIC_PROJECTION_AT,
        ended_at=PUBLIC_PROJECTION_AT,
        output_summary="needs_retry",
        metadata={"reasons": ["missing_inline_citations"]},
    )
    retry = LoopStep(
        step_id="step_strict_retry",
        phase=LoopPhase.RETRY,
        decision=LoopDecision.RETRY,
        started_at=PUBLIC_PROJECTION_AT,
        ended_at=PUBLIC_PROJECTION_AT,
        retry_count=1,
        metadata={
            "retry_reason": "self_check",
            "reasons": ["missing_inline_citations"],
            "retry_count": 1,
            "max_retries": 1,
            "retry_trigger_step_id": trigger.step_id,
        },
    )
    retry = replace(
        retry,
        metadata={**dict(retry.metadata), field_name: True},
    )
    retry_steps = tuple(
        replace(step, retry_count=1) for step in report.run.steps
    )
    initial_draft = replace(
        report.run.steps[0],
        step_id="step_strict_retry_initial_draft",
    )
    report = replace(
        report,
        run=replace(
            report.run,
            policy=LoopPolicy(max_retries=1),
            steps=(initial_draft, trigger, retry, *retry_steps),
        ),
    )

    with pytest.raises(PublicProjectionError, match="retry trigger binding"):
        report.to_public_dict()


def test_public_projection_preserves_only_typed_memory_provenance():
    answer = "Answer"
    report = _visible_report(answer=answer)
    report = replace(
        report,
        run=replace(
            report.run,
            run_id="run_memory_provenance",
            metadata={
                "conversation_context_turns": 2,
                "semantic_memory_turns": 1,
                "semantic_memory_status": "retrieved",
                "semantic_memory_text": "SECRET_MEMORY_TEXT",
            },
        ),
    )

    public_run = report.to_public_dict()["run"]

    assert public_run["conversation_context_count"] == 2
    assert public_run["semantic_memory_count"] == 1
    assert public_run["semantic_memory_status"] == "retrieved"
    assert "SECRET_MEMORY_TEXT" not in json.dumps(public_run)


@pytest.mark.parametrize(
    "metadata",
    [
        {"conversation_context_turns": 1},
        {
            "conversation_context_turns": 0,
            "semantic_memory_turns": 0,
            "semantic_memory_status": "retrieved",
        },
        {
            "conversation_context_turns": 13,
            "semantic_memory_turns": 0,
            "semantic_memory_status": "empty",
        },
        {
            "conversation_context_turns": 0,
            "semantic_memory_turns": 1,
            "semantic_memory_status": "empty",
        },
    ],
)
def test_public_projection_rejects_incoherent_memory_provenance(metadata):
    report = _visible_report(metadata=metadata)

    with pytest.raises(PublicProjectionError, match="memory provenance"):
        report.to_public_dict()


def test_public_projection_rejects_verification_on_nonverify_phase():
    report = _visible_report()
    hostile_step = LoopStep(
        step_id="step_hostile_verification",
        phase=LoopPhase.DRAFT,
        decision=LoopDecision.CONTINUE,
        started_at=PUBLIC_PROJECTION_AT,
        verification=VerificationResult(
            outcome=VerificationOutcome.NOT_VERIFIED,
        ),
    )
    report = replace(
        report,
        run=replace(report.run, steps=(hostile_step, *report.run.steps)),
    )

    with pytest.raises(PublicProjectionError, match="verification has invalid phase"):
        report.to_public_dict()


def test_public_projection_rejects_false_same_model_relationship():
    report = _supported_report()
    verify_step = report.run.steps[1]
    verification = replace(
        verify_step.verification,
        same_model_as_drafter=True,
        verifier_model_label="DifferentModel",
    )
    verify_step = replace(
        verify_step,
        model_label="DifferentModel",
        verification=verification,
    )
    report = replace(
        report,
        run=replace(
            report.run,
            steps=(report.run.steps[0], verify_step, report.run.steps[2]),
        ),
    )

    with pytest.raises(PublicProjectionError, match="same-model verifier"):
        report.to_public_dict()


@pytest.mark.parametrize(
    ("target", "field_name", "hostile_value"),
    [
        ("run", "run_id", "../hostile"),
        ("run", "started_at", datetime(2026, 1, 1)),
        ("policy", "max_retries", 2),
        ("step", "phase", "final"),
        ("step", "decision", "not_verified"),
    ],
)
def test_public_projection_rejects_mutated_record_fields(
    target,
    field_name,
    hostile_value,
):
    report = _visible_report()
    target_object = {
        "run": report.run,
        "policy": report.run.policy,
        "step": report.run.steps[-1],
    }[target]
    object.__setattr__(target_object, field_name, hostile_value)

    with pytest.raises(PublicProjectionError):
        report.to_public_dict()


def test_public_projection_uses_detached_snapshot(monkeypatch):
    report = _visible_report(answer="Snapshot answer")
    original_validator = public_projection_module._validate_public_record_contract

    def mutate_source_while_validating(snapshot):
        object.__setattr__(report.run, "final_answer", "MUTATED AFTER SNAPSHOT")
        original_validator(snapshot)

    monkeypatch.setattr(
        public_projection_module,
        "_validate_public_record_contract",
        mutate_source_while_validating,
    )

    payload = report.to_public_dict()

    assert payload["run"]["final_answer"] == "Snapshot answer"


@pytest.mark.parametrize("provider_location", ("context", "evidence"))
def test_public_projection_rejects_content_bearing_provider_identifiers(
    provider_location,
):
    report = _supported_report()
    if provider_location == "context":
        report = replace(
            report,
            run=replace(report.run, context_provider="https://secret.example/token"),
        )
    else:
        hostile_evidence = replace(
            report.run.evidence[0],
            provider="https://secret.example/token",
        )
        report = replace(
            report,
            run=replace(report.run, evidence=(hostile_evidence,)),
        )

    with pytest.raises(PublicProjectionError, match="provider identifier"):
        report.to_public_dict()


@pytest.mark.parametrize(
    ("target", "field_name", "hostile_value", "error_match"),
    [
        ("reference", "evidence_id", "evidence_not-a-digest", "snapshot"),
        ("reference", "evidence_id", "evidence_" + ("A" * 64), "snapshot"),
        ("reference", "citation_id", True, "citation id"),
        ("reference", "citation_id", 0, "snapshot"),
        ("reference", "citation_id", 1.0, "citation id"),
        (
            "reference",
            "provider",
            "https://secret.example/evidence",
            "provider identifier",
        ),
        ("reference", "provider", None, "provider identifier"),
        (
            "reference",
            "locator",
            {"page": 1, "chunk_index": 0},
            "evidence locator",
        ),
        ("locator", "page", True, "locator page"),
        ("locator", "page", -1, "snapshot"),
        ("locator", "chunk_index", "0", "locator chunk_index"),
        ("locator", "chunk_index", -1, "snapshot"),
    ],
)
def test_public_projection_revalidates_mutated_evidence_fields(
    target,
    field_name,
    hostile_value,
    error_match,
):
    report = _supported_report()
    reference = report.run.evidence[0]
    target_object = reference if target == "reference" else reference.locator
    object.__setattr__(target_object, field_name, hostile_value)

    with pytest.raises(PublicProjectionError, match=error_match):
        report.to_public_dict()


@pytest.mark.parametrize(
    "hostile_evidence",
    [
        lambda reference: [reference],
        lambda _reference: (object(),),
    ],
)
def test_public_projection_revalidates_mutated_evidence_collection(hostile_evidence):
    report = _supported_report()
    object.__setattr__(
        report.run,
        "evidence",
        hostile_evidence(report.run.evidence[0]),
    )

    with pytest.raises(PublicProjectionError, match="public evidence"):
        report.to_public_dict()


def test_public_projection_requires_concrete_evidence_locator_type():
    class HostileEvidenceLocator(EvidenceLocator):
        pass

    report = _supported_report()
    object.__setattr__(
        report.run.evidence[0],
        "locator",
        HostileEvidenceLocator(page=1, chunk_index=0),
    )

    with pytest.raises(PublicProjectionError, match="evidence locator"):
        report.to_public_dict()


def test_loop_run_rejects_duplicate_evidence_identity():
    first = EvidenceReference.from_source(
        citation_id=1,
        provider="document",
        source_identity="same.txt",
        page=1,
        chunk_index=0,
        excerpt="Same evidence.",
    )
    duplicate = EvidenceReference.from_source(
        citation_id=2,
        provider="document",
        source_identity="same.txt",
        page=1,
        chunk_index=0,
        excerpt="Same evidence.",
    )
    with pytest.raises(ValueError, match="duplicate evidence_id"):
        LoopRun(
            run_id="run_duplicate_evidence",
            user_input="question",
            context_provider="document",
            backend="ollama",
            model_label="Model",
            evidence=(first, duplicate),
            final_decision=LoopDecision.NOT_VERIFIED,
            terminal_reason=LoopTerminalReason.NOT_VERIFIED,
            final_answer="answer",
        )


def test_public_projection_suppresses_human_review_body_and_evidence():
    secret = "SECRET_HUMAN_REVIEW_BODY"
    evidence = EvidenceReference.from_source(
        citation_id=1,
        provider="document",
        source_identity=secret,
        page=None,
        chunk_index=0,
        excerpt=secret,
    )
    review = HumanReviewRequest(
        reason=secret,
        instructions=secret,
        created_at=PUBLIC_PROJECTION_AT,
        metadata={"secret": secret},
    )
    report = LoopReport(
        run=LoopRun(
            run_id="run_review",
            user_input=secret,
            context_provider="document",
            backend="openai-compatible",
            model_label=secret,
            steps=(
                LoopStep(
                    phase=LoopPhase.ERROR,
                    decision=LoopDecision.REQUIRES_REVIEW,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                    name=secret,
                    human_review=review,
                ),
            ),
            started_at=PUBLIC_PROJECTION_AT,
            completed_at=PUBLIC_PROJECTION_AT,
            evidence=(evidence,),
            final_decision=LoopDecision.REQUIRES_REVIEW,
            terminal_reason=LoopTerminalReason.HUMAN_REVIEW_REQUIRED,
            final_answer=secret,
        )
    )

    payload = report.to_public_dict()
    step = payload["run"]["steps"][0]

    assert payload["public_redaction"]["applied"] is True
    assert payload["run"]["final_answer"] is None
    assert payload["run"]["evidence"] == []
    assert payload["run"]["backend"] is None
    assert payload["run"]["model_label"] is None
    assert step["human_review_required"] is True
    assert secret not in json.dumps(payload)


def test_public_projection_rejects_requires_review_steps_without_requests():
    review = HumanReviewRequest(
        reason="manual approval required",
        instructions="Review the terminal decision.",
        created_at=PUBLIC_PROJECTION_AT,
    )
    guardrail_step = LoopStep(
        step_id="step_review_guardrail",
        phase=LoopPhase.ERROR,
        decision=LoopDecision.REQUIRES_REVIEW,
        human_review=review,
        started_at=PUBLIC_PROJECTION_AT,
        ended_at=PUBLIC_PROJECTION_AT,
        metadata={"guardrail_decision": "requires_review"},
    )
    final_step = LoopStep(
        step_id="step_review_final",
        phase=LoopPhase.FINAL,
        decision=LoopDecision.REQUIRES_REVIEW,
        human_review=review,
        started_at=PUBLIC_PROJECTION_AT,
        ended_at=PUBLIC_PROJECTION_AT,
    )
    report = LoopReport(
        run=LoopRun(
            run_id="run_mutated_review",
            user_input="question",
            context_provider="none",
            backend="mock",
            model_label="MockLLM",
            started_at=PUBLIC_PROJECTION_AT,
            completed_at=PUBLIC_PROJECTION_AT,
            steps=(
                replace(guardrail_step, human_review=None),
                replace(final_step, human_review=None),
            ),
            final_decision=LoopDecision.REQUIRES_REVIEW,
            terminal_reason=LoopTerminalReason.HUMAN_REVIEW_REQUIRED,
            final_answer="Review required.",
        )
    )

    with pytest.raises(PublicProjectionError, match="human review request"):
        report.to_public_dict()


def test_public_projection_rejects_requires_review_without_causal_step():
    report = LoopReport(
        run=LoopRun(
            run_id="run_review_without_step",
            user_input="question",
            context_provider="none",
            backend="mock",
            model_label="MockLLM",
            started_at=PUBLIC_PROJECTION_AT,
            completed_at=PUBLIC_PROJECTION_AT,
            final_decision=LoopDecision.REQUIRES_REVIEW,
            terminal_reason=LoopTerminalReason.HUMAN_REVIEW_REQUIRED,
            final_answer="Review required.",
        )
    )

    with pytest.raises(PublicProjectionError, match="causal human review"):
        report.to_public_dict()


def test_public_projection_rejects_human_review_created_before_run():
    review = HumanReviewRequest(
        request_id="review_before_run",
        reason="manual approval required",
        instructions="Review the terminal decision.",
        requested_by_step_id="step_review_before_run_guardrail",
        created_at=PUBLIC_PROJECTION_AT - timedelta(days=1),
    )
    report = LoopReport(
        run=LoopRun(
            run_id="run_review_before_start",
            user_input="question",
            context_provider="none",
            backend="mock",
            model_label="MockLLM",
            started_at=PUBLIC_PROJECTION_AT,
            completed_at=PUBLIC_PROJECTION_AT,
            steps=(
                LoopStep(
                    step_id="step_review_before_run_guardrail",
                    phase=LoopPhase.ERROR,
                    decision=LoopDecision.REQUIRES_REVIEW,
                    human_review=review,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                    metadata={"guardrail_decision": "requires_review"},
                ),
                LoopStep(
                    step_id="step_review_before_run_final",
                    phase=LoopPhase.FINAL,
                    decision=LoopDecision.REQUIRES_REVIEW,
                    human_review=review,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                ),
            ),
            final_decision=LoopDecision.REQUIRES_REVIEW,
            terminal_reason=LoopTerminalReason.HUMAN_REVIEW_REQUIRED,
            final_answer="Review required.",
        )
    )

    with pytest.raises(PublicProjectionError, match="human review timestamp"):
        report.to_public_dict()


def test_public_projection_rejects_review_created_after_requesting_step():
    guardrail_started_at = PUBLIC_PROJECTION_AT
    guardrail_ended_at = PUBLIC_PROJECTION_AT + timedelta(seconds=1)
    review_created_at = PUBLIC_PROJECTION_AT + timedelta(seconds=2)
    run_completed_at = PUBLIC_PROJECTION_AT + timedelta(seconds=3)
    review = HumanReviewRequest(
        request_id="review_after_requesting_step",
        reason="manual approval required",
        instructions="Review the terminal decision.",
        requested_by_step_id="step_completed_review_guardrail",
        created_at=review_created_at,
    )
    report = LoopReport(
        run=LoopRun(
            run_id="run_review_after_requesting_step",
            user_input="question",
            context_provider="none",
            backend="mock",
            model_label="MockLLM",
            started_at=guardrail_started_at,
            completed_at=run_completed_at,
            steps=(
                LoopStep(
                    step_id="step_completed_review_guardrail",
                    phase=LoopPhase.ERROR,
                    decision=LoopDecision.REQUIRES_REVIEW,
                    human_review=review,
                    started_at=guardrail_started_at,
                    ended_at=guardrail_ended_at,
                    metadata={"guardrail_decision": "requires_review"},
                ),
                LoopStep(
                    step_id="step_late_review_final",
                    phase=LoopPhase.FINAL,
                    decision=LoopDecision.REQUIRES_REVIEW,
                    human_review=review,
                    started_at=review_created_at,
                    ended_at=run_completed_at,
                ),
            ),
            final_decision=LoopDecision.REQUIRES_REVIEW,
            terminal_reason=LoopTerminalReason.HUMAN_REVIEW_REQUIRED,
            final_answer="Review required.",
        )
    )

    with pytest.raises(PublicProjectionError, match="requesting step"):
        report.to_public_dict()


@pytest.mark.parametrize("divergent_request_points_to_own_step", [False, True])
def test_public_projection_rejects_divergent_repeated_review_identity(
    divergent_request_points_to_own_step,
):
    review = HumanReviewRequest(
        request_id="review_same_identity",
        reason="first review reason",
        instructions="First review instructions.",
        requested_by_step_id="step_same_review_guardrail",
        created_at=PUBLIC_PROJECTION_AT,
        metadata={"source": "guardrail"},
    )
    divergent_review = replace(
        review,
        reason="different review reason",
        instructions="Different review instructions.",
        requested_by_step_id=(
            "step_divergent_review_final"
            if divergent_request_points_to_own_step
            else review.requested_by_step_id
        ),
        metadata={"source": "final"},
    )
    report = LoopReport(
        run=LoopRun(
            run_id="run_divergent_review_identity",
            user_input="question",
            context_provider="none",
            backend="mock",
            model_label="MockLLM",
            started_at=PUBLIC_PROJECTION_AT,
            completed_at=PUBLIC_PROJECTION_AT,
            steps=(
                LoopStep(
                    step_id="step_same_review_guardrail",
                    phase=LoopPhase.ERROR,
                    decision=LoopDecision.REQUIRES_REVIEW,
                    human_review=review,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                    metadata={"guardrail_decision": "requires_review"},
                ),
                LoopStep(
                    step_id="step_divergent_review_final",
                    phase=LoopPhase.FINAL,
                    decision=LoopDecision.REQUIRES_REVIEW,
                    human_review=divergent_review,
                    started_at=PUBLIC_PROJECTION_AT,
                    ended_at=PUBLIC_PROJECTION_AT,
                ),
            ),
            final_decision=LoopDecision.REQUIRES_REVIEW,
            terminal_reason=LoopTerminalReason.HUMAN_REVIEW_REQUIRED,
            final_answer="Review required.",
        )
    )

    with pytest.raises(PublicProjectionError, match="divergent request data"):
        report.to_public_dict()


@pytest.mark.parametrize(
    ("mutation", "error_match"),
    [
        ("divergent", "divergent request data"),
        ("late", "requesting step"),
    ],
)
def test_public_projection_binds_implicit_human_review_source(
    mutation,
    error_match,
):
    source_started_at = PUBLIC_PROJECTION_AT
    source_ended_at = PUBLIC_PROJECTION_AT + timedelta(seconds=1)
    final_started_at = PUBLIC_PROJECTION_AT + timedelta(seconds=2)
    completed_at = PUBLIC_PROJECTION_AT + timedelta(seconds=3)
    review = HumanReviewRequest(
        request_id="review_implicit_source",
        reason="first review reason",
        instructions="Review the terminal decision.",
        requested_by_step_id=None,
        created_at=(
            final_started_at if mutation == "late" else source_started_at
        ),
    )
    final_review = (
        replace(review, reason="different final review reason")
        if mutation == "divergent"
        else review
    )
    report = LoopReport(
        run=LoopRun(
            run_id=f"run_implicit_review_{mutation}",
            user_input="question",
            context_provider="none",
            backend="mock",
            model_label="MockLLM",
            started_at=source_started_at,
            completed_at=completed_at,
            steps=(
                LoopStep(
                    step_id="step_implicit_review_source",
                    phase=LoopPhase.ERROR,
                    decision=LoopDecision.REQUIRES_REVIEW,
                    human_review=review,
                    started_at=source_started_at,
                    ended_at=source_ended_at,
                    metadata={"guardrail_decision": "requires_review"},
                ),
                LoopStep(
                    step_id="step_implicit_review_final",
                    phase=LoopPhase.FINAL,
                    decision=LoopDecision.REQUIRES_REVIEW,
                    human_review=final_review,
                    started_at=final_started_at,
                    ended_at=completed_at,
                ),
            ),
            final_decision=LoopDecision.REQUIRES_REVIEW,
            terminal_reason=LoopTerminalReason.HUMAN_REVIEW_REQUIRED,
            final_answer="Review required.",
        )
    )

    with pytest.raises(PublicProjectionError, match=error_match):
        report.to_public_dict()


def test_public_projection_rejects_completed_run_without_terminal_decision():
    report = LoopReport(
        run=LoopRun(
            run_id="run_completed_without_decision",
            user_input="question",
            context_provider="none",
            backend="mock",
            model_label="MockLLM",
            started_at=PUBLIC_PROJECTION_AT,
            completed_at=PUBLIC_PROJECTION_AT,
        )
    )

    with pytest.raises(PublicProjectionError, match="completion timestamp"):
        report.to_public_dict()


def test_public_projection_rejects_later_step_after_unfinished_predecessor():
    report = LoopReport(
        run=LoopRun(
            run_id="run_step_after_unfinished",
            user_input="question",
            context_provider="none",
            backend="mock",
            model_label="MockLLM",
            started_at=PUBLIC_PROJECTION_AT,
            steps=(
                LoopStep(
                    step_id="step_unfinished_input",
                    phase=LoopPhase.INPUT,
                    decision=LoopDecision.CONTINUE,
                    started_at=PUBLIC_PROJECTION_AT,
                ),
                LoopStep(
                    step_id="step_after_unfinished",
                    phase=LoopPhase.CONTEXT_SELECT,
                    decision=LoopDecision.CONTINUE,
                    started_at=PUBLIC_PROJECTION_AT + timedelta(seconds=1),
                ),
            ),
        )
    )

    with pytest.raises(PublicProjectionError, match="unfinished step"):
        report.to_public_dict()


def test_loop_session_rejects_duplicate_run_identity():
    report = LoopReport(
        run=LoopRun(
            run_id="run_duplicate",
            session_id="thread_duplicate",
            user_input="one",
            context_provider="none",
            backend="mock",
            model_label="MockLLM",
        )
    )

    with pytest.raises(ValueError, match="duplicate run ids"):
        LoopSession(
            session_id="thread_duplicate",
            reports=(report, report),
        )


def test_loop_session_constructor_rejects_unsupported_schema():
    with pytest.raises(ValueError, match="Unsupported loop session schema"):
        LoopSession(schema_version="bogus-session/v999")


def test_loop_report_constructor_rejects_unsupported_schema():
    with pytest.raises(ValueError, match="Unsupported loop report schema"):
        LoopReport(
            run=LoopRun(
                user_input="question",
                context_provider="none",
                backend="mock",
                model_label="MockLLM",
            ),
            schema_version="bogus-report/v999",
        )


@pytest.mark.parametrize("field_name", ["allow_tool_calls", "allow_mock_supported"])
def test_loop_policy_rejects_string_false_booleans(field_name):
    payload = LoopPolicy().to_dict()
    payload[field_name] = "false"

    with pytest.raises(ValueError, match=f"{field_name} must be a JSON boolean"):
        LoopPolicy.from_dict(payload)


def test_loop_policy_constructor_rejects_non_boolean_guardrail_values():
    with pytest.raises(ValueError, match="allow_tool_calls must be a boolean"):
        LoopPolicy(allow_tool_calls="false")


@pytest.mark.parametrize("value", [-1, True, 1.5, "1"])
def test_loop_policy_rejects_invalid_max_retries(value):
    with pytest.raises(ValueError, match="max_retries"):
        LoopPolicy(max_retries=value)

    payload = LoopPolicy().to_dict()
    payload["max_retries"] = value
    with pytest.raises(ValueError, match="max_retries"):
        LoopPolicy.from_dict(payload)


def test_guardrail_decision_round_trips_human_review_request():
    review = HumanReviewRequest(
        request_id="review_guardrail",
        reason="tool output needs approval",
        instructions="Inspect untrusted tool output before continuing.",
        created_at=utc("2026-06-23T12:00:00"),
    )
    decision = GuardrailDecision(
        decision=LoopDecision.REQUIRES_REVIEW,
        reason="untrusted_tool_output",
        human_review=review,
        metadata={"source": "middleware"},
    )

    restored = GuardrailDecision.from_dict(decision.to_dict())

    assert restored == decision
    assert restored.can_continue is False


def test_guardrail_decision_rejects_non_guardrail_outcome():
    with pytest.raises(ValueError, match="supported is not a guardrail decision"):
        GuardrailDecision(decision=LoopDecision.SUPPORTED)


def test_guardrail_decision_rejects_malformed_typed_fields():
    with pytest.raises(ValueError, match="reason must be a string"):
        GuardrailDecision(decision=LoopDecision.BLOCK, reason={"bad": "reason"})

    with pytest.raises(ValueError, match="human_review"):
        GuardrailDecision(decision=LoopDecision.BLOCK, human_review={"bad": True})

    with pytest.raises(ValueError, match="canonical JSON"):
        GuardrailDecision(
            decision=LoopDecision.BLOCK,
            metadata={"opaque": object()},
        )


def test_guardrail_decision_requires_exact_human_review_coherence():
    review = HumanReviewRequest(
        reason="manual approval required",
        instructions="Review the terminal decision.",
    )

    with pytest.raises(ValueError, match="requires a human_review request"):
        GuardrailDecision(decision=LoopDecision.REQUIRES_REVIEW)

    with pytest.raises(ValueError, match="only valid for a requires_review"):
        GuardrailDecision(
            decision=LoopDecision.BLOCK,
            human_review=review,
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {"opaque": object()},
        {"score": float("nan")},
        {"tuple": (1, 2)},
    ],
)
def test_human_review_request_rejects_noncanonical_metadata(metadata):
    with pytest.raises(ValueError, match="canonical JSON"):
        HumanReviewRequest(
            reason="manual approval required",
            instructions="Review the terminal decision.",
            metadata=metadata,
        )


def test_loop_run_add_step_and_complete_are_immutable():
    run = LoopRun(
        run_id="run_1",
        user_input="What changed?",
        context_provider="document",
        backend="mock",
        model_label="MockLLM (explicit demo)",
    )
    step = LoopStep(
        step_id="step_1",
        phase=LoopPhase.DRAFT,
        decision=LoopDecision.NOT_VERIFIED,
        output_summary="drafted answer",
    )

    with_step = run.with_step(step)
    completed = with_step.complete(
        final_decision=LoopDecision.NOT_VERIFIED,
        terminal_reason=LoopTerminalReason.NOT_VERIFIED,
        final_answer="Draft answer [1].",
        metadata={"retry_count": 0},
    )

    assert run.steps == ()
    assert with_step.steps == (step,)
    assert completed.final_decision == LoopDecision.NOT_VERIFIED
    assert completed.terminal_reason == LoopTerminalReason.NOT_VERIFIED
    assert completed.final_answer == "Draft answer [1]."
    assert completed.metadata == {"retry_count": 0}
    assert completed.completed_at is not None


def test_human_review_request_and_policy_keep_safety_boundaries_explicit():
    review = HumanReviewRequest(
        request_id="review_1",
        reason="tool call requires approval",
        instructions="Approve only after checking destination and payload.",
        requested_by_step_id="step_tool",
        created_at=utc("2026-06-23T11:00:00"),
        metadata={"tool": "filesystem_write"},
    )
    step = LoopStep(
        step_id="step_tool",
        phase=LoopPhase.MECHANICAL_CHECK,
        decision=LoopDecision.REQUIRES_REVIEW,
        human_review=review,
    )
    policy = LoopPolicy()

    restored_step = LoopStep.from_dict(step.to_dict())

    assert policy.allow_tool_calls is False
    assert policy.require_human_review_for_tools is True
    assert policy.require_verifier_for_supported is True
    assert restored_step.human_review == review
    assert restored_step.decision == LoopDecision.REQUIRES_REVIEW
