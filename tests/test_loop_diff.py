import io
import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from src import loop_replay
from src.loop_engine import (
    EVIDENCE_SET_SHA256_METADATA_KEY,
    EvidenceReference,
    HumanReviewRequest,
    LoopDecision,
    LoopPhase,
    LoopReport,
    LoopRun,
    LoopStep,
    LoopTerminalReason,
    VerificationOutcome,
    evidence_set_sha256,
)
from test_loop_replay import AT, SECRET, blocked_report, retry_report, sample_report


ROOT = Path(__file__).resolve().parents[1]


def write_artifact(tmp_path, name, *reports):
    path = tmp_path / name
    path.write_text(
        "".join(json.dumps(report.to_dict(), sort_keys=True) + "\n" for report in reports),
        encoding="utf-8",
    )
    return path


def compare(tmp_path, before, after, **options):
    before_path = write_artifact(tmp_path, "before.jsonl", before)
    after_path = write_artifact(tmp_path, "after.jsonl", after)
    return loop_replay.diff_artifacts(before_path, after_path, **options)


def invoke(before_path, after_path, *options):
    output, errors = io.StringIO(), io.StringIO()
    code = loop_replay.main(
        ["diff", str(before_path), str(after_path), *options],
        output_stream=output,
        error_stream=errors,
    )
    return code, output.getvalue(), errors.getvalue()


def bind_evidence(report, evidence):
    """Keep the fixture's verifier bound to the evidence actually compared."""
    return replace(report, run=replace(
        report.run,
        evidence=tuple(evidence),
        steps=tuple(replace(step, metadata={
            **step.metadata,
            EVIDENCE_SET_SHA256_METADATA_KEY: evidence_set_sha256(evidence),
        }) for step in report.run.steps),
    ))


def assert_material_field(payload, field):
    assert payload["has_material_changes"] is True
    assert any(field in change["field"] for change in payload["material_changes"]), payload
    for change in payload["material_changes"]:
        assert {"field", "before", "after", "kind"} <= change.keys()


def test_identical_run_diff_has_versioned_provenance_and_explicit_limits(tmp_path):
    report = sample_report()
    payload = compare(tmp_path, report, report)

    assert payload["schema_version"] == "loop-diff/v1"
    assert payload["public"] is True
    assert payload["same_task_verified"] is False
    assert payload["has_material_changes"] is False
    assert payload["material_changes"] == []
    assert payload["timing_changes"] == []
    for side in ("before", "after"):
        assert payload[side]["source_path"] == str(tmp_path / f"{side}.jsonl")
        assert payload[side]["source_jsonl_line"] == 1
        assert payload[side]["run_id"] == report.run.run_id
        assert payload[side]["session_id"] == report.run.session_id
        assert "raw_report" not in payload[side]
    assert payload["unavailable_fields"]  # The fixture did not record memory provenance.
    assert payload["limitations"]
    assert all(isinstance(item, str) for item in payload["limitations"])
    assert SECRET not in json.dumps(payload)
    assert compare(tmp_path, report, report) == payload


@pytest.mark.parametrize("public", [True, False])
def test_generated_ids_and_absolute_times_do_not_become_material_changes(tmp_path, public):
    before = retry_report()
    shift = timedelta(days=7)
    step_ids = {step.step_id: f"new_{step.step_id}" for step in before.run.steps}
    steps = []
    for step in before.run.steps:
        metadata = dict(step.metadata)
        if "retry_trigger_step_id" in metadata:
            metadata["retry_trigger_step_id"] = step_ids[metadata["retry_trigger_step_id"]]
        steps.append(replace(
            step, step_id=step_ids[step.step_id],
            started_at=step.started_at + shift, ended_at=step.ended_at + shift,
            metadata=metadata,
        ))
    after = replace(before, run=replace(
        before.run, run_id="another_run", session_id="another_session",
        started_at=before.run.started_at + shift,
        completed_at=before.run.completed_at + shift, steps=tuple(steps),
    ))

    payload = compare(tmp_path, before, after, public=public)

    assert payload["has_material_changes"] is False
    assert payload["material_changes"] == []
    assert payload["timing_changes"] == []
    assert payload["same_task_verified"] is False


def test_duration_changes_are_reported_separately(tmp_path):
    before = sample_report()
    after = replace(before, run=replace(
        before.run, completed_at=AT + timedelta(milliseconds=250),
        steps=(*before.run.steps[:-1], replace(
            before.run.steps[-1], ended_at=AT + timedelta(milliseconds=250),
        )),
    ))

    payload = compare(tmp_path, before, after)

    assert payload["has_material_changes"] is False
    assert payload["material_changes"] == []
    assert payload["timing_changes"]
    assert any(change["before"] == 0 and change["after"] == 250
               for change in payload["timing_changes"])


@pytest.mark.parametrize("field,value", [
    ("max_retries", 0),
    ("require_citations", False),
    ("require_verifier_for_supported", False),
    ("allow_mock_supported", True),
    ("allow_tool_calls", True),
    ("require_human_review_for_tools", False),
])
def test_every_operational_policy_setting_is_compared(tmp_path, field, value):
    before = sample_report()
    after = replace(before, run=replace(
        before.run, policy=replace(before.run.policy, **{field: value}),
    ))

    payload = compare(tmp_path, before, after)

    assert_material_field(payload, field)
    assert payload["timing_changes"] == []


def test_final_answer_difference_is_material(tmp_path):
    before = sample_report()
    after = sample_report(answer="Launch is in July [1].")
    payload = compare(tmp_path, before, after)
    assert_material_field(payload, "outcome.answer")
    assert any(change["before"] == before.run.final_answer
               and change["after"] == after.run.final_answer
               for change in payload["material_changes"])


def test_drafter_model_and_verifier_identity_are_compared(tmp_path):
    before = sample_report()
    after = replace(before, run=replace(
        before.run, model_label="Model B", steps=tuple(replace(
            step, model_label="Model B",
            verification=replace(
                step.verification, verifier_model_label="Model B",
            ) if step.verification else None,
        ) for step in before.run.steps),
    ))
    payload = compare(tmp_path, before, after)
    assert_material_field(payload, "model.label")
    assert_material_field(payload, "verification.model.label")


def test_different_verifier_relationship_is_material(tmp_path):
    before = sample_report()
    verification = replace(
        before.run.steps[-2], model_label="Verifier B",
        verification=replace(
            before.run.steps[-2].verification,
            verifier_model_label="Verifier B", same_model_as_drafter=False,
        ),
    )
    after = replace(before, run=replace(
        before.run, steps=(*before.run.steps[:-2], verification, before.run.steps[-1]),
    ))
    payload = compare(tmp_path, before, after)
    assert_material_field(payload, "verification.model.label")
    assert_material_field(payload, "same_model_as_drafter")


def test_recorded_verifier_outcome_is_compared_even_when_terminal_content_is_redacted(tmp_path):
    report = sample_report(answer=SECRET)
    verification = replace(
        report.run.steps[-2], decision=LoopDecision.NOT_VERIFIED,
        verification=replace(
            report.run.steps[-2].verification, outcome=VerificationOutcome.UNSUPPORTED,
        ),
    )
    refusal = LoopStep(
        step_id="refusal", phase=LoopPhase.REFUSE, decision=LoopDecision.REFUSE,
        started_at=AT, ended_at=AT,
    )
    final = replace(report.run.steps[-1], decision=LoopDecision.REFUSE)
    before = replace(report, run=replace(
        report.run, steps=(*report.run.steps[:-2], verification, refusal, final),
        final_decision=LoopDecision.REFUSE, terminal_reason=LoopTerminalReason.VERIFICATION_FAILED,
    ))
    after = replace(before, run=replace(before.run, steps=tuple(
        replace(step, verification=replace(
            step.verification, outcome=VerificationOutcome.INSUFFICIENT,
        )) if step.verification else step
        for step in before.run.steps
    )))
    payload = compare(tmp_path, before, after)
    assert_material_field(payload, "verification.outcome")
    assert SECRET not in json.dumps(payload)
    assert "Model A" not in json.dumps(payload)


def test_error_presence_is_material_without_comparing_error_text(tmp_path):
    report = blocked_report()
    before = replace(report, run=replace(
        report.run, error_message=None,
        steps=tuple(replace(step, error_message=None) for step in report.run.steps),
    ))
    payload = compare(tmp_path, before, report)
    assert_material_field(payload, "outcome.error_present")
    assert_material_field(payload, "step.error_present")
    assert SECRET not in json.dumps(payload)


def test_human_review_presence_is_material_but_request_identity_and_text_are_not(tmp_path):
    before = blocked_report()
    review = HumanReviewRequest(
        request_id="review_one", reason=SECRET, instructions=SECRET, created_at=AT,
    )
    after = replace(before, run=replace(
        before.run, final_decision=LoopDecision.REQUIRES_REVIEW,
        terminal_reason=LoopTerminalReason.HUMAN_REVIEW_REQUIRED,
        steps=(before.run.steps[0], replace(
            before.run.steps[-1], decision=LoopDecision.REQUIRES_REVIEW, human_review=review,
        )),
    ))
    payload = compare(tmp_path, before, after)
    assert_material_field(payload, "human_review_required")
    assert SECRET not in json.dumps(payload)

    changed_review = replace(review, request_id="review_two", reason="Another reason", instructions="Other instructions")
    changed = replace(after, run=replace(after.run, steps=(
        after.run.steps[0], replace(after.run.steps[-1], human_review=changed_review),
    )))
    payload = compare(tmp_path, after, changed)
    assert payload["has_material_changes"] is False
    assert payload["material_changes"] == []


@pytest.mark.parametrize("field,value", [
    ("conversation_context_turns", 2),
    ("semantic_memory_turns", 1),
])
def test_recorded_memory_provenance_changes_are_material(tmp_path, field, value):
    report = sample_report()
    metadata = {
        **report.run.metadata, "conversation_context_turns": 0,
        "semantic_memory_turns": 0, "semantic_memory_status": "empty",
    }
    before = replace(report, run=replace(report.run, metadata=metadata))
    after_metadata = {**metadata, field: value}
    if field == "semantic_memory_turns":
        after_metadata["semantic_memory_status"] = "retrieved"
    after = replace(before, run=replace(before.run, metadata=after_metadata))
    payload = compare(tmp_path, before, after)
    assert_material_field(payload, field.replace("_turns", "_count"))


def test_context_provider_difference_is_material(tmp_path):
    before = sample_report()
    after = replace(before, run=replace(before.run, context_provider="web"))
    after = bind_evidence(after, tuple(replace(item, provider="web") for item in after.run.evidence))
    payload = compare(tmp_path, before, after)
    assert_material_field(payload, "context_provider")


def test_evidence_list_order_is_ignored_but_citation_mapping_is_preserved(tmp_path):
    before = sample_report()
    second = EvidenceReference.from_source(
        citation_id=2, provider="document", source_identity="second source",
        page=3, chunk_index=1, excerpt="Second source content.",
    )
    before = bind_evidence(before, (*before.run.evidence, second))
    reordered = bind_evidence(before, tuple(reversed(before.run.evidence)))
    payload = compare(tmp_path, before, reordered)
    assert payload["has_material_changes"] is False
    assert payload["material_changes"] == []

    remapped = bind_evidence(before, tuple(
        replace(item, citation_id=3 - item.citation_id) for item in before.run.evidence
    ))
    payload = compare(tmp_path, before, remapped)
    assert_material_field(payload, "evidence")


@pytest.mark.parametrize("component", ["identity", "locator"])
def test_evidence_identity_and_locator_changes_are_material(tmp_path, component):
    before = sample_report()
    item = before.run.evidence[0]
    if component == "identity":
        item = replace(item, evidence_id="evidence_" + "a" * 64)
    else:
        item = replace(item, locator=replace(item.locator, page=4, chunk_index=2))
    after = bind_evidence(before, (item,))
    payload = compare(tmp_path, before, after)
    assert_material_field(payload, "evidence")


def test_inserted_phase_is_an_explicit_addition_without_shifting_later_steps(tmp_path):
    before = sample_report()
    extra = LoopStep(
        step_id="extra_context", phase=LoopPhase.CONTEXT_SELECT,
        decision=LoopDecision.CONTINUE, started_at=AT, ended_at=AT,
    )
    after = replace(before, run=replace(before.run, steps=(extra, *before.run.steps)))
    payload = compare(tmp_path, before, after)
    assert payload["has_material_changes"] is True
    assert any(change["kind"] == "added" and "context_select" in json.dumps(change)
               for change in payload["material_changes"])
    assert not any("Model A" in json.dumps(change)
                   for change in payload["material_changes"])
    reverse = compare(tmp_path, after, before)
    assert any(change["kind"] == "removed" and "context_select" in json.dumps(change)
               for change in reverse["material_changes"])


def test_retries_and_phase_order_remain_material(tmp_path):
    payload = compare(tmp_path, sample_report(), retry_report())
    assert payload["has_material_changes"] is True
    assert "retry" in json.dumps(payload["material_changes"])
    assert any(change["kind"] in {"added", "removed"}
               for change in payload["material_changes"])

    before = sample_report()
    context = LoopStep(
        step_id="context", phase=LoopPhase.CONTEXT_SELECT,
        started_at=AT, ended_at=AT,
    )
    before = replace(before, run=replace(before.run, steps=(context, *before.run.steps)))
    after = replace(before, run=replace(before.run, steps=(
        before.run.steps[1], context, *before.run.steps[2:],
    )))
    assert compare(tmp_path, before, after)["has_material_changes"] is True


def test_terminal_decision_and_reason_remain_visible_when_details_are_redacted(tmp_path):
    payload = compare(tmp_path, sample_report(), blocked_report())
    assert_material_field(payload, "outcome.decision")
    assert_material_field(payload, "terminal_reason")
    assert payload["unavailable_fields"]
    assert "redacted" in json.dumps(payload["unavailable_fields"])
    assert SECRET not in json.dumps(payload)


@pytest.mark.parametrize("output_format", ["text", "json"])
def test_redacted_raw_differences_never_become_public_changes(tmp_path, output_format):
    before = blocked_report()
    after = replace(before, run=replace(
        before.run, user_input="SECOND_PRIVATE_SENTINEL", model_label="Hidden Model B",
        final_answer="SECOND_PRIVATE_SENTINEL", error_message="SECOND_PRIVATE_SENTINEL",
        evidence=(), metadata={"private": "SECOND_PRIVATE_SENTINEL"},
        steps=tuple(replace(
            step, model_label="Hidden Model B", input_summary="SECOND_PRIVATE_SENTINEL",
            output_summary="SECOND_PRIVATE_SENTINEL",
        ) for step in before.run.steps),
    ))
    before_path = write_artifact(tmp_path, "before.jsonl", before)
    after_path = write_artifact(tmp_path, "after.jsonl", after)
    code, output, errors = invoke(before_path, after_path, "--format", output_format)

    assert code == 0, errors
    assert errors == ""
    for hidden in (SECRET, "SECOND_PRIVATE_SENTINEL", "Hidden Model B", "Model A", "evidence_"):
        assert hidden not in output
    payload = loop_replay.diff_artifacts(before_path, after_path)
    assert payload["has_material_changes"] is False
    assert payload["material_changes"] == []
    assert "redacted" in json.dumps(payload["unavailable_fields"])


def test_unknown_memory_is_not_reported_as_known_zero(tmp_path):
    before = sample_report()
    after = replace(before, run=replace(before.run, metadata={
        **before.run.metadata, "conversation_context_turns": 0,
        "semantic_memory_turns": 0, "semantic_memory_status": "empty",
    }))
    payload = compare(tmp_path, before, after)
    assert payload["unavailable_fields"]
    assert any(
        change["field"] == "memory.conversation_context_count"
        and change["before"] == {"availability": "unknown"}
        and change["after"] == 0
        for change in payload["material_changes"]
    )
    assert payload["same_task_verified"] is False


def test_incomplete_run_and_step_completion_are_not_ignored_as_timestamps(tmp_path):
    open_step = LoopStep(
        step_id="context", phase=LoopPhase.CONTEXT_SELECT, started_at=AT,
    )
    before = LoopReport(run=LoopRun(
        run_id="open_run", session_id="open_session", user_input="Question",
        context_provider="none", backend="mock", model_label="MockLLM",
        started_at=AT, steps=(open_step,),
    ))
    after = replace(before, run=replace(
        before.run, steps=(replace(open_step, ended_at=AT),),
    ))
    payload = compare(tmp_path, before, after)
    assert payload["has_material_changes"] is True
    assert_material_field(payload, "step.completed")
    assert "unknown" in json.dumps(payload)

    completed = compare(tmp_path, before, sample_report())
    assert_material_field(completed, "outcome.completed")


@pytest.mark.parametrize("public", [True, False])
def test_arbitrary_raw_metadata_and_private_text_are_not_semantic_equivalence(tmp_path, public):
    before = sample_report()
    after = replace(before, run=replace(
        before.run, user_input="A completely different task",
        metadata={"recipe": {"instructions": "Changed instruction"}, "private": "Changed private value"},
        policy=replace(before.run.policy, metadata={"private_policy": "Changed"}),
        steps=tuple(replace(
            step, name="Changed name", input_summary="Changed prompt",
            metadata={**step.metadata, "private": "Changed private value"},
            verification=replace(
                step.verification, reasons=("Changed reason",), raw_response="Changed raw response",
            ) if step.verification else None,
        ) for step in before.run.steps),
    ))
    payload = compare(tmp_path, before, after, public=public)
    assert payload["has_material_changes"] is False
    assert payload["material_changes"] == []
    assert payload["same_task_verified"] is False
    limits = " ".join(payload["limitations"]).lower()
    assert "task" in limits
    assert "metadata" in limits or "recipe" in limits
    if public:
        assert "Changed private value" not in json.dumps(payload)
    else:
        assert payload["before"]["raw_report"] == before.to_dict()
        assert payload["after"]["raw_report"] == after.to_dict()


@pytest.mark.parametrize("output_format", ["text", "json"])
def test_raw_cli_requires_explicit_opt_in_and_contains_full_reports(tmp_path, output_format):
    before, after = blocked_report(), sample_report()
    before_path = write_artifact(tmp_path, "before.jsonl", before)
    after_path = write_artifact(tmp_path, "after.jsonl", after)
    code, output, errors = invoke(before_path, after_path, "--raw", "--format", output_format)
    assert code == 0, errors
    assert SECRET in output
    if output_format == "json":
        payload = json.loads(output)
        assert payload["public"] is False
        assert payload["before"]["raw_report"] == before.to_dict()
        assert payload["after"]["raw_report"] == after.to_dict()
    else:
        assert "RAW" in output


def test_legacy_raw_omissions_stay_unknown_instead_of_known_empty_values():
    path = ROOT / "tests/fixtures/loop-report-v1-legacy.jsonl"
    code, output, errors = invoke(path, path)
    assert code == 2
    assert output == ""
    assert "line 1" in errors

    payload = loop_replay.diff_artifacts(path, path, public=False)
    assert payload["has_material_changes"] is False
    unknown = {(item["field"], item["availability"])
               for item in payload["unavailable_fields"]}
    assert ("evidence", "not_recorded") in unknown
    assert ("outcome.terminal_reason", "unknown") in unknown
    assert payload["same_task_verified"] is False


@pytest.mark.parametrize("side", ["before", "after"])
def test_multiple_reports_require_explicit_selection_on_each_side(tmp_path, side):
    paths = {}
    for label in ("before", "after"):
        reports = (sample_report("run_one"), sample_report("run_two")) if label == side else (sample_report(),)
        paths[label] = write_artifact(tmp_path, f"{label}.jsonl", *reports)
    code, output, errors = invoke(paths["before"], paths["after"])
    assert code == 2
    assert output == ""
    assert f"--{side}-report-index" in errors

    code, output, errors = invoke(
        paths["before"], paths["after"], f"--{side}-report-index", "2", "--format", "json",
    )
    assert code == 0, errors
    payload = json.loads(output)
    assert payload[side]["source_jsonl_line"] == 2
    assert payload[side]["run_id"] == "run_two"


@pytest.mark.parametrize("side", ["before", "after"])
@pytest.mark.parametrize("index", ["0", "-1", "2"])
def test_out_of_range_report_selection_is_rejected(tmp_path, side, index):
    before = write_artifact(tmp_path, "before.jsonl", sample_report())
    after = write_artifact(tmp_path, "after.jsonl", sample_report())
    code, output, errors = invoke(before, after, f"--{side}-report-index", index)
    assert code == 2
    assert output == ""
    assert f"--{side}-report-index" in errors


@pytest.mark.parametrize("side", ["before", "after"])
def test_missing_session_requires_explicit_identity_for_that_side(tmp_path, side):
    report = sample_report()
    missing = replace(report, run=replace(report.run, session_id=None))
    paths = {label: write_artifact(
        tmp_path, f"{label}.jsonl", missing if label == side else report,
    ) for label in ("before", "after")}
    code, output, errors = invoke(paths["before"], paths["after"])
    assert code == 2
    assert output == ""
    assert f"--{side}-session-id" in errors

    code, output, errors = invoke(
        paths["before"], paths["after"], f"--{side}-session-id", "supplied_session", "--format", "json",
    )
    assert code == 0, errors
    assert json.loads(output)[side]["session_id"] == "supplied_session"


@pytest.mark.parametrize("side", ["before", "after"])
@pytest.mark.parametrize("invalid", ["malformed_json", "invalid_public_projection"])
def test_unselected_later_invalid_line_rejects_both_artifacts_before_output(tmp_path, side, invalid):
    paths = {label: write_artifact(tmp_path, f"{label}.jsonl", sample_report())
             for label in ("before", "after")}
    if invalid == "malformed_json":
        bad_line = "{"
    else:
        report = sample_report("invalid_report")
        report = replace(report, run=replace(report.run, terminal_reason=LoopTerminalReason.BLOCKED))
        bad_line = json.dumps(report.to_dict())
    path = paths[side]
    path.write_text(path.read_text(encoding="utf-8") + bad_line + "\n", encoding="utf-8")
    code, output, errors = invoke(
        paths["before"], paths["after"], "--before-report-index", "1", "--after-report-index", "1",
    )
    assert code == 2
    assert output == ""
    assert "line 2" in errors


@pytest.mark.parametrize("side", ["before", "after"])
@pytest.mark.parametrize("bad_line", ["", "[]", '{"schema_version":NaN}', '{"run":{},"run":{}}'])
def test_strict_jsonl_validation_is_preserved_for_both_inputs(tmp_path, side, bad_line):
    paths = {label: write_artifact(tmp_path, f"{label}.jsonl", sample_report())
             for label in ("before", "after")}
    path = paths[side]
    path.write_text(path.read_text(encoding="utf-8") + bad_line + "\n", encoding="utf-8")
    code, output, errors = invoke(
        paths["before"], paths["after"], "--before-report-index", "1", "--after-report-index", "1",
    )
    assert code == 2
    assert output == ""
    assert "line 2" in errors


@pytest.mark.parametrize("side", ["before", "after"])
@pytest.mark.parametrize("missing", [True, False])
def test_missing_and_empty_diff_inputs_fail_cleanly(tmp_path, side, missing):
    paths = {label: write_artifact(tmp_path, f"{label}.jsonl", sample_report())
             for label in ("before", "after")}
    if missing:
        paths[side].unlink()
    else:
        paths[side].write_text("", encoding="utf-8")
    code, output, errors = invoke(paths["before"], paths["after"])
    assert code == 2
    assert output == ""
    assert errors


def test_valid_cli_diff_returns_success_even_with_material_changes(tmp_path):
    before = write_artifact(tmp_path, "before.jsonl", sample_report())
    after = write_artifact(tmp_path, "after.jsonl", sample_report(answer="Launch is in July [1]."))
    code, output, errors = invoke(before, after, "--format", "json")
    assert code == 0
    assert errors == ""
    assert json.loads(output)["has_material_changes"] is True
    assert output == invoke(before, after, "--format", "json")[1]


def test_readable_diff_escapes_artifact_terminal_controls(tmp_path):
    before = write_artifact(tmp_path, "before.jsonl", sample_report())
    answer = "Café\nforged line\x1b[2J\u202e\u2028 [1]."
    after = write_artifact(tmp_path, "after.jsonl", sample_report(answer=answer))
    code, output, errors = invoke(before, after)
    assert code == 0, errors
    assert "Café" in output
    assert "\\nforged line" in output
    for character in ("\x1b", "\u202e", "\u2028"):
        assert character not in output

    malformed = sample_report().to_dict()
    malformed["run"]["steps"][0]["phase"] = "\x1b[2J\nforged error"
    after.write_text(json.dumps(malformed) + "\n", encoding="utf-8")
    code, output, errors = invoke(before, after)
    assert code == 2
    assert output == ""
    assert "\x1b" not in errors
    assert errors.count("\n") == 1


def test_diff_needs_only_standard_library_and_never_connects(tmp_path):
    before = write_artifact(tmp_path, "before.jsonl", sample_report())
    after = write_artifact(tmp_path, "after.jsonl", sample_report(answer="Launch is in July [1]."))
    script = """
import runpy
import socket
import sys

def forbid_network(*args, **kwargs):
    raise AssertionError('Offline diff must not connect to a model or network')

socket.socket = forbid_network
socket.create_connection = forbid_network
sys.argv = ['src.loop_replay', 'diff', sys.argv[1], sys.argv[2], '--format', 'json']
runpy.run_module('src.loop_replay', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", script, str(before), str(after)],
        cwd=ROOT,
        env={**os.environ, "LLM_BACKEND": "invalid", "OLLAMA_BASE_URL": "https://invalid.test"},
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["public"] is True
    assert result.stderr == ""


def test_diff_help_describes_selection_raw_mode_and_limits(capsys):
    with pytest.raises(SystemExit) as exc:
        loop_replay.main(["diff", "--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    for option in ("--raw", "--before-report-index", "--after-report-index",
                   "--before-session-id", "--after-session-id"):
        assert option in output
    assert "task equivalence is not inferred" in output.lower()
