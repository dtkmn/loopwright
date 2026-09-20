import io
import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.loop_engine import (
    ANSWER_CANDIDATE_SHA256_METADATA_KEY,
    EVIDENCE_SET_SHA256_METADATA_KEY,
    EvidenceReference,
    LoopDecision,
    LoopPhase,
    LoopReport,
    LoopRun,
    LoopStep,
    LoopTerminalReason,
    VerificationOutcome,
    VerificationResult,
    answer_candidate_sha256,
    evidence_set_sha256,
)
from src.loop_replay import inspect_artifact, main


AT = datetime(2026, 9, 20, tzinfo=timezone.utc)
SECRET = "PRIVATE_DIAGNOSTIC_SENTINEL"
ROOT = Path(__file__).resolve().parents[1]


def sample_report(run_id="run_supported", *, answer="Launch is in June [1]."):
    evidence = EvidenceReference.from_source(
        citation_id=1, provider="document", source_identity=SECRET,
        page=2, chunk_index=0, excerpt=SECRET,
    )
    binding = {
        ANSWER_CANDIDATE_SHA256_METADATA_KEY: answer_candidate_sha256(answer),
        EVIDENCE_SET_SHA256_METADATA_KEY: evidence_set_sha256((evidence,)),
        "private": SECRET,
    }
    steps = []
    for phase, decision in (
        (LoopPhase.RETRIEVE, LoopDecision.CONTINUE),
        (LoopPhase.DRAFT, LoopDecision.CONTINUE),
        (LoopPhase.MECHANICAL_CHECK, LoopDecision.CONTINUE),
        (LoopPhase.VERIFY, LoopDecision.SUPPORTED),
        (LoopPhase.FINAL, LoopDecision.SUPPORTED),
    ):
        steps.append(LoopStep(
            step_id=f"step_{phase.value}", phase=phase, decision=decision,
            started_at=AT, ended_at=AT, backend="ollama", model_label="Model A",
            name=SECRET, input_summary=SECRET,
            output_summary=(
                "mechanical_checks_passed" if phase == LoopPhase.MECHANICAL_CHECK
                else answer if phase in {LoopPhase.DRAFT, LoopPhase.FINAL} else SECRET
            ),
            metadata=binding,
            verification=(VerificationResult(
                outcome=VerificationOutcome.SUPPORTED, verifier="ollama",
                verifier_backend="ollama", verifier_model_label="Model A",
                same_model_as_drafter=True, reasons=(SECRET,), raw_response=SECRET,
            ) if phase == LoopPhase.VERIFY else None),
        ))
    return LoopReport(run=LoopRun(
        run_id=run_id, session_id="session_inspect", user_input=SECRET,
        context_provider="document", backend="ollama", model_label="Model A",
        started_at=AT, completed_at=AT, steps=tuple(steps), evidence=(evidence,),
        final_decision=LoopDecision.SUPPORTED,
        terminal_reason=LoopTerminalReason.COMPLETED, final_answer=answer,
        metadata={"private": SECRET, "model_thinking": SECRET},
    ))


def blocked_report():
    report = sample_report("run_blocked", answer=SECRET)
    block = LoopStep(
        step_id="step_block", phase=LoopPhase.ERROR, decision=LoopDecision.BLOCK,
        started_at=AT, ended_at=AT, error_message=SECRET,
        input_summary=SECRET, output_summary=SECRET,
    )
    return replace(report, run=replace(
        report.run, steps=(report.run.steps[1], block),
        final_decision=LoopDecision.BLOCK, terminal_reason=LoopTerminalReason.BLOCKED,
        error_message=SECRET,
    ))


def retry_report():
    report = sample_report("run_retry")
    initial_draft = replace(report.run.steps[1], step_id="step_initial_draft")
    trigger = LoopStep(
        step_id="step_failed_check", phase=LoopPhase.MECHANICAL_CHECK,
        decision=LoopDecision.RETRY, started_at=AT, ended_at=AT,
        output_summary="needs_retry", metadata={"reasons": ["missing_inline_citations"]},
    )
    retry = LoopStep(
        step_id="step_retry", phase=LoopPhase.RETRY, decision=LoopDecision.RETRY,
        started_at=AT, ended_at=AT, retry_count=1,
        metadata={
            "retry_reason": "self_check", "reasons": ["missing_inline_citations"],
            "retry_count": 1, "max_retries": 1, "retry_trigger_step_id": trigger.step_id,
        },
    )
    return replace(report, run=replace(report.run, steps=(
        report.run.steps[0], initial_draft, trigger, retry,
        *(replace(step, retry_count=1) for step in report.run.steps[1:]),
    )))


def write_reports(tmp_path, *reports):
    path = tmp_path / "session.jsonl"
    path.write_text(
        "".join(json.dumps(report.to_dict(), sort_keys=True) + "\n" for report in reports),
        encoding="utf-8",
    )
    return path


def invoke(path, *options):
    output, errors = io.StringIO(), io.StringIO()
    code = main(
        ["inspect", str(path), *options], output_stream=output, error_stream=errors,
    )
    return code, output.getvalue(), errors.getvalue()


def test_readable_inspection_explains_recorded_evidence_and_checks(tmp_path):
    report = sample_report()
    path = write_reports(tmp_path, report)

    code, output, errors = invoke(path)

    assert code == 0, errors
    assert errors == ""
    assert "Run run_supported [JSONL line 1]" in output
    assert "Final decision: supported" in output
    assert "Terminal reason: completed" in output
    assert "Context provider: document" in output
    assert "Model: ollama / Model A" in output
    assert "mechanical_check -> continue" in output
    assert "Verification recorded: supported" in output
    assert "Verifier: ollama / Model A; same model as drafter: yes" in output
    assert report.run.evidence[0].evidence_id in output
    assert "page 2, chunk 0" in output
    assert "Highest recorded retry count: 0; limit: 1" in output
    assert "Final answer: Launch is in June [1]." in output
    assert "no model execution or independent verification" in output
    assert SECRET not in output
    assert output == invoke(path)[1]


def test_json_inspection_uses_exact_public_projection_and_keeps_line_provenance(tmp_path):
    first, second = sample_report("run_one"), sample_report("run_two")
    path = write_reports(tmp_path, first, second)

    code, output, errors = invoke(path, "--format", "json", "--report-index", "2")

    assert code == 0
    assert errors == ""
    payload = json.loads(output)
    assert payload["schema_version"] == "loop-inspection/v1"
    assert payload["source_path"] == str(path)
    assert payload["session_id"] == "session_inspect"
    assert payload["input_report_count"] == 2
    assert payload["report_count"] == 1
    assert payload["public"] is True
    assert payload["reports"] == [{"source_jsonl_line": 2, "report": second.to_public_dict()}]
    assert SECRET not in output
    assert output == invoke(path, "--format", "json", "--report-index", "2")[1]


def test_readable_inspection_shows_failed_check_then_retry_in_order(tmp_path):
    path = write_reports(tmp_path, retry_report())
    code, output, errors = invoke(path)
    assert code == 0, errors
    assert "Highest recorded retry count: 1; limit: 1" in output
    failed = output.index("mechanical_check -> retry")
    retry = output.index("retry -> retry")
    final = output.index("final -> supported")
    assert failed < retry < final
    assert "Final decision: supported" in output


def test_refusal_shows_verifier_outcome_but_withholds_verifier_identity_and_answer(tmp_path):
    report = sample_report(answer=SECRET)
    verification = replace(
        report.run.steps[-2], decision=LoopDecision.NOT_VERIFIED,
        verification=replace(
            report.run.steps[-2].verification, outcome=VerificationOutcome.UNSUPPORTED,
        ),
    )
    refusal = LoopStep(
        step_id="step_refuse", phase=LoopPhase.REFUSE, decision=LoopDecision.REFUSE,
        started_at=AT, ended_at=AT,
    )
    final = replace(report.run.steps[-1], decision=LoopDecision.REFUSE)
    report = replace(report, run=replace(
        report.run, steps=(*report.run.steps[:-2], verification, refusal, final),
        final_decision=LoopDecision.REFUSE, terminal_reason=LoopTerminalReason.VERIFICATION_FAILED,
    ))
    path = write_reports(tmp_path, report)
    code, output, errors = invoke(path)
    assert code == 0, errors
    assert "Final decision: refuse" in output
    assert "Terminal reason: verification_failed" in output
    assert "Verification recorded: unsupported" in output
    assert "Verifier identity: withheld by terminal public redaction" in output
    assert SECRET not in output
    assert "Model A" not in output


@pytest.mark.parametrize("output_format", ["text", "json"])
def test_terminal_redaction_never_uses_raw_fallbacks(tmp_path, output_format):
    path = write_reports(tmp_path, blocked_report())

    code, output, errors = invoke(path, "--format", output_format)

    assert code == 0
    assert errors == ""
    assert SECRET not in output
    assert "Model A" not in output
    assert "evidence_" not in output
    if output_format == "text":
        assert "Final decision: block" in output
        assert "Evidence: withheld by terminal public redaction (count unknown)" in output
        assert "Final answer: withheld by terminal public redaction" in output
    else:
        assert json.loads(output)["reports"][0]["report"] == blocked_report().to_public_dict()


@pytest.mark.parametrize("output_format", ["text", "json"])
def test_raw_mode_explicitly_includes_diagnostics(tmp_path, output_format):
    report = blocked_report()
    path = write_reports(tmp_path, report)

    code, output, errors = invoke(path, "--raw", "--format", output_format)

    assert code == 0
    assert errors == ""
    assert SECRET in output
    if output_format == "json":
        payload = json.loads(output)
        assert payload["public"] is False
        assert payload["reports"][0]["report"] == report.to_dict()
    else:
        assert "RAW local diagnostics" in output
        assert "not validated for public serving" in output


def test_legacy_artifact_requires_raw_and_keeps_unknown_provenance():
    path = ROOT / "tests/fixtures/loop-report-v1-legacy.jsonl"
    code, output, errors = invoke(path)
    assert code == 2
    assert output == ""
    assert "line 1" in errors

    code, output, errors = invoke(path, "--raw", "--format", "json")
    assert code == 0
    assert errors == ""
    run = json.loads(output)["reports"][0]["report"]["run"]
    assert run["terminal_reason"] == "unspecified"
    assert run["evidence"] == []
    verification = next(step["verification"] for step in run["steps"] if step["verification"])
    assert verification["verifier_backend"] is None
    assert verification["same_model_as_drafter"] is None


@pytest.mark.parametrize("selected", [[], ["--report-index", "1"]])
def test_invalid_later_projection_rejects_entire_input_without_partial_output(tmp_path, selected):
    valid = sample_report()
    invalid = replace(valid, run=replace(
        valid.run, run_id="run_invalid", terminal_reason=LoopTerminalReason.BLOCKED,
    ))
    path = write_reports(tmp_path, valid, invalid)

    code, output, errors = invoke(path, *selected)

    assert code == 2
    assert output == ""
    assert "line 2" in errors
    assert "terminal" in errors


@pytest.mark.parametrize("bad_line", ["", "{", "[]", '{"schema_version":NaN}', '{"run":{},"run":{}}'])
def test_malformed_input_reports_exact_line_without_partial_output(tmp_path, bad_line):
    path = write_reports(tmp_path, sample_report())
    path.write_text(path.read_text() + bad_line + "\n", encoding="utf-8")

    code, output, errors = invoke(path)

    assert code == 2
    assert output == ""
    assert "line 2" in errors


def test_unknown_fields_and_public_artifact_inputs_are_rejected(tmp_path):
    path = tmp_path / "input.jsonl"
    for payload in (sample_report().to_public_dict(), {**sample_report().to_dict(), "extra": True}):
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        code, output, errors = invoke(path)
        assert code == 2
        assert output == ""
        assert "line 1" in errors


@pytest.mark.parametrize("index", ["0", "-1", "2"])
def test_invalid_report_selection_fails(tmp_path, index):
    path = write_reports(tmp_path, sample_report())
    code, output, errors = invoke(path, "--report-index", index)
    assert code == 2
    assert output == ""
    assert "--report-index must be between 1 and 1" in errors


def test_missing_session_id_requires_explicit_identity_without_rewriting_report(tmp_path):
    report = sample_report()
    report = replace(report, run=replace(report.run, session_id=None))
    path = write_reports(tmp_path, report)
    code, output, errors = invoke(path)
    assert code == 2
    assert output == ""
    assert "--session-id" in errors
    assert "line 1" in errors

    payload = inspect_artifact(str(path), session_id="session_supplied")
    assert payload["session_id"] == "session_supplied"
    assert payload["reports"][0]["report"]["run"]["session_id"] is None


def test_incomplete_report_keeps_unknown_outcome(tmp_path):
    report = LoopReport(run=LoopRun(
        run_id="run_incomplete", session_id="session_inspect", user_input=SECRET,
        context_provider="none", backend="mock", model_label="MockLLM",
        started_at=AT,
    ))
    path = write_reports(tmp_path, report)
    code, output, errors = invoke(path)
    assert code == 0
    assert errors == ""
    assert "Final decision: unknown" in output
    assert "Terminal reason: unknown" in output
    assert "Steps: none recorded" in output
    assert "Final answer: not available" in output


def test_text_and_errors_escape_artifact_terminal_controls(tmp_path):
    answer = "Café\nforged line\x1b[2J\u202e\u2028 [1]."
    path = write_reports(tmp_path, sample_report(answer=answer))
    code, output, errors = invoke(path)
    assert code == 0
    assert errors == ""
    assert "Café" in output
    assert "\\nforged line" in output
    for character in ("\x1b", "\u202e", "\u2028"):
        assert character not in output

    bad = sample_report().to_dict()
    bad["run"]["steps"][0]["phase"] = "\x1b[2J\nforged error"
    path.write_text(json.dumps(bad), encoding="utf-8")
    code, output, errors = invoke(path)
    assert code == 2
    assert output == ""
    assert "\x1b" not in errors
    assert errors.count("\n") == 1


def test_inspection_needs_only_standard_library_and_never_connects(tmp_path):
    path = write_reports(tmp_path, sample_report())
    script = """
import runpy
import socket
import sys

def forbid_network(*args, **kwargs):
    raise AssertionError('Inspection must not connect to a model or network')

socket.socket = forbid_network
socket.create_connection = forbid_network
sys.argv = ['src.loop_replay', 'inspect', sys.argv[1], '--format', 'json']
runpy.run_module('src.loop_replay', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", script, str(path)],
        cwd=ROOT, env={**os.environ, "LLM_BACKEND": "invalid", "OLLAMA_BASE_URL": "https://invalid.test"},
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["public"] is True
    assert result.stderr == ""


def test_runtime_mock_session_export_can_be_inspected(tmp_path, monkeypatch):
    from src.ai_loop_engine import AILoopEngine

    for name in (
        "LLM_BACKEND", "LLM_MODEL", "EMBEDDINGS_MODEL", "OLLAMA_BASE_URL",
        "OLLAMA_MODEL", "OLLAMA_EMBED_MODEL", "OPENAI_COMPAT_BASE_URL",
        "OPENAI_COMPAT_MODEL", "OPENAI_COMPAT_EMBED_MODEL", "OPENAI_COMPAT_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    engine = AILoopEngine(
        fast_mode=True, llm_backend="mock", embeddings_model="local-hashing-384",
    )
    result = engine.query_with_trace(
        "Rewrite: hello there", context_provider="none", session_id="session_inspect",
    )
    assert result.loop_report is not None
    path = tmp_path / "runtime-session.jsonl"
    engine.export_loop_session_jsonl(path, session_id="session_inspect")

    code, output, errors = invoke(path)
    assert code == 0, errors
    assert "Final decision: not_verified" in output
    assert "Context provider: none" in output
    assert "Evidence references recorded: 0" in output
    assert "Verification recorded: supported" not in output


def test_missing_and_empty_artifacts_fail_cleanly(tmp_path):
    path = tmp_path / "missing.jsonl"
    code, output, errors = invoke(path)
    assert code == 2
    assert output == ""
    assert "Cannot inspect" in errors
    path.write_text("", encoding="utf-8")
    code, output, errors = invoke(path)
    assert code == 2
    assert output == ""
    assert "No loop reports" in errors


def test_inspect_help_describes_limits_and_explicit_raw(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["inspect", "--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "--raw" in output
    assert "public projection" in output
    assert "not independent proof" in output


def test_diff_help_describes_explicit_pairing_and_no_execution(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["diff", "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--before-report-index" in help_text
    assert "--after-report-index" in help_text
    assert "task equivalence is not inferred" in help_text
