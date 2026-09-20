import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

import src.loop_export as loop_export_module
from src.loop_engine import (
    ANSWER_CANDIDATE_SHA256_METADATA_KEY,
    EVIDENCE_SET_SHA256_METADATA_KEY,
    LoopDecision,
    LoopPhase,
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
from src.loop_export import export_payload, main, parse_args


# Serialized by the pre-provenance v1 writer, not rebuilt from today's records.
LEGACY_V1_JSONL = Path(__file__).parent / "fixtures" / "loop-report-v1-legacy.jsonl"


def utc(value):
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def not_verified_steps(*, answer: str, started_at: datetime) -> tuple[LoopStep, ...]:
    binding = {
        ANSWER_CANDIDATE_SHA256_METADATA_KEY: answer_candidate_sha256(answer),
        EVIDENCE_SET_SHA256_METADATA_KEY: evidence_set_sha256(()),
    }
    provenance = {
        "backend": "mock",
        "model_label": "MockLLM (explicit demo)",
    }
    return (
        LoopStep(
            step_id="step_draft",
            phase=LoopPhase.DRAFT,
            decision=LoopDecision.CONTINUE,
            started_at=started_at,
            ended_at=started_at,
            output_summary=answer,
            retry_count=0,
            metadata=binding,
            **provenance,
        ),
        LoopStep(
            step_id="step_verify",
            phase=LoopPhase.VERIFY,
            decision=LoopDecision.NOT_VERIFIED,
            started_at=started_at,
            ended_at=started_at,
            retry_count=0,
            verification=VerificationResult(
                outcome=VerificationOutcome.NOT_VERIFIED,
            ),
            metadata=binding,
            **provenance,
        ),
        LoopStep(
            step_id="step_final",
            phase=LoopPhase.FINAL,
            decision=LoopDecision.NOT_VERIFIED,
            started_at=started_at,
            ended_at=started_at,
            output_summary=answer,
            retry_count=0,
            metadata=binding,
            **provenance,
        ),
    )


def sample_report(run_id: str, *, secret: str | None = None) -> LoopReport:
    started_at = utc("2026-06-24T05:00:00")
    if secret:
        return LoopReport(
            run=LoopRun(
                run_id=run_id,
                session_id="session_cli",
                user_input=secret,
                context_provider="document",
                backend="mock",
                model_label="MockLLM (explicit demo)",
                started_at=started_at,
                completed_at=started_at,
                steps=(
                    LoopStep(
                        step_id="step_block",
                        phase=LoopPhase.ERROR,
                        decision=LoopDecision.BLOCK,
                        started_at=started_at,
                        ended_at=started_at,
                        output_summary=secret,
                        metadata={"secret": secret},
                    ),
                ),
                final_decision=LoopDecision.BLOCK,
                final_answer=secret,
                error_message=secret,
                metadata={"secret": secret},
            )
        )
    answer = "Project Phoenix launches in June 2026."
    return LoopReport(
        run=LoopRun(
            run_id=run_id,
            session_id="session_cli",
            user_input="When does Project Phoenix launch?",
            context_provider="document",
            backend="mock",
            model_label="MockLLM (explicit demo)",
            started_at=started_at,
            completed_at=started_at,
            steps=not_verified_steps(answer=answer, started_at=started_at),
            final_decision=LoopDecision.NOT_VERIFIED,
            terminal_reason=LoopTerminalReason.NOT_VERIFIED,
            final_answer=answer,
        )
    )


def sample_report_without_session(run_id: str) -> LoopReport:
    started_at = utc("2026-06-24T05:15:00")
    answer = "Project Phoenix launches in June 2026."
    return LoopReport(
        run=LoopRun(
            run_id=run_id,
            user_input="When does Project Phoenix launch?",
            context_provider="document",
            backend="mock",
            model_label="MockLLM (explicit demo)",
            started_at=started_at,
            completed_at=started_at,
            steps=not_verified_steps(answer=answer, started_at=started_at),
            final_decision=LoopDecision.NOT_VERIFIED,
            terminal_reason=LoopTerminalReason.NOT_VERIFIED,
            final_answer=answer,
        )
    )


def write_jsonl(tmp_path, *reports: LoopReport):
    session = LoopSession(session_id="session_cli")
    for report in reports:
        session = session.add_report(report)
    path = tmp_path / "loop-session.jsonl"
    path.write_text(session.to_jsonl(), encoding="utf-8")
    return path


def test_loop_export_writes_langgraph_session_to_stdout(tmp_path):
    input_path = write_jsonl(tmp_path, sample_report("run_one"), sample_report("run_two"))
    output = io.StringIO()

    exit_code = main(
        [
            "--adapter",
            "langgraph-manifest",
            "--input",
            str(input_path),
        ],
        output_stream=output,
    )

    payload = json.loads(output.getvalue())
    assert exit_code == 0
    assert payload["adapter_name"] == "langgraph_manifest"
    assert payload["source_projection_schema_version"] == "loop-public-report/v1"
    assert payload["thread_id"] == "session_cli"
    assert payload["manifest_count"] == 2
    assert [manifest["source_jsonl_line"] for manifest in payload["manifests"]] == [
        1,
        2,
    ]


def test_loop_export_writes_one_openai_report_to_file(tmp_path):
    input_path = write_jsonl(tmp_path, sample_report("run_one"), sample_report("run_two"))
    output_path = tmp_path / "trace.json"

    exit_code = main(
        [
            "--adapter",
            "openai-trace",
            "--input",
            str(input_path),
            "--report-index",
            "2",
            "--output",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["adapter_name"] == "openai_trace"
    assert payload["source_projection_schema_version"] == "loop-public-report/v1"
    assert payload["trace"]["trace_id"].startswith("trace_")
    assert payload["trace"]["metadata"]["loopwright_run_id"] == "run_two"
    assert payload["trace"]["source_jsonl_line"] == 2
    assert "traces" not in payload


def test_loop_export_preserves_selected_line_for_single_langgraph_manifest(tmp_path):
    input_path = write_jsonl(tmp_path, sample_report("run_one"), sample_report("run_two"))
    output = io.StringIO()

    exit_code = main(
        [
            "--adapter",
            "langgraph-manifest",
            "--input",
            str(input_path),
            "--report-index",
            "2",
        ],
        output_stream=output,
    )

    payload = json.loads(output.getvalue())
    manifest = payload["manifest"]
    assert exit_code == 0
    assert manifest["run_id"] == "run_two"
    assert manifest["source_jsonl_line"] == 2
    assert manifest["checkpoints"][0]["source_jsonl_line"] == 2


def test_loop_export_uses_supplied_session_id_for_langgraph_manifest(tmp_path):
    input_path = write_jsonl(tmp_path, sample_report_without_session("run_no_session"))
    output = io.StringIO()

    exit_code = main(
        [
            "--adapter",
            "langgraph-manifest",
            "--input",
            str(input_path),
            "--session-id",
            "session_supplied",
        ],
        output_stream=output,
    )

    payload = json.loads(output.getvalue())
    assert exit_code == 0
    assert payload["thread_id"] == "session_supplied"
    assert payload["manifests"][0]["thread_id"] == "session_supplied"


def test_loop_export_uses_supplied_session_id_for_single_openai_trace(tmp_path):
    input_path = write_jsonl(tmp_path, sample_report_without_session("run_no_session"))
    output = io.StringIO()

    exit_code = main(
        [
            "--adapter",
            "openai-trace",
            "--input",
            str(input_path),
            "--session-id",
            "session_supplied",
            "--report-index",
            "1",
        ],
        output_stream=output,
    )

    payload = json.loads(output.getvalue())
    assert exit_code == 0
    assert payload["trace"]["group_id"] == "session_supplied"


def test_loop_export_requires_session_id_for_sessionless_jsonl(tmp_path):
    input_path = write_jsonl(
        tmp_path,
        sample_report_without_session("run_no_session"),
    )
    errors = io.StringIO()

    exit_code = main(
        [
            "--adapter",
            "openai-trace",
            "--input",
            str(input_path),
        ],
        error_stream=errors,
    )

    assert exit_code == 2
    assert "Invalid loop session JSONL at line 1" in errors.getvalue()
    assert "supply --session-id explicitly" in errors.getvalue()


def test_loop_export_does_not_infer_missing_later_report_session(tmp_path):
    input_path = write_jsonl(
        tmp_path,
        sample_report("run_with_session"),
        sample_report_without_session("run_without_session"),
    )
    errors = io.StringIO()

    exit_code = main(
        [
            "--adapter",
            "langgraph-manifest",
            "--input",
            str(input_path),
        ],
        error_stream=errors,
    )

    assert exit_code == 2
    assert "Invalid loop session JSONL at line 2" in errors.getvalue()
    assert "supply --session-id explicitly" in errors.getvalue()


def test_loop_export_rejects_explicit_empty_session_id(tmp_path):
    input_path = write_jsonl(tmp_path, sample_report("run_one"))
    errors = io.StringIO()

    exit_code = main(
        [
            "--adapter",
            "openai-trace",
            "--input",
            str(input_path),
            "--session-id",
            "",
        ],
        error_stream=errors,
    )

    assert exit_code == 2
    assert "--session-id must be a canonical identity or None" in errors.getvalue()


def test_loop_export_public_default_redacts_terminal_content(tmp_path):
    secret = "cli secret should not leak"
    input_path = write_jsonl(tmp_path, sample_report("run_secret", secret=secret))
    output = io.StringIO()

    exit_code = main(
        [
            "--adapter",
            "langgraph-manifest",
            "--input",
            str(input_path),
        ],
        output_stream=output,
    )

    assert exit_code == 0
    assert secret not in output.getvalue()


def test_loop_export_raw_requires_explicit_flag(tmp_path):
    secret = "cli raw secret"
    input_path = write_jsonl(tmp_path, sample_report("run_secret", secret=secret))
    output = io.StringIO()

    exit_code = main(
        [
            "--adapter",
            "langgraph-manifest",
            "--input",
            str(input_path),
            "--raw",
        ],
        output_stream=output,
    )

    assert exit_code == 0
    assert json.loads(output.getvalue())["source_projection_schema_version"] is None
    assert secret in output.getvalue()


@pytest.mark.parametrize("adapter", ["openai-trace", "langgraph-manifest"])
def test_loop_export_raw_preserves_historical_v1_report(adapter):
    original = json.loads(LEGACY_V1_JSONL.read_text(encoding="utf-8"))
    output = io.StringIO()
    errors = io.StringIO()

    exit_code = main(
        [
            "--adapter", adapter,
            "--input", str(LEGACY_V1_JSONL),
            "--report-index", "1",
            "--raw",
        ],
        output_stream=output,
        error_stream=errors,
    )

    assert exit_code == 0, errors.getvalue()
    payload = json.loads(output.getvalue())
    assert payload["public"] is False
    assert payload["source_projection_schema_version"] is None
    migrated = payload["source_report"]
    assert migrated["run"].pop("evidence") == []
    assert migrated["run"].pop("terminal_reason") == "unspecified"
    for step in migrated["run"]["steps"]:
        if step["verification"] is not None:
            verification = step["verification"]
            assert verification.pop("verifier_backend") is None
            assert verification.pop("verifier_model_label") is None
            assert verification.pop("same_model_as_drafter") is None
    # Every original field, including raw verifier text, survives unchanged.
    assert migrated == original


@pytest.mark.parametrize("visibility_args", [[], ["--public"]])
def test_loop_export_historical_v1_requires_explicit_raw(visibility_args):
    output = io.StringIO()
    errors = io.StringIO()

    exit_code = main(
        [
            "--adapter", "openai-trace",
            "--input", str(LEGACY_V1_JSONL),
            *visibility_args,
        ],
        output_stream=output,
        error_stream=errors,
    )

    assert exit_code == 2
    assert output.getvalue() == ""
    assert "must exactly match its canonical" in errors.getvalue()


@pytest.mark.parametrize("adapter", ["openai-trace", "langgraph-manifest"])
def test_raw_legacy_loading_does_not_invent_public_provenance(adapter):
    session = loop_export_module.load_session(
        str(LEGACY_V1_JSONL), allow_legacy_raw=True
    )

    with pytest.raises(ValueError, match="verifier identity.*backend provenance"):
        export_payload(session, adapter_name=adapter, public=True)


def test_loop_export_raw_accepts_mixed_historical_and_current_session(tmp_path):
    legacy = json.loads(LEGACY_V1_JSONL.read_text(encoding="utf-8"))
    current = sample_report("run_current").to_dict()
    current["run"]["session_id"] = legacy["run"]["session_id"]
    input_path = tmp_path / "mixed-session.jsonl"
    input_path.write_text(
        json.dumps(legacy) + "\n" + json.dumps(current) + "\n",
        encoding="utf-8",
    )
    output = io.StringIO()
    errors = io.StringIO()

    exit_code = main(
        ["--adapter", "openai-trace", "--input", str(input_path), "--raw"],
        output_stream=output,
        error_stream=errors,
    )

    assert exit_code == 0, errors.getvalue()
    traces = json.loads(output.getvalue())["traces"]
    assert [trace["source_jsonl_line"] for trace in traces] == [1, 2]
    assert [trace["metadata"]["loopwright_run_id"] for trace in traces] == [
        "run_legacy_v1", "run_current"
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_policy",
        "unknown_report_field",
        "unknown_run_field",
        "unknown_step_field",
        "unknown_verifier_field",
        "partial_evidence",
        "partial_terminal_reason",
        "partial_verifier_backend",
        "partial_verifier_model_label",
        "partial_same_model_as_drafter",
        "boolean_duration",
        "boolean_retry_budget",
        "normalized_timestamp",
    ],
)
def test_raw_legacy_loading_rejects_malformed_or_mixed_shapes(tmp_path, mutation):
    payload = json.loads(LEGACY_V1_JSONL.read_text(encoding="utf-8"))
    run = payload["run"]
    verifier = run["steps"][2]["verification"]
    if mutation == "missing_policy":
        run.pop("policy")
    elif mutation == "unknown_report_field":
        payload["unknown"] = "unexpected"
    elif mutation == "unknown_run_field":
        run["unknown"] = "unexpected"
    elif mutation == "unknown_step_field":
        run["steps"][0]["unknown"] = "unexpected"
    elif mutation == "unknown_verifier_field":
        verifier["unknown"] = "unexpected"
    elif mutation == "partial_evidence":
        run["evidence"] = []
    elif mutation == "partial_terminal_reason":
        run["terminal_reason"] = "unspecified"
    elif mutation.startswith("partial_"):
        verifier[mutation.removeprefix("partial_")] = None
    elif mutation == "boolean_duration":
        run["steps"][0]["duration_ms"] = False
    elif mutation == "boolean_retry_budget":
        run["policy"]["max_retries"] = True
    elif mutation == "normalized_timestamp":
        run["started_at"] = run["started_at"].replace("Z", "+00:00")
    input_path = tmp_path / "malformed-legacy.jsonl"
    input_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    output = io.StringIO()
    errors = io.StringIO()

    exit_code = main(
        ["--adapter", "openai-trace", "--input", str(input_path), "--raw"],
        output_stream=output,
        error_stream=errors,
    )

    assert exit_code == 2
    assert output.getvalue() == ""
    assert "Invalid loop report JSONL at line 1" in errors.getvalue()


@pytest.mark.parametrize("mutation", ["duplicate_key", "nonfinite_value"])
def test_raw_legacy_loading_retains_strict_json_parsing(tmp_path, mutation):
    serialized = LEGACY_V1_JSONL.read_text(encoding="utf-8")
    if mutation == "duplicate_key":
        serialized = serialized.replace(
            '"run_id": "run_legacy_v1"',
            '"run_id": "other", "run_id": "run_legacy_v1"',
        )
    else:
        serialized = serialized.replace('"max_retries": 1', '"max_retries": NaN')
    input_path = tmp_path / "invalid-json-legacy.jsonl"
    input_path.write_text(serialized, encoding="utf-8")
    output = io.StringIO()
    errors = io.StringIO()

    exit_code = main(
        ["--adapter", "openai-trace", "--input", str(input_path), "--raw"],
        output_stream=output,
        error_stream=errors,
    )

    assert exit_code == 2
    assert output.getvalue() == ""
    assert "Invalid loop report JSONL at line 1" in errors.getvalue()


def test_export_payload_does_not_invent_line_provenance_for_memory_session():
    session = LoopSession(session_id="session_cli").add_report(
        sample_report("run_one")
    )

    payload = export_payload(
        session,
        adapter_name="openai-trace",
        public=True,
        report_index=1,
    )

    assert payload["trace"]["source_jsonl_line"] is None


def test_loop_export_help_describes_projection_without_replay_claim(capsys):
    with pytest.raises(SystemExit) as exc_info:
        parse_args(["--help"])

    help_text = capsys.readouterr().out
    assert exc_info.value.code == 0
    assert "replay artifacts" not in help_text
    assert "does not replay or execute a run" in help_text
    assert "not access control" in help_text
    assert "general secret/PII scrub" in help_text


def test_loop_export_rejects_duplicate_run_identity(tmp_path):
    report = sample_report("run_duplicate")
    input_path = tmp_path / "duplicate-run.jsonl"
    serialized = json.dumps(report.to_dict())
    input_path.write_text(f"{serialized}\n{serialized}\n", encoding="utf-8")
    errors = io.StringIO()

    exit_code = main(
        [
            "--adapter",
            "openai-trace",
            "--input",
            str(input_path),
        ],
        error_stream=errors,
    )

    assert exit_code == 2
    assert "Invalid loop session JSONL at line 2" in errors.getvalue()
    assert "first seen at line 1" in errors.getvalue()
    assert "duplicate run ids" in errors.getvalue()


def test_loop_export_rejects_duplicate_keys_within_one_jsonl_record(tmp_path):
    payload = sample_report("run_one").to_dict()
    serialized = json.dumps(payload, separators=(",", ":"))
    canonical_identity = '"run_id":"run_one"'
    assert serialized.count(canonical_identity) == 1
    duplicate_key_json = serialized.replace(
        canonical_identity,
        '"run_id":"run_foreign","run_id":"run_one"',
        1,
    )
    input_path = tmp_path / "duplicate-key.jsonl"
    input_path.write_text(duplicate_key_json + "\n", encoding="utf-8")
    errors = io.StringIO()

    exit_code = main(
        [
            "--adapter",
            "openai-trace",
            "--input",
            str(input_path),
        ],
        error_stream=errors,
    )

    assert exit_code == 2
    assert "Invalid loop report JSONL at line 1" in errors.getvalue()
    assert "duplicate JSON object key: 'run_id'" in errors.getvalue()


def test_loop_export_rejects_record_without_explicit_source_schema(tmp_path):
    payload = sample_report("run_one").to_dict()
    payload.pop("schema_version")
    input_path = tmp_path / "missing-schema.jsonl"
    input_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    errors = io.StringIO()

    exit_code = main(
        [
            "--adapter",
            "openai-trace",
            "--input",
            str(input_path),
        ],
        error_stream=errors,
    )

    assert exit_code == 2
    assert "schema_version must be exactly 'loop-report/v1'" in errors.getvalue()


def test_loop_export_rejects_record_that_omits_defaulted_policy(tmp_path):
    payload = sample_report("run_one").to_dict()
    payload["run"].pop("policy")
    input_path = tmp_path / "missing-policy.jsonl"
    input_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    errors = io.StringIO()

    exit_code = main(
        [
            "--adapter",
            "langgraph-manifest",
            "--input",
            str(input_path),
        ],
        error_stream=errors,
    )

    assert exit_code == 2
    assert "must exactly match its canonical" in errors.getvalue()


def test_loop_export_rejects_record_with_normalized_timestamp_spelling(tmp_path):
    payload = sample_report("run_one").to_dict()
    payload["run"]["started_at"] = payload["run"]["started_at"].replace(
        "Z",
        "+00:00",
    )
    input_path = tmp_path / "normalized-timestamp.jsonl"
    input_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    errors = io.StringIO()

    exit_code = main(
        [
            "--adapter",
            "openai-trace",
            "--input",
            str(input_path),
        ],
        error_stream=errors,
    )

    assert exit_code == 2
    assert "must exactly match its canonical" in errors.getvalue()


@pytest.mark.parametrize("nonfinite_token", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_loop_export_rejects_nonfinite_json_numbers(tmp_path, nonfinite_token):
    payload = sample_report("run_one").to_dict()
    payload["run"]["metadata"]["hostile_number"] = 0.0
    serialized = json.dumps(payload, separators=(",", ":"))
    serialized = serialized.replace(
        '"hostile_number":0.0',
        f'"hostile_number":{nonfinite_token}',
        1,
    )
    input_path = tmp_path / "nonfinite.jsonl"
    input_path.write_text(serialized + "\n", encoding="utf-8")
    errors = io.StringIO()

    exit_code = main(
        [
            "--adapter",
            "langgraph-manifest",
            "--input",
            str(input_path),
        ],
        error_stream=errors,
    )

    assert exit_code == 2
    assert "Invalid loop report JSONL at line 1" in errors.getvalue()
    assert (
        "non-standard JSON constant" in errors.getvalue()
        or "non-finite JSON number" in errors.getvalue()
    )


def test_loop_export_never_serializes_nonfinite_adapter_output(
    tmp_path,
    monkeypatch,
):
    input_path = write_jsonl(tmp_path, sample_report("run_one"))
    output = io.StringIO()
    errors = io.StringIO()
    monkeypatch.setattr(
        loop_export_module,
        "export_payload",
        lambda *args, **kwargs: {"hostile_number": float("nan")},
    )

    exit_code = main(
        [
            "--adapter",
            "openai-trace",
            "--input",
            str(input_path),
        ],
        output_stream=output,
        error_stream=errors,
    )

    assert exit_code == 2
    assert output.getvalue() == ""
    assert "JSON compliant" in errors.getvalue()


def test_loop_export_rejects_empty_jsonl(tmp_path):
    input_path = tmp_path / "empty.jsonl"
    input_path.write_text("", encoding="utf-8")
    errors = io.StringIO()

    exit_code = main(
        [
            "--adapter",
            "openai-trace",
            "--input",
            str(input_path),
        ],
        error_stream=errors,
    )

    assert exit_code == 2
    assert "No loop reports found" in errors.getvalue()


def test_loop_export_rejects_blank_lines_to_preserve_line_provenance(tmp_path):
    input_path = tmp_path / "blank-lines.jsonl"
    input_path.write_text(
        "\n" + json.dumps(sample_report("run_one").to_dict()) + "\n",
        encoding="utf-8",
    )
    errors = io.StringIO()

    exit_code = main(
        [
            "--adapter",
            "langgraph-manifest",
            "--input",
            str(input_path),
            "--report-index",
            "1",
        ],
        error_stream=errors,
    )

    assert exit_code == 2
    assert "Invalid loop report JSONL at line 1: blank line" in errors.getvalue()


def test_loop_export_rejects_malformed_report_jsonl_with_line_number(tmp_path):
    input_path = tmp_path / "malformed.jsonl"
    input_path.write_text('{"not":"a loop report"}\n', encoding="utf-8")
    errors = io.StringIO()

    exit_code = main(
        [
            "--adapter",
            "openai-trace",
            "--input",
            str(input_path),
        ],
        error_stream=errors,
    )

    assert exit_code == 2
    assert "Invalid loop report JSONL at line 1" in errors.getvalue()
    assert "run" in errors.getvalue()


@pytest.mark.parametrize(
    ("identity_path", "bad_identity"),
    [
        ("run_id", None),
        ("run_id", 7),
        ("run_id", True),
        ("run_id", []),
        ("run_id", {}),
        ("session_id", 7),
        ("session_id", True),
        ("session_id", []),
        ("session_id", {}),
        ("step_id", None),
        ("step_id", 7),
        ("step_id", True),
        ("step_id", []),
        ("step_id", {}),
    ],
)
def test_loop_export_rejects_non_string_json_identities(
    tmp_path,
    identity_path,
    bad_identity,
):
    payload = sample_report("run_one").to_dict()
    if identity_path == "step_id":
        payload["run"]["steps"][0]["step_id"] = bad_identity
        expected_field = "run.steps[0].step_id"
    else:
        payload["run"][identity_path] = bad_identity
        expected_field = f"run.{identity_path}"
    input_path = tmp_path / "bad-identity.jsonl"
    input_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    errors = io.StringIO()

    exit_code = main(
        [
            "--adapter",
            "openai-trace",
            "--input",
            str(input_path),
        ],
        error_stream=errors,
    )

    assert exit_code == 2
    assert "Invalid loop report JSONL at line 1" in errors.getvalue()
    assert expected_field in errors.getvalue()
    assert "non-empty string" in errors.getvalue()


@pytest.mark.parametrize("payload", ['"not a report"', "[]", "123", "null"])
def test_loop_export_rejects_non_object_jsonl_with_line_number(tmp_path, payload):
    input_path = tmp_path / "non-object.jsonl"
    input_path.write_text(payload + "\n", encoding="utf-8")
    errors = io.StringIO()

    exit_code = main(
        [
            "--adapter",
            "openai-trace",
            "--input",
            str(input_path),
        ],
        error_stream=errors,
    )

    assert exit_code == 2
    assert "Invalid loop report JSONL at line 1" in errors.getvalue()
    assert "expected object" in errors.getvalue()


def test_loop_export_rejects_invalid_json_with_line_number(tmp_path):
    input_path = tmp_path / "invalid.jsonl"
    input_path.write_text(
        json.dumps(sample_report("run_one").to_dict()) + "\n{bad json}\n",
        encoding="utf-8",
    )
    errors = io.StringIO()

    exit_code = main(
        [
            "--adapter",
            "openai-trace",
            "--input",
            str(input_path),
        ],
        error_stream=errors,
    )

    assert exit_code == 2
    assert "Invalid loop report JSONL at line 2" in errors.getvalue()


def test_loop_export_rejects_out_of_range_report_index(tmp_path):
    input_path = write_jsonl(tmp_path, sample_report("run_one"))
    errors = io.StringIO()

    exit_code = main(
        [
            "--adapter",
            "openai-trace",
            "--input",
            str(input_path),
            "--report-index",
            "2",
        ],
        error_stream=errors,
    )

    assert exit_code == 2
    assert "--report-index must be between 1 and 1" in errors.getvalue()


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("trailing_newline", [False, True])
def test_load_session_preserves_unicode_separators_inside_json_strings(
    tmp_path, newline, trailing_newline
):
    first = sample_report("run_one").to_dict()
    first["run"]["user_input"] = "first\u0085second\u2028third\u2029fourth"
    second = sample_report("run_two").to_dict()
    content = newline.join(
        json.dumps(payload, ensure_ascii=False) for payload in (first, second)
    )
    if trailing_newline:
        content += newline
    input_path = tmp_path / "unicode-separators.jsonl"
    input_path.write_bytes(content.encode("utf-8"))

    session = loop_export_module.load_session(str(input_path))

    assert session.report_count == 2
    assert session.reports[0].run.user_input == first["run"]["user_input"]
    assert session.reports[1].run.run_id == "run_two"


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_load_session_rejects_trailing_blank_record_with_physical_line(
    tmp_path, newline
):
    input_path = tmp_path / "trailing-blank.jsonl"
    serialized = json.dumps(sample_report("run_one").to_dict())
    input_path.write_bytes((serialized + newline + newline).encode("utf-8"))

    with pytest.raises(ValueError, match="JSONL at line 2: blank line"):
        loop_export_module.load_session(str(input_path))


@pytest.mark.parametrize("raw_args", [[], ["--raw"]])
@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_loop_export_reports_invalid_utf8_at_physical_line(
    tmp_path, raw_args, newline
):
    first = sample_report("run_one").to_dict()
    first["run"]["user_input"] = "unicode\u0085separators\u2028remain\u2029data"
    input_path = tmp_path / "invalid-utf8.jsonl"
    input_path.write_bytes(
        json.dumps(first, ensure_ascii=False).encode("utf-8")
        + newline
        + b"\xff"
        + newline
    )
    output = io.StringIO()
    errors = io.StringIO()

    exit_code = main(
        ["--adapter", "openai-trace", "--input", str(input_path), *raw_args],
        output_stream=output,
        error_stream=errors,
    )

    assert exit_code == 2
    assert output.getvalue() == ""
    assert errors.getvalue() == (
        "Invalid loop report JSONL at line 2: invalid UTF-8\n"
    )


@pytest.mark.parametrize("raw_args", [[], ["--raw"]])
def test_loop_export_rejects_excessive_json_nesting_with_line(
    tmp_path, raw_args
):
    input_path = tmp_path / "deeply-nested.jsonl"
    serialized = json.dumps(sample_report("run_one").to_dict())
    input_path.write_text(
        serialized + "\n" + "[" * 10000 + "0" + "]" * 10000 + "\n",
        encoding="utf-8",
    )
    output = io.StringIO()
    errors = io.StringIO()

    exit_code = main(
        ["--adapter", "langgraph-manifest", "--input", str(input_path), *raw_args],
        output_stream=output,
        error_stream=errors,
    )

    assert exit_code == 2
    assert output.getvalue() == ""
    assert errors.getvalue() == (
        "Invalid loop report JSONL at line 2: "
        "JSON nesting exceeds the supported depth\n"
    )


@pytest.mark.parametrize(
    ("supplied_session_id", "expected_line"),
    [(None, 2), ("session_cli", 2), ("session_other", 1)],
)
def test_load_session_reports_conflicting_session_at_source_line(
    tmp_path, supplied_session_id, expected_line
):
    first = sample_report("run_one").to_dict()
    second = sample_report("run_two").to_dict()
    second["run"]["session_id"] = "session_other"
    input_path = tmp_path / "conflicting-session.jsonl"
    input_path.write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8"
    )

    with pytest.raises(
        ValueError,
        match=(
            f"Invalid loop session JSONL at line {expected_line}: "
            "LoopSession cannot contain reports from another session"
        ),
    ):
        loop_export_module.load_session(
            str(input_path), session_id=supplied_session_id
        )
