import argparse
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

from src.adapters.base import (
    require_optional_session_id,
    source_jsonl_lines_for_session,
)
from src.adapters.langgraph_manifest import LangGraphManifestAdapter
from src.adapters.openai_trace import OpenAITraceAdapter
from src.loop_engine import SCHEMA_VERSION, LoopReport, LoopSession


ADAPTERS = {
    "openai-trace": OpenAITraceAdapter,
    "openai_trace": OpenAITraceAdapter,
    "langgraph-manifest": LangGraphManifestAdapter,
    "langgraph_manifest": LangGraphManifestAdapter,
}


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_json_constant(value: str):
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number is not allowed")
    return parsed


def _strict_json_loads(value: str):
    return json.loads(
        value,
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_nonstandard_json_constant,
        parse_float=_parse_finite_json_float,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert newline-delimited Loopwright loop-report artifacts to "
            "offline framework-shaped JSON. This does not replay or execute a run."
        ),
        epilog=(
            "The public artifact projection is not access control or a general "
            "secret/PII scrub. Inspect every export before sharing it."
        ),
    )
    parser.add_argument(
        "--adapter",
        required=True,
        choices=sorted(ADAPTERS),
        help="Adapter export shape to produce.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to newline-delimited loop-report/v1 records.",
    )
    parser.add_argument(
        "--output",
        help="Write JSON to this path. Defaults to stdout.",
    )
    parser.add_argument(
        "--session-id",
        help="Supply a session id. Required when input reports omit one.",
    )
    parser.add_argument(
        "--report-index",
        type=int,
        help="Export one report by 1-based JSONL line index instead of the session.",
    )
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument(
        "--public",
        dest="public",
        action="store_true",
        default=True,
        help=(
            "Export the versioned public artifact projection (default). This is "
            "field projection, not access control or a general secret/PII scrub."
        ),
    )
    visibility.add_argument(
        "--raw",
        dest="public",
        action="store_false",
        help="Export the unprojected local diagnostic form. Treat output as sensitive.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write compact JSON instead of indented JSON.",
    )
    return parser.parse_args(argv)


def load_session(
    input_path: str,
    *,
    session_id: Optional[str] = None,
    allow_legacy_raw: bool = False,
) -> LoopSession:
    path = Path(input_path)
    content = path.read_text(encoding="utf-8")
    reports = _read_reports_jsonl(content, allow_legacy_raw=allow_legacy_raw)
    if not reports:
        raise ValueError("No loop reports found in input JSONL.")
    session_id = require_optional_session_id(
        session_id,
        field_name="--session-id",
    )
    if session_id is None:
        if any(report.run.session_id is None for report in reports):
            raise ValueError(
                "Input reports omit session_id; supply --session-id explicitly."
            )
        inferred_session_id = reports[0].run.session_id
    else:
        inferred_session_id = session_id
    try:
        return LoopSession(
            session_id=inferred_session_id,
            reports=tuple(reports),
        )
    except ValueError as exc:
        raise ValueError(f"Invalid loop session JSONL: {exc}") from exc


def _read_reports_jsonl(
    content: str, *, allow_legacy_raw: bool = False
) -> list[LoopReport]:
    reports = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            raise ValueError(
                f"Invalid loop report JSONL at line {line_number}: blank line"
            )
        try:
            payload = _strict_json_loads(line)
            if not isinstance(payload, Mapping):
                raise TypeError(
                    f"expected object, got {type(payload).__name__}"
                )
            _validate_raw_report_identities(payload)
            if (
                type(payload.get("schema_version")) is not str
                or payload.get("schema_version") != SCHEMA_VERSION
            ):
                raise ValueError(
                    f"schema_version must be exactly {SCHEMA_VERSION!r}"
                )
            report = LoopReport.from_dict(payload)
            canonical = report.to_dict()
            if not (
                _canonical_json_values_match(payload, canonical)
                or (
                    allow_legacy_raw is True
                    and _legacy_v1_json_values_match(payload, canonical)
                )
            ):
                raise ValueError(
                    "loop report record must exactly match its canonical "
                    "loop-report/v1 form"
                )
            reports.append(report)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid loop report JSONL at line {line_number}: {exc.msg}"
            ) from exc
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid loop report JSONL at line {line_number}: {exc}"
            ) from exc
    return reports


def _legacy_v1_json_values_match(source: dict, canonical: dict) -> bool:
    """Accept only the complete historical wire shape for raw diagnostics.

    The v1 writer originally omitted evidence, terminal reason, and verifier
    provenance. Compare that exact shape rather than accepting arbitrary
    defaulted omissions. Reconstruction leaves those additions empty/unknown;
    it must never manufacture the bindings required by public projection.
    """

    legacy = dict(canonical)
    run = dict(canonical["run"])
    legacy["run"] = run
    run.pop("evidence")
    run.pop("terminal_reason")
    run["steps"] = []
    for step in canonical["run"]["steps"]:
        legacy_step = dict(step)
        if step["verification"] is not None:
            verification = dict(step["verification"])
            for field in (
                "verifier_backend",
                "verifier_model_label",
                "same_model_as_drafter",
            ):
                verification.pop(field)
            legacy_step["verification"] = verification
        run["steps"].append(legacy_step)
    return _canonical_json_values_match(source, legacy)


def _canonical_json_values_match(source, canonical) -> bool:
    """Compare parsed JSON without Python's bool/int equality shortcuts."""

    if type(source) is not type(canonical):
        return False
    if isinstance(source, dict):
        return source.keys() == canonical.keys() and all(
            _canonical_json_values_match(source[key], canonical[key])
            for key in source
        )
    if isinstance(source, list):
        return len(source) == len(canonical) and all(
            _canonical_json_values_match(source_item, canonical_item)
            for source_item, canonical_item in zip(source, canonical)
        )
    return source == canonical


def _validate_raw_report_identities(payload: Mapping) -> None:
    run = payload.get("run")
    if not isinstance(run, Mapping):
        raise TypeError("run must be an object")

    _require_json_identity(run.get("run_id"), "run.run_id")
    session_id = run.get("session_id")
    if session_id is not None:
        _require_json_identity(session_id, "run.session_id")

    steps = run.get("steps", ())
    if not isinstance(steps, (list, tuple)):
        raise TypeError("run.steps must be an array")
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise TypeError(f"run.steps[{index}] must be an object")
        _require_json_identity(step.get("step_id"), f"run.steps[{index}].step_id")

    evidence = run.get("evidence", ())
    if not isinstance(evidence, (list, tuple)):
        raise TypeError("run.evidence must be an array")
    for index, reference in enumerate(evidence):
        if not isinstance(reference, Mapping):
            raise TypeError(f"run.evidence[{index}] must be an object")
        _require_json_identity(
            reference.get("evidence_id"),
            f"run.evidence[{index}].evidence_id",
        )
        _require_json_identity(
            reference.get("provider"),
            f"run.evidence[{index}].provider",
        )


def _require_json_identity(value, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{field_name} must be a non-empty string")


def export_payload(
    session: LoopSession,
    *,
    adapter_name: str,
    public: bool,
    report_index: Optional[int] = None,
    source_jsonl_lines: Optional[Sequence[Optional[int]]] = None,
) -> dict:
    source_lines = source_jsonl_lines_for_session(
        session.report_count,
        source_jsonl_lines,
    )
    adapter = ADAPTERS[adapter_name]()
    if report_index is None:
        return adapter.export_session(
            session,
            public=public,
            source_jsonl_lines=(
                source_lines if source_jsonl_lines is not None else None
            ),
        )
    if report_index < 1 or report_index > session.report_count:
        raise ValueError(
            f"--report-index must be between 1 and {session.report_count}."
        )
    return adapter.export_report(
        session.reports[report_index - 1],
        public=public,
        session_id=session.session_id,
        source_jsonl_line=source_lines[report_index - 1],
    )


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    output_stream=None,
    error_stream=None,
) -> int:
    output_stream = output_stream or sys.stdout
    error_stream = error_stream or sys.stderr
    try:
        args = parse_args(argv)
        session = load_session(
            args.input,
            session_id=args.session_id,
            allow_legacy_raw=not args.public,
        )
        source_jsonl_lines = tuple(range(1, session.report_count + 1))
        payload = export_payload(
            session,
            adapter_name=args.adapter,
            public=args.public,
            report_index=args.report_index,
            source_jsonl_lines=source_jsonl_lines,
        )
        indent = None if args.compact else 2
        serialized = json.dumps(
            payload,
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(serialized, encoding="utf-8")
        else:
            print(serialized, end="", file=output_stream)
    except (OSError, ValueError) as exc:
        print(str(exc), file=error_stream)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
