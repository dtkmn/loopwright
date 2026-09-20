"""Read-only inspection and comparison of recorded loops; no model execution."""

import argparse
import json
import sys
import unicodedata
from typing import Optional, Sequence

from src.loop_diff import compare_reports
from src.loop_export import load_session


INSPECTION_SCHEMA_VERSION = "loop-inspection/v1"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect and compare recorded Loopwright runs offline. No model execution.",
        epilog="Deterministic model re-execution is not implemented.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser(
        "inspect",
        help="Explain runs from a local raw loop-report/v1 JSONL artifact.",
        description=(
            "Read recorded evidence and decisions using the public projection by "
            "default. Recorded verifier claims are not independent proof of correctness."
        ),
        epilog=(
            "Public projection is data minimization, not access control or a "
            "general secret/PII scrub. Output can still contain sensitive answers."
        ),
    )
    inspect.add_argument("input", help="Path to raw loop-report/v1 JSONL records.")
    _add_output_options(inspect)
    inspect.add_argument(
        "--report-index", type=int,
        help="Show one run by 1-based JSONL line; the entire input is still validated.",
    )
    inspect.add_argument(
        "--session-id", help="Required if any input report omits its session id.",
    )
    diff = commands.add_parser(
        "diff",
        help="Compare two selected runs from local raw JSONL artifacts.",
        description=(
            "Compare recorded operational fields; task equivalence is not inferred. "
            "Generated IDs and absolute timestamps do not count as material changes."
        ),
    )
    diff.add_argument("before", help="Raw session JSONL containing the before run.")
    diff.add_argument("after", help="Raw session JSONL containing the after run.")
    for side in ("before", "after"):
        diff.add_argument(
            f"--{side}-report-index", type=int,
            help=f"1-based {side} JSONL line; required if that file contains multiple runs.",
        )
        diff.add_argument(
            f"--{side}-session-id",
            help=f"Explicit session id if {side} reports omit one.",
        )
    _add_output_options(diff)
    return parser.parse_args(argv)


def _add_output_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format", choices=("text", "json"), default="text",
        help="Readable summary (default) or a versioned JSON envelope.",
    )
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument(
        "--public", dest="public", action="store_true", default=True,
        help="Use the shared versioned public projection (default).",
    )
    visibility.add_argument(
        "--raw", dest="public", action="store_false",
        help=(
            "Use unprojected local diagnostics, including suppressed content. Also accepts "
            "historical v1 records; absent provenance stays unknown."
        ),
    )


def inspect_artifact(
    input_path: str,
    *,
    public: bool = True,
    report_index: Optional[int] = None,
    session_id: Optional[str] = None,
) -> dict:
    """Validate the whole artifact before returning any selected report data."""

    session = load_session(
        input_path, session_id=session_id, allow_legacy_raw=not public,
    )
    if report_index is not None and (
        type(report_index) is not int or not 1 <= report_index <= session.report_count
    ):
        raise ValueError(f"--report-index must be between 1 and {session.report_count}.")

    reports = []
    for line_number, report in enumerate(session.reports, start=1):
        try:
            payload = report.to_public_dict() if public else report.to_dict()
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError(
                f"Invalid loop report JSONL at line {line_number}: {exc}"
            ) from exc
        if report_index is None or report_index == line_number:
            reports.append({"source_jsonl_line": line_number, "report": payload})
    return {
        "schema_version": INSPECTION_SCHEMA_VERSION,
        "source_path": str(input_path),
        "public": public,
        "session_id": session.session_id,
        "input_report_count": session.report_count,
        "report_count": len(reports),
        "reports": reports,
    }


def _select_diff_report(inspection: dict, index: Optional[int], side: str) -> dict:
    count = inspection["input_report_count"]
    if index is None:
        if count != 1:
            raise ValueError(
                f"{side} artifact {inspection['source_path']} contains {count} runs; "
                f"select one with --{side}-report-index."
            )
        index = 1
    if type(index) is not int or not 1 <= index <= count:
        raise ValueError(f"--{side}-report-index must be between 1 and {count}.")
    return inspection["reports"][index - 1]


def _diff_source(inspection: dict, entry: dict) -> dict:
    report = entry["report"]
    source = {
        "source_path": inspection["source_path"],
        "source_jsonl_line": entry["source_jsonl_line"],
        "run_id": report["run"]["run_id"],
        "session_id": inspection["session_id"],
        "projection_schema_version": report.get("projection_schema_version"),
    }
    if not inspection["public"]:
        source["raw_report"] = report
    return source


def diff_artifacts(
    before_path: str,
    after_path: str,
    *,
    public: bool = True,
    before_report_index: Optional[int] = None,
    after_report_index: Optional[int] = None,
    before_session_id: Optional[str] = None,
    after_session_id: Optional[str] = None,
) -> dict:
    """Validate both complete artifacts, then compare one selected run per side."""

    inspections = []
    for side, path, session_id in (
        ("before", before_path, before_session_id),
        ("after", after_path, after_session_id),
    ):
        try:
            inspections.append(inspect_artifact(path, public=public, session_id=session_id))
        except (OSError, ValueError) as exc:
            detail = str(exc).replace("--session-id", f"--{side}-session-id")
            raise ValueError(f"{side} artifact {path}: {detail}") from exc
    before_inspection, after_inspection = inspections
    before = _select_diff_report(before_inspection, before_report_index, "before")
    after = _select_diff_report(after_inspection, after_report_index, "after")
    return {
        **compare_reports(before["report"], after["report"], public=public),
        "before": _diff_source(before_inspection, before),
        "after": _diff_source(after_inspection, after),
    }


def _display(value) -> str:
    """Keep artifact text from injecting terminal controls or extra output lines."""

    if value is None:
        return "unknown"
    if type(value) is bool:
        return "yes" if value else "no"
    escaped = json.dumps(str(value), ensure_ascii=False)[1:-1]
    return _escape_controls(escaped)


def _escape_controls(value: str) -> str:
    return "".join(
        json.dumps(character, ensure_ascii=True)[1:-1]
        if unicodedata.category(character) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
        else character
        for character in value
    )


def _render_step(index: int, step: dict, *, redacted: bool) -> list[str]:
    lines = [
        f"    {index}. {_display(step['phase'])} -> {_display(step['decision'])} "
        f"[step {_display(step['step_id'])}; retry {step['retry_count']}; "
        f"duration {_display(step['duration_ms'])} ms]"
    ]
    verification = step["verification"]
    if verification is not None:
        lines.append(f"       Verification recorded: {_display(verification['outcome'])}")
        if redacted:
            lines.append("       Verifier identity: withheld by terminal public redaction")
        else:
            lines.append(
                f"       Verifier: {_display(verification['verifier_backend'])} / "
                f"{_display(verification['verifier_model_label'])}; "
                f"same model as drafter: {_display(verification['same_model_as_drafter'])}"
            )
    if step.get("human_review_required") or step.get("human_review") is not None:
        lines.append("       Human review requested")
    if step.get("error_present") or step.get("error_message"):
        lines.append("       Error recorded")
    return lines


def _render_report(entry: dict, *, public: bool) -> list[str]:
    report = entry["report"]
    run = report["run"]
    redaction = report.get("public_redaction", {})
    redacted = redaction.get("applied", False)
    if not public:
        projection = "not applied (raw diagnostics; not validated for public serving)"
    elif redacted:
        projection = f"{report['projection_schema_version']}; terminal content withheld"
    else:
        projection = report["projection_schema_version"]
    steps = run["steps"]
    retries = max((step["retry_count"] for step in steps), default=0)
    lines = [
        "",
        f"Run {_display(run['run_id'])} [JSONL line {entry['source_jsonl_line']}]",
        f"  Projection: {projection}",
        f"  Final decision: {_display(run['final_decision'])}",
        f"  Terminal reason: {_display(run['terminal_reason'])}",
        f"  Started: {_display(run['started_at'])}; completed: {_display(run['completed_at'])}",
        f"  Context provider: {_display(run['context_provider'])}",
        f"  Highest recorded retry count: {retries}; limit: {run['policy']['max_retries']}",
        f"  Error recorded: {_display(bool(run.get('error_present') or run.get('error_message')))}",
    ]
    if redacted:
        lines.extend([
            "  Model: withheld by terminal public redaction",
            "  Evidence: withheld by terminal public redaction (count unknown)",
        ])
    else:
        lines.extend([
            f"  Model: {_display(run['backend'])} / {_display(run['model_label'])}",
            f"  Evidence references recorded: {len(run['evidence'])}",
        ])
        for reference in run["evidence"]:
            locator = reference["locator"]
            lines.append(
                f"    [{reference['citation_id']}] {_display(reference['provider'])}: "
                f"{_display(reference['evidence_id'])}; "
                f"page {_display(locator['page'])}, chunk {_display(locator['chunk_index'])}"
            )
    lines.append("  Steps (recorded order):" if steps else "  Steps: none recorded")
    for index, step in enumerate(steps, start=1):
        lines.extend(_render_step(index, step, redacted=redacted))
    if redacted:
        lines.append("  Final answer: withheld by terminal public redaction")
    elif run["final_answer"] is None:
        lines.append("  Final answer: not available")
    else:
        lines.append(f"  Final answer: {_display(run['final_answer'])}")
    if not public:
        lines.append("  Full raw diagnostic report:")
        lines.extend("    " + line for line in _json(report).splitlines())
    return lines


def _json(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)


def render_text(inspection: dict) -> str:
    visibility = "public projection" if inspection["public"] else "RAW local diagnostics"
    lines = [
        f"Loopwright run inspection ({visibility})",
        f"Source: {_display(inspection['source_path'])}",
        f"Session: {_display(inspection['session_id'])}",
        f"Reports shown: {inspection['report_count']} of {inspection['input_report_count']}",
        "Recorded observations only; no model execution or independent verification.",
        "Different or unknown verifier identity does not establish independence.",
    ]
    for entry in inspection["reports"]:
        lines.extend(_render_report(entry, public=inspection["public"]))
    return "\n".join(lines) + "\n"


def _diff_value(value) -> str:
    if isinstance(value, dict) and set(value) == {"availability"}:
        return f"<{value['availability']}>"
    # JSON retains types and distinguishes literal strings from availability labels.
    return _escape_controls(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


def _step_summary(step: dict) -> str:
    summary = (
        f"{_display(step['phase'])} -> {_display(step['decision'])} "
        f"(retry {step['retry_count']}, completed: {_display(step['completed'])})"
    )
    verification = step["verification"]
    if verification is not None and "outcome" in verification:
        summary += f"; verification recorded: {_display(verification['outcome'])}"
        summary += f"; verifier: {_diff_value(verification['model'])}"
    if step["human_review_required"]:
        summary += "; human review requested"
    if step["error_present"]:
        summary += "; error recorded"
    return summary


def _render_changes(changes: list[dict]) -> list[str]:
    lines = []
    for change in changes:
        if change["kind"] in {"added", "removed"}:
            side = "after" if change["kind"] == "added" else "before"
            index = change[f"{side}_step"]["index"]
            lines.append(
                f"  {change['kind'].capitalize()} {side} step {index}: "
                f"{_step_summary(change[side])}"
            )
            continue
        location = change["field"]
        if "before_step" in change:
            before_index = (change["before_step"] or {}).get("index", "absent")
            after_index = (change["after_step"] or {}).get("index", "absent")
            location += f" [before step {before_index}, after step {after_index}]"
        lines.append(f"  {_display(location)} ({change['kind']}):")
        lines.append(f"    before: {_diff_value(change['before'])}")
        lines.append(f"    after:  {_diff_value(change['after'])}")
    return lines


def render_diff_text(comparison: dict) -> str:
    visibility = "public projection" if comparison["public"] else "RAW local diagnostics"
    lines = [f"Loopwright run comparison ({visibility})"]
    for side in ("before", "after"):
        source = comparison[side]
        lines.append(
            f"{side.capitalize()}: {_display(source['source_path'])} "
            f"[JSONL line {source['source_jsonl_line']}; run {_display(source['run_id'])}; "
            f"session {_display(source['session_id'])}]"
        )
    lines.extend([
        "Selected runs only; same task not established. No model execution or independent verification.",
        "Outcome: " + " -> ".join(
            _diff_value(comparison["summary"][side]["decision"]) for side in ("before", "after")
        ),
        "Terminal reason: " + " -> ".join(
            _diff_value(comparison["summary"][side]["terminal_reason"]) for side in ("before", "after")
        ),
        "Highest recorded retry count: " + " -> ".join(
            str(comparison["summary"][side]["retry_count"]) for side in ("before", "after")
        ),
        "",
        (
            f"Material changes in recorded fields: {len(comparison['material_changes'])}"
            if comparison["has_material_changes"]
            else "No observed material changes in comparable recorded fields."
        ),
    ])
    lines.extend(_render_changes(comparison["material_changes"]))
    lines.append(f"Timing changes (separate from material changes): {len(comparison['timing_changes'])}")
    lines.extend(_render_changes(comparison["timing_changes"]))
    if comparison["unavailable_fields"]:
        lines.append("Unavailable fields (cannot establish equality):")
        for field in comparison["unavailable_fields"]:
            lines.append(f"  {field['side']}.{field['field']}: {field['availability']}")
    lines.append("Comparison limits:")
    lines.extend(f"  {note}" for note in comparison["limitations"])
    return "\n".join(lines) + "\n"


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    output_stream=None,
    error_stream=None,
) -> int:
    output_stream = output_stream or sys.stdout
    error_stream = error_stream or sys.stderr
    args = parse_args(argv)
    try:
        if args.command == "inspect":
            payload = inspect_artifact(
                args.input, public=args.public, report_index=args.report_index,
                session_id=args.session_id,
            )
            renderer = render_text
        else:
            payload = diff_artifacts(
                args.before, args.after, public=args.public,
                before_report_index=args.before_report_index,
                after_report_index=args.after_report_index,
                before_session_id=args.before_session_id,
                after_session_id=args.after_session_id,
            )
            renderer = render_diff_text
        output = _json(payload) + "\n" if args.format == "json" else renderer(payload)
        print(output, end="", file=output_stream)
    except (OSError, ValueError) as exc:
        action = f"inspect {_display(args.input)}" if args.command == "inspect" else "compare runs"
        print(f"Cannot {action}: {_display(str(exc))}", file=error_stream)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
