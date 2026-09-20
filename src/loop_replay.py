"""Read-only inspection of recorded loops; no replay or model execution."""

import argparse
import json
import sys
import unicodedata
from typing import Optional, Sequence

from src.loop_export import load_session


INSPECTION_SCHEMA_VERSION = "loop-inspection/v1"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect recorded Loopwright runs offline. No model execution.",
        epilog="Semantic diff and deterministic re-execution are not implemented.",
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
    inspect.add_argument(
        "--format", choices=("text", "json"), default="text",
        help="Readable summary (default) or a versioned JSON inspection envelope.",
    )
    inspect.add_argument(
        "--report-index", type=int,
        help="Show one run by 1-based JSONL line; the entire input is still validated.",
    )
    inspect.add_argument(
        "--session-id", help="Required if any input report omits its session id.",
    )
    visibility = inspect.add_mutually_exclusive_group()
    visibility.add_argument(
        "--public", dest="public", action="store_true", default=True,
        help="Use the shared versioned public projection (default).",
    )
    visibility.add_argument(
        "--raw", dest="public", action="store_false",
        help=(
            "Include full local diagnostics, including suppressed content. Also accepts "
            "historical v1 records; absent provenance stays unknown."
        ),
    )
    return parser.parse_args(argv)


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


def _display(value) -> str:
    """Keep artifact text from injecting terminal controls or extra output lines."""

    if value is None:
        return "unknown"
    if type(value) is bool:
        return "yes" if value else "no"
    escaped = json.dumps(str(value), ensure_ascii=False)[1:-1]
    return "".join(
        json.dumps(character, ensure_ascii=True)[1:-1]
        if unicodedata.category(character) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
        else character
        for character in escaped
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
        inspection = inspect_artifact(
            args.input, public=args.public, report_index=args.report_index,
            session_id=args.session_id,
        )
        output = (
            _json(inspection) + "\n"
            if args.format == "json" else render_text(inspection)
        )
        print(output, end="", file=output_stream)
    except (OSError, ValueError) as exc:
        print(f"Cannot inspect {_display(args.input)}: {_display(str(exc))}", file=error_stream)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
