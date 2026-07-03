import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

from src.adapters.langgraph_manifest import LangGraphManifestAdapter
from src.adapters.openai_trace import OpenAITraceAdapter
from src.loop_engine import LoopReport, LoopSession


ADAPTERS = {
    "openai-trace": OpenAITraceAdapter,
    "openai_trace": OpenAITraceAdapter,
    "langgraph-manifest": LangGraphManifestAdapter,
    "langgraph_manifest": LangGraphManifestAdapter,
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Loopwright JSONL replay artifacts to adapter JSON.",
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
        help="Path to a LoopSession JSONL artifact.",
    )
    parser.add_argument(
        "--output",
        help="Write JSON to this path. Defaults to stdout.",
    )
    parser.add_argument(
        "--session-id",
        help="Supply a session id when reports omit one.",
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
        help="Export public/redacted payloads. This is the default.",
    )
    visibility.add_argument(
        "--raw",
        dest="public",
        action="store_false",
        help="Export raw local diagnostics. Treat output as sensitive.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write compact JSON instead of indented JSON.",
    )
    return parser.parse_args(argv)


def load_session(input_path: str, *, session_id: Optional[str] = None) -> LoopSession:
    path = Path(input_path)
    content = path.read_text(encoding="utf-8")
    reports = _read_reports_jsonl(content)
    if not reports:
        raise ValueError("No loop reports found in input JSONL.")
    inferred_session_id = session_id or reports[0].run.session_id or "default"
    try:
        return LoopSession(
            session_id=inferred_session_id,
            reports=tuple(reports),
        )
    except ValueError as exc:
        raise ValueError(f"Invalid loop session JSONL: {exc}") from exc


def _read_reports_jsonl(content: str) -> list[LoopReport]:
    reports = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            raise ValueError(
                f"Invalid loop report JSONL at line {line_number}: blank line"
            )
        try:
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise TypeError(
                    f"expected object, got {type(payload).__name__}"
                )
            reports.append(LoopReport.from_dict(payload))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid loop report JSONL at line {line_number}: {exc.msg}"
            ) from exc
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid loop report JSONL at line {line_number}: {exc}"
            ) from exc
    return reports


def export_payload(
    session: LoopSession,
    *,
    adapter_name: str,
    public: bool,
    report_index: Optional[int] = None,
) -> dict:
    adapter = ADAPTERS[adapter_name]()
    if report_index is None:
        return adapter.export_session(session, public=public)
    if report_index < 1 or report_index > session.report_count:
        raise ValueError(
            f"--report-index must be between 1 and {session.report_count}."
        )
    return adapter.export_report(
        session.reports[report_index - 1],
        public=public,
        session_id=session.session_id,
        source_jsonl_line=report_index,
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
        session = load_session(args.input, session_id=args.session_id)
        payload = export_payload(
            session,
            adapter_name=args.adapter,
            public=args.public,
            report_index=args.report_index,
        )
        indent = None if args.compact else 2
        serialized = json.dumps(payload, indent=indent, sort_keys=True) + "\n"
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
