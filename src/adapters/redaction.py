from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from src.adapters.base import canonical_raw_report_payload
from src.loop_engine import LoopReport
from src.public_projection import project_public_report


def require_public_bool(public: bool) -> None:
    if type(public) is not bool:
        raise ValueError("public must be a boolean")


def report_payload(report: LoopReport, *, public: bool = True) -> Dict[str, Any]:
    require_public_bool(public)
    if not public:
        return canonical_raw_report_payload(report)

    return project_public_report(report)


def run_with_session_fallback(
    run: Mapping[str, Any],
    session_id: Optional[str],
) -> Mapping[str, Any]:
    if run.get("session_id") or not session_id:
        return run
    run_with_session = dict(run)
    run_with_session["session_id"] = session_id
    return run_with_session
