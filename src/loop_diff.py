"""Semantic comparison of validated recorded runs, without executing a model.

Only named operational fields participate. Raw mode does not turn arbitrary
metadata or recipe text into a configuration/equivalence contract.
"""

from datetime import datetime, timedelta
from difflib import SequenceMatcher


DIFF_SCHEMA_VERSION = "loop-diff/v1"
_POLICY_FIELDS = (
    "max_retries", "require_citations", "require_verifier_for_supported",
    "allow_mock_supported", "allow_tool_calls", "require_human_review_for_tools",
)


def _unavailable(reason="unknown") -> dict:
    return {"availability": reason}


def _known(value):
    return _unavailable() if value is None else value


def _model(record: dict, *, redacted: bool) -> dict:
    if redacted:
        return _unavailable("redacted")
    return {"backend": _known(record["backend"]), "label": _known(record["model_label"])}


def _step(step: dict, *, redacted: bool) -> dict:
    verification = step["verification"]
    if verification is None:
        verification_value = (
            _unavailable("not_recorded") if step["phase"] == "verify" else None
        )
    else:
        verification_value = {
            "outcome": verification["outcome"],
            "model": (
                _unavailable("redacted") if redacted else {
                    "backend": _known(verification["verifier_backend"]),
                    "label": _known(verification["verifier_model_label"]),
                    "same_model_as_drafter": _known(verification["same_model_as_drafter"]),
                }
            ),
        }
    return {
        "phase": step["phase"],
        "decision": step["decision"],
        "retry_count": step["retry_count"],
        "completed": step["ended_at"] is not None,
        "model": _model(step, redacted=redacted),
        "verification": verification_value,
        "human_review_required": (
            step.get("human_review_required", step.get("human_review") is not None)
        ),
        "error_present": bool(step.get("error_present") or step.get("error_message")),
    }


def _operational_fields(report: dict, *, public: bool) -> dict:
    run = report["run"]
    redacted = report.get("public_redaction", {}).get("applied", False)
    if redacted:
        evidence = _unavailable("redacted")
    elif not public and not run["evidence"]:
        # Historical raw v1 reports reconstruct a missing evidence field as [].
        # Empty raw references cannot establish that a run used no context.
        evidence = _unavailable("not_recorded")
    else:
        evidence = {
            str(reference["citation_id"]): reference
            for reference in sorted(run["evidence"], key=lambda item: item["citation_id"])
        }
    memory = {
        key: _known(run.get(key))
        for key in ("conversation_context_count", "semantic_memory_count", "semantic_memory_status")
    }
    return {
        "outcome": {
            "decision": _known(run["final_decision"]),
            "retry_count": max((step["retry_count"] for step in run["steps"]), default=0),
            "terminal_reason": (
                _unavailable() if run["terminal_reason"] in {None, "unspecified"}
                else run["terminal_reason"]
            ),
            "completed": run["completed_at"] is not None,
            "answer": (
                _unavailable("redacted") if redacted else _known(run["final_answer"])
            ),
            "error_present": bool(run.get("error_present") or run.get("error_message")),
        },
        "terminal_redaction": redacted if public else _unavailable("not_applied"),
        "context_provider": run["context_provider"],
        "memory": memory,
        "model": _model(run, redacted=redacted),
        "policy": {key: run["policy"][key] for key in _POLICY_FIELDS},
        "evidence": evidence,
    }


def _is_unavailable(value) -> bool:
    return isinstance(value, dict) and set(value) == {"availability"}


def _differences(before, after, field: str) -> list[dict]:
    if before == after:
        return []
    if _is_unavailable(before) or _is_unavailable(after):
        return [{"kind": "availability", "field": field, "before": before, "after": after}]
    if isinstance(before, dict) and isinstance(after, dict):
        changes = []
        for key in dict.fromkeys((*before, *after)):
            changes.extend(_differences(
                before.get(key, _unavailable("absent")),
                after.get(key, _unavailable("absent")), f"{field}.{key}" if field else key,
            ))
        return changes
    return [{"kind": "value", "field": field, "before": before, "after": after}]


def _unavailable_fields(value, field: str, side: str) -> list[dict]:
    if _is_unavailable(value):
        return [{"side": side, "field": field, **value}]
    if not isinstance(value, dict):
        return []
    return [
        entry
        for key, item in value.items()
        for entry in _unavailable_fields(item, f"{field}.{key}" if field else key, side)
    ]


def _elapsed_ms(start: str, end: str | None):
    if end is None:
        return _unavailable()
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)) // timedelta(milliseconds=1)


def _step_reference(steps: list[dict], index: int | None):
    if index is None:
        return None
    return {"index": index + 1, "step_id": steps[index]["step_id"]}


def _compare_steps(before: dict, after: dict) -> tuple[list, list, list]:
    before_steps, after_steps = before["run"]["steps"], after["run"]["steps"]
    left = [_step(step, redacted=before.get("public_redaction", {}).get("applied", False)) for step in before_steps]
    right = [_step(step, redacted=after.get("public_redaction", {}).get("applied", False)) for step in after_steps]
    unavailable = [
        entry
        for side, steps in (("before", left), ("after", right))
        for index, step in enumerate(steps, start=1)
        for entry in _unavailable_fields(step, f"steps[{index}]", side)
    ]
    alignment = SequenceMatcher(
        a=[(step["phase"], step["retry_count"]) for step in left],
        b=[(step["phase"], step["retry_count"]) for step in right],
        autojunk=False,
    )
    changes, timing = [], []
    for tag, first_start, first_end, second_start, second_end in alignment.get_opcodes():
        if tag == "equal":
            for first, second in zip(range(first_start, first_end), range(second_start, second_end)):
                refs = {
                    "before_step": _step_reference(before_steps, first),
                    "after_step": _step_reference(after_steps, second),
                }
                changes.extend({**change, **refs} for change in _differences(left[first], right[second], "step"))
                timing.extend({**change, **refs} for change in _differences(
                    _known(before_steps[first]["duration_ms"]),
                    _known(after_steps[second]["duration_ms"]), "step.duration_ms",
                ))
        else:
            for side, steps, start, end in (
                ("before", left, first_start, first_end),
                ("after", right, second_start, second_end),
            ):
                for index in range(start, end):
                    changes.append({
                        "kind": "removed" if side == "before" else "added",
                        "field": "step",
                        "before": steps[index] if side == "before" else _unavailable("absent"),
                        "after": steps[index] if side == "after" else _unavailable("absent"),
                        "before_step": _step_reference(before_steps, index if side == "before" else None),
                        "after_step": _step_reference(after_steps, index if side == "after" else None),
                    })
    return changes, timing, unavailable


def compare_reports(before: dict, after: dict, *, public: bool) -> dict:
    """Compare records already loaded and projected by the inspection boundary."""

    left, right = (_operational_fields(report, public=public) for report in (before, after))
    changes = _differences(left, right, "")
    timing = _differences(
        _elapsed_ms(before["run"]["started_at"], before["run"]["completed_at"]),
        _elapsed_ms(after["run"]["started_at"], after["run"]["completed_at"]),
        "run.duration_ms",
    )
    step_changes, step_timing, step_unavailable = _compare_steps(before, after)
    changes.extend(step_changes)
    timing.extend(step_timing)
    limitations = [
        "Only recorded operational fields are compared; task equivalence is not established.",
        "No observed change does not establish correctness, verifier independence, or full run equivalence.",
        "Generated identities and absolute timestamps are provenance only; durations are reported separately.",
        "Steps align by phase and retry count in recorded order; this does not establish causal identity.",
        "Prompts, recipe/configuration metadata, summaries, verifier reasons, raw errors, and causal metadata references are outside the comparison scope.",
        "Configuration comparison covers recorded policy and model fields only; redacted or unknown fields cannot establish equality.",
    ]
    if not public:
        limitations.extend([
            "Raw diagnostics are not validated for public serving; complete raw reports are included in JSON for manual inspection.",
            "Raw memory metadata is not projected; empty raw evidence and unspecified terminal reasons remain unavailable, including historical v1 omissions.",
        ])
    return {
        "schema_version": DIFF_SCHEMA_VERSION,
        "public": public,
        "same_task_verified": False,
        "summary": {
            side: {key: fields["outcome"][key] for key in ("decision", "terminal_reason", "retry_count")}
            for side, fields in (("before", left), ("after", right))
        },
        "has_material_changes": bool(changes),
        "material_changes": changes,
        "timing_changes": timing,
        "unavailable_fields": (
            _unavailable_fields(left, "", "before")
            + _unavailable_fields(right, "", "after") + step_unavailable
        ),
        "limitations": limitations,
    }
