from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
import re
from typing import Any, Dict, Optional

try:
    from .loop_engine import (
        ANSWER_CANDIDATE_SHA256_METADATA_KEY,
        EVIDENCE_SET_SHA256_METADATA_KEY,
        PUBLIC_REDACTION_REASON,
        SCHEMA_VERSION,
        EvidenceLocator,
        EvidenceReference,
        HumanReviewRequest,
        LoopDecision,
        LoopPhase,
        LoopPolicy,
        LoopReport,
        LoopRun,
        LoopStep,
        LoopTerminalReason,
        VerificationOutcome,
        VerificationResult,
        answer_candidate_sha256,
        evidence_set_sha256,
    )
except ImportError:
    from loop_engine import (
        ANSWER_CANDIDATE_SHA256_METADATA_KEY,
        EVIDENCE_SET_SHA256_METADATA_KEY,
        PUBLIC_REDACTION_REASON,
        SCHEMA_VERSION,
        EvidenceLocator,
        EvidenceReference,
        HumanReviewRequest,
        LoopDecision,
        LoopPhase,
        LoopPolicy,
        LoopReport,
        LoopRun,
        LoopStep,
        LoopTerminalReason,
        VerificationOutcome,
        VerificationResult,
        answer_candidate_sha256,
        evidence_set_sha256,
    )


PUBLIC_REPORT_PROJECTION_SCHEMA_VERSION = "loop-public-report/v1"
_ANSWER_CANDIDATE_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_EVIDENCE_ID_PATTERN = re.compile(r"evidence_[0-9a-f]{64}")
# Match the runtime's citation-marker grammar exactly. Attached bracket
# expressions such as ``fact[1]`` and ``arr[0]`` are ordinary answer text, not
# evidence claims; standalone ``[1]`` markers remain bounded and validated.
_INLINE_CITATION_PATTERN = re.compile(
    r"(?<!\S)\[(\d+)\](?=$|[\s.,;:!?)])"
)
_PUBLIC_IDENTITY_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}")
_PUBLIC_BACKENDS = frozenset({"auto", "mock", "ollama", "openai-compatible"})
_PUBLIC_CONTEXT_PROVIDERS = frozenset({"document", "none", "web"})
_PUBLIC_EVIDENCE_PROVIDERS = frozenset({"document", "web"})
_PUBLIC_MEMORY_STATUSES = frozenset(
    {"empty", "not_requested", "retrieved", "unavailable"}
)
# Keep display-label blankness language-neutral. Python ``str.strip`` and
# JavaScript ``String.trim`` disagree on NEL (U+0085) and BOM (U+FEFF), which
# can otherwise make a canonical server projection impossible to restore in
# the browser. This explicit union is mirrored in ``src/web_static/app.js``.
_PUBLIC_DISPLAY_BLANK_CODE_POINTS = frozenset(
    {
        0x0020,
        0x0085,
        0x00A0,
        0x1680,
        *range(0x2000, 0x200B),
        0x2028,
        0x2029,
        0x202F,
        0x205F,
        0x3000,
        0xFEFF,
    }
)
# Unicode 15.0 General_Category=Cf ranges. Keep this explicit and mirrored in
# the browser so different Python/JavaScript Unicode tables cannot disagree.
_PUBLIC_DISPLAY_FORMAT_CODE_POINT_RANGES = (
    (0x00AD, 0x00AD),
    (0x0600, 0x0605),
    (0x061C, 0x061C),
    (0x06DD, 0x06DD),
    (0x070F, 0x070F),
    (0x0890, 0x0891),
    (0x08E2, 0x08E2),
    (0x180E, 0x180E),
    (0x200B, 0x200F),
    (0x202A, 0x202E),
    (0x2060, 0x2064),
    (0x2066, 0x206F),
    (0xFEFF, 0xFEFF),
    (0xFFF9, 0xFFFB),
    (0x110BD, 0x110BD),
    (0x110CD, 0x110CD),
    (0x13430, 0x1343F),
    (0x1BCA0, 0x1BCA3),
    (0x1D173, 0x1D17A),
    (0xE0001, 0xE0001),
    (0xE0020, 0xE007F),
)
# Unicode 15.0.0 General_Category=M (Mn, Mc, or Me). These are explicitly
# pinned rather than delegated to the host Unicode database so the browser can
# mirror the exact same visible-base rule. Variation selectors are included in
# these ranges. Marks remain valid when a label also contains a visible base.
_PUBLIC_DISPLAY_MARK_UNICODE_VERSION = "15.0.0"
_PUBLIC_DISPLAY_MARK_CODE_POINT_RANGES = tuple(
    (
        int(bounds[0], 16),
        int(bounds[-1], 16),
    )
    for token in """
0300-036F 0483-0489 0591-05BD 05BF 05C1-05C2 05C4-05C5 05C7 0610-061A 064B-065F 0670
06D6-06DC 06DF-06E4 06E7-06E8 06EA-06ED 0711 0730-074A 07A6-07B0 07EB-07F3 07FD 0816-0819
081B-0823 0825-0827 0829-082D 0859-085B 0898-089F 08CA-08E1 08E3-0903 093A-093C 093E-094F 0951-0957
0962-0963 0981-0983 09BC 09BE-09C4 09C7-09C8 09CB-09CD 09D7 09E2-09E3 09FE 0A01-0A03
0A3C 0A3E-0A42 0A47-0A48 0A4B-0A4D 0A51 0A70-0A71 0A75 0A81-0A83 0ABC 0ABE-0AC5
0AC7-0AC9 0ACB-0ACD 0AE2-0AE3 0AFA-0AFF 0B01-0B03 0B3C 0B3E-0B44 0B47-0B48 0B4B-0B4D 0B55-0B57
0B62-0B63 0B82 0BBE-0BC2 0BC6-0BC8 0BCA-0BCD 0BD7 0C00-0C04 0C3C 0C3E-0C44 0C46-0C48
0C4A-0C4D 0C55-0C56 0C62-0C63 0C81-0C83 0CBC 0CBE-0CC4 0CC6-0CC8 0CCA-0CCD 0CD5-0CD6 0CE2-0CE3
0CF3 0D00-0D03 0D3B-0D3C 0D3E-0D44 0D46-0D48 0D4A-0D4D 0D57 0D62-0D63 0D81-0D83 0DCA
0DCF-0DD4 0DD6 0DD8-0DDF 0DF2-0DF3 0E31 0E34-0E3A 0E47-0E4E 0EB1 0EB4-0EBC 0EC8-0ECE
0F18-0F19 0F35 0F37 0F39 0F3E-0F3F 0F71-0F84 0F86-0F87 0F8D-0F97 0F99-0FBC 0FC6
102B-103E 1056-1059 105E-1060 1062-1064 1067-106D 1071-1074 1082-108D 108F 109A-109D 135D-135F
1712-1715 1732-1734 1752-1753 1772-1773 17B4-17D3 17DD 180B-180D 180F 1885-1886 18A9
1920-192B 1930-193B 1A17-1A1B 1A55-1A5E 1A60-1A7C 1A7F 1AB0-1ACE 1B00-1B04 1B34-1B44 1B6B-1B73
1B80-1B82 1BA1-1BAD 1BE6-1BF3 1C24-1C37 1CD0-1CD2 1CD4-1CE8 1CED 1CF4 1CF7-1CF9 1DC0-1DFF
20D0-20F0 2CEF-2CF1 2D7F 2DE0-2DFF 302A-302F 3099-309A A66F-A672 A674-A67D A69E-A69F A6F0-A6F1
A802 A806 A80B A823-A827 A82C A880-A881 A8B4-A8C5 A8E0-A8F1 A8FF A926-A92D
A947-A953 A980-A983 A9B3-A9C0 A9E5 AA29-AA36 AA43 AA4C-AA4D AA7B-AA7D AAB0 AAB2-AAB4
AAB7-AAB8 AABE-AABF AAC1 AAEB-AAEF AAF5-AAF6 ABE3-ABEA ABEC-ABED FB1E FE00-FE0F FE20-FE2F
101FD 102E0 10376-1037A 10A01-10A03 10A05-10A06 10A0C-10A0F 10A38-10A3A 10A3F 10AE5-10AE6 10D24-10D27
10EAB-10EAC 10EFD-10EFF 10F46-10F50 10F82-10F85 11000-11002 11038-11046 11070 11073-11074 1107F-11082 110B0-110BA
110C2 11100-11102 11127-11134 11145-11146 11173 11180-11182 111B3-111C0 111C9-111CC 111CE-111CF 1122C-11237
1123E 11241 112DF-112EA 11300-11303 1133B-1133C 1133E-11344 11347-11348 1134B-1134D 11357 11362-11363
11366-1136C 11370-11374 11435-11446 1145E 114B0-114C3 115AF-115B5 115B8-115C0 115DC-115DD 11630-11640 116AB-116B7
1171D-1172B 1182C-1183A 11930-11935 11937-11938 1193B-1193E 11940 11942-11943 119D1-119D7 119DA-119E0 119E4
11A01-11A0A 11A33-11A39 11A3B-11A3E 11A47 11A51-11A5B 11A8A-11A99 11C2F-11C36 11C38-11C3F 11C92-11CA7 11CA9-11CB6
11D31-11D36 11D3A 11D3C-11D3D 11D3F-11D45 11D47 11D8A-11D8E 11D90-11D91 11D93-11D97 11EF3-11EF6 11F00-11F01
11F03 11F34-11F3A 11F3E-11F42 13440 13447-13455 16AF0-16AF4 16B30-16B36 16F4F 16F51-16F87 16F8F-16F92
16FE4 16FF0-16FF1 1BC9D-1BC9E 1CF00-1CF2D 1CF30-1CF46 1D165-1D169 1D16D-1D172 1D17B-1D182 1D185-1D18B 1D1AA-1D1AD
1D242-1D244 1DA00-1DA36 1DA3B-1DA6C 1DA75 1DA84 1DA9B-1DA9F 1DAA1-1DAAF 1E000-1E006 1E008-1E018 1E01B-1E021
1E023-1E024 1E026-1E02A 1E08F 1E130-1E136 1E2AE 1E2EC-1E2EF 1E4EC-1E4EF 1E8D0-1E8D6 1E944-1E94A E0100-E01EF
""".split()
    for bounds in (token.split("-", 1),)
)
_PUBLIC_MAX_RETRIES = 1
_PUBLIC_MAX_MEMORY_TURNS = 12
_PUBLIC_MAX_JSON_INTEGER = (1 << 53) - 1
_VISIBLE_FINAL_DECISIONS = frozenset(
    {LoopDecision.FINAL, LoopDecision.SUPPORTED, LoopDecision.NOT_VERIFIED}
)
_PUBLIC_PHASE_DECISIONS = {
    LoopPhase.INPUT: {LoopDecision.CONTINUE, LoopDecision.BLOCK},
    LoopPhase.CONTEXT_SELECT: {LoopDecision.CONTINUE},
    LoopPhase.RETRIEVE: {LoopDecision.CONTINUE},
    LoopPhase.DRAFT: {LoopDecision.CONTINUE, LoopDecision.ERROR},
    LoopPhase.FORMAT_CHECK: {
        LoopDecision.CONTINUE,
        LoopDecision.RETRY,
        LoopDecision.ERROR,
    },
    LoopPhase.MECHANICAL_CHECK: {
        LoopDecision.CONTINUE,
        LoopDecision.RETRY,
        LoopDecision.NOT_VERIFIED,
    },
    LoopPhase.VERIFY: {LoopDecision.SUPPORTED, LoopDecision.NOT_VERIFIED},
    LoopPhase.RETRY: {LoopDecision.RETRY},
    LoopPhase.REFUSE: {LoopDecision.REFUSE},
    LoopPhase.FINAL: {
        LoopDecision.FINAL,
        LoopDecision.SUPPORTED,
        LoopDecision.NOT_VERIFIED,
        LoopDecision.REFUSE,
        LoopDecision.BLOCK,
        LoopDecision.REQUIRES_REVIEW,
        LoopDecision.ERROR,
    },
    LoopPhase.ERROR: {
        LoopDecision.ERROR,
        LoopDecision.RETRY,
        LoopDecision.REFUSE,
        LoopDecision.BLOCK,
        LoopDecision.REQUIRES_REVIEW,
    },
}


class PublicProjectionError(ValueError):
    """Raised when a raw loop report cannot be projected without ambiguity."""


_FINAL_REASON_CONTRACT = {
    None: {None},
    LoopDecision.FINAL: {
        LoopTerminalReason.COMPLETED,
        LoopTerminalReason.UNSPECIFIED,
    },
    LoopDecision.SUPPORTED: {
        LoopTerminalReason.COMPLETED,
        LoopTerminalReason.UNSPECIFIED,
    },
    LoopDecision.NOT_VERIFIED: {
        LoopTerminalReason.NOT_VERIFIED,
        LoopTerminalReason.TRACE_UNAVAILABLE,
        LoopTerminalReason.UNSPECIFIED,
    },
    LoopDecision.REFUSE: {
        LoopTerminalReason.VERIFICATION_FAILED,
        LoopTerminalReason.RETRY_BUDGET_EXHAUSTED,
        LoopTerminalReason.POLICY_REFUSED,
        LoopTerminalReason.UNSPECIFIED,
    },
    LoopDecision.BLOCK: {
        LoopTerminalReason.BLOCKED,
        LoopTerminalReason.UNSPECIFIED,
    },
    LoopDecision.REQUIRES_REVIEW: {
        LoopTerminalReason.HUMAN_REVIEW_REQUIRED,
        LoopTerminalReason.UNSPECIFIED,
    },
    LoopDecision.ERROR: {
        LoopTerminalReason.ERROR,
        LoopTerminalReason.RETRY_BUDGET_EXHAUSTED,
        LoopTerminalReason.TRACE_UNAVAILABLE,
        LoopTerminalReason.UNSPECIFIED,
    },
}

_TERMINAL_DECISIONS = frozenset(
    {
        LoopDecision.REFUSE,
        LoopDecision.BLOCK,
        LoopDecision.REQUIRES_REVIEW,
    }
)
_TERMINAL_REASONS = frozenset(
    {
        LoopTerminalReason.POLICY_REFUSED,
        LoopTerminalReason.BLOCKED,
        LoopTerminalReason.HUMAN_REVIEW_REQUIRED,
    }
)
_TERMINAL_STEP_TO_FINAL = {
    LoopDecision.RETRY: LoopDecision.BLOCK,
    LoopDecision.REFUSE: LoopDecision.REFUSE,
    LoopDecision.BLOCK: LoopDecision.BLOCK,
    LoopDecision.REQUIRES_REVIEW: LoopDecision.REQUIRES_REVIEW,
}


def project_public_report(
    report: LoopReport,
    *,
    expected_run_id: Optional[str] = None,
    expected_session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the sole versioned, allowlist-only public LoopReport artifact.

    This projection deliberately preserves typed operational provenance. It is
    not authorization and it is not a general PII/secret scrubber. Arbitrary
    prompt, step, verifier, error, recipe, and evidence text is excluded.
    """

    report = _detached_public_report_snapshot(report)
    if expected_run_id is not None and type(expected_run_id) is not str:
        raise PublicProjectionError("malformed expected run identity")
    if expected_session_id is not None and type(expected_session_id) is not str:
        raise PublicProjectionError("malformed expected session identity")
    if expected_run_id is not None and report.run.run_id != expected_run_id:
        raise PublicProjectionError("loop report run identity mismatch")
    if expected_session_id is not None and report.run.session_id != expected_session_id:
        raise PublicProjectionError("loop report session identity mismatch")

    _validate_public_record_contract(report)
    _validate_provider_contract(report)
    _validate_terminal_contract(report)
    _validate_terminal_causality(report)
    _validate_unique_identities(report)
    _validate_step_relationships(report)
    _validate_retry_contract(report)
    terminal_redaction = terminal_public_redaction_required(report)
    _validate_visible_answer_contract(
        report,
        terminal_redaction=terminal_redaction,
    )
    _validate_supported_contract(report)
    _validate_public_model_provenance(
        report,
        terminal_redaction=terminal_redaction,
    )
    _validate_run_timeline(report)
    _validate_completed_timeline_contract(report)
    run = report.run
    memory_provenance = _public_memory_provenance(run.metadata)

    return {
        "schema_version": report.schema_version,
        "projection_schema_version": PUBLIC_REPORT_PROJECTION_SCHEMA_VERSION,
        "public": True,
        "public_redaction": {
            "applied": terminal_redaction,
            "reason": PUBLIC_REDACTION_REASON if terminal_redaction else None,
        },
        "run": {
            "run_id": run.run_id,
            "session_id": run.session_id,
            "context_provider": run.context_provider,
            **memory_provenance,
            "backend": None if terminal_redaction else run.backend,
            "model_label": None if terminal_redaction else run.model_label,
            "policy": {
                "max_retries": run.policy.max_retries,
                "require_citations": run.policy.require_citations,
                "require_verifier_for_supported": (
                    run.policy.require_verifier_for_supported
                ),
                "allow_mock_supported": run.policy.allow_mock_supported,
                "allow_tool_calls": run.policy.allow_tool_calls,
                "require_human_review_for_tools": (
                    run.policy.require_human_review_for_tools
                ),
            },
            "started_at": _iso_value(run.started_at),
            "completed_at": _iso_value(run.completed_at),
            "steps": [
                _project_step(step, terminal_redaction=terminal_redaction)
                for step in run.steps
            ],
            "evidence": (
                []
                if terminal_redaction
                else [reference.to_dict() for reference in run.evidence]
            ),
            "final_decision": (
                run.final_decision.value if run.final_decision is not None else None
            ),
            "terminal_reason": (
                run.terminal_reason.value if run.terminal_reason is not None else None
            ),
            "final_answer": (
                run.final_answer
                if _public_final_answer_allowed(report, terminal_redaction)
                else None
            ),
            "error_present": bool(run.error_message),
        },
    }


def _detached_public_report_snapshot(report: LoopReport) -> LoopReport:
    """Detach hostile caller-owned records before semantic validation.

    Frozen dataclasses can still be mutated with ``object.__setattr__`` and
    their metadata dictionaries remain mutable. Validate the exact source
    record shapes first, then JSON-detach and reconstruct a canonical typed
    snapshot. All later reads and projection operate only on that snapshot.
    """

    if type(report) is not LoopReport:
        raise TypeError("report must be a LoopReport")
    _validate_public_source_shape(report)
    if (
        type(report.schema_version) is not str
        or report.schema_version != SCHEMA_VERSION
    ):
        raise PublicProjectionError("unsupported loop report source schema")
    try:
        detached_payload = json.loads(
            json.dumps(
                report.to_dict(),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        snapshot = LoopReport.from_dict(detached_payload)
        if snapshot.to_dict() != detached_payload:
            raise ValueError("loop report snapshot is not canonical")
    except Exception as exc:
        raise PublicProjectionError(
            "malformed public loop report snapshot"
        ) from exc
    return snapshot


def _validate_public_source_shape(report: LoopReport) -> None:
    """Reject mutated/subclassed dataclass fields before invoking serializers."""

    run = report.run
    if type(run) is not LoopRun:
        raise PublicProjectionError("malformed public loop run")
    if type(run.policy) is not LoopPolicy:
        raise PublicProjectionError("malformed public loop policy")
    if type(run.steps) is not tuple:
        raise PublicProjectionError("malformed public loop step collection")
    if type(run.evidence) is not tuple:
        raise PublicProjectionError("malformed public evidence collection")
    if type(run.metadata) is not dict or type(run.policy.metadata) is not dict:
        raise PublicProjectionError("malformed public record metadata")
    if type(run.started_at) is not datetime or (
        run.completed_at is not None and type(run.completed_at) is not datetime
    ):
        raise PublicProjectionError("malformed public run timestamp")
    for value, location in (
        (report.schema_version, "report schema"),
        (run.run_id, "run id"),
        (run.user_input, "run user input"),
        (run.context_provider, "run context provider"),
        (run.backend, "run backend"),
        (run.model_label, "run model label"),
    ):
        if type(value) is not str:
            raise PublicProjectionError(f"malformed public string: {location}")
    if run.session_id is not None and type(run.session_id) is not str:
        raise PublicProjectionError("malformed public string: run session id")
    for value, location in (
        (run.final_answer, "final answer"),
        (run.error_message, "run error"),
    ):
        if value is not None and type(value) is not str:
            raise PublicProjectionError(f"malformed public string: {location}")
    if run.final_decision is not None and type(run.final_decision) is not LoopDecision:
        raise PublicProjectionError("malformed public final decision")
    if run.terminal_reason is not None and type(run.terminal_reason) is not LoopTerminalReason:
        raise PublicProjectionError("malformed public terminal reason")
    if type(run.policy.max_retries) is not int:
        raise PublicProjectionError("malformed public retry policy")
    for field_name in (
        "require_citations",
        "require_verifier_for_supported",
        "allow_mock_supported",
        "allow_tool_calls",
        "require_human_review_for_tools",
    ):
        if type(getattr(run.policy, field_name)) is not bool:
            raise PublicProjectionError(
                f"malformed public loop policy: {field_name}"
            )
    for index, step in enumerate(run.steps):
        _validate_public_step_source_shape(step, location=f"step[{index}]")
    for index, reference in enumerate(run.evidence):
        _validate_evidence_source_shape(
            reference,
            location=f"evidence[{index}]",
        )


def _validate_public_step_source_shape(step: LoopStep, *, location: str) -> None:
    if type(step) is not LoopStep:
        raise PublicProjectionError(f"malformed public loop step: {location}")
    if type(step.phase) is not LoopPhase:
        raise PublicProjectionError(f"malformed public step phase: {location}")
    if type(step.decision) is not LoopDecision:
        raise PublicProjectionError(f"malformed public step decision: {location}")
    if type(step.started_at) is not datetime or (
        step.ended_at is not None and type(step.ended_at) is not datetime
    ):
        raise PublicProjectionError(f"malformed public step timestamp: {location}")
    if type(step.retry_count) is not int:
        raise PublicProjectionError(f"malformed public retry count: {location}")
    if type(step.metadata) is not dict:
        raise PublicProjectionError(f"malformed public step metadata: {location}")
    for field_name in (
        "name",
        "input_summary",
        "output_summary",
        "backend",
        "model_label",
        "error_message",
    ):
        value = getattr(step, field_name)
        if value is not None and type(value) is not str:
            raise PublicProjectionError(
                f"malformed public string: {location} {field_name}"
            )
    if step.verification is not None:
        verification = step.verification
        if type(verification) is not VerificationResult:
            raise PublicProjectionError(
                f"malformed public verification: {location}"
            )
        if type(verification.outcome) is not VerificationOutcome:
            raise PublicProjectionError(
                f"malformed public verification outcome: {location}"
            )
        if type(verification.reasons) is not tuple or any(
            type(reason) is not str for reason in verification.reasons
        ):
            raise PublicProjectionError(
                f"malformed public verification reasons: {location}"
            )
        for field_name in (
            "verifier",
            "verifier_backend",
            "verifier_model_label",
            "raw_response",
        ):
            value = getattr(verification, field_name)
            if value is not None and type(value) is not str:
                raise PublicProjectionError(
                    f"malformed public verification string: {location}"
                )
        if (
            verification.same_model_as_drafter is not None
            and type(verification.same_model_as_drafter) is not bool
        ):
            raise PublicProjectionError(
                f"malformed public verifier relationship: {location}"
            )
        if type(verification.metadata) is not dict:
            raise PublicProjectionError(
                f"malformed public verification metadata: {location}"
            )
    if step.human_review is not None:
        review = step.human_review
        if type(review) is not HumanReviewRequest:
            raise PublicProjectionError(
                f"malformed public human review: {location}"
            )
        for field_name in ("request_id", "reason", "instructions"):
            if type(getattr(review, field_name)) is not str:
                raise PublicProjectionError(
                    f"malformed public human review string: {location}"
                )
        if (
            review.requested_by_step_id is not None
            and type(review.requested_by_step_id) is not str
        ):
            raise PublicProjectionError(
                f"malformed public human review identity: {location}"
            )
        if type(review.created_at) is not datetime or type(review.metadata) is not dict:
            raise PublicProjectionError(
                f"malformed public human review record: {location}"
            )


def _validate_evidence_source_shape(
    reference: EvidenceReference,
    *,
    location: str,
) -> None:
    if type(reference) is not EvidenceReference:
        raise PublicProjectionError(f"malformed public evidence reference: {location}")
    if type(reference.evidence_id) is not str:
        raise PublicProjectionError(f"malformed public evidence identity: {location}")
    if type(reference.citation_id) is not int:
        raise PublicProjectionError(f"malformed public evidence citation id: {location}")
    if type(reference.provider) is not str:
        raise PublicProjectionError(
            "unsupported public evidence provider identifier"
        )
    if type(reference.locator) is not EvidenceLocator:
        raise PublicProjectionError(f"malformed public evidence locator: {location}")
    for field_name, value in (
        ("page", reference.locator.page),
        ("chunk_index", reference.locator.chunk_index),
    ):
        if value is not None and type(value) is not int:
            raise PublicProjectionError(
                f"malformed public evidence locator {field_name}: {location}"
            )


def _validate_public_record_contract(report: LoopReport) -> None:
    run = report.run
    if type(run) is not LoopRun:
        raise PublicProjectionError("malformed public loop run")
    _validate_public_identity(run.run_id, location="run id")
    if run.session_id is not None:
        _validate_public_identity(run.session_id, location="session id")
    if type(run.policy) is not LoopPolicy:
        raise PublicProjectionError("malformed public loop policy")
    if (
        type(run.policy.max_retries) is not int
        or run.policy.max_retries < 0
        or run.policy.max_retries > _PUBLIC_MAX_RETRIES
    ):
        raise PublicProjectionError("malformed public retry policy")
    for field_name in (
        "require_citations",
        "require_verifier_for_supported",
        "allow_mock_supported",
        "allow_tool_calls",
        "require_human_review_for_tools",
    ):
        if type(getattr(run.policy, field_name)) is not bool:
            raise PublicProjectionError(
                f"malformed public loop policy: {field_name}"
            )
    if type(run.metadata) is not dict:
        raise PublicProjectionError("malformed public run metadata")
    _validate_public_datetime(run.started_at, location="run started_at")
    if run.completed_at is not None:
        _validate_public_datetime(run.completed_at, location="run completed_at")
        if run.completed_at < run.started_at:
            raise PublicProjectionError(
                "inconsistent run timeline: completion precedes start"
            )
    if run.final_decision is not None and type(run.final_decision) is not LoopDecision:
        raise PublicProjectionError("malformed public final decision")
    if run.terminal_reason is not None and type(run.terminal_reason) is not LoopTerminalReason:
        raise PublicProjectionError("malformed public terminal reason")
    _validate_optional_string(run.final_answer, location="final answer")
    _validate_optional_string(run.error_message, location="run error")
    _validate_string(run.backend, location="run backend", allow_empty=False)
    _validate_string(run.model_label, location="run model label", allow_empty=False)
    if type(run.steps) is not tuple:
        raise PublicProjectionError("malformed public loop step collection")
    for index, step in enumerate(run.steps):
        _validate_public_step(step, location=f"step[{index}]")


def _validate_public_step(step: LoopStep, *, location: str) -> None:
    if type(step) is not LoopStep:
        raise PublicProjectionError(f"malformed public loop step: {location}")
    _validate_public_identity(step.step_id, location=f"{location} id")
    if type(step.phase) is not LoopPhase:
        raise PublicProjectionError(f"malformed public step phase: {location}")
    if type(step.decision) is not LoopDecision:
        raise PublicProjectionError(f"malformed public step decision: {location}")
    _validate_public_datetime(step.started_at, location=f"{location} started_at")
    if step.ended_at is not None:
        _validate_public_datetime(step.ended_at, location=f"{location} ended_at")
        if step.ended_at < step.started_at:
            raise PublicProjectionError(
                f"inconsistent step timeline: {location} ends before it starts"
            )
    if (
        type(step.retry_count) is not int
        or step.retry_count < 0
        or step.retry_count > _PUBLIC_MAX_JSON_INTEGER
    ):
        raise PublicProjectionError(f"malformed public retry count: {location}")
    _validate_optional_string(step.backend, location=f"{location} backend")
    _validate_optional_string(step.model_label, location=f"{location} model label")
    _validate_optional_string(step.error_message, location=f"{location} error")
    if type(step.metadata) is not dict:
        raise PublicProjectionError(f"malformed public step metadata: {location}")
    if step.verification is not None:
        _validate_public_verification(
            step.verification,
            location=f"{location} verification",
        )
    if step.human_review is not None:
        _validate_public_human_review(
            step.human_review,
            location=f"{location} human review",
        )


def _validate_public_verification(
    verification: VerificationResult,
    *,
    location: str,
) -> None:
    if type(verification) is not VerificationResult:
        raise PublicProjectionError(f"malformed public verification: {location}")
    if type(verification.outcome) is not VerificationOutcome:
        raise PublicProjectionError(
            f"malformed public verification outcome: {location}"
        )
    _validate_optional_string(
        verification.verifier_backend,
        location=f"{location} backend",
    )
    _validate_optional_string(
        verification.verifier_model_label,
        location=f"{location} model label",
    )
    if (
        verification.same_model_as_drafter is not None
        and type(verification.same_model_as_drafter) is not bool
    ):
        raise PublicProjectionError(
            f"malformed public verifier relationship: {location}"
        )


def _validate_public_human_review(
    review: HumanReviewRequest,
    *,
    location: str,
) -> None:
    if type(review) is not HumanReviewRequest:
        raise PublicProjectionError(f"malformed public human review: {location}")
    _validate_public_identity(review.request_id, location=f"{location} request id")
    _validate_string(review.reason, location=f"{location} reason", allow_empty=False)
    _validate_string(
        review.instructions,
        location=f"{location} instructions",
        allow_empty=False,
    )
    if review.requested_by_step_id is not None:
        _validate_public_identity(
            review.requested_by_step_id,
            location=f"{location} requested step id",
        )
    _validate_public_datetime(review.created_at, location=f"{location} created_at")
    if type(review.metadata) is not dict:
        raise PublicProjectionError(
            f"malformed public human review metadata: {location}"
        )


def _validate_public_model_provenance(
    report: LoopReport,
    *,
    terminal_redaction: bool,
) -> None:
    if terminal_redaction:
        return
    run = report.run
    _validate_public_backend(run.backend, location="run backend")
    _validate_public_display_label(run.model_label, location="run model label")
    for index, step in enumerate(run.steps):
        if step.backend is not None:
            _validate_public_backend(step.backend, location=f"step[{index}] backend")
        if step.model_label is not None:
            _validate_public_display_label(
                step.model_label,
                location=f"step[{index}] model label",
            )
        if step.verification is None:
            continue
        if step.verification.verifier_backend is not None:
            _validate_public_backend(
                step.verification.verifier_backend,
                location=f"step[{index}] verifier backend",
            )
        if step.verification.verifier_model_label is not None:
            _validate_public_display_label(
                step.verification.verifier_model_label,
                location=f"step[{index}] verifier model label",
            )


def _validate_public_identity(value: Any, *, location: str) -> None:
    if type(value) is not str or _PUBLIC_IDENTITY_PATTERN.fullmatch(value) is None:
        raise PublicProjectionError(f"malformed public identity: {location}")


def _validate_string(value: Any, *, location: str, allow_empty: bool = True) -> None:
    if type(value) is not str or (not allow_empty and not value.strip()):
        raise PublicProjectionError(f"malformed public string: {location}")


def _validate_optional_string(value: Any, *, location: str) -> None:
    if value is not None:
        _validate_string(value, location=location)


def _validate_public_datetime(value: Any, *, location: str) -> None:
    if type(value) is not datetime:
        raise PublicProjectionError(f"malformed public timestamp: {location}")
    try:
        offset = value.utcoffset()
        value.astimezone(timezone.utc)
    except (OverflowError, TypeError, ValueError) as exc:
        raise PublicProjectionError(
            f"malformed public timestamp: {location}"
        ) from exc
    if offset is None:
        raise PublicProjectionError(f"malformed public timestamp: {location}")


def _validate_public_backend(value: str, *, location: str) -> None:
    if value not in _PUBLIC_BACKENDS:
        raise PublicProjectionError(f"unsupported public backend identifier: {location}")


def _validate_public_display_label(value: str, *, location: str) -> None:
    if (
        not value
        or all(
            _public_display_label_character_is_blank(character)
            for character in value
        )
        or len(value) > 512
        or "://" in value
        or any(
            ord(character) < 32
            or 0x007F <= ord(character) <= 0x009F
            for character in value
        )
    ):
        raise PublicProjectionError(f"malformed public display label: {location}")


def _public_display_label_character_is_blank(character: str) -> bool:
    code_point = ord(character)
    return (
        code_point in _PUBLIC_DISPLAY_BLANK_CODE_POINTS
        or 0x0080 <= code_point <= 0x009F
        or any(
            start <= code_point <= end
            for start, end in _PUBLIC_DISPLAY_FORMAT_CODE_POINT_RANGES
        )
        or any(
            start <= code_point <= end
            for start, end in _PUBLIC_DISPLAY_MARK_CODE_POINT_RANGES
        )
    )


def terminal_public_redaction_required(report: LoopReport) -> bool:
    run = report.run
    if run.final_decision in _TERMINAL_DECISIONS:
        return True
    if run.terminal_reason in _TERMINAL_REASONS:
        return True
    for step in run.steps:
        if (
            step.decision in _TERMINAL_DECISIONS
            or step.phase.value == "refuse"
            or step.human_review is not None
        ):
            return True
        if _terminal_guardrail_decision(
            step.metadata,
            location=f"step {step.step_id!r}",
        ) is not None:
            return True
    if _terminal_guardrail_decision(run.metadata, location="run") is not None:
        return True
    return _after_run_guardrail_applied(run.metadata)


def _validate_terminal_contract(report: LoopReport) -> None:
    run = report.run
    allowed_reasons = _FINAL_REASON_CONTRACT.get(run.final_decision)
    if allowed_reasons is None or run.terminal_reason not in allowed_reasons:
        decision = run.final_decision.value if run.final_decision else None
        reason = run.terminal_reason.value if run.terminal_reason else None
        raise PublicProjectionError(
            f"inconsistent terminal contract: decision={decision!r}, reason={reason!r}"
        )

    terminal_step_decisions = set()
    for step in run.steps:
        if step.decision in _TERMINAL_DECISIONS:
            terminal_step_decisions.add(step.decision)
        if step.phase.value == "refuse":
            terminal_step_decisions.add(LoopDecision.REFUSE)
        guardrail_decision = _terminal_guardrail_decision(
            step.metadata,
            location=f"step {step.step_id!r}",
        )
        if guardrail_decision is not None:
            terminal_step_decisions.add(guardrail_decision)
        if step.human_review is not None:
            terminal_step_decisions.add(LoopDecision.REQUIRES_REVIEW)

    run_guardrail_decision = _terminal_guardrail_decision(
        run.metadata,
        location="run",
    )
    if run_guardrail_decision is not None:
        terminal_step_decisions.add(run_guardrail_decision)

    if (
        _after_run_guardrail_applied(run.metadata)
        and run.final_decision not in _TERMINAL_DECISIONS
    ):
        raise PublicProjectionError(
            "inconsistent terminal contract: after-run guardrail does not match "
            "the final decision"
        )

    for terminal_decision in terminal_step_decisions:
        expected_final = _TERMINAL_STEP_TO_FINAL[terminal_decision]
        if run.final_decision != expected_final:
            raise PublicProjectionError(
                "inconsistent terminal contract: terminal step does not match "
                "the final decision"
            )

    if run.final_decision == LoopDecision.REQUIRES_REVIEW and not any(
        step.decision == LoopDecision.REQUIRES_REVIEW
        and step.human_review is not None
        for step in run.steps
    ):
        raise PublicProjectionError(
            "inconsistent terminal contract: requires_review lacks a causal "
            "human review request"
        )

    final_indexes = [
        index for index, step in enumerate(run.steps) if step.phase == LoopPhase.FINAL
    ]
    final_index = final_indexes[0] if final_indexes else None
    terminal_boundary_indexes = [
        index
        for index, step in enumerate(run.steps)
        if _is_terminal_boundary_step(step)
    ]
    if final_index is not None:
        boundaries_before_final = [
            index for index in terminal_boundary_indexes if index < final_index
        ]
        if boundaries_before_final:
            first_boundary = boundaries_before_final[0]
            if any(
                not _is_terminal_boundary_step(step)
                for step in run.steps[first_boundary + 1 : final_index]
            ):
                raise PublicProjectionError(
                    "inconsistent terminal contract: execution continued after a "
                    "terminal boundary"
                )
        if final_index < len(run.steps) - 1:
            post_final_steps = run.steps[final_index + 1 :]
            final_step_guardrail = (
                len(post_final_steps) == 1
                and _terminal_guardrail_decision(
                    post_final_steps[0].metadata,
                    location=f"step {post_final_steps[0].step_id!r}",
                )
                is not None
            )
            if (
                not (
                    _after_run_guardrail_applied(run.metadata)
                    or final_step_guardrail
                )
                or any(
                    not _is_terminal_boundary_step(step)
                    for step in post_final_steps
                )
            ):
                raise PublicProjectionError(
                    "inconsistent terminal contract: steps follow terminal final "
                    "without an after-run guardrail"
                )
    elif terminal_boundary_indexes:
        first_boundary = terminal_boundary_indexes[0]
        if any(
            not _is_terminal_boundary_step(step)
            for step in run.steps[first_boundary + 1 :]
        ):
            raise PublicProjectionError(
                "inconsistent terminal contract: execution continued after a "
                "terminal boundary"
            )

    if _after_run_guardrail_applied(run.metadata):
        if final_index is None or final_index != len(run.steps) - 2:
            raise PublicProjectionError(
                "inconsistent terminal contract: after-run guardrail lacks its "
                "post-final causal boundary"
            )
        guardrail_step = run.steps[-1]
        guardrail_decision = _terminal_guardrail_decision(
            guardrail_step.metadata,
            location=f"step {guardrail_step.step_id!r}",
        )
        if (
            guardrail_step.phase != LoopPhase.ERROR
            or guardrail_decision is None
            or _TERMINAL_STEP_TO_FINAL[guardrail_decision] != run.final_decision
        ):
            raise PublicProjectionError(
                "inconsistent terminal contract: after-run guardrail lacks its "
                "post-final causal boundary"
            )


def _validate_terminal_causality(report: LoopReport) -> None:
    """Bind non-generic terminal reasons to the step evidence that caused them."""

    run = report.run
    reason = run.terminal_reason
    if reason in {
        None,
        LoopTerminalReason.UNSPECIFIED,
        LoopTerminalReason.COMPLETED,
        LoopTerminalReason.NOT_VERIFIED,
        LoopTerminalReason.HUMAN_REVIEW_REQUIRED,
        LoopTerminalReason.RETRY_BUDGET_EXHAUSTED,
    }:
        return

    final_indexes = [
        index for index, step in enumerate(run.steps) if step.phase == LoopPhase.FINAL
    ]
    terminal_index = final_indexes[0] if final_indexes else len(run.steps)
    non_final_steps = [
        (index, step)
        for index, step in enumerate(run.steps[:terminal_index])
        if step.phase != LoopPhase.FINAL
    ]
    all_non_final_steps = [
        (index, step)
        for index, step in enumerate(run.steps)
        if step.phase != LoopPhase.FINAL
    ]
    if reason == LoopTerminalReason.VERIFICATION_FAILED:
        has_terminal_refusal = bool(non_final_steps) and (
            non_final_steps[-1][1].phase == LoopPhase.REFUSE
            and non_final_steps[-1][1].decision == LoopDecision.REFUSE
        )
        causal_step = non_final_steps[-2][1] if len(non_final_steps) >= 2 else None
        failed_terminal_check = causal_step is not None and (
            (
                causal_step.phase == LoopPhase.VERIFY
                and causal_step.decision == LoopDecision.NOT_VERIFIED
                and causal_step.verification is not None
                and causal_step.verification.outcome
                in {
                    VerificationOutcome.UNSUPPORTED,
                    VerificationOutcome.INSUFFICIENT,
                    VerificationOutcome.ERROR,
                }
            )
            or (
                causal_step.phase == LoopPhase.MECHANICAL_CHECK
                and causal_step.decision == LoopDecision.NOT_VERIFIED
                and causal_step.output_summary != "mechanical_checks_passed"
            )
        )
        if (
            not has_terminal_refusal
            or not failed_terminal_check
            or causal_step.retry_count
            != sum(1 for step in run.steps if step.phase == LoopPhase.RETRY)
        ):
            raise PublicProjectionError(
                "inconsistent terminal causality: verification_failed lacks a "
                "current failed verifier or mechanical check before refusal"
            )
        return

    if reason == LoopTerminalReason.POLICY_REFUSED:
        if not any(
            _terminal_guardrail_decision(
                step.metadata,
                location=f"step {step.step_id!r}",
            )
            == LoopDecision.REFUSE
            for _index, step in all_non_final_steps
        ):
            raise PublicProjectionError(
                "inconsistent terminal causality: policy_refused lacks a causal "
                "refusing guardrail"
            )
        return

    if reason == LoopTerminalReason.BLOCKED:
        if not any(
            step.decision == LoopDecision.BLOCK
            or _terminal_guardrail_decision(
                step.metadata,
                location=f"step {step.step_id!r}",
            )
            in {LoopDecision.BLOCK, LoopDecision.RETRY}
            for _index, step in all_non_final_steps
        ):
            raise PublicProjectionError(
                "inconsistent terminal causality: blocked lacks a causal blocking "
                "step or guardrail"
            )
        return

    if reason == LoopTerminalReason.ERROR:
        causal_step = non_final_steps[-1][1] if non_final_steps else None
        if (
            causal_step is None
            or causal_step.decision != LoopDecision.ERROR
            or causal_step.phase not in {LoopPhase.DRAFT, LoopPhase.ERROR}
            or causal_step.retry_count
            != sum(1 for step in run.steps if step.phase == LoopPhase.RETRY)
        ):
            raise PublicProjectionError(
                "inconsistent terminal causality: error lacks a current causal "
                "failed step"
            )
        return

    if reason == LoopTerminalReason.TRACE_UNAVAILABLE:
        retry_count = sum(1 for step in run.steps if step.phase == LoopPhase.RETRY)
        draft_steps = [
            (index, step)
            for index, step in non_final_steps
            if step.phase == LoopPhase.DRAFT
        ]
        verify_steps = [
            (index, step)
            for index, step in non_final_steps
            if step.phase == LoopPhase.VERIFY
        ]
        terminal_draft = draft_steps[-1] if draft_steps else None
        terminal_verify = verify_steps[-1] if verify_steps else None
        if run.final_decision == LoopDecision.NOT_VERIFIED:
            has_trace_boundary = bool(terminal_draft and terminal_verify) and (
                terminal_draft[0] < terminal_verify[0]
                and terminal_draft[1].decision == LoopDecision.CONTINUE
                and terminal_draft[1].retry_count == retry_count
                and terminal_draft[1].metadata.get("trace_available") is False
                and terminal_verify[1].decision == LoopDecision.NOT_VERIFIED
                and terminal_verify[1].retry_count == retry_count
                and terminal_verify[1].metadata.get("trace_available") is False
                and terminal_verify[1].metadata.get("verifier_skipped") is True
            )
        else:
            causal_step = non_final_steps[-1][1] if non_final_steps else None
            has_trace_boundary = (
                causal_step is not None
                and causal_step.phase == LoopPhase.ERROR
                and causal_step.decision == LoopDecision.ERROR
                and causal_step.retry_count == retry_count
                and causal_step.metadata.get("trace_available") is False
            )
        if not has_trace_boundary:
            raise PublicProjectionError(
                "inconsistent terminal causality: trace_unavailable lacks a causal "
                "trace-loss boundary"
            )
        return

    raise PublicProjectionError("unsupported public terminal causality")


def _is_terminal_boundary_step(step: LoopStep) -> bool:
    return (
        step.decision in _TERMINAL_DECISIONS
        or step.phase == LoopPhase.REFUSE
        or step.human_review is not None
        or _terminal_guardrail_decision(
            step.metadata,
            location=f"step {step.step_id!r}",
        )
        is not None
    )


def _terminal_guardrail_decision(
    metadata: Mapping[str, Any],
    *,
    location: str,
) -> Optional[LoopDecision]:
    if "guardrail_decision" not in metadata:
        return None
    raw_value = metadata.get("guardrail_decision")
    if isinstance(raw_value, Mapping):
        raise PublicProjectionError(
            f"malformed guardrail decision metadata on {location}"
        )
    try:
        decision = (
            raw_value
            if isinstance(raw_value, LoopDecision)
            else LoopDecision(raw_value)
        )
    except (TypeError, ValueError) as exc:
        raise PublicProjectionError(
            f"malformed guardrail decision metadata on {location}"
        ) from exc
    if decision in {*_TERMINAL_DECISIONS, LoopDecision.RETRY}:
        return decision
    if decision == LoopDecision.CONTINUE:
        return None
    raise PublicProjectionError(
        f"malformed guardrail decision metadata on {location}"
    )


def _after_run_guardrail_applied(metadata: Mapping[str, Any]) -> bool:
    if "after_run_guardrail" not in metadata:
        return False
    value = metadata.get("after_run_guardrail")
    if type(value) is not bool:
        raise PublicProjectionError("malformed after-run guardrail metadata")
    return value


def _validate_provider_contract(report: LoopReport) -> None:
    run = report.run
    if (
        type(run.context_provider) is not str
        or run.context_provider not in _PUBLIC_CONTEXT_PROVIDERS
    ):
        raise PublicProjectionError("unsupported public context provider identifier")
    if type(run.evidence) is not tuple:
        raise PublicProjectionError("malformed public evidence collection")
    if run.context_provider == "none" and run.evidence:
        raise PublicProjectionError(
            "inconsistent evidence contract: no-context run contains evidence"
        )
    if (
        run.context_provider == "none"
        and run.final_decision in _VISIBLE_FINAL_DECISIONS
        and run.final_decision != LoopDecision.NOT_VERIFIED
    ):
        raise PublicProjectionError(
            "inconsistent evidence contract: no-context visible completion must "
            "be not_verified"
        )
    for index, reference in enumerate(run.evidence):
        _validate_evidence_reference(
            reference,
            run_context_provider=run.context_provider,
            location=f"evidence[{index}]",
        )


def _validate_evidence_reference(
    reference: EvidenceReference,
    *,
    run_context_provider: str,
    location: str,
) -> None:
    if type(reference) is not EvidenceReference:
        raise PublicProjectionError(f"malformed public evidence reference: {location}")
    if (
        type(reference.evidence_id) is not str
        or _EVIDENCE_ID_PATTERN.fullmatch(reference.evidence_id) is None
    ):
        raise PublicProjectionError(
            f"malformed public evidence identity: {location}"
        )
    if (
        type(reference.citation_id) is not int
        or reference.citation_id < 1
        or reference.citation_id > _PUBLIC_MAX_JSON_INTEGER
    ):
        raise PublicProjectionError(
            f"malformed public evidence citation id: {location}"
        )
    if (
        type(reference.provider) is not str
        or reference.provider not in _PUBLIC_EVIDENCE_PROVIDERS
    ):
        raise PublicProjectionError(
            "unsupported public evidence provider identifier"
        )
    if reference.provider != run_context_provider:
        raise PublicProjectionError(
            "inconsistent evidence contract: evidence provider does not match "
            "the selected context provider"
        )
    locator = reference.locator
    if type(locator) is not EvidenceLocator:
        raise PublicProjectionError(
            f"malformed public evidence locator: {location}"
        )
    for field_name, value in (
        ("page", locator.page),
        ("chunk_index", locator.chunk_index),
    ):
        if value is not None and (
            type(value) is not int
            or value < 0
            or value > _PUBLIC_MAX_JSON_INTEGER
        ):
            raise PublicProjectionError(
                f"malformed public evidence locator {field_name}: {location}"
            )


def _public_memory_provenance(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    provenance_keys = {
        "conversation_context_turns",
        "semantic_memory_turns",
        "semantic_memory_status",
    }
    present_keys = provenance_keys.intersection(metadata)
    if present_keys and present_keys != provenance_keys:
        raise PublicProjectionError(
            "inconsistent public memory provenance: fields must be all present or "
            "all absent"
        )
    if not present_keys:
        return {
            "conversation_context_count": None,
            "semantic_memory_count": None,
            "semantic_memory_status": None,
        }

    def optional_count(key: str) -> Optional[int]:
        if key not in metadata:
            return None
        value = metadata.get(key)
        if (
            type(value) is not int
            or value < 0
            or value > _PUBLIC_MAX_MEMORY_TURNS
        ):
            raise PublicProjectionError(f"malformed public memory provenance: {key}")
        return value

    status = metadata.get("semantic_memory_status")
    if "semantic_memory_status" not in metadata:
        status = None
    elif type(status) is not str or status not in _PUBLIC_MEMORY_STATUSES:
        raise PublicProjectionError(
            "malformed public memory provenance: semantic_memory_status"
        )
    semantic_memory_count = optional_count("semantic_memory_turns")
    if semantic_memory_count is not None and semantic_memory_count > 0:
        if status != "retrieved":
            raise PublicProjectionError(
                "inconsistent public memory provenance: nonempty semantic memory "
                "must be retrieved"
            )
    if status == "retrieved" and (
        semantic_memory_count is None or semantic_memory_count < 1
    ):
        raise PublicProjectionError(
            "inconsistent public memory provenance: retrieved semantic memory "
            "must be nonempty"
        )
    return {
        "conversation_context_count": optional_count(
            "conversation_context_turns"
        ),
        "semantic_memory_count": semantic_memory_count,
        "semantic_memory_status": status,
    }


def _validate_step_relationships(report: LoopReport) -> None:
    run = report.run
    verification_decisions = {
        VerificationOutcome.SUPPORTED: {LoopDecision.SUPPORTED},
        VerificationOutcome.NOT_VERIFIED: {LoopDecision.NOT_VERIFIED},
        VerificationOutcome.UNSUPPORTED: {LoopDecision.NOT_VERIFIED},
        VerificationOutcome.INSUFFICIENT: {LoopDecision.NOT_VERIFIED},
        VerificationOutcome.ERROR: {LoopDecision.NOT_VERIFIED},
    }
    final_steps = [step for step in run.steps if step.phase == LoopPhase.FINAL]
    if len(final_steps) > 1:
        raise PublicProjectionError(
            "inconsistent step contract: multiple final steps"
        )
    if final_steps and run.steps[-1] is final_steps[0]:
        if final_steps[0].decision != run.final_decision:
            raise PublicProjectionError(
                "inconsistent step contract: final step decision mismatch"
            )
    implicit_review_sources = {}
    for source_index, source_step in enumerate(run.steps):
        if source_step.human_review is not None:
            implicit_review_sources.setdefault(
                source_step.human_review.request_id,
                (source_index, source_step),
            )
    for index, step in enumerate(run.steps):
        location = f"step[{index}]"
        if step.decision not in _PUBLIC_PHASE_DECISIONS[step.phase]:
            raise PublicProjectionError(
                f"inconsistent step contract: invalid decision for {location} phase"
            )
        guardrail_decision = _terminal_guardrail_decision(
            step.metadata,
            location=f"step {step.step_id!r}",
        )
        if guardrail_decision is not None:
            expected_decision = (
                _TERMINAL_STEP_TO_FINAL[guardrail_decision]
                if step.phase == LoopPhase.FINAL
                else guardrail_decision
            )
            if (
                step.phase not in {LoopPhase.ERROR, LoopPhase.FINAL}
                or step.decision != expected_decision
            ):
                raise PublicProjectionError(
                    "inconsistent step contract: guardrail marker does not match "
                    "its public step boundary"
                )
            if "guardrail_reason" in step.metadata:
                guardrail_reason = step.metadata.get("guardrail_reason")
                if guardrail_reason is not None and type(guardrail_reason) is not str:
                    raise PublicProjectionError(
                        "inconsistent step contract: guardrail reason is malformed"
                    )
                if step.phase == LoopPhase.ERROR and (
                    step.error_message != guardrail_reason
                    or step.output_summary
                    != (guardrail_reason or guardrail_decision.value)
                ):
                    raise PublicProjectionError(
                        "inconsistent step contract: guardrail reason does not match "
                        "its public step boundary"
                    )
        if (step.phase == LoopPhase.VERIFY) != (step.verification is not None):
            raise PublicProjectionError(
                "inconsistent step contract: verification has invalid phase"
            )
        if step.verification is not None:
            if step.decision not in verification_decisions[step.verification.outcome]:
                raise PublicProjectionError(
                    "inconsistent step contract: verification outcome does not "
                    "match its decision"
                )
            _validate_verifier_relationship(
                run,
                step_index=index,
                step=step,
                strict_provenance=not terminal_public_redaction_required(report),
            )
        if (step.decision == LoopDecision.REQUIRES_REVIEW) != (
            step.human_review is not None
        ):
            raise PublicProjectionError(
                "inconsistent step contract: requires_review and human review "
                "request must appear together"
            )
        if step.human_review is not None:
            identity_source = implicit_review_sources.get(
                step.human_review.request_id
            )
            if (
                identity_source is None
                or identity_source[1].human_review != step.human_review
            ):
                raise PublicProjectionError(
                    "inconsistent step contract: repeated human review identity "
                    "has divergent request data"
                )
            if step.human_review.requested_by_step_id is not None:
                requested_by = next(
                    (
                        (candidate_index, candidate)
                        for candidate_index, candidate in enumerate(run.steps)
                        if candidate.step_id
                        == step.human_review.requested_by_step_id
                    ),
                    None,
                )
            else:
                requested_by = implicit_review_sources.get(
                    step.human_review.request_id
                )
            if (
                requested_by is None
                or requested_by[0] > index
                or requested_by[1].human_review is None
                or requested_by[1].human_review.request_id
                != step.human_review.request_id
            ):
                raise PublicProjectionError(
                    "inconsistent step contract: human review request cites an "
                    "unrelated step"
                )
            requested_step = requested_by[1]
            if requested_step.human_review != step.human_review:
                raise PublicProjectionError(
                    "inconsistent step contract: repeated human review identity "
                    "has divergent request data"
                )
            if (
                step.human_review.created_at < requested_step.started_at
                or (
                    requested_step.ended_at is not None
                    and step.human_review.created_at > requested_step.ended_at
                )
            ):
                raise PublicProjectionError(
                    "inconsistent step contract: human review timestamp falls "
                    "outside its requesting step"
                )
        if (
            step.phase == LoopPhase.FORMAT_CHECK
            and step.metadata.get("sanitized_internal_labels") is True
        ):
            preceding_step = run.steps[index - 1] if index else None
            preceding_draft = run.steps[index - 2] if index >= 2 else None
            source_answer_digest = (
                preceding_draft.metadata.get(ANSWER_CANDIDATE_SHA256_METADATA_KEY)
                if preceding_draft is not None
                else None
            )
            source_evidence_digest = (
                preceding_draft.metadata.get(EVIDENCE_SET_SHA256_METADATA_KEY)
                if preceding_draft is not None
                else None
            )
            if (
                preceding_step is None
                or preceding_step.phase != LoopPhase.FORMAT_CHECK
                or preceding_step.decision != LoopDecision.ERROR
                or preceding_step.output_summary != "needs_retry"
                or preceding_step.metadata.get("reasons")
                != ["internal_verification_label"]
                or preceding_step.metadata.get("retry_denied") is not True
                or type(preceding_step.metadata.get("retry_unavailable")) is not bool
                or type(
                    preceding_step.metadata.get("retry_budget_exhausted")
                )
                is not bool
                or preceding_step.metadata.get("retry_budget_exhausted")
                != (
                    sum(
                        1
                        for prior in run.steps[:index]
                        if prior.phase == LoopPhase.RETRY
                    )
                    >= run.policy.max_retries
                )
                or preceding_step.metadata.get("retry_denied")
                != (
                    preceding_step.metadata.get("retry_unavailable")
                    or preceding_step.metadata.get("retry_budget_exhausted")
                )
                or preceding_draft is None
                or preceding_draft.phase != LoopPhase.DRAFT
                or preceding_draft.decision != LoopDecision.CONTINUE
                or preceding_draft.retry_count != step.retry_count
                or preceding_step.retry_count != step.retry_count
                or step.output_summary != "format_sanitized"
                or step.metadata.get("reasons")
                != ["internal_verification_label"]
                or step.metadata.get("resolved_format_step_id")
                != preceding_step.step_id
                or step.metadata.get("format_resolution")
                != "deterministic_format_sanitizer"
                or type(source_answer_digest) is not str
                or _ANSWER_CANDIDATE_SHA256_PATTERN.fullmatch(source_answer_digest)
                is None
                or step.metadata.get("sanitized_from_answer_sha256")
                != source_answer_digest
                or type(
                    step.metadata.get(ANSWER_CANDIDATE_SHA256_METADATA_KEY)
                )
                is not str
                or _ANSWER_CANDIDATE_SHA256_PATTERN.fullmatch(
                    step.metadata.get(ANSWER_CANDIDATE_SHA256_METADATA_KEY, "")
                )
                is None
                or step.metadata.get(ANSWER_CANDIDATE_SHA256_METADATA_KEY)
                == source_answer_digest
                or type(source_evidence_digest) is not str
                or _ANSWER_CANDIDATE_SHA256_PATTERN.fullmatch(source_evidence_digest)
                is None
                or step.metadata.get(EVIDENCE_SET_SHA256_METADATA_KEY)
                != source_evidence_digest
            ):
                raise PublicProjectionError(
                    "inconsistent step contract: sanitized candidate lacks its "
                    "deterministic format-check cause"
                )


def _validate_verifier_relationship(
    run: LoopRun,
    *,
    step_index: int,
    step: LoopStep,
    strict_provenance: bool,
) -> None:
    verification = step.verification
    if verification is None:
        return
    verifier_pair = (
        verification.verifier_backend,
        verification.verifier_model_label,
    )
    if (verifier_pair[0] is None) != (verifier_pair[1] is None):
        raise PublicProjectionError(
            "inconsistent step contract: verifier provenance must be all present "
            "or all absent"
        )
    if strict_provenance and verification.verifier is not None:
        if (
            type(verification.verifier) is not str
            or verification.verifier_backend is None
            or verification.verifier != verification.verifier_backend
        ):
            raise PublicProjectionError(
                "inconsistent step contract: verifier identity does not match "
                "its backend provenance"
            )
    step_pair = (step.backend, step.model_label)
    if strict_provenance and verifier_pair[0] is not None:
        if (
            step_pair[0] is None
            or step_pair[1] is None
            or step_pair != verifier_pair
        ):
            raise PublicProjectionError(
                "inconsistent step contract: verify-step provenance does not "
                "match its verifier provenance"
            )
    relationship = verification.same_model_as_drafter
    if relationship is None:
        return
    preceding_drafts = [
        candidate
        for candidate in run.steps[:step_index]
        if candidate.phase == LoopPhase.DRAFT
    ]
    if not preceding_drafts:
        raise PublicProjectionError(
            "inconsistent step contract: verifier relationship lacks a drafter"
        )
    draft = preceding_drafts[-1]
    draft_pair = (draft.backend, draft.model_label)
    complete_pairs = (draft_pair, step_pair, verifier_pair)
    if any(
        backend is None
        or model_label is None
        or not backend
        or not model_label
        for backend, model_label in complete_pairs
    ):
        raise PublicProjectionError(
            "inconsistent step contract: verifier relationship lacks complete "
            "model provenance"
        )
    all_equal = draft_pair == step_pair == verifier_pair
    if relationship is True and not all_equal:
        raise PublicProjectionError(
            "inconsistent step contract: same-model verifier provenance does not "
            "match the drafter"
        )
    if relationship is False and all_equal:
        raise PublicProjectionError(
            "inconsistent step contract: different-model verifier provenance "
            "matches the drafter"
        )


def _validate_retry_contract(report: LoopReport) -> None:
    run = report.run
    retry_steps = [step for step in run.steps if step.phase == LoopPhase.RETRY]
    if len(retry_steps) > run.policy.max_retries:
        raise PublicProjectionError(
            "inconsistent retry contract: retry budget exceeded"
        )
    for retry_number, step in enumerate(retry_steps, start=1):
        if step.decision != LoopDecision.RETRY:
            raise PublicProjectionError(
                "inconsistent retry contract: retry phase did not request retry"
            )
        if step.retry_count != retry_number:
            raise PublicProjectionError(
                "inconsistent retry contract: retry count is not monotonic"
            )
        step_index = run.steps.index(step)
        _validate_retry_trigger(run, step_index=step_index, step=step)
    retry_epoch = 0
    for step in run.steps:
        expected_retry_count = (
            retry_epoch + 1
            if step.phase == LoopPhase.RETRY
            else retry_epoch
        )
        if step.retry_count != expected_retry_count:
            step_kind = (
                "verifier" if step.phase == LoopPhase.VERIFY else "step"
            )
            raise PublicProjectionError(
                f"inconsistent retry contract: {step_kind} retry count does not "
                "match the current retry epoch"
            )
        if step.phase == LoopPhase.RETRY:
            retry_epoch += 1
    pending_retry_index = None
    seen_draft = False
    retries_seen = 0
    for index, step in enumerate(run.steps):
        if step.phase == LoopPhase.RETRY:
            retries_seen += 1
            if pending_retry_index is not None:
                raise PublicProjectionError(
                    "inconsistent retry contract: retry has no new draft candidate"
                )
            pending_retry_index = index
            continue
        if step.phase != LoopPhase.DRAFT:
            continue
        if step.retry_count != retries_seen:
            raise PublicProjectionError(
                "inconsistent retry contract: draft retry count does not match "
                "the recorded retry sequence"
            )
        if seen_draft and pending_retry_index is None:
            raise PublicProjectionError(
                "inconsistent retry contract: new draft lacks a recorded retry"
            )
        seen_draft = True
        pending_retry_index = None
    if pending_retry_index is not None:
        later_steps = run.steps[pending_retry_index + 1 :]
        retry_step = run.steps[pending_retry_index]
        explicit_terminal_interruption = (
            run.final_decision not in _VISIBLE_FINAL_DECISIONS
            and bool(later_steps)
            and (
                _is_terminal_boundary_step(later_steps[0])
                or _is_causal_failed_retry_attempt(
                    retry_step=retry_step,
                    failed_step=later_steps[0],
                    run=run,
                )
            )
        )
        if not explicit_terminal_interruption:
            raise PublicProjectionError(
                "inconsistent retry contract: retry has no new draft candidate"
            )
    pending_retry_request = None
    for index, step in enumerate(run.steps):
        if step.decision == LoopDecision.RETRY and step.phase != LoopPhase.RETRY:
            terminal_guardrail_retry = (
                _terminal_guardrail_decision(
                    step.metadata,
                    location=f"step {step.step_id!r}",
                )
                == LoopDecision.RETRY
                and run.final_decision == LoopDecision.BLOCK
            )
            if terminal_guardrail_retry:
                if (
                    pending_retry_request is not None
                    and index != pending_retry_request[0] + 1
                ):
                    raise PublicProjectionError(
                        "inconsistent retry contract: execution continued before "
                        "a terminal retry guardrail"
                    )
                pending_retry_request = None
                continue
            if pending_retry_request is not None:
                raise PublicProjectionError(
                    "inconsistent retry contract: one retry record cannot resolve "
                    "multiple retry requests"
                )
            pending_retry_request = (index, step)
            continue
        if step.phase == LoopPhase.RETRY and pending_retry_request is not None:
            _request_index, request_step = pending_retry_request
            if step.metadata.get("retry_trigger_step_id") != request_step.step_id:
                raise PublicProjectionError(
                    "inconsistent retry contract: retry token does not consume "
                    "the pending request"
                )
            pending_retry_request = None
    if pending_retry_request is not None:
        request_index, _step = pending_retry_request
        terminal_interruption = (
            run.final_decision not in _VISIBLE_FINAL_DECISIONS
            and request_index + 1 < len(run.steps)
            and _is_terminal_boundary_step(run.steps[request_index + 1])
        )
        if not terminal_interruption:
            raise PublicProjectionError(
                "inconsistent retry contract: retry request was not resolved"
            )

    if run.terminal_reason == LoopTerminalReason.RETRY_BUDGET_EXHAUSTED:
        causal_denial_steps = []
        declared_denial_steps = []
        for index, step in enumerate(run.steps):
            metadata = step.metadata
            retry_unavailable = metadata.get("retry_unavailable")
            retry_budget_exhausted = metadata.get("retry_budget_exhausted")
            retry_denied = metadata.get("retry_denied")
            reasons = metadata.get("reasons")
            preceding_step = run.steps[index - 1] if index else None
            preceding_candidate = run.steps[index - 2] if index >= 2 else None
            format_request_lineage = (
                step.phase == LoopPhase.FORMAT_CHECK
                and preceding_step is not None
                and preceding_step.phase == LoopPhase.DRAFT
                and preceding_step.decision == LoopDecision.CONTINUE
                and preceding_step.retry_count == step.retry_count
            )
            mechanical_request_lineage = (
                step.phase == LoopPhase.MECHANICAL_CHECK
                and preceding_step is not None
                and preceding_step.phase == LoopPhase.FORMAT_CHECK
                and preceding_step.decision == LoopDecision.CONTINUE
                and preceding_step.retry_count == step.retry_count
                and (
                    (
                        preceding_step.output_summary == "format_passed"
                        and preceding_candidate is not None
                        and preceding_candidate.phase == LoopPhase.DRAFT
                        and preceding_candidate.decision == LoopDecision.CONTINUE
                        and preceding_candidate.retry_count == step.retry_count
                    )
                    or (
                        preceding_step.output_summary == "format_sanitized"
                        and preceding_step.metadata.get(
                            "sanitized_internal_labels"
                        )
                        is True
                    )
                )
            )
            retry_request_shape = (
                type(reasons) is list
                and bool(reasons)
                and all(type(reason) is str and bool(reason) for reason in reasons)
                and step.output_summary == "needs_retry"
                and (
                    (
                        step.phase == LoopPhase.FORMAT_CHECK
                        and step.decision == LoopDecision.ERROR
                    )
                    or (
                        step.phase == LoopPhase.MECHANICAL_CHECK
                        and step.decision == LoopDecision.NOT_VERIFIED
                    )
                )
            )
            retry_requesting_check = retry_request_shape and (
                format_request_lineage or mechanical_request_lineage
            )
            failed_retry_attempt = (
                step.phase == LoopPhase.FINAL
                and index >= 2
                and bool(retry_steps)
                and run.steps[index - 2].step_id == retry_steps[-1].step_id
                and _is_causal_failed_retry_attempt(
                    retry_step=retry_steps[-1],
                    failed_step=run.steps[index - 1],
                    run=run,
                )
            )
            valid_denial_flags = (
                retry_denied is True
                and retry_budget_exhausted is True
                and type(retry_unavailable) is bool
                and retry_denied
                == (retry_unavailable or retry_budget_exhausted)
                and step.retry_count == run.policy.max_retries
            )
            if valid_denial_flags and retry_request_shape:
                declared_denial_steps.append((index, step))
            if valid_denial_flags and (
                retry_requesting_check or failed_retry_attempt
            ):
                causal_denial_steps.append((index, step))
        if not causal_denial_steps:
            final_indexes = [
                index
                for index, step in enumerate(run.steps)
                if step.phase == LoopPhase.FINAL
            ]
            final_index = final_indexes[0] if final_indexes else len(run.steps)
            declared_pre_final_denials = [
                item for item in declared_denial_steps if item[0] < final_index
            ]
            if declared_pre_final_denials:
                denial_index, _denial_step = declared_pre_final_denials[-1]
                terminal_suffix = run.steps[denial_index + 1 : final_index]
                unresolved_denial = all(
                    (
                        step.phase == LoopPhase.REFUSE
                        and step.decision == LoopDecision.REFUSE
                    )
                    or (
                        step.phase == LoopPhase.ERROR
                        and step.decision == LoopDecision.ERROR
                    )
                    for step in terminal_suffix
                )
                if not unresolved_denial:
                    raise PublicProjectionError(
                        "inconsistent retry contract: retry-budget terminal "
                        "cause uses a stale or resolved retry denial"
                    )
            raise PublicProjectionError(
                "inconsistent retry contract: retry-budget terminal cause lacks "
                "a causal denied retry request"
            )
        final_indexes = [
            index
            for index, step in enumerate(run.steps)
            if step.phase == LoopPhase.FINAL
        ]
        final_index = final_indexes[0] if final_indexes else len(run.steps)
        pre_final_denials = [
            item for item in causal_denial_steps if item[0] < final_index
        ]
        denial_index, denial_step = (
            pre_final_denials[-1]
            if pre_final_denials
            else causal_denial_steps[-1]
        )
        if denial_step.retry_count != len(retry_steps):
            raise PublicProjectionError(
                "inconsistent retry contract: retry-budget terminal cause uses a "
                "stale retry denial"
            )
        if denial_index == final_index:
            preceding_step = run.steps[final_index - 1] if final_index else None
            unresolved_denial = (
                preceding_step is not None
                and preceding_step.retry_count == denial_step.retry_count
                and preceding_step.decision == LoopDecision.ERROR
                and preceding_step.phase in {LoopPhase.DRAFT, LoopPhase.ERROR}
            )
        else:
            terminal_suffix = run.steps[denial_index + 1 : final_index]
            unresolved_denial = all(
                (
                    step.phase == LoopPhase.REFUSE
                    and step.decision == LoopDecision.REFUSE
                )
                or (
                    step.phase == LoopPhase.ERROR
                    and step.decision == LoopDecision.ERROR
                )
                for step in terminal_suffix
            )
        if not unresolved_denial:
            raise PublicProjectionError(
                "inconsistent retry contract: retry-budget terminal cause uses a "
                "stale or resolved retry denial"
            )


def _is_causal_failed_retry_attempt(
    *,
    retry_step: LoopStep,
    failed_step: LoopStep,
    run: LoopRun,
) -> bool:
    """Recognize the exact web-retrieval failure emitted by a prepared retry."""

    return (
        run.final_decision == LoopDecision.ERROR
        and run.terminal_reason == LoopTerminalReason.RETRY_BUDGET_EXHAUSTED
        and retry_step.phase == LoopPhase.RETRY
        and failed_step.phase == LoopPhase.ERROR
        and failed_step.decision == LoopDecision.ERROR
        and failed_step.retry_count == retry_step.retry_count
        and failed_step.output_summary == "web_search_failed"
        and failed_step.error_message == "web_search_failed"
        and failed_step.metadata.get("planned_phase") == LoopPhase.DRAFT.value
        and failed_step.metadata.get("reasons") == ["web_search_failed"]
    )


def _validate_retry_trigger(
    run: LoopRun,
    *,
    step_index: int,
    step: LoopStep,
) -> None:
    if step_index == 0:
        raise PublicProjectionError(
            "inconsistent retry contract: retry lacks a causal trigger"
        )
    trigger = run.steps[step_index - 1]
    metadata = step.metadata
    reasons = metadata.get("reasons")
    trigger_reasons = trigger.metadata.get("reasons")
    if (
        metadata.get("retry_trigger_step_id") != trigger.step_id
        or type(metadata.get("retry_count")) is not int
        or metadata.get("retry_count") != step.retry_count
        or type(metadata.get("max_retries")) is not int
        or metadata.get("max_retries") != run.policy.max_retries
        or type(reasons) is not list
        or not reasons
        or any(type(reason) is not str or not reason for reason in reasons)
        or reasons != trigger_reasons
        or trigger.retry_count != step.retry_count - 1
    ):
        raise PublicProjectionError(
            "inconsistent retry contract: retry trigger binding is invalid"
        )

    retry_reason = metadata.get("retry_reason")
    if retry_reason == "format_check":
        valid_trigger = (
            trigger.phase == LoopPhase.FORMAT_CHECK
            and trigger.decision == LoopDecision.RETRY
            and trigger.output_summary == "needs_retry"
        )
    elif retry_reason == "self_check":
        valid_trigger = (
            trigger.phase == LoopPhase.MECHANICAL_CHECK
            and trigger.decision == LoopDecision.RETRY
            and trigger.output_summary == "needs_retry"
        )
    elif retry_reason in {"web_verifier", "smart_web_direct_fallback"}:
        verifier_trigger = (
            trigger.phase == LoopPhase.VERIFY
            and trigger.decision == LoopDecision.NOT_VERIFIED
            and trigger.verification is not None
            and trigger.verification.outcome
            in {VerificationOutcome.UNSUPPORTED, VerificationOutcome.INSUFFICIENT}
        )
        search_error_trigger = (
            retry_reason == "smart_web_direct_fallback"
            and trigger.phase == LoopPhase.ERROR
            and trigger.decision == LoopDecision.ERROR
            and trigger.output_summary == "web_search_failed"
        )
        valid_trigger = verifier_trigger or search_error_trigger
    else:
        valid_trigger = False
    if not valid_trigger:
        raise PublicProjectionError(
            "inconsistent retry contract: retry reason does not match its trigger"
        )


def _validate_visible_answer_contract(
    report: LoopReport,
    *,
    terminal_redaction: bool,
) -> None:
    run = report.run
    if terminal_redaction or run.final_decision not in _VISIBLE_FINAL_DECISIONS:
        return
    final_answer = run.final_answer
    if type(final_answer) is not str or not final_answer:
        raise PublicProjectionError(
            "inconsistent visible answer contract: final answer is required"
        )
    if not run.steps or run.steps[-1].phase != LoopPhase.FINAL:
        raise PublicProjectionError(
            "inconsistent visible answer contract: terminal final step is required"
        )
    final_step = run.steps[-1]
    if final_step.decision != run.final_decision:
        raise PublicProjectionError(
            "inconsistent visible answer contract: terminal final decision mismatch"
        )
    _require_answer_candidate_digest(
        final_step,
        expected_digest=answer_candidate_sha256(final_answer),
        location="terminal final step",
    )
    _require_evidence_set_digest(
        final_step,
        expected_digest=evidence_set_sha256(run.evidence),
        location="terminal final step",
    )
    if (final_step.backend, final_step.model_label) != (
        run.backend,
        run.model_label,
    ):
        raise PublicProjectionError(
            "inconsistent visible answer contract: terminal final provenance "
            "does not match the run drafter"
        )
    retries_used = sum(
        1 for step in run.steps if step.phase == LoopPhase.RETRY
    )
    if final_step.retry_count != retries_used:
        raise PublicProjectionError(
            "inconsistent visible answer contract: terminal final retry count "
            "does not match the executed retry sequence"
        )
    inline_citation_ids = _inline_citation_ids(final_answer)
    evidence_citation_ids = {
        reference.citation_id for reference in run.evidence
    }
    if not inline_citation_ids.issubset(evidence_citation_ids):
        raise PublicProjectionError(
            "inconsistent visible answer contract: final answer cites unknown evidence"
        )

    candidate_steps = [
        (index, step)
        for index, step in enumerate(run.steps[:-1])
        if step.phase == LoopPhase.DRAFT
        and step.decision == LoopDecision.CONTINUE
        or (
            step.phase == LoopPhase.FORMAT_CHECK
            and step.decision == LoopDecision.CONTINUE
            and step.metadata.get("sanitized_internal_labels") is True
        )
    ]
    verify_steps = [
        (index, step)
        for index, step in enumerate(run.steps[:-1])
        if step.phase == LoopPhase.VERIFY
    ]
    if not candidate_steps:
        if verify_steps or not _is_valid_identity_answer_shortcut(run, final_step):
            raise PublicProjectionError(
                "inconsistent visible answer contract: terminal answer lacks a "
                "bound candidate"
            )
        return

    candidate_index, candidate_step = candidate_steps[-1]
    final_answer_digest = answer_candidate_sha256(final_answer)
    final_evidence_digest = evidence_set_sha256(run.evidence)
    _require_answer_candidate_digest(
        candidate_step,
        expected_digest=final_answer_digest,
        location="terminal candidate",
    )
    _require_evidence_set_digest(
        candidate_step,
        expected_digest=final_evidence_digest,
        location="terminal candidate",
    )
    source_drafter_step = candidate_step
    if candidate_step.phase == LoopPhase.FORMAT_CHECK:
        source_drafter_step = next(
            (
                step
                for step in reversed(run.steps[:candidate_index])
                if step.phase == LoopPhase.DRAFT
            ),
            candidate_step,
        )
    for provenance_step, location in (
        (source_drafter_step, "source draft"),
        (candidate_step, "terminal candidate"),
    ):
        if (provenance_step.backend, provenance_step.model_label) != (
            run.backend,
            run.model_label,
        ):
            raise PublicProjectionError(
                "inconsistent visible answer contract: "
                f"{location} provenance does not match the run drafter"
            )
    if candidate_step.retry_count != retries_used:
        raise PublicProjectionError(
            "inconsistent visible answer contract: terminal candidate retry count "
            "does not match the executed retry sequence"
        )

    if run.final_decision == LoopDecision.FINAL:
        if verify_steps:
            raise PublicProjectionError(
                "inconsistent visible answer contract: unverified verifier step "
                "precedes a final decision"
            )
        _validate_candidate_completion_suffix(
            run,
            candidate_index=candidate_index,
            boundary_index=len(run.steps) - 1,
        )
        return

    if not verify_steps:
        raise PublicProjectionError(
            "inconsistent visible answer contract: terminal verifier is required"
        )
    verify_index, verify_step = verify_steps[-1]
    if verify_index <= candidate_index:
        raise PublicProjectionError(
            "inconsistent visible answer contract: terminal candidate was not "
            "verified"
        )
    _validate_candidate_completion_suffix(
        run,
        candidate_index=candidate_index,
        boundary_index=verify_index,
    )
    _require_answer_candidate_digest(
        verify_step,
        expected_digest=final_answer_digest,
        location="terminal verifier step",
    )
    _require_evidence_set_digest(
        verify_step,
        expected_digest=final_evidence_digest,
        location="terminal verifier step",
    )
    if verify_step.retry_count != retries_used:
        raise PublicProjectionError(
            "inconsistent visible answer contract: terminal verifier retry count "
            "does not match the executed retry sequence"
        )
    if run.final_decision == LoopDecision.NOT_VERIFIED and (
        verify_step.decision != LoopDecision.NOT_VERIFIED
    ):
        raise PublicProjectionError(
            "inconsistent visible answer contract: terminal verifier decision "
            "does not match the final decision"
        )
    if run.steps[verify_index + 1 : -1]:
        raise PublicProjectionError(
            "inconsistent visible answer contract: terminal verifier does not bind "
            "the terminal completion"
        )


def _validate_candidate_completion_suffix(
    run: LoopRun,
    *,
    candidate_index: int,
    boundary_index: int,
) -> None:
    for step in run.steps[candidate_index + 1 : boundary_index]:
        valid_format_check = (
            step.phase == LoopPhase.FORMAT_CHECK
            and step.decision == LoopDecision.CONTINUE
            and step.output_summary == "format_passed"
        )
        valid_mechanical_check = (
            step.phase == LoopPhase.MECHANICAL_CHECK
            and step.decision == LoopDecision.CONTINUE
            and step.output_summary == "mechanical_checks_passed"
        )
        if not (valid_format_check or valid_mechanical_check):
            raise PublicProjectionError(
                "inconsistent visible answer contract: unresolved step follows "
                "the terminal candidate"
            )


def _is_valid_identity_answer_shortcut(
    run: LoopRun,
    final_step: LoopStep,
) -> bool:
    document_name = run.metadata.get("document_name")
    return (
        run.final_decision == LoopDecision.FINAL
        and run.context_provider == "document"
        and not run.evidence
        and run.metadata.get("identity_answer") is True
        and final_step.metadata.get("identity_answer") is True
        and type(document_name) is str
        and bool(document_name)
        and run.final_answer == f"The indexed file is `{document_name}`."
        and all(
            step.phase in {LoopPhase.INPUT, LoopPhase.CONTEXT_SELECT}
            and step.decision == LoopDecision.CONTINUE
            for step in run.steps[:-1]
        )
    )


def _inline_citation_ids(answer: str) -> set[int]:
    citation_ids = set()
    for match in _INLINE_CITATION_PATTERN.finditer(answer):
        token = match.group(1)
        if len(token) > 16:
            raise PublicProjectionError(
                "malformed public inline citation identifier"
            )
        citation_id = int(token)
        if citation_id < 1 or citation_id > _PUBLIC_MAX_JSON_INTEGER:
            raise PublicProjectionError(
                "malformed public inline citation identifier"
            )
        citation_ids.add(citation_id)
    return citation_ids


def _validate_supported_contract(report: LoopReport) -> None:
    run = report.run
    if run.final_decision != LoopDecision.SUPPORTED:
        return

    final_answer = run.final_answer
    if type(final_answer) is not str or not final_answer:
        raise PublicProjectionError(
            "inconsistent supported contract: final answer is required"
        )
    if not run.evidence:
        raise PublicProjectionError(
            "inconsistent supported contract: context evidence is required"
        )

    inline_citation_ids = _inline_citation_ids(final_answer)
    evidence_citation_ids = {
        reference.citation_id for reference in run.evidence
    }
    if not inline_citation_ids.issubset(evidence_citation_ids):
        raise PublicProjectionError(
            "inconsistent supported contract: final answer cites unknown evidence"
        )
    if run.policy.require_citations and not inline_citation_ids:
        raise PublicProjectionError(
            "inconsistent supported contract: cited evidence is required"
        )
    if run.context_provider == "none":
        raise PublicProjectionError(
            "inconsistent supported contract: context evidence is required"
        )

    if not run.steps or run.steps[-1].phase != LoopPhase.FINAL:
        raise PublicProjectionError(
            "inconsistent supported contract: terminal final step is required"
        )
    final_step = run.steps[-1]
    if final_step.decision != LoopDecision.SUPPORTED:
        raise PublicProjectionError(
            "inconsistent supported contract: final step did not record support"
        )
    final_digest = answer_candidate_sha256(final_answer)
    final_evidence_digest = evidence_set_sha256(run.evidence)
    _require_answer_candidate_digest(
        final_step,
        expected_digest=final_digest,
        location="final step",
    )
    _require_evidence_set_digest(
        final_step,
        expected_digest=final_evidence_digest,
        location="final step",
    )

    verify_steps = [step for step in run.steps if step.phase == LoopPhase.VERIFY]
    final_verify_step = verify_steps[-1] if verify_steps else None
    verifier_backend = None
    if final_verify_step is not None:
        verification = final_verify_step.verification
        if (
            final_verify_step.decision != LoopDecision.SUPPORTED
            or verification is None
            or verification.outcome != VerificationOutcome.SUPPORTED
        ):
            raise PublicProjectionError(
                "inconsistent supported contract: final verifier did not support "
                "the final answer"
            )
        verifier_backend = verification.verifier_backend
        _require_answer_candidate_digest(
            final_verify_step,
            expected_digest=final_digest,
            location="final verifier step",
        )
        _require_evidence_set_digest(
            final_verify_step,
            expected_digest=final_evidence_digest,
            location="final verifier step",
        )
    else:
        raise PublicProjectionError(
            "inconsistent supported contract: verifier support is required"
        )

    draft_steps = [
        step for step in run.steps[:-1] if step.phase == LoopPhase.DRAFT
    ]
    if not draft_steps:
        raise PublicProjectionError(
            "inconsistent supported contract: draft candidate is required"
        )
    candidate_steps = [
        step
        for step in run.steps[:-1]
        if step.phase == LoopPhase.DRAFT
        or (
            step.phase == LoopPhase.FORMAT_CHECK
            and step.metadata.get("sanitized_internal_labels") is True
        )
    ]
    if not candidate_steps:
        raise PublicProjectionError(
            "inconsistent supported contract: draft candidate is required"
        )
    final_candidate_step = candidate_steps[-1]
    final_draft_step = draft_steps[-1]
    if (
        final_draft_step.decision != LoopDecision.CONTINUE
        or final_candidate_step.decision != LoopDecision.CONTINUE
    ):
        raise PublicProjectionError(
            "inconsistent supported contract: final draft candidate did not "
            "complete successfully"
        )
    _require_answer_candidate_digest(
        final_candidate_step,
        expected_digest=final_digest,
        location="final draft candidate",
    )
    _require_evidence_set_digest(
        final_candidate_step,
        expected_digest=final_evidence_digest,
        location="final draft candidate",
    )
    expected_drafter_pair = (run.backend, run.model_label)
    for candidate_step, location in (
        (final_draft_step, "final draft"),
        (final_candidate_step, "final candidate"),
    ):
        if (candidate_step.backend, candidate_step.model_label) != expected_drafter_pair:
            raise PublicProjectionError(
                "inconsistent supported contract: "
                f"{location} provenance does not match the run drafter"
            )
    if final_verify_step is not None:
        final_candidate_index = run.steps.index(final_candidate_step)
        final_verify_index = run.steps.index(final_verify_step)
        if final_candidate_index >= final_verify_index:
            raise PublicProjectionError(
                "inconsistent supported contract: final candidate was not verified"
            )
        if (
            final_candidate_step.ended_at is not None
            and final_verify_step.started_at < final_candidate_step.ended_at
        ):
            raise PublicProjectionError(
                "inconsistent supported contract: verifier started before the "
                "final candidate completed"
            )
        post_verifier_steps = run.steps[final_verify_index + 1 : -1]
        if any(
            step.phase == LoopPhase.RETRY or step.decision == LoopDecision.RETRY
            for step in post_verifier_steps
        ):
            raise PublicProjectionError(
                "inconsistent supported contract: unresolved retry follows the "
                "final verifier"
            )
        if post_verifier_steps:
            raise PublicProjectionError(
                "inconsistent supported contract: final verifier does not bind "
                "the terminal completion"
            )

    if not isinstance(verifier_backend, str) or not verifier_backend.strip():
        raise PublicProjectionError(
            "inconsistent supported contract: real verifier backend provenance "
            "is required"
        )

    if run.backend not in {"ollama", "openai-compatible"}:
        raise PublicProjectionError(
            "inconsistent supported contract: concrete real drafter backend is "
            "required"
        )
    if verifier_backend not in {"ollama", "openai-compatible"}:
        raise PublicProjectionError(
            "inconsistent supported contract: concrete real verifier backend is "
            "required"
        )


def _require_answer_candidate_digest(
    step: LoopStep,
    *,
    expected_digest: str,
    location: str,
) -> None:
    value = step.metadata.get(ANSWER_CANDIDATE_SHA256_METADATA_KEY)
    if (
        type(value) is not str
        or _ANSWER_CANDIDATE_SHA256_PATTERN.fullmatch(value) is None
        or value != expected_digest
    ):
        raise PublicProjectionError(
            f"inconsistent answer binding contract: {location} is not bound to the "
            "final answer"
        )


def _require_evidence_set_digest(
    step: LoopStep,
    *,
    expected_digest: str,
    location: str,
) -> None:
    value = step.metadata.get(EVIDENCE_SET_SHA256_METADATA_KEY)
    if (
        type(value) is not str
        or _ANSWER_CANDIDATE_SHA256_PATTERN.fullmatch(value) is None
        or value != expected_digest
    ):
        raise PublicProjectionError(
            f"inconsistent supported contract: {location} is not bound to the "
            "final evidence set"
        )


def _validate_run_timeline(report: LoopReport) -> None:
    run = report.run
    previous_started_at = None
    previous_ended_at = None
    previous_step = None
    for step in run.steps:
        if step.started_at < run.started_at:
            raise PublicProjectionError(
                "inconsistent run timeline: step starts before the run"
            )
        if (
            previous_started_at is not None
            and step.started_at < previous_started_at
        ):
            raise PublicProjectionError(
                "inconsistent run timeline: step tuple is not chronological"
            )
        if previous_step is not None and previous_step.ended_at is None:
            raise PublicProjectionError(
                "inconsistent run timeline: a later step follows an unfinished step"
            )
        if previous_ended_at is not None and step.started_at < previous_ended_at:
            raise PublicProjectionError(
                "inconsistent run timeline: sequential steps overlap"
            )
        if run.completed_at is not None:
            if step.started_at > run.completed_at:
                raise PublicProjectionError(
                    "inconsistent run timeline: step starts after run completion"
                )
            if step.ended_at is not None and step.ended_at > run.completed_at:
                raise PublicProjectionError(
                    "inconsistent run timeline: step ends after run completion"
                )
        previous_started_at = step.started_at
        previous_ended_at = step.ended_at
        previous_step = step
        if step.human_review is not None:
            review_created_at = step.human_review.created_at
            if review_created_at < run.started_at or (
                run.completed_at is not None
                and review_created_at > run.completed_at
            ):
                raise PublicProjectionError(
                    "inconsistent run timeline: human review timestamp falls "
                    "outside the run"
                )


def _validate_completed_timeline_contract(report: LoopReport) -> None:
    run = report.run
    if run.final_decision is None:
        if run.completed_at is not None:
            raise PublicProjectionError(
                "inconsistent run timeline: completion timestamp lacks a terminal "
                "decision"
            )
        return
    if run.completed_at is None:
        raise PublicProjectionError(
            "inconsistent run timeline: terminal run lacks completion timestamp"
        )
    if any(step.ended_at is None for step in run.steps):
        raise PublicProjectionError(
            "inconsistent run timeline: terminal run contains an unfinished step"
        )


def _validate_unique_identities(report: LoopReport) -> None:
    step_ids = [step.step_id for step in report.run.steps]
    if len(step_ids) != len(set(step_ids)):
        raise PublicProjectionError("loop report contains duplicate step identities")
    citation_ids = [reference.citation_id for reference in report.run.evidence]
    if len(citation_ids) != len(set(citation_ids)):
        raise PublicProjectionError("loop report contains duplicate citation identities")
    evidence_ids = [reference.evidence_id for reference in report.run.evidence]
    if len(evidence_ids) != len(set(evidence_ids)):
        raise PublicProjectionError("loop report contains duplicate evidence identities")


def _project_step(step, *, terminal_redaction: bool) -> Dict[str, Any]:
    verification = step.verification
    return {
        "step_id": step.step_id,
        "phase": step.phase.value,
        "decision": step.decision.value,
        "started_at": _iso_value(step.started_at),
        "ended_at": _iso_value(step.ended_at),
        "duration_ms": step.duration_ms,
        "backend": None if terminal_redaction else step.backend,
        "model_label": None if terminal_redaction else step.model_label,
        "retry_count": step.retry_count,
        "error_present": bool(step.error_message),
        "verification": (
            {
                "outcome": verification.outcome.value,
                "verifier_backend": (
                    None if terminal_redaction else verification.verifier_backend
                ),
                "verifier_model_label": (
                    None if terminal_redaction else verification.verifier_model_label
                ),
                "same_model_as_drafter": (
                    None if terminal_redaction else verification.same_model_as_drafter
                ),
            }
            if verification is not None
            else None
        ),
        "human_review_required": step.human_review is not None,
    }


def _public_final_answer_allowed(
    report: LoopReport,
    terminal_redaction: bool,
) -> bool:
    return (
        not terminal_redaction
        and report.run.final_decision
        in {LoopDecision.FINAL, LoopDecision.SUPPORTED, LoopDecision.NOT_VERIFIED}
    )


def _iso_value(value) -> Optional[str]:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
