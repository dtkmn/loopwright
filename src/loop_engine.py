from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, Tuple
from uuid import uuid4


SCHEMA_VERSION = "loop-report/v1"
LOOP_SESSION_SCHEMA_VERSION = "loop-session/v1"
LOOP_RECIPE_SCHEMA_VERSION = "loop-recipe/v1"
DEFAULT_LOOP_RECIPE_ID = "recipe_general_loop"
DEFAULT_LOOP_RECIPE_NAME = "General assistant loop"
DEFAULT_LOOP_RECIPE_GOAL = (
    "Answer the current user request clearly, using smart evidence selection "
    "to choose web evidence, uploaded files, or direct model knowledge as "
    "appropriate."
)
DEFAULT_LOOP_RECIPE_INSTRUCTIONS = (
    "Prefer useful, specific answers. Use same-thread memory only to resolve "
    "references. Prefer web evidence for current or general lookup questions, "
    "use uploaded files only when the question is about those files, do not "
    "invent citations, and surface uncertainty when evidence is missing."
)
DEFAULT_LOOP_RECIPE_SUCCESS_CRITERIA = (
    "The answer addresses the current request.",
    "Evidence-grounded claims use retrieved web or file evidence and citations.",
    "Model-knowledge answers are marked not_verified instead of supported.",
)
DEFAULT_LOOP_RECIPE_STOP_CONDITION = (
    "Stop when the answer passes mechanical checks, is refused, or reaches the "
    "configured retry limit."
)
GUARDRAIL_DECISION_VALUES = frozenset(
    {
        "continue",
        "retry",
        "refuse",
        "block",
        "requires_review",
    }
)
TERMINAL_PUBLIC_REDACTION_DECISION_VALUES = frozenset(
    {
        "refuse",
        "block",
        "requires_review",
    }
)
PUBLIC_REDACTION_REASON = "terminal_public_redaction"
PUBLIC_REDACTION_TEXT = "[redacted: terminal decision]"


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _datetime_to_json(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.astimezone(timezone.utc)
    return normalized.isoformat().replace("+00:00", "Z")


def _datetime_from_json(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def _metadata_dict(metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(metadata or {})


def _string_tuple(values) -> Tuple[str, ...]:
    return tuple(str(value) for value in (values or ()))


def _clean_text(value: Any, *, max_chars: int = 4000) -> str:
    return " ".join(str(value or "").split()).strip()[:max_chars]


def _json_bool(data: Mapping[str, Any], key: str, default: bool) -> bool:
    if key not in data:
        return default
    value = data[key]
    if type(value) is not bool:
        raise ValueError(f"{key} must be a JSON boolean")
    return value


def _require_bool(value: bool, field_name: str) -> None:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be a boolean")


class LoopPhase(str, Enum):
    INPUT = "input"
    CONTEXT_SELECT = "context_select"
    RETRIEVE = "retrieve"
    DRAFT = "draft"
    FORMAT_CHECK = "format_check"
    MECHANICAL_CHECK = "mechanical_check"
    VERIFY = "verify"
    RETRY = "retry"
    REFUSE = "refuse"
    FINAL = "final"
    ERROR = "error"


class LoopDecision(str, Enum):
    CONTINUE = "continue"
    RETRY = "retry"
    REFUSE = "refuse"
    BLOCK = "block"
    REQUIRES_REVIEW = "requires_review"
    SUPPORTED = "supported"
    NOT_VERIFIED = "not_verified"
    FINAL = "final"
    ERROR = "error"


class VerificationOutcome(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    INSUFFICIENT = "insufficient"
    NOT_VERIFIED = "not_verified"
    ERROR = "error"


@dataclass(frozen=True)
class LoopRecipe:
    name: str
    goal: str
    instructions: str = ""
    success_criteria: Tuple[str, ...] = ()
    stop_condition: str = DEFAULT_LOOP_RECIPE_STOP_CONDITION
    context_provider: str = "smart"
    model_profile: str = "quality"
    verifier: str = "default"
    recipe_id: str = field(default_factory=lambda: _new_id("recipe"))
    description: str = ""
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = LOOP_RECIPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        recipe_id = _clean_text(self.recipe_id, max_chars=96)
        name = _clean_text(self.name, max_chars=96)
        goal = _clean_text(self.goal, max_chars=2000)
        if not recipe_id:
            raise ValueError("Loop recipe id must not be empty")
        if not name:
            raise ValueError("Loop recipe name must not be empty")
        if not goal:
            raise ValueError("Loop recipe goal must not be empty")
        criteria = tuple(
            criterion
            for criterion in (
                _clean_text(value, max_chars=500)
                for value in self.success_criteria
            )
            if criterion
        )
        object.__setattr__(self, "recipe_id", recipe_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "goal", goal)
        object.__setattr__(
            self,
            "instructions",
            _clean_text(self.instructions, max_chars=4000),
        )
        object.__setattr__(self, "success_criteria", criteria[:12])
        object.__setattr__(
            self,
            "stop_condition",
            _clean_text(self.stop_condition, max_chars=1000)
            or DEFAULT_LOOP_RECIPE_STOP_CONDITION,
        )
        object.__setattr__(
            self,
            "context_provider",
            _clean_text(self.context_provider, max_chars=64) or "smart",
        )
        object.__setattr__(
            self,
            "model_profile",
            _clean_text(self.model_profile, max_chars=64) or "quality",
        )
        object.__setattr__(
            self,
            "verifier",
            _clean_text(self.verifier, max_chars=64) or "default",
        )
        object.__setattr__(
            self,
            "description",
            _clean_text(self.description, max_chars=500),
        )
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))
        if self.schema_version != LOOP_RECIPE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported loop recipe schema: {self.schema_version}")

    @property
    def is_default(self) -> bool:
        return self.recipe_id == DEFAULT_LOOP_RECIPE_ID

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "recipe_id": self.recipe_id,
            "name": self.name,
            "description": self.description,
            "goal": self.goal,
            "context_provider": self.context_provider,
            "model_profile": self.model_profile,
            "verifier": self.verifier,
            "is_default": self.is_default,
            "updated_at": _datetime_to_json(self.updated_at),
        }

    def runtime_dict(self) -> Dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "name": self.name,
            "goal": self.goal,
            "instructions": self.instructions,
            "success_criteria": list(self.success_criteria),
            "stop_condition": self.stop_condition,
            "context_provider": self.context_provider,
            "model_profile": self.model_profile,
            "verifier": self.verifier,
        }

    def to_dict(self) -> Dict[str, Any]:
        payload = self.runtime_dict()
        payload.update(
            {
                "schema_version": self.schema_version,
                "description": self.description,
                "created_at": _datetime_to_json(self.created_at),
                "updated_at": _datetime_to_json(self.updated_at),
                "metadata": _metadata_dict(self.metadata),
                "is_default": self.is_default,
            }
        )
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LoopRecipe":
        schema_version = str(data.get("schema_version", LOOP_RECIPE_SCHEMA_VERSION))
        if schema_version != LOOP_RECIPE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported loop recipe schema: {schema_version}")
        return cls(
            schema_version=schema_version,
            recipe_id=str(data.get("recipe_id") or data.get("id") or ""),
            name=str(data.get("name") or ""),
            description=str(data.get("description") or ""),
            goal=str(data.get("goal") or ""),
            instructions=str(data.get("instructions") or ""),
            success_criteria=_string_tuple(data.get("success_criteria")),
            stop_condition=str(
                data.get("stop_condition") or DEFAULT_LOOP_RECIPE_STOP_CONDITION
            ),
            context_provider=str(data.get("context_provider") or "smart"),
            model_profile=str(data.get("model_profile") or "quality"),
            verifier=str(data.get("verifier") or "default"),
            created_at=_datetime_from_json(data.get("created_at")) or utc_now(),
            updated_at=_datetime_from_json(data.get("updated_at")) or utc_now(),
            metadata=_metadata_dict(data.get("metadata")),
        )


def default_loop_recipe(*, created_at: Optional[datetime] = None) -> LoopRecipe:
    timestamp = created_at or utc_now()
    return LoopRecipe(
        recipe_id=DEFAULT_LOOP_RECIPE_ID,
        name=DEFAULT_LOOP_RECIPE_NAME,
        description="Default evidence loop behavior.",
        goal=DEFAULT_LOOP_RECIPE_GOAL,
        instructions=DEFAULT_LOOP_RECIPE_INSTRUCTIONS,
        success_criteria=DEFAULT_LOOP_RECIPE_SUCCESS_CRITERIA,
        stop_condition=DEFAULT_LOOP_RECIPE_STOP_CONDITION,
        context_provider="smart",
        model_profile="quality",
        verifier="default",
        created_at=timestamp,
        updated_at=timestamp,
        metadata={"built_in": True},
    )


@dataclass(frozen=True)
class GuardrailDecision:
    decision: LoopDecision = LoopDecision.CONTINUE
    reason: Optional[str] = None
    human_review: Optional[HumanReviewRequest] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.decision, LoopDecision):
            object.__setattr__(self, "decision", LoopDecision(self.decision))
        if self.decision.value not in GUARDRAIL_DECISION_VALUES:
            raise ValueError(f"{self.decision.value} is not a guardrail decision")
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))

    @property
    def can_continue(self) -> bool:
        return self.decision == LoopDecision.CONTINUE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "reason": self.reason,
            "human_review": (
                self.human_review.to_dict() if self.human_review else None
            ),
            "metadata": _metadata_dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GuardrailDecision":
        human_review = data.get("human_review")
        return cls(
            decision=LoopDecision(data.get("decision", LoopDecision.CONTINUE.value)),
            reason=data.get("reason"),
            human_review=(
                HumanReviewRequest.from_dict(human_review) if human_review else None
            ),
            metadata=_metadata_dict(data.get("metadata")),
        )


class LoopMiddleware(Protocol):
    def before_run(self, run: "LoopRun") -> Optional[GuardrailDecision]:
        ...

    def before_step(
        self, run: "LoopRun", step: "LoopStep"
    ) -> Optional[GuardrailDecision]:
        ...

    def after_step(
        self, run: "LoopRun", step: "LoopStep"
    ) -> Optional[GuardrailDecision]:
        ...

    def after_run(self, run: "LoopRun") -> Optional[GuardrailDecision]:
        ...

    def on_error(self, run: "LoopRun", error: Exception) -> Optional[GuardrailDecision]:
        ...


@dataclass(frozen=True)
class VerificationResult:
    outcome: VerificationOutcome
    reasons: Tuple[str, ...] = ()
    verifier: Optional[str] = None
    raw_response: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, VerificationOutcome):
            object.__setattr__(self, "outcome", VerificationOutcome(self.outcome))
        object.__setattr__(self, "reasons", _string_tuple(self.reasons))
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "reasons": list(self.reasons),
            "verifier": self.verifier,
            "raw_response": self.raw_response,
            "metadata": _metadata_dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VerificationResult":
        return cls(
            outcome=VerificationOutcome(data["outcome"]),
            reasons=_string_tuple(data.get("reasons")),
            verifier=data.get("verifier"),
            raw_response=data.get("raw_response"),
            metadata=_metadata_dict(data.get("metadata")),
        )


@dataclass(frozen=True)
class HumanReviewRequest:
    reason: str
    instructions: str
    requested_by_step_id: Optional[str] = None
    request_id: str = field(default_factory=lambda: _new_id("review"))
    created_at: datetime = field(default_factory=utc_now)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "reason": self.reason,
            "instructions": self.instructions,
            "requested_by_step_id": self.requested_by_step_id,
            "created_at": _datetime_to_json(self.created_at),
            "metadata": _metadata_dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HumanReviewRequest":
        created_at = _datetime_from_json(data.get("created_at"))
        return cls(
            request_id=str(data["request_id"]),
            reason=str(data["reason"]),
            instructions=str(data["instructions"]),
            requested_by_step_id=data.get("requested_by_step_id"),
            created_at=created_at or utc_now(),
            metadata=_metadata_dict(data.get("metadata")),
        )


@dataclass(frozen=True)
class LoopPolicy:
    max_retries: int = 1
    require_citations: bool = True
    require_verifier_for_supported: bool = True
    allow_mock_supported: bool = False
    allow_tool_calls: bool = False
    require_human_review_for_tools: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_bool(self.require_citations, "require_citations")
        _require_bool(
            self.require_verifier_for_supported,
            "require_verifier_for_supported",
        )
        _require_bool(self.allow_mock_supported, "allow_mock_supported")
        _require_bool(self.allow_tool_calls, "allow_tool_calls")
        _require_bool(
            self.require_human_review_for_tools,
            "require_human_review_for_tools",
        )
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_retries": self.max_retries,
            "require_citations": self.require_citations,
            "require_verifier_for_supported": self.require_verifier_for_supported,
            "allow_mock_supported": self.allow_mock_supported,
            "allow_tool_calls": self.allow_tool_calls,
            "require_human_review_for_tools": self.require_human_review_for_tools,
            "metadata": _metadata_dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "LoopPolicy":
        if not data:
            return cls()
        return cls(
            max_retries=int(data.get("max_retries", 1)),
            require_citations=_json_bool(data, "require_citations", True),
            require_verifier_for_supported=_json_bool(
                data, "require_verifier_for_supported", True
            ),
            allow_mock_supported=_json_bool(
                data, "allow_mock_supported", False
            ),
            allow_tool_calls=_json_bool(data, "allow_tool_calls", False),
            require_human_review_for_tools=_json_bool(
                data, "require_human_review_for_tools", True
            ),
            metadata=_metadata_dict(data.get("metadata")),
        )


@dataclass(frozen=True)
class LoopStep:
    phase: LoopPhase
    decision: LoopDecision = LoopDecision.CONTINUE
    name: Optional[str] = None
    step_id: str = field(default_factory=lambda: _new_id("step"))
    started_at: datetime = field(default_factory=utc_now)
    ended_at: Optional[datetime] = None
    input_summary: Optional[str] = None
    output_summary: Optional[str] = None
    backend: Optional[str] = None
    model_label: Optional[str] = None
    retry_count: int = 0
    error_message: Optional[str] = None
    verification: Optional[VerificationResult] = None
    human_review: Optional[HumanReviewRequest] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> Optional[int]:
        if self.ended_at is None:
            return None
        duration = self.ended_at - self.started_at
        return int(duration.total_seconds() * 1000)

    def complete(
        self,
        *,
        decision: Optional[LoopDecision] = None,
        output_summary: Optional[str] = None,
        error_message: Optional[str] = None,
        ended_at: Optional[datetime] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "LoopStep":
        merged_metadata = _metadata_dict(self.metadata)
        merged_metadata.update(metadata or {})
        return replace(
            self,
            decision=decision or self.decision,
            output_summary=output_summary
            if output_summary is not None
            else self.output_summary,
            error_message=error_message
            if error_message is not None
            else self.error_message,
            ended_at=ended_at or utc_now(),
            metadata=merged_metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "phase": self.phase.value,
            "decision": self.decision.value,
            "name": self.name,
            "started_at": _datetime_to_json(self.started_at),
            "ended_at": _datetime_to_json(self.ended_at),
            "duration_ms": self.duration_ms,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "backend": self.backend,
            "model_label": self.model_label,
            "retry_count": self.retry_count,
            "error_message": self.error_message,
            "verification": (
                self.verification.to_dict() if self.verification else None
            ),
            "human_review": (
                self.human_review.to_dict() if self.human_review else None
            ),
            "metadata": _metadata_dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LoopStep":
        verification = data.get("verification")
        human_review = data.get("human_review")
        return cls(
            step_id=str(data["step_id"]),
            phase=LoopPhase(data["phase"]),
            decision=LoopDecision(data.get("decision", LoopDecision.CONTINUE.value)),
            name=data.get("name"),
            started_at=_datetime_from_json(data.get("started_at")) or utc_now(),
            ended_at=_datetime_from_json(data.get("ended_at")),
            input_summary=data.get("input_summary"),
            output_summary=data.get("output_summary"),
            backend=data.get("backend"),
            model_label=data.get("model_label"),
            retry_count=int(data.get("retry_count", 0)),
            error_message=data.get("error_message"),
            verification=(
                VerificationResult.from_dict(verification) if verification else None
            ),
            human_review=(
                HumanReviewRequest.from_dict(human_review) if human_review else None
            ),
            metadata=_metadata_dict(data.get("metadata")),
        )


@dataclass(frozen=True)
class LoopRun:
    user_input: str
    context_provider: str
    backend: str
    model_label: str
    policy: LoopPolicy = field(default_factory=LoopPolicy)
    run_id: str = field(default_factory=lambda: _new_id("run"))
    session_id: Optional[str] = None
    started_at: datetime = field(default_factory=utc_now)
    completed_at: Optional[datetime] = None
    steps: Tuple[LoopStep, ...] = ()
    final_decision: Optional[LoopDecision] = None
    final_answer: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def with_step(self, step: LoopStep) -> "LoopRun":
        return replace(self, steps=(*self.steps, step))

    def complete(
        self,
        *,
        final_decision: LoopDecision,
        final_answer: Optional[str] = None,
        error_message: Optional[str] = None,
        completed_at: Optional[datetime] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "LoopRun":
        merged_metadata = _metadata_dict(self.metadata)
        merged_metadata.update(metadata or {})
        return replace(
            self,
            completed_at=completed_at or utc_now(),
            final_decision=final_decision,
            final_answer=final_answer,
            error_message=error_message,
            metadata=merged_metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "user_input": self.user_input,
            "context_provider": self.context_provider,
            "backend": self.backend,
            "model_label": self.model_label,
            "policy": self.policy.to_dict(),
            "started_at": _datetime_to_json(self.started_at),
            "completed_at": _datetime_to_json(self.completed_at),
            "steps": [step.to_dict() for step in self.steps],
            "final_decision": (
                self.final_decision.value if self.final_decision else None
            ),
            "final_answer": self.final_answer,
            "error_message": self.error_message,
            "metadata": _metadata_dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LoopRun":
        final_decision = data.get("final_decision")
        return cls(
            run_id=str(data["run_id"]),
            session_id=data.get("session_id"),
            user_input=str(data["user_input"]),
            context_provider=str(data["context_provider"]),
            backend=str(data["backend"]),
            model_label=str(data["model_label"]),
            policy=LoopPolicy.from_dict(data.get("policy")),
            started_at=_datetime_from_json(data.get("started_at")) or utc_now(),
            completed_at=_datetime_from_json(data.get("completed_at")),
            steps=tuple(LoopStep.from_dict(step) for step in data.get("steps", ())),
            final_decision=LoopDecision(final_decision) if final_decision else None,
            final_answer=data.get("final_answer"),
            error_message=data.get("error_message"),
            metadata=_metadata_dict(data.get("metadata")),
        )


@dataclass(frozen=True)
class LoopReport:
    run: LoopRun
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run": self.run.to_dict(),
        }

    def to_public_dict(self) -> Dict[str, Any]:
        report = self.to_dict()
        run = report["run"]
        steps = run.get("steps", [])
        final_decision = run.get("final_decision")
        if final_decision not in TERMINAL_PUBLIC_REDACTION_DECISION_VALUES:
            return report

        run["user_input"] = PUBLIC_REDACTION_TEXT
        run["final_answer"] = PUBLIC_REDACTION_TEXT
        run["error_message"] = PUBLIC_REDACTION_REASON
        run["metadata"] = {
            "redacted": True,
            "redaction_reason": PUBLIC_REDACTION_REASON,
        }
        run["steps"] = [_redact_public_step(step) for step in steps]
        report["public_redaction"] = {
            "applied": True,
            "reason": PUBLIC_REDACTION_REASON,
        }
        return report

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LoopReport":
        schema_version = str(data.get("schema_version", SCHEMA_VERSION))
        if schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported loop report schema: {schema_version}")
        return cls(
            schema_version=schema_version,
            run=LoopRun.from_dict(data["run"]),
        )


@dataclass(frozen=True)
class LoopSession:
    session_id: str = "default"
    reports: Tuple[LoopReport, ...] = ()
    schema_version: str = LOOP_SESSION_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        session_id = str(self.session_id or "default").strip() or "default"
        reports = tuple(self.reports or ())
        for report in reports:
            report_session_id = report.run.session_id
            if report_session_id and report_session_id != session_id:
                raise ValueError(
                    "LoopSession cannot contain reports from another session: "
                    f"{report_session_id!r}"
                )
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "reports", reports)
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))

    @property
    def report_count(self) -> int:
        return len(self.reports)

    def add_report(
        self,
        report: LoopReport,
        *,
        max_reports: Optional[int] = None,
    ) -> "LoopSession":
        if report.run.session_id and report.run.session_id != self.session_id:
            raise ValueError(
                "Cannot add loop report from another session: "
                f"{report.run.session_id!r}"
            )
        reports = (*self.reports, report)
        if max_reports is not None and max_reports >= 0:
            reports = reports[-max_reports:] if max_reports else ()
        return replace(self, reports=reports)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "report_count": self.report_count,
            "reports": [report.to_dict() for report in self.reports],
            "metadata": _metadata_dict(self.metadata),
        }

    def to_jsonl(self, *, public: bool = False) -> str:
        lines = []
        for report in self.reports:
            payload = report.to_public_dict() if public else report.to_dict()
            lines.append(json.dumps(payload, sort_keys=True))
        return "\n".join(lines) + ("\n" if lines else "")

    def write_jsonl(self, path: str | Path, *, public: bool = False) -> Path:
        artifact_path = Path(path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(self.to_jsonl(public=public), encoding="utf-8")
        return artifact_path

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LoopSession":
        schema_version = str(data.get("schema_version", LOOP_SESSION_SCHEMA_VERSION))
        if schema_version != LOOP_SESSION_SCHEMA_VERSION:
            raise ValueError(f"Unsupported loop session schema: {schema_version}")
        return cls(
            schema_version=schema_version,
            session_id=str(data.get("session_id") or "default"),
            reports=tuple(
                LoopReport.from_dict(report) for report in data.get("reports", ())
            ),
            metadata=_metadata_dict(data.get("metadata")),
        )

    @classmethod
    def from_jsonl(
        cls,
        content: str,
        *,
        session_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "LoopSession":
        reports = []
        for line in content.splitlines():
            if not line.strip():
                continue
            reports.append(LoopReport.from_dict(json.loads(line)))
        first_report_session_id = reports[0].run.session_id if reports else None
        inferred_session_id = session_id or first_report_session_id or "default"
        return cls(
            session_id=inferred_session_id,
            reports=tuple(reports),
            metadata=metadata or {},
        )


def _redact_public_step(step: Mapping[str, Any]) -> Dict[str, Any]:
    redacted = dict(step)
    if redacted.get("input_summary") is not None:
        redacted["input_summary"] = PUBLIC_REDACTION_TEXT
    if redacted.get("output_summary") is not None:
        redacted["output_summary"] = PUBLIC_REDACTION_TEXT
    if redacted.get("error_message") is not None:
        redacted["error_message"] = PUBLIC_REDACTION_REASON
    redacted["metadata"] = {
        "redacted": True,
        "redaction_reason": PUBLIC_REDACTION_REASON,
    }
    if redacted.get("human_review") is not None:
        redacted["human_review"] = None
    verification = redacted.get("verification")
    if verification:
        redacted_verification = dict(verification)
        redacted_verification["verifier"] = None
        redacted_verification["reasons"] = [PUBLIC_REDACTION_TEXT]
        redacted_verification["raw_response"] = None
        redacted_verification["metadata"] = {
            "redacted": True,
            "redaction_reason": PUBLIC_REDACTION_REASON,
        }
        redacted["verification"] = redacted_verification
    return redacted
