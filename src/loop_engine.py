from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple
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
ANSWER_CANDIDATE_SHA256_METADATA_KEY = "answer_candidate_sha256"
EVIDENCE_SET_SHA256_METADATA_KEY = "evidence_set_sha256"


def answer_candidate_sha256(answer: str) -> str:
    """Return a deterministic correlation digest for exact answer bytes.

    The digest binds raw loop steps to the same answer candidate during public
    projection. It is not an authentication, confidentiality, or signature
    mechanism.
    """

    if type(answer) is not str:
        raise TypeError("answer must be a string")
    return hashlib.sha256(answer.encode("utf-8")).hexdigest()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _datetime_to_json(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    _require_aware_datetime(value, "datetime")
    normalized = value.astimezone(timezone.utc)
    return normalized.isoformat().replace("+00:00", "Z")


def _datetime_from_json(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("datetime must be a string")
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    _require_aware_datetime(parsed, "datetime")
    return parsed


def _require_aware_datetime(value: Any, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{field_name} must be a datetime")
    try:
        offset = value.utcoffset()
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be timezone-aware") from exc
    if offset is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    try:
        value.astimezone(timezone.utc)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must normalize to UTC") from exc
    return value


def _require_timestamp_order(
    started_at: datetime,
    ended_at: Optional[datetime],
    *,
    started_field_name: str = "started_at",
    ended_field_name: str,
) -> None:
    if ended_at is not None and ended_at < started_at:
        raise ValueError(
            f"{ended_field_name} must not be before {started_field_name}"
        )


def _reject_nonfinite_json_constant(value: str):
    raise ValueError(f"JSON must not contain non-finite number {value}")


def _strict_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("JSON number is outside the finite float range")
    return parsed


def _strict_json_object(pairs) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"JSON object contains duplicate key {key!r}")
        payload[key] = value
    return payload


def _metadata_dict(metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a JSON object")
    return dict(metadata)


def _strict_metadata_dict(metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    payload = _metadata_dict(metadata)
    try:
        detached = json.loads(
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            parse_constant=_reject_nonfinite_json_constant,
            parse_float=_strict_json_float,
            object_pairs_hook=_strict_json_object,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must contain canonical JSON values") from exc
    if detached != payload:
        raise ValueError("metadata must contain canonical JSON values")
    return detached


def _string_tuple(values) -> Tuple[str, ...]:
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise ValueError("string collection must be a JSON array")
    if not all(isinstance(value, str) for value in values):
        raise ValueError("string collection entries must be strings")
    return tuple(values)


def _require_string(
    value: Any,
    field_name: str,
    *,
    allow_empty: bool = True,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


def _optional_string(value: Any, field_name: str) -> Optional[str]:
    if value is None:
        return None
    return _require_string(value, field_name)


def _require_identity(value: Any, field_name: str) -> str:
    identity = _require_string(value, field_name, allow_empty=False)
    if identity != identity.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace")
    return identity


def _json_array(value: Any, field_name: str) -> tuple:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a JSON array")
    return tuple(value)


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


class LoopTerminalReason(str, Enum):
    COMPLETED = "completed"
    NOT_VERIFIED = "not_verified"
    VERIFICATION_FAILED = "verification_failed"
    RETRY_BUDGET_EXHAUSTED = "retry_budget_exhausted"
    TRACE_UNAVAILABLE = "trace_unavailable"
    POLICY_REFUSED = "policy_refused"
    BLOCKED = "blocked"
    HUMAN_REVIEW_REQUIRED = "human_review_required"
    ERROR = "error"
    UNSPECIFIED = "unspecified"


@dataclass(frozen=True)
class EvidenceLocator:
    """Non-secret coordinates for locating evidence within its provider."""

    page: Optional[int] = None
    chunk_index: Optional[int] = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("page", self.page),
            ("chunk_index", self.chunk_index),
        ):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{field_name} must be a non-negative integer or null")

    def to_dict(self) -> Dict[str, Optional[int]]:
        return {
            "page": self.page,
            "chunk_index": self.chunk_index,
        }

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "EvidenceLocator":
        if data is not None and not isinstance(data, Mapping):
            raise ValueError("evidence locator must be a JSON object")
        locator = data or {}
        return cls(
            page=locator.get("page"),
            chunk_index=locator.get("chunk_index"),
        )


@dataclass(frozen=True)
class EvidenceReference:
    """Stable evidence identity without retaining source labels or excerpts."""

    evidence_id: str
    citation_id: int
    provider: str
    locator: EvidenceLocator = field(default_factory=EvidenceLocator)

    def __post_init__(self) -> None:
        evidence_id = _require_identity(self.evidence_id, "evidence_id").lower()
        provider = _require_string(
            self.provider,
            "provider",
            allow_empty=False,
        ).strip().casefold()[:64]
        if type(self.citation_id) is not int or self.citation_id < 1:
            raise ValueError("citation_id must be a positive integer")
        digest = evidence_id.removeprefix("evidence_")
        if (
            not evidence_id.startswith("evidence_")
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("evidence_id must be an evidence_ prefixed SHA-256 digest")
        if not provider:
            raise ValueError("provider must not be empty")
        if not isinstance(self.locator, EvidenceLocator):
            object.__setattr__(
                self,
                "locator",
                EvidenceLocator.from_dict(self.locator),
            )
        object.__setattr__(self, "evidence_id", evidence_id)
        object.__setattr__(self, "provider", provider)

    @classmethod
    def from_source(
        cls,
        *,
        citation_id: int,
        provider: str,
        source_identity: str,
        page: Optional[int],
        chunk_index: Optional[int],
        excerpt: str,
    ) -> "EvidenceReference":
        """Derive reference/content identity without retaining the raw inputs."""

        normalized_provider = _clean_text(provider, max_chars=64).casefold()
        identity_payload = json.dumps(
            {
                "provider": normalized_provider,
                "source_identity": str(source_identity or ""),
                "page": page,
                "chunk_index": chunk_index,
                "excerpt": str(excerpt or ""),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return cls(
            evidence_id=f"evidence_{hashlib.sha256(identity_payload).hexdigest()}",
            citation_id=citation_id,
            provider=normalized_provider,
            locator=EvidenceLocator(page=page, chunk_index=chunk_index),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "citation_id": self.citation_id,
            "provider": self.provider,
            "locator": self.locator.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceReference":
        if not isinstance(data, Mapping):
            raise ValueError("evidence reference must be a JSON object")
        return cls(
            evidence_id=_require_identity(data["evidence_id"], "evidence_id"),
            citation_id=data["citation_id"],
            provider=_require_string(
                data["provider"],
                "provider",
                allow_empty=False,
            ),
            locator=EvidenceLocator.from_dict(data.get("locator")),
        )


def evidence_set_sha256(evidence: Sequence[EvidenceReference]) -> str:
    """Return a deterministic correlation digest for an exact evidence set.

    The digest lets projection verify that draft, verifier, and final steps refer
    to the same normalized reference fields. It is not semantic comparison,
    authentication, confidentiality, or a signature.
    """

    if not isinstance(evidence, (list, tuple)):
        raise TypeError("evidence must be a sequence")
    canonical = []
    for reference in evidence:
        if not isinstance(reference, EvidenceReference):
            raise TypeError("evidence must contain EvidenceReference records")
        canonical.append(
            {
                "evidence_id": reference.evidence_id,
                "citation_id": reference.citation_id,
                "provider": reference.provider,
                "locator": {
                    "page": reference.locator.page,
                    "chunk_index": reference.locator.chunk_index,
                },
            }
        )
    canonical.sort(
        key=lambda item: (
            item["citation_id"],
            item["evidence_id"],
            item["provider"],
            -1 if item["locator"]["page"] is None else item["locator"]["page"],
            -1
            if item["locator"]["chunk_index"] is None
            else item["locator"]["chunk_index"],
        )
    )
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
        _require_aware_datetime(self.created_at, "created_at")
        _require_aware_datetime(self.updated_at, "updated_at")
        _require_timestamp_order(
            self.created_at,
            self.updated_at,
            started_field_name="created_at",
            ended_field_name="updated_at",
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
        _optional_string(self.reason, "reason")
        if (
            self.human_review is not None
            and type(self.human_review) is not HumanReviewRequest
        ):
            raise ValueError("human_review must be a HumanReviewRequest or None")
        if (
            self.decision == LoopDecision.REQUIRES_REVIEW
            and self.human_review is None
        ):
            raise ValueError(
                "requires_review guardrail decision requires a human_review request"
            )
        if (
            self.decision != LoopDecision.REQUIRES_REVIEW
            and self.human_review is not None
        ):
            raise ValueError(
                "human_review is only valid for a requires_review guardrail decision"
            )
        object.__setattr__(self, "metadata", _strict_metadata_dict(self.metadata))

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
            "metadata": _strict_metadata_dict(self.metadata),
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
            metadata=_strict_metadata_dict(data.get("metadata")),
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
    verifier_backend: Optional[str] = None
    verifier_model_label: Optional[str] = None
    same_model_as_drafter: Optional[bool] = None
    raw_response: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, VerificationOutcome):
            object.__setattr__(self, "outcome", VerificationOutcome(self.outcome))
        if self.same_model_as_drafter is not None:
            _require_bool(self.same_model_as_drafter, "same_model_as_drafter")
        for field_name in (
            "verifier",
            "verifier_backend",
            "verifier_model_label",
            "raw_response",
        ):
            _optional_string(getattr(self, field_name), field_name)
        object.__setattr__(self, "reasons", _string_tuple(self.reasons))
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "reasons": list(self.reasons),
            "verifier": self.verifier,
            "verifier_backend": self.verifier_backend,
            "verifier_model_label": self.verifier_model_label,
            "same_model_as_drafter": self.same_model_as_drafter,
            "raw_response": self.raw_response,
            "metadata": _metadata_dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VerificationResult":
        if not isinstance(data, Mapping):
            raise ValueError("verification must be a JSON object")
        outcome = _require_string(data["outcome"], "verification.outcome")
        return cls(
            outcome=VerificationOutcome(outcome),
            reasons=_string_tuple(data.get("reasons")),
            verifier=_optional_string(data.get("verifier"), "verifier"),
            verifier_backend=_optional_string(
                data.get("verifier_backend"), "verifier_backend"
            ),
            verifier_model_label=_optional_string(
                data.get("verifier_model_label"), "verifier_model_label"
            ),
            same_model_as_drafter=data.get("same_model_as_drafter"),
            raw_response=_optional_string(data.get("raw_response"), "raw_response"),
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

    def __post_init__(self) -> None:
        for field_name in ("request_id", "reason", "instructions"):
            _require_string(
                getattr(self, field_name),
                field_name,
                allow_empty=False,
            )
        _optional_string(self.requested_by_step_id, "requested_by_step_id")
        _require_aware_datetime(self.created_at, "created_at")
        object.__setattr__(self, "metadata", _strict_metadata_dict(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "reason": self.reason,
            "instructions": self.instructions,
            "requested_by_step_id": self.requested_by_step_id,
            "created_at": _datetime_to_json(self.created_at),
            "metadata": _strict_metadata_dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HumanReviewRequest":
        if not isinstance(data, Mapping):
            raise ValueError("human review must be a JSON object")
        if data.get("created_at") is None:
            raise ValueError("human review created_at must be a string")
        created_at = _datetime_from_json(data.get("created_at"))
        return cls(
            request_id=_require_string(
                data["request_id"], "request_id", allow_empty=False
            ),
            reason=_require_string(data["reason"], "reason", allow_empty=False),
            instructions=_require_string(
                data["instructions"], "instructions", allow_empty=False
            ),
            requested_by_step_id=_optional_string(
                data.get("requested_by_step_id"), "requested_by_step_id"
            ),
            created_at=created_at,
            metadata=_strict_metadata_dict(data.get("metadata")),
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
        if type(self.max_retries) is not int:
            raise ValueError("max_retries must be an integer")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
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
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError("policy must be a JSON object")
        if not data:
            return cls()
        return cls(
            max_retries=data.get("max_retries", 1),
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

    def __post_init__(self) -> None:
        if not isinstance(self.phase, LoopPhase):
            object.__setattr__(self, "phase", LoopPhase(self.phase))
        if not isinstance(self.decision, LoopDecision):
            object.__setattr__(self, "decision", LoopDecision(self.decision))
        _require_identity(self.step_id, "step_id")
        for field_name in (
            "name",
            "input_summary",
            "output_summary",
            "backend",
            "model_label",
            "error_message",
        ):
            _optional_string(getattr(self, field_name), field_name)
        if type(self.retry_count) is not int or self.retry_count < 0:
            raise ValueError("retry_count must be a non-negative integer")
        _require_aware_datetime(self.started_at, "started_at")
        if self.ended_at is not None:
            _require_aware_datetime(self.ended_at, "ended_at")
        _require_timestamp_order(
            self.started_at,
            self.ended_at,
            ended_field_name="ended_at",
        )
        if self.verification is not None and not isinstance(
            self.verification, VerificationResult
        ):
            raise ValueError("verification must be a VerificationResult or null")
        if self.human_review is not None and not isinstance(
            self.human_review, HumanReviewRequest
        ):
            raise ValueError("human_review must be a HumanReviewRequest or null")
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))

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
        if not isinstance(data, Mapping):
            raise ValueError("loop step must be a JSON object")
        verification = data.get("verification")
        if verification is not None and not isinstance(verification, Mapping):
            raise ValueError("verification must be a JSON object or null")
        human_review = data.get("human_review")
        if human_review is not None and not isinstance(human_review, Mapping):
            raise ValueError("human_review must be a JSON object or null")
        phase = _require_string(data["phase"], "step.phase")
        decision = _require_string(
            data.get("decision", LoopDecision.CONTINUE.value),
            "step.decision",
        )
        if data.get("started_at") is None:
            raise ValueError("step.started_at must be a string")
        return cls(
            step_id=_require_identity(data["step_id"], "step_id"),
            phase=LoopPhase(phase),
            decision=LoopDecision(decision),
            name=_optional_string(data.get("name"), "step.name"),
            started_at=_datetime_from_json(data.get("started_at")),
            ended_at=_datetime_from_json(data.get("ended_at")),
            input_summary=_optional_string(
                data.get("input_summary"), "step.input_summary"
            ),
            output_summary=_optional_string(
                data.get("output_summary"), "step.output_summary"
            ),
            backend=_optional_string(data.get("backend"), "step.backend"),
            model_label=_optional_string(
                data.get("model_label"), "step.model_label"
            ),
            retry_count=data.get("retry_count", 0),
            error_message=_optional_string(
                data.get("error_message"), "step.error_message"
            ),
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
    evidence: Tuple[EvidenceReference, ...] = ()
    final_decision: Optional[LoopDecision] = None
    terminal_reason: Optional[LoopTerminalReason] = None
    final_answer: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identity(self.run_id, "run_id")
        _require_string(self.user_input, "user_input")
        _require_string(self.context_provider, "context_provider", allow_empty=False)
        _require_string(self.backend, "backend", allow_empty=False)
        _require_string(self.model_label, "model_label", allow_empty=False)
        if self.session_id is not None:
            _require_identity(self.session_id, "session_id")
        _optional_string(self.final_answer, "final_answer")
        _optional_string(self.error_message, "error_message")
        if not isinstance(self.policy, LoopPolicy):
            raise ValueError("policy must be a LoopPolicy")
        _require_aware_datetime(self.started_at, "started_at")
        if self.completed_at is not None:
            _require_aware_datetime(self.completed_at, "completed_at")
        _require_timestamp_order(
            self.started_at,
            self.completed_at,
            ended_field_name="completed_at",
        )
        steps = tuple(self.steps or ())
        if not all(isinstance(step, LoopStep) for step in steps):
            raise ValueError("steps must contain LoopStep records")
        step_ids = [step.step_id for step in steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("steps must not contain duplicate step_id values")
        object.__setattr__(self, "steps", steps)
        if self.final_decision is not None and not isinstance(
            self.final_decision, LoopDecision
        ):
            object.__setattr__(
                self,
                "final_decision",
                LoopDecision(self.final_decision),
            )
        if self.terminal_reason is not None and not isinstance(
            self.terminal_reason, LoopTerminalReason
        ):
            object.__setattr__(
                self,
                "terminal_reason",
                LoopTerminalReason(self.terminal_reason),
            )
        if self.final_decision is not None and self.terminal_reason is None:
            object.__setattr__(
                self,
                "terminal_reason",
                LoopTerminalReason.UNSPECIFIED,
            )
        evidence = tuple(self.evidence or ())
        if not all(isinstance(item, EvidenceReference) for item in evidence):
            raise ValueError("evidence must contain EvidenceReference records")
        citation_ids = [item.citation_id for item in evidence]
        if len(citation_ids) != len(set(citation_ids)):
            raise ValueError("evidence must not contain duplicate citation_id values")
        evidence_ids = [item.evidence_id for item in evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence must not contain duplicate evidence_id values")
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))

    def with_step(self, step: LoopStep) -> "LoopRun":
        return replace(self, steps=(*self.steps, step))

    def with_evidence(self, evidence: Tuple[EvidenceReference, ...]) -> "LoopRun":
        return replace(self, evidence=tuple(evidence or ()))

    def complete(
        self,
        *,
        final_decision: LoopDecision,
        terminal_reason: LoopTerminalReason = LoopTerminalReason.UNSPECIFIED,
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
            terminal_reason=terminal_reason,
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
            "evidence": [item.to_dict() for item in self.evidence],
            "final_decision": (
                self.final_decision.value if self.final_decision else None
            ),
            "terminal_reason": (
                self.terminal_reason.value if self.terminal_reason else None
            ),
            "final_answer": self.final_answer,
            "error_message": self.error_message,
            "metadata": _metadata_dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LoopRun":
        if not isinstance(data, Mapping):
            raise ValueError("run must be a JSON object")
        final_decision = data.get("final_decision")
        if final_decision is not None:
            final_decision = _require_string(final_decision, "final_decision")
        terminal_reason = data.get("terminal_reason")
        if terminal_reason is not None:
            terminal_reason = _require_string(terminal_reason, "terminal_reason")
        policy = data.get("policy")
        if policy is not None and not isinstance(policy, Mapping):
            raise ValueError("policy must be a JSON object")
        steps = _json_array(data.get("steps"), "steps")
        evidence = _json_array(data.get("evidence"), "evidence")
        if data.get("started_at") is None:
            raise ValueError("started_at must be a string")
        return cls(
            run_id=_require_identity(data["run_id"], "run_id"),
            session_id=(
                _require_identity(data["session_id"], "session_id")
                if data.get("session_id") is not None
                else None
            ),
            user_input=_require_string(data["user_input"], "user_input"),
            context_provider=_require_string(
                data["context_provider"], "context_provider", allow_empty=False
            ),
            backend=_require_string(data["backend"], "backend", allow_empty=False),
            model_label=_require_string(
                data["model_label"], "model_label", allow_empty=False
            ),
            policy=LoopPolicy.from_dict(policy),
            started_at=_datetime_from_json(data.get("started_at")),
            completed_at=_datetime_from_json(data.get("completed_at")),
            steps=tuple(LoopStep.from_dict(step) for step in steps),
            evidence=tuple(
                EvidenceReference.from_dict(item)
                for item in evidence
            ),
            final_decision=LoopDecision(final_decision) if final_decision else None,
            terminal_reason=(
                LoopTerminalReason(terminal_reason) if terminal_reason else None
            ),
            final_answer=_optional_string(data.get("final_answer"), "final_answer"),
            error_message=_optional_string(data.get("error_message"), "error_message"),
            metadata=_metadata_dict(data.get("metadata")),
        )


@dataclass(frozen=True)
class LoopReport:
    run: LoopRun
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.run, LoopRun):
            raise ValueError("run must be a LoopRun")
        schema_version = _require_string(
            self.schema_version,
            "schema_version",
            allow_empty=False,
        )
        if schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported loop report schema: {schema_version}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run": self.run.to_dict(),
        }

    def to_public_dict(self) -> Dict[str, Any]:
        try:
            from .public_projection import project_public_report
        except ImportError:
            from public_projection import project_public_report

        return project_public_report(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LoopReport":
        if not isinstance(data, Mapping):
            raise ValueError("loop report must be a JSON object")
        schema_version = _require_string(
            data.get("schema_version", SCHEMA_VERSION),
            "schema_version",
            allow_empty=False,
        )
        if schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported loop report schema: {schema_version}")
        run = data.get("run")
        if not isinstance(run, Mapping):
            raise ValueError("run must be a JSON object")
        return cls(
            schema_version=schema_version,
            run=LoopRun.from_dict(run),
        )


@dataclass(frozen=True)
class LoopSession:
    session_id: str = "default"
    reports: Tuple[LoopReport, ...] = ()
    schema_version: str = LOOP_SESSION_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        schema_version = _require_string(
            self.schema_version,
            "schema_version",
            allow_empty=False,
        )
        if schema_version != LOOP_SESSION_SCHEMA_VERSION:
            raise ValueError(f"Unsupported loop session schema: {schema_version}")
        session_id = _require_string(
            self.session_id,
            "session_id",
            allow_empty=False,
        )
        if session_id != session_id.strip():
            raise ValueError("session_id must not contain surrounding whitespace")
        reports = tuple(self.reports or ())
        if not all(isinstance(report, LoopReport) for report in reports):
            raise ValueError("reports must contain LoopReport records")
        run_ids = [report.run.run_id for report in reports]
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("LoopSession cannot contain duplicate run ids")
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
        if any(existing.run.run_id == report.run.run_id for existing in self.reports):
            raise ValueError(
                f"LoopSession already contains run id {report.run.run_id!r}"
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
            lines.append(json.dumps(payload, sort_keys=True, allow_nan=False))
        return "\n".join(lines) + ("\n" if lines else "")

    def write_jsonl(self, path: str | Path, *, public: bool = False) -> Path:
        artifact_path = Path(path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(self.to_jsonl(public=public), encoding="utf-8")
        return artifact_path

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LoopSession":
        if not isinstance(data, Mapping):
            raise ValueError("loop session must be a JSON object")
        schema_version = _require_string(
            data.get("schema_version", LOOP_SESSION_SCHEMA_VERSION),
            "schema_version",
            allow_empty=False,
        )
        if schema_version != LOOP_SESSION_SCHEMA_VERSION:
            raise ValueError(f"Unsupported loop session schema: {schema_version}")
        reports = _json_array(data.get("reports"), "reports")
        return cls(
            schema_version=schema_version,
            session_id=_require_string(
                data.get("session_id", "default"),
                "session_id",
                allow_empty=False,
            ),
            reports=tuple(
                LoopReport.from_dict(report) for report in reports
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
            payload = json.loads(
                line,
                parse_constant=_reject_nonfinite_json_constant,
                parse_float=_strict_json_float,
                object_pairs_hook=_strict_json_object,
            )
            if isinstance(payload, Mapping) and (
                "projection_schema_version" in payload
                or payload.get("public") is True
            ):
                raise ValueError(
                    "Public projection JSONL is non-rehydratable; "
                    "raw loop-report/v1 JSONL is required."
                )
            reports.append(LoopReport.from_dict(payload))
        first_report_session_id = reports[0].run.session_id if reports else None
        inferred_session_id = session_id or first_report_session_id or "default"
        return cls(
            session_id=inferred_session_id,
            reports=tuple(reports),
            metadata=metadata or {},
        )
