import json
import math
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    from .loop_engine import LoopDecision, VerificationOutcome, VerificationResult
    from .retrieval_types import AnswerCitation
except ImportError:
    from loop_engine import LoopDecision, VerificationOutcome, VerificationResult
    from retrieval_types import AnswerCitation


SELF_CHECK_REFUSAL_ANSWER = (
    "I could not find enough relevant information in the provided evidence to answer that."
)
FORMAT_CHECK_FAILURE_ANSWER = (
    "The model answer failed formatting checks after retry. Please try again."
)
SELF_CHECK_PASS_OUTCOMES = {"supported", "not_verified"}
VERIFIER_OUTCOMES = {"supported", "unsupported", "insufficient"}
WEB_VERIFIER_RETRY_REASONS = {
    "llm_verifier_unsupported",
    "llm_verifier_insufficient",
}

ANSWER_SUPPORT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "based",
    "be",
    "because",
    "by",
    "can",
    "configured",
    "context",
    "currently",
    "demonstration",
    "did",
    "do",
    "does",
    "document",
    "for",
    "from",
    "has",
    "have",
    "how",
    "i",
    "identify",
    "in",
    "information",
    "is",
    "it",
    "language",
    "model",
    "no",
    "of",
    "on",
    "or",
    "provided",
    "real",
    "related",
    "response",
    "that",
    "the",
    "this",
    "to",
    "using",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


@dataclass(frozen=True)
class AnswerSelfCheck:
    outcome: str
    reasons: List[str]
    retry_attempted: bool = False


@dataclass(frozen=True)
class AnswerFormatCheck:
    outcome: str
    reasons: List[str]
    retry_attempted: bool = False


def answer_is_refusal(answer: str) -> bool:
    lowered = answer.lower()
    refusal_markers = (
        "could not find",
        "not in the context",
        "not enough relevant information",
        "not provided in the context",
        "cannot answer",
        "can't answer",
        "insufficient information",
    )
    return any(marker in lowered for marker in refusal_markers)


INLINE_CITATION_MARKER_PATTERN = re.compile(
    r"(?<!\S)\[(\d+)\](?=$|[\s.,;:!?)])"
)


def inline_citation_ids(answer: str) -> List[int]:
    citation_ids = []
    for match in INLINE_CITATION_MARKER_PATTERN.finditer(answer):
        try:
            citation_ids.append(int(match.group(1)))
        except ValueError:
            continue
    return citation_ids


def format_check_answer(
    answer: str, *, retry_attempted: bool = False
) -> AnswerFormatCheck:
    clean_answer = answer.strip()
    check_text = re.sub(r"```.*?```", " ", clean_answer, flags=re.DOTALL)
    check_text = re.sub(r"`[^`\n]+`", " ", check_text)
    reasons: List[str] = []

    if re.search(
        r"\b(?:not_verified|mechanical_checks_passed|needs_retry|needs_refusal|"
        r"llm_verifier_[a-z_]+|self_check_[a-z_]+)\b",
        check_text,
    ):
        reasons.append("internal_verification_label")

    compact_labeled_list_pattern = re.compile(
        r"(?:^|\s)1[.)]\s+(?:\*\*)?[^:\n]{1,80}:(?:\*\*)?"
        r".+?\s+2[.)]\s+(?:\*\*)?[^:\n]{1,80}:(?:\*\*)?",
        flags=re.DOTALL,
    )
    compact_numbered_list_pattern = re.compile(
        r"(?:^|:)\s*1[.)]\s+\S[^\n]{2,}?\s+2[.)]\s+\S",
        flags=re.DOTALL,
    )
    for line in check_text.splitlines():
        if compact_labeled_list_pattern.search(
            line
        ) or compact_numbered_list_pattern.search(line):
            reasons.append("compact_ordered_list")
            break

    if reasons:
        return AnswerFormatCheck(
            outcome="needs_retry",
            reasons=reasons,
            retry_attempted=retry_attempted,
        )

    return AnswerFormatCheck(
        outcome="format_passed",
        reasons=["clean_web_markdown"],
        retry_attempted=retry_attempted,
    )


def sanitize_internal_verification_labels(answer: str) -> str:
    text = str(answer or "").strip()
    protected_blocks: List[str] = []

    def protect(match: re.Match[str]) -> str:
        protected_blocks.append(match.group(0))
        return f"@@AI_LOOP_PROTECTED_{len(protected_blocks) - 1}@@"

    text = re.sub(r"```.*?```", protect, text, flags=re.DOTALL)
    text = re.sub(r"`[^`\n]+`", protect, text)
    text = re.sub(
        r"^\s*(?:[-*]\s*)?(?:\*\*)?\s*Answer\s*:\s*(?:\*\*)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    text = re.sub(
        r"\s*(?:\*\*)?\s*Note\s*:\s*(?:\*\*)?\s*"
        r"(?:This|The)\s+(?:answer|response)\s+[^.?!]*\b"
        r"(?:not_verified|mechanical_checks_passed|needs_retry|needs_refusal|"
        r"llm_verifier_[a-z_]+|self_check_[a-z_]+)\b[^.?!]*(?:[.?!]|$)",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s*(?:This|The)\s+(?:answer|response)\s+[^.?!]*\b"
        r"(?:not_verified|mechanical_checks_passed|needs_retry|needs_refusal|"
        r"llm_verifier_[a-z_]+|self_check_[a-z_]+)\b[^.?!]*(?:[.?!]|$)",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    replacements = {
        "not_verified": "not verified",
        "mechanical_checks_passed": "mechanical checks passed",
        "needs_retry": "needs retry",
        "needs_refusal": "needs refusal",
    }

    def replace_label(match: re.Match[str]) -> str:
        label = match.group(0)
        return replacements.get(label.lower(), label.replace("_", " "))

    text = re.sub(
        r"\b(?:not_verified|mechanical_checks_passed|needs_retry|needs_refusal|"
        r"llm_verifier_[a-z_]+|self_check_[a-z_]+)\b",
        replace_label,
        text,
        flags=re.IGNORECASE,
    )
    for index, value in enumerate(protected_blocks):
        text = text.replace(f"@@AI_LOOP_PROTECTED_{index}@@", value)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def loop_decision_for_format_check(
    format_check: AnswerFormatCheck,
) -> LoopDecision:
    if format_check.outcome == "format_passed":
        return LoopDecision.CONTINUE
    if format_check.outcome == "needs_retry":
        return LoopDecision.RETRY
    return LoopDecision.ERROR


def citation_ids_are_valid(
    inline_ids: List[int], citations: List[AnswerCitation]
) -> bool:
    valid_citation_ids = {citation.citation_id for citation in citations}
    return set(inline_ids).issubset(valid_citation_ids)


def cited_citations_for_answer(
    answer: str, citations: List[AnswerCitation]
) -> List[AnswerCitation]:
    citation_ids = set(inline_citation_ids(answer))
    return [citation for citation in citations if citation.citation_id in citation_ids]


def support_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def support_tokens(text: str) -> set:
    normalized_text = re.sub(r"\[\d+\]", " ", text.lower())
    tokens = re.findall(r"\w+", normalized_text, flags=re.UNICODE)
    return {
        support_token(token)
        for token in tokens
        if (not token.isascii() or len(token) > 2)
        and token not in ANSWER_SUPPORT_STOPWORDS
    }


def normalize_support_text(text: str) -> str:
    without_citations = re.sub(r"\[\d+\]", " ", text.lower())
    normalized_chars = []
    for char in without_citations:
        if char.isalnum():
            normalized_chars.append(char)
        else:
            normalized_chars.append(" ")
    return " ".join("".join(normalized_chars).split())


def matched_claim_is_denied(
    normalized_answer: str,
    normalized_evidence: str,
    raw_evidence: str = "",
) -> bool:
    if not normalized_answer:
        return False

    match_starts = []
    search_start = 0
    while True:
        start = normalized_evidence.find(normalized_answer, search_start)
        if start < 0:
            break
        match_starts.append(start)
        search_start = start + len(normalized_answer)

    if not match_starts:
        return False

    if raw_evidence:
        claim_pattern = r"\b" + r"\W+".join(
            re.escape(token) for token in normalized_answer.split()
        )
        qa_denial_pattern = (
            rf"{claim_pattern}\b\s*(?:\?|\:|[-–—])\s*"
            r"(?:no|nope)\b(?=\s*(?:[.!?,;:]|$))"
        )
        if re.search(qa_denial_pattern, raw_evidence, flags=re.IGNORECASE):
            return True

    prefix_denial_markers = (
        "it is false that",
        "it is not true that",
        "it is incorrect that",
        "it is wrong that",
        "it was false that",
        "it was not true that",
        "it was incorrect that",
        "it was wrong that",
        "false that",
        "not true that",
        "incorrect that",
        "wrong that",
        "denied that",
        "refuted that",
    )

    denial_markers = (
        "is false",
        "is not true",
        "is incorrect",
        "is wrong",
        "is denied",
        "is refuted",
        "is rejected",
        "is debunked",
        "is contradicted",
        "is untrue",
        "is unsupported",
        "is not supported",
        "is disputed",
        "is inaccurate",
        "is baseless",
        "is unfounded",
        "is disproven",
        "is disproved",
        "has been denied",
        "has been refuted",
        "has been rejected",
        "has been debunked",
        "has been contradicted",
        "has been unsupported",
        "has been disputed",
        "has been disproven",
        "has been disproved",
        "have been denied",
        "have been refuted",
        "have been rejected",
        "have been debunked",
        "have been contradicted",
        "have been unsupported",
        "have been disputed",
        "have been disproven",
        "have been disproved",
        "had been denied",
        "had been refuted",
        "had been rejected",
        "had been debunked",
        "had been contradicted",
        "had been unsupported",
        "had been disputed",
        "had been disproven",
        "had been disproved",
        "was false",
        "was not true",
        "was incorrect",
        "was wrong",
        "was denied",
        "was refuted",
        "was rejected",
        "was debunked",
        "was contradicted",
        "was untrue",
        "was unsupported",
        "was not supported",
        "was disputed",
        "was inaccurate",
        "was baseless",
        "was unfounded",
        "was disproven",
        "was disproved",
        "are false",
        "are not true",
        "are incorrect",
        "are wrong",
        "are denied",
        "are refuted",
        "are rejected",
        "are debunked",
        "are contradicted",
        "are untrue",
        "are unsupported",
        "are not supported",
        "are disputed",
        "are inaccurate",
        "are baseless",
        "are unfounded",
        "are disproven",
        "are disproved",
        "were false",
        "were not true",
        "were incorrect",
        "were wrong",
        "were denied",
        "were refuted",
        "were rejected",
        "were debunked",
        "were contradicted",
        "were untrue",
        "were unsupported",
        "were not supported",
        "were disputed",
        "were inaccurate",
        "were baseless",
        "were unfounded",
        "were disproven",
        "were disproved",
        "not true",
        "not supported",
        "false",
        "incorrect",
        "wrong",
        "denied",
        "refuted",
        "rejected",
        "debunked",
        "contradicted",
        "untrue",
        "unsupported",
        "disputed",
        "inaccurate",
        "baseless",
        "unfounded",
        "disproven",
        "disproved",
    )

    def has_denial_marker(tokens: List[str]) -> bool:
        text = " ".join(tokens[:8])
        return any(
            text == marker or text.startswith(f"{marker} ")
            for marker in denial_markers
        )

    referent_tokens = {"it", "that", "this", "they", "these", "those"}
    referential_determiners = {"a", "an", "the"}
    referential_nouns = {
        "answer",
        "assertion",
        "assertions",
        "claim",
        "claims",
        "idea",
        "ideas",
        "premise",
        "premises",
        "report",
        "reports",
        "statement",
        "statements",
    }
    referential_modifiers = {"above", "previous", "prior", "same"}

    def has_referential_denial_marker(tokens: List[str]) -> bool:
        if has_denial_marker(tokens):
            return True

        if tokens and tokens[0] in referent_tokens:
            if has_denial_marker(tokens[1:]):
                return True
            if len(tokens) > 1 and tokens[1] in referential_nouns:
                return has_denial_marker(tokens[2:])

        if tokens and tokens[0] in referential_determiners:
            noun_tokens = tokens[1:]
            if noun_tokens and noun_tokens[0] in referential_modifiers:
                noun_tokens = noun_tokens[1:]
            if noun_tokens and noun_tokens[0] in referential_nouns:
                return has_denial_marker(noun_tokens[1:])

        return False

    discourse_connectors = {
        "although",
        "but",
        "however",
        "that",
        "this",
        "though",
        "which",
        "yet",
    }

    for start in match_starts:
        before_match = normalized_evidence[:start]
        previous_tokens = before_match.split()[-6:]
        previous_text = " ".join(previous_tokens)
        if any(
            previous_text == marker or previous_text.endswith(f" {marker}")
            for marker in prefix_denial_markers
        ):
            return True

        after_match = normalized_evidence[start + len(normalized_answer) :]
        following_tokens = after_match.split()[:8]
        following_text = " ".join(following_tokens)
        if following_tokens and following_tokens[0] in {"no", "nope"}:
            if len(following_tokens) == 1:
                return True
            if following_tokens[1] in {"it", "this", "that", "instead", "rather"}:
                return True
            if following_text.startswith(("no not true", "no false", "no incorrect")):
                return True

        if has_referential_denial_marker(following_tokens):
            return True

        if following_tokens and following_tokens[0] in discourse_connectors:
            connector_tail = following_tokens[1:]
            if has_referential_denial_marker(connector_tail):
                return True

    return False


def citation_text_refutes_answer(
    answer: str, citations: List[AnswerCitation], question: str
) -> bool:
    cited_citations = cited_citations_for_answer(answer, citations)
    if not cited_citations:
        return False

    evidence_text_parts = []
    for citation in cited_citations:
        evidence_text_parts.append(citation.excerpt)

    answer_tokens = support_tokens(answer)
    if not answer_tokens:
        return False

    normalized_answer = normalize_support_text(answer)
    normalized_evidence = normalize_support_text(" ".join(evidence_text_parts))
    if not normalized_answer or normalized_answer not in normalized_evidence:
        return False
    return matched_claim_is_denied(
        normalized_answer,
        normalized_evidence,
        " ".join(evidence_text_parts),
    )


def mechanical_self_check_answer(
    answer: str,
    citations: List[AnswerCitation],
    *,
    question: str = "",
    retry_attempted: bool = False,
) -> AnswerSelfCheck:
    clean_answer = answer.strip()
    if len(clean_answer) < 3:
        return AnswerSelfCheck(
            outcome="needs_retry",
            reasons=["answer_too_short"],
            retry_attempted=retry_attempted,
        )

    if not citations:
        if answer_is_refusal(clean_answer):
            return AnswerSelfCheck(
                outcome="not_verified",
                reasons=["refused_without_prompt_evidence"],
                retry_attempted=retry_attempted,
            )
        return AnswerSelfCheck(
            outcome="needs_refusal",
            reasons=["no_prompt_evidence"],
            retry_attempted=retry_attempted,
        )

    if answer_is_refusal(clean_answer):
        return AnswerSelfCheck(
            outcome="needs_retry",
            reasons=["answer_refused_despite_prompt_evidence"],
            retry_attempted=retry_attempted,
        )

    citation_ids = inline_citation_ids(clean_answer)
    if not citation_ids:
        return AnswerSelfCheck(
            outcome="needs_retry",
            reasons=["missing_inline_citation"],
            retry_attempted=retry_attempted,
        )

    if not citation_ids_are_valid(citation_ids, citations):
        return AnswerSelfCheck(
            outcome="needs_retry",
            reasons=["invalid_inline_citation"],
            retry_attempted=retry_attempted,
        )

    if citation_text_refutes_answer(clean_answer, citations, question):
        return AnswerSelfCheck(
            outcome="needs_refusal",
            reasons=[
                "citation_text_does_not_support_answer",
                "deterministic_refutation_detected",
            ],
            retry_attempted=retry_attempted,
        )

    return AnswerSelfCheck(
        outcome="mechanical_checks_passed",
        reasons=["mechanical_checks_passed"],
        retry_attempted=retry_attempted,
    )


def loop_decision_for_self_check(self_check: AnswerSelfCheck) -> LoopDecision:
    if self_check.outcome == "mechanical_checks_passed":
        return LoopDecision.CONTINUE
    if self_check.outcome == "supported":
        return LoopDecision.SUPPORTED
    if self_check.outcome == "not_verified":
        return LoopDecision.NOT_VERIFIED
    if self_check.outcome == "needs_retry":
        return LoopDecision.RETRY
    if self_check.outcome == "needs_refusal":
        return LoopDecision.REFUSE
    return LoopDecision.ERROR


def verification_result_for_self_check(
    self_check: AnswerSelfCheck, *, verifier: Optional[str]
) -> VerificationResult:
    reasons = tuple(self_check.reasons)
    if self_check.outcome == "supported":
        outcome = VerificationOutcome.SUPPORTED
    elif self_check.outcome == "not_verified":
        outcome = VerificationOutcome.NOT_VERIFIED
    elif "llm_verifier_unsupported" in reasons:
        outcome = VerificationOutcome.UNSUPPORTED
    elif "llm_verifier_insufficient" in reasons:
        outcome = VerificationOutcome.INSUFFICIENT
    else:
        outcome = VerificationOutcome.ERROR

    return VerificationResult(
        outcome=outcome,
        reasons=reasons,
        verifier=verifier,
        metadata={
            "retry_attempted": self_check.retry_attempted,
            "self_check_outcome": self_check.outcome,
        },
    )


def verifier_prompt(
    *, question: str, answer: str, citations: List[AnswerCitation]
) -> str:
    cited_citations = cited_citations_for_answer(answer, citations)
    cited_excerpts = "\n\n".join(
        f"[{citation.citation_id}] {citation.excerpt.strip()}"
        for citation in cited_citations
    )
    return (
        "You are a strict citation verifier for an evidence-grounded AI loop.\n"
        "Use only the cited excerpts. Do not use outside knowledge.\n"
        "Decide whether every factual claim in the answer is directly supported "
        "by the cited excerpts.\n"
        "Return only JSON with this schema: "
        '{"outcome":"supported|unsupported|insufficient","reason":"short reason"}.\n'
        "Use unsupported when the cited excerpts contradict any answer claim.\n"
        "Use insufficient when the cited excerpts do not contain enough evidence.\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Cited excerpts:\n{cited_excerpts}"
    )


def _reject_duplicate_verifier_json_keys(pairs):
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate verifier JSON key: {key!r}")
        payload[key] = value
    return payload


def _reject_nonfinite_verifier_json_constant(value: str):
    raise ValueError(f"non-finite verifier JSON number: {value}")


def _parse_finite_verifier_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("verifier JSON number is outside the finite range")
    return parsed


def parse_verifier_response(raw_response: str) -> Tuple[Optional[str], str]:
    raw_payload = raw_response.strip()
    if "{" not in raw_payload:
        return None, "missing_json"
    try:
        payload = json.loads(
            raw_payload,
            object_pairs_hook=_reject_duplicate_verifier_json_keys,
            parse_constant=_reject_nonfinite_verifier_json_constant,
            parse_float=_parse_finite_verifier_json_float,
        )
    except (json.JSONDecodeError, ValueError):
        return None, "invalid_json"
    if type(payload) is not dict:
        return None, "invalid_json"

    outcome = payload.get("outcome")
    if type(outcome) is not str or outcome not in VERIFIER_OUTCOMES:
        return None, "invalid_outcome"
    reason = payload.get("reason")
    if type(reason) is not str or not reason.strip():
        return None, "invalid_reason"
    return outcome, reason


def verify_answer_with_llm(
    *,
    llm,
    question: str,
    answer: str,
    citations: List[AnswerCitation],
    retry_attempted: bool,
    logger=None,
) -> AnswerSelfCheck:
    if llm is None:
        return AnswerSelfCheck(
            outcome="needs_refusal",
            reasons=["llm_verifier_unavailable"],
            retry_attempted=retry_attempted,
        )

    prompt = verifier_prompt(
        question=question,
        answer=answer,
        citations=citations,
    )
    try:
        raw_response = str(llm.invoke(prompt)).strip()
    except Exception as exc:
        if logger is not None:
            logger.warning("LLM verifier failed: %s", exc)
        return AnswerSelfCheck(
            outcome="needs_refusal",
            reasons=["llm_verifier_error"],
            retry_attempted=retry_attempted,
        )

    verifier_outcome, parse_reason = parse_verifier_response(raw_response)
    if verifier_outcome is None:
        return AnswerSelfCheck(
            outcome="needs_refusal",
            reasons=["llm_verifier_parse_failed", parse_reason],
            retry_attempted=retry_attempted,
        )

    if verifier_outcome == "supported":
        return AnswerSelfCheck(
            outcome="supported",
            reasons=["mechanical_checks_passed", "llm_verifier_supported"],
            retry_attempted=retry_attempted,
        )

    return AnswerSelfCheck(
        outcome="needs_refusal",
        reasons=[f"llm_verifier_{verifier_outcome}"],
        retry_attempted=retry_attempted,
    )


def fail_closed_self_check(self_check: AnswerSelfCheck) -> AnswerSelfCheck:
    reasons = list(self_check.reasons)
    if "self_check_failed_closed" not in reasons:
        reasons.insert(0, "self_check_failed_closed")
    return AnswerSelfCheck(
        outcome="needs_refusal",
        reasons=reasons,
        retry_attempted=self_check.retry_attempted,
    )


def self_check_retry_instruction(self_check: AnswerSelfCheck) -> str:
    reasons = ", ".join(self_check.reasons)
    return (
        "Self-check retry instruction: the previous answer failed checks "
        f"({reasons}). Answer again using only the context above. Include at least "
        "one bracketed citation like [1] for supported claims. If the context does "
        "not contain the answer, say that clearly."
    )


def should_retry_web_verifier_failure(self_check: AnswerSelfCheck) -> bool:
    return (
        self_check.outcome == "needs_refusal"
        and not self_check.retry_attempted
        and bool(WEB_VERIFIER_RETRY_REASONS & set(self_check.reasons))
    )


def web_evidence_retry_instruction(self_check: AnswerSelfCheck) -> str:
    reasons = ", ".join(self_check.reasons)
    return (
        "Web evidence retry instruction: the previous answer could not be verified "
        f"against the snippets ({reasons}). Answer again using only facts stated "
        "plainly in the web search snippets above. Keep the answer short, cite every "
        "factual claim with bracketed source numbers like [1], and omit any claim "
        "that is not directly supported by the snippets. If the snippets still do "
        "not contain enough evidence, say that clearly."
    )


def format_check_retry_instruction(format_check: AnswerFormatCheck) -> str:
    reasons = ", ".join(format_check.reasons)
    return (
        "Format retry instruction: the previous answer was hard to render in a "
        f"web chat UI ({reasons}). Keep the facts and citations unchanged, but "
        "rewrite the answer using clean Markdown. Put each numbered-list item "
        "on its own line, use short paragraphs, avoid raw internal verification "
        "labels such as not_verified or needs_retry, and do not emit HTML."
    )
