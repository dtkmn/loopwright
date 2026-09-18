from dataclasses import dataclass
from typing import Tuple

from langchain_core.embeddings import Embeddings


GOLDEN_DOCUMENT_TEXT = """# Project Phoenix Brief

Project Phoenix launches in June 2026.
The approved Project Phoenix budget is $42 million.
Alex Rivera owns the Project Phoenix rollout.
Unsupported claims must be refused instead of invented.
"""


@dataclass(frozen=True)
class GoldenEvalCase:
    case_id: str
    question: str
    expected_terms: Tuple[str, ...] = ()
    forbidden_terms: Tuple[str, ...] = ()
    accepted_answer_patterns: Tuple[str, ...] = ()
    expect_refusal: bool = False


GOLDEN_EVAL_CASES = (
    GoldenEvalCase(
        case_id="launch_date",
        question="When does Project Phoenix launch?",
        expected_terms=("June 2026",),
        forbidden_terms=("Lunar Base Alpha",),
        accepted_answer_patterns=(
            r"(?:According to (?:the )?(?:brief|document),\s*)?"
            r"(?:Project Phoenix\s+)?(?:launches|is scheduled to launch) "
            r"in June 2026\s*\[1\]\.?",
        ),
    ),
    GoldenEvalCase(
        case_id="budget",
        question="What is the approved Project Phoenix budget?",
        expected_terms=("$42 million",),
        forbidden_terms=("$999 million",),
        accepted_answer_patterns=(
            r"(?:The )?(?:approved )?(?:Project Phoenix )?budget is "
            r"\$42 million\s*\[1\]\.?",
        ),
    ),
    GoldenEvalCase(
        case_id="owner",
        question="Who owns the Project Phoenix rollout?",
        expected_terms=("Alex Rivera",),
        accepted_answer_patterns=(
            r"(?:Alex Rivera(?: owns the Project Phoenix rollout)?|"
            r"The Project Phoenix rollout is owned by Alex Rivera)\s*\[1\]\.?",
        ),
    ),
    GoldenEvalCase(
        case_id="unsupported_venue",
        question="Where is the Project Phoenix launch venue?",
        expect_refusal=True,
    ),
)


class GoldenEvalEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        lower = text.lower()
        return [
            float(len(text)),
            float(lower.count("phoenix")),
            float(lower.count("launch")),
            float(lower.count("budget")),
            float(lower.count("owner") + lower.count("owns")),
            float(lower.count("venue") + lower.count("location")),
        ]
