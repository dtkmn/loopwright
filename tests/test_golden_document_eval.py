import json

import pytest
from langchain_core.language_models.llms import LLM
from pydantic import Field

from src.ai_loop_engine import AILoopEngine, SELF_CHECK_REFUSAL_ANSWER
from src.golden_eval import GOLDEN_DOCUMENT_TEXT, GoldenEvalEmbeddings
from src.loop_engine import LoopDecision, LoopPhase


class GoldenEvalLLM(LLM):
    scenario: str
    calls: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "golden-eval"

    def _call(self, prompt: str, stop=None, **kwargs) -> str:
        self.calls.append(prompt)
        if "strict citation verifier" in prompt:
            return self._verifier_response(prompt)
        return self._answer_response(prompt)

    def _answer_response(self, prompt: str) -> str:
        question = self._question_from_prompt(prompt)
        if self.scenario == "missing_citation_then_supported":
            if "Self-check retry instruction" in prompt:
                return "Project Phoenix launches in June 2026 [1]."
            return "Project Phoenix launches in June 2026."
        if self.scenario == "unsupported_answer":
            return "Project Phoenix launches from Lunar Base Alpha [1]."
        if "budget" in question:
            return "The approved Project Phoenix budget is $42 million [1]."
        if "owner" in question or "owns" in question:
            return "Alex Rivera owns the Project Phoenix rollout [1]."
        return "Project Phoenix launches in June 2026 [1]."

    def _question_from_prompt(self, prompt: str) -> str:
        if "Question:" not in prompt:
            return ""
        question_section = prompt.split("Question:", 1)[1]
        question = question_section.split("Self-check retry instruction:", 1)[0]
        question = question.split("Answer:", 1)[0]
        return question.strip().lower()

    def _verifier_response(self, prompt: str) -> str:
        if "Lunar Base Alpha" in prompt:
            return json.dumps(
                {
                    "outcome": "insufficient",
                    "reason": "venue is not in the cited excerpts",
                }
            )
        return json.dumps(
            {
                "outcome": "supported",
                "reason": "answer is directly supported by cited excerpts",
            }
        )


@pytest.fixture(autouse=True)
def clear_provider_env(monkeypatch):
    for env_var in (
        "LLM_BACKEND",
        "LLM_MODEL",
        "EMBEDDINGS_MODEL",
        "OLLAMA_MODEL",
        "OLLAMA_EMBED_MODEL",
        "OLLAMA_BASE_URL",
        "OPENAI_COMPAT_BASE_URL",
        "OPENAI_COMPAT_MODEL",
        "OPENAI_COMPAT_EMBED_MODEL",
        "OPENAI_COMPAT_API_KEY",
        "OPENAI_COMPAT_TIMEOUT",
    ):
        monkeypatch.delenv(env_var, raising=False)


def build_golden_qa(tmp_path, scenario="supported"):
    document = tmp_path / "project_phoenix_brief.md"
    document.write_text(GOLDEN_DOCUMENT_TEXT, encoding="utf-8")

    qa = AILoopEngine(fast_mode=True, llm_backend="ollama")
    qa.llm = GoldenEvalLLM(scenario=scenario)
    qa.active_llm_backend = "ollama"
    qa.loaded_model_id = "golden-eval"
    qa.loaded_model_label = "Golden eval model"
    qa.embeddings = GoldenEvalEmbeddings()
    qa.process_document(str(document))
    return qa, document


def phase_sequence(result):
    return [step.phase for step in result.loop_report.run.steps]


def steps_for_phase(result, phase):
    return [step for step in result.loop_report.run.steps if step.phase == phase]


def only_step(result, phase):
    steps = steps_for_phase(result, phase)
    assert len(steps) == 1
    return steps[0]


def test_golden_document_supported_answer_is_cited_and_verified(tmp_path):
    qa, document = build_golden_qa(tmp_path)
    status = qa.status()

    result = qa.query_with_trace(
        "When does Project Phoenix launch?",
        context_provider="document",
    )

    assert status.ready_for_queries is True
    assert status.active_backend == "ollama"
    assert status.active_model_label == "Golden eval model"
    assert status.processing_report.success is True
    assert status.processing_report.active_document_name == document.name
    assert status.processing_report.phase == "complete"
    assert status.processing_report.backend == "ollama"
    assert result.answer == "Project Phoenix launches in June 2026 [1]."
    assert result.trace.document_name == document.name
    assert result.trace.backend == "ollama"
    assert result.trace.model_label == "Golden eval model"
    assert result.trace.retrieved_chunk_count == len(result.trace.citations) == 1
    assert result.trace.self_check.outcome == "supported"
    assert result.trace.self_check.reasons == [
        "mechanical_checks_passed",
        "llm_verifier_supported",
    ]
    assert result.trace.self_check.retry_attempted is False
    assert result.trace.citations[0].citation_id == 1
    assert result.trace.citations[0].source_name == document.name
    assert "June 2026" in result.trace.citations[0].excerpt
    assert result.loop_report.run.final_decision == LoopDecision.SUPPORTED
    assert result.loop_report.run.final_answer == result.answer
    assert phase_sequence(result) == [
        LoopPhase.CONTEXT_SELECT,
        LoopPhase.RETRIEVE,
        LoopPhase.DRAFT,
        LoopPhase.FORMAT_CHECK,
        LoopPhase.MECHANICAL_CHECK,
        LoopPhase.VERIFY,
        LoopPhase.FINAL,
    ]
    assert only_step(result, LoopPhase.RETRIEVE).decision == LoopDecision.CONTINUE
    assert only_step(result, LoopPhase.DRAFT).decision == LoopDecision.CONTINUE
    assert only_step(result, LoopPhase.FORMAT_CHECK).decision == LoopDecision.CONTINUE
    mechanical_step = only_step(result, LoopPhase.MECHANICAL_CHECK)
    verify_step = only_step(result, LoopPhase.VERIFY)
    assert mechanical_step.decision == LoopDecision.CONTINUE
    assert mechanical_step.metadata["reasons"] == ["mechanical_checks_passed"]
    assert verify_step.decision == LoopDecision.SUPPORTED
    assert verify_step.verification.outcome.value == "supported"
    assert verify_step.verification.reasons == (
        "mechanical_checks_passed",
        "llm_verifier_supported",
    )
    assert only_step(result, LoopPhase.FINAL).decision == LoopDecision.SUPPORTED
    assert qa.chat_history[-1]["self_check"]["outcome"] == "supported"


def test_golden_document_unsupported_answer_fails_closed(tmp_path):
    qa, _document = build_golden_qa(tmp_path, scenario="unsupported_answer")

    result = qa.query_with_trace(
        "Where is the Project Phoenix launch venue?",
        context_provider="document",
    )

    assert result.answer == SELF_CHECK_REFUSAL_ANSWER
    assert result.trace.self_check.outcome == "needs_refusal"
    assert result.trace.self_check.reasons == ["llm_verifier_insufficient"]
    assert result.trace.self_check.retry_attempted is False
    assert result.trace.retrieved_chunk_count == len(result.trace.citations) == 1
    assert result.loop_report.run.final_decision == LoopDecision.REFUSE
    assert phase_sequence(result) == [
        LoopPhase.CONTEXT_SELECT,
        LoopPhase.RETRIEVE,
        LoopPhase.DRAFT,
        LoopPhase.FORMAT_CHECK,
        LoopPhase.MECHANICAL_CHECK,
        LoopPhase.VERIFY,
        LoopPhase.REFUSE,
        LoopPhase.FINAL,
    ]
    verify_step = only_step(result, LoopPhase.VERIFY)
    refuse_step = only_step(result, LoopPhase.REFUSE)
    assert verify_step.decision == LoopDecision.NOT_VERIFIED
    assert verify_step.verification.outcome.value == "insufficient"
    assert verify_step.metadata["reasons"] == ["llm_verifier_insufficient"]
    assert refuse_step.decision == LoopDecision.REFUSE
    assert refuse_step.metadata["reasons"] == ["llm_verifier_insufficient"]
    assert only_step(result, LoopPhase.FINAL).decision == LoopDecision.REFUSE
    assert qa.chat_history[-1]["answer"] == SELF_CHECK_REFUSAL_ANSWER


def test_golden_document_missing_citation_retries_then_passes(tmp_path):
    qa, _document = build_golden_qa(
        tmp_path, scenario="missing_citation_then_supported"
    )

    result = qa.query_with_trace(
        "When does Project Phoenix launch?",
        context_provider="document",
    )

    assert result.answer == "Project Phoenix launches in June 2026 [1]."
    assert result.trace.self_check.outcome == "supported"
    assert result.trace.self_check.retry_attempted is True
    assert result.loop_report.run.final_decision == LoopDecision.SUPPORTED
    assert phase_sequence(result) == [
        LoopPhase.CONTEXT_SELECT,
        LoopPhase.RETRIEVE,
        LoopPhase.DRAFT,
        LoopPhase.FORMAT_CHECK,
        LoopPhase.MECHANICAL_CHECK,
        LoopPhase.RETRY,
        LoopPhase.DRAFT,
        LoopPhase.FORMAT_CHECK,
        LoopPhase.MECHANICAL_CHECK,
        LoopPhase.VERIFY,
        LoopPhase.FINAL,
    ]
    draft_steps = steps_for_phase(result, LoopPhase.DRAFT)
    mechanical_steps = steps_for_phase(result, LoopPhase.MECHANICAL_CHECK)
    retry_step = only_step(result, LoopPhase.RETRY)
    verify_step = only_step(result, LoopPhase.VERIFY)
    assert len(draft_steps) == 2
    assert draft_steps[0].retry_count == 0
    assert draft_steps[1].retry_count == 1
    assert mechanical_steps[0].decision == LoopDecision.RETRY
    assert mechanical_steps[0].metadata["reasons"] == ["missing_inline_citation"]
    assert retry_step.decision == LoopDecision.RETRY
    assert retry_step.metadata["reasons"] == ["missing_inline_citation"]
    assert mechanical_steps[1].decision == LoopDecision.CONTINUE
    assert mechanical_steps[1].metadata["retry_attempted"] is True
    assert verify_step.decision == LoopDecision.SUPPORTED
    assert verify_step.retry_count == 1
    assert verify_step.verification.outcome.value == "supported"
    assert any("missing_inline_citation" in call for call in qa.llm.calls)
    answer_calls = [
        call for call in qa.llm.calls if "You are a helpful AI assistant" in call
    ]
    assert len(answer_calls) == 2
