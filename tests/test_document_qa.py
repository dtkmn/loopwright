import json
import logging
import threading
import typing
import urllib.error
from types import SimpleNamespace

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

import src.answer_loop as answer_loop_module
import src.DocumentQA as document_qa_module
import src.ai_loop_runtime as ai_loop_runtime_module
import src.context_providers as context_providers_module
import src.document_config as document_config_module
import src.document_ingestion as document_ingestion_module
import src.document_text as document_text_module
import src.model_adapters as model_adapters_module
import src.retrieval as retrieval_module
import src.retrieval_types as retrieval_types_module
import src.runtime_config as runtime_config_module
import src.web_contract as web_contract_module
import src.web_search as web_search_module
from src.ai_loop_engine import AILoopEngine, DocumentQA as PublicDocumentQA
from src.DocumentQA import (
    DEFAULT_OLLAMA_EMBEDDINGS_MODEL,
    DEFAULT_OLLAMA_MODEL,
    AnswerCitation,
    AnswerSelfCheck,
    DocumentContextProvider,
    DocumentQA,
    FaissVectorStore,
    LocalHashingEmbeddings,
    MockLLM,
    OllamaEmbeddings,
    OllamaLLM,
    OpenAICompatibleEmbeddings,
    OpenAICompatibleLLM,
    normalize_ollama_base_url,
    normalize_openai_compatible_base_url,
    safe_ollama_base_url_for_error,
    safe_openai_compatible_base_url_for_error,
)
from src.loop_engine import GuardrailDecision, LoopDecision, LoopPhase, LoopReport


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        lower = text.lower()
        return [
            float(len(text)),
            float(lower.count("project")),
            float(lower.count("phoenix")),
            float(lower.count("launch")),
        ]


@pytest.fixture(autouse=True)
def clear_model_provider_env(monkeypatch):
    for name in (
        "LLM_BACKEND",
        "LLM_MODEL",
        "EMBEDDINGS_MODEL",
        "EMBEDDINGS_DEVICE",
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
        "OLLAMA_EMBED_MODEL",
        "OLLAMA_THINK_LEVEL",
        "OPENAI_COMPAT_BASE_URL",
        "OPENAI_COMPAT_MODEL",
        "OPENAI_COMPAT_EMBED_MODEL",
        "OPENAI_COMPAT_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_ai_loop_engine_is_canonical_runtime_alias():
    from src import AILoopEngine as RootAILoopEngine

    assert document_qa_module is ai_loop_runtime_module
    assert RootAILoopEngine is AILoopEngine
    assert DocumentQA is AILoopEngine
    assert PublicDocumentQA is AILoopEngine
    assert normalize_ollama_base_url is runtime_config_module.normalize_ollama_base_url
    assert (
        normalize_openai_compatible_base_url
        is runtime_config_module.normalize_openai_compatible_base_url
    )
    assert MockLLM is model_adapters_module.MockLLM
    assert OllamaEmbeddings is model_adapters_module.OllamaEmbeddings
    assert OllamaLLM is model_adapters_module.OllamaLLM
    assert OpenAICompatibleEmbeddings is model_adapters_module.OpenAICompatibleEmbeddings
    assert OpenAICompatibleLLM is model_adapters_module.OpenAICompatibleLLM
    assert DocumentContextProvider is context_providers_module.DocumentContextProvider
    assert AnswerCitation is retrieval_types_module.AnswerCitation
    assert retrieval_module.AnswerCitation is retrieval_types_module.AnswerCitation
    assert FaissVectorStore is retrieval_module.FaissVectorStore
    assert AnswerSelfCheck is answer_loop_module.AnswerSelfCheck
    assert (
        document_qa_module.SELF_CHECK_REFUSAL_ANSWER
        == answer_loop_module.SELF_CHECK_REFUSAL_ANSWER
    )
    assert (
        document_qa_module.SELF_CHECK_PASS_OUTCOMES
        == answer_loop_module.SELF_CHECK_PASS_OUTCOMES
    )
    assert ai_loop_runtime_module.MAX_DOCUMENT_BYTES == (
        document_config_module.MAX_DOCUMENT_BYTES
    )
    assert ai_loop_runtime_module.MAX_DOCUMENT_CHUNKS == (
        document_config_module.MAX_DOCUMENT_CHUNKS
    )
    assert document_ingestion_module.MAX_DOCUMENT_BYTES == (
        document_config_module.MAX_DOCUMENT_BYTES
    )
    assert (
        document_qa_module.SUPPORTED_EXTENSIONS
        == document_config_module.SUPPORTED_EXTENSIONS
    )
    assert (
        document_qa_module.TEXT_ENCODING_FALLBACKS
        == document_config_module.TEXT_ENCODING_FALLBACKS
    )
    assert (
        document_qa_module.normalize_encoding_name
        is document_config_module.normalize_encoding_name
    )
    assert document_qa_module.decode_supported_text(b"ok", "utf-8") == (
        document_text_module.decode_supported_text(b"ok", "utf-8")
    )
    assert (
        document_qa_module.open_ollama_request_no_proxy
        is model_adapters_module.open_ollama_request_no_proxy
    )
    assert (
        document_qa_module.open_openai_compatible_request
        is model_adapters_module.open_openai_compatible_request
    )


def test_legacy_star_import_and_type_hints_include_retrieval_exports():
    namespace = {}
    exec("from src.DocumentQA import *", {}, namespace)

    assert namespace["AnswerCitation"] is retrieval_types_module.AnswerCitation
    assert namespace["AnswerSelfCheck"] is answer_loop_module.AnswerSelfCheck
    assert namespace["FaissVectorStore"] is retrieval_module.FaissVectorStore
    hints = typing.get_type_hints(document_qa_module.AnswerTrace)
    assert typing.get_args(hints["citations"])[0] is retrieval_types_module.AnswerCitation
    typing.get_type_hints(document_qa_module.AILoopEngine._commit_active_document_state)
    typing.get_type_hints(document_qa_module.AILoopEngine._build_retrieval_chain)
    typing.get_type_hints(document_qa_module.AILoopEngine._load_documents)


def test_legacy_vector_store_module_reassignment_intercepts_index(
    monkeypatch, tmp_path
):
    document = tmp_path / "phoenix.txt"
    document.write_text("Project Phoenix launches in June 2026.", encoding="utf-8")
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.embeddings = FakeEmbeddings()
    calls = []

    class PatchedVectorStore:
        @classmethod
        def from_documents(cls, documents, embedding):
            calls.append(len(documents))
            raise ValueError("patched index boom")

    monkeypatch.setattr(document_qa_module, "FaissVectorStore", PatchedVectorStore)

    with pytest.raises(RuntimeError, match="patched index boom"):
        qa.process_document(str(document))

    assert calls
    assert qa.vector_store is None
    assert qa.retrieval_chain is None


def test_legacy_retrieval_factory_module_reassignment_intercepts_chain(
    monkeypatch, tmp_path
):
    document = tmp_path / "phoenix.txt"
    document.write_text("Project Phoenix launches in June 2026.", encoding="utf-8")
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.embeddings = FakeEmbeddings()
    calls = []

    def patched_build_document_retrieval_chain(
        *, vector_store, llm, document_name, profile
    ):
        calls.append(
            {
                "vector_store": vector_store,
                "llm": llm,
                "document_name": document_name,
                "profile": profile,
            }
        )
        return SimpleNamespace(invoke=lambda question: "patched answer")

    monkeypatch.setattr(
        document_qa_module,
        "build_document_retrieval_chain",
        patched_build_document_retrieval_chain,
    )

    status = qa.process_document(str(document))

    assert status.ready_for_queries is True
    assert calls
    assert calls[0]["document_name"] == "phoenix.txt"
    assert qa.retrieval_chain.invoke("anything") == "patched answer"


def create_processed_mock_qa(tmp_path):
    document = tmp_path / "phoenix.txt"
    document.write_text(
        "Project Phoenix is a document QA assistant. "
        "The launch date is June 2026.",
        encoding="utf-8",
    )
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.embeddings = FakeEmbeddings()
    qa.process_document(str(document))
    return qa, document


def test_query_before_document_runs_no_context_loop():
    qa = DocumentQA(fast_mode=True, llm_backend="mock")

    assert "mock response" in qa.query("What is this?", context_provider="none")
    result = qa.query_with_trace(
        "What is this?",
        session_id="direct",
        context_provider="none",
    )
    assert "mock response" in result.answer
    assert result.trace.question == "What is this?"
    assert result.trace.document_name is None
    assert result.trace.retrieved_chunk_count == 0
    assert result.trace.citations == []
    assert result.trace.error_message is None
    assert result.trace.self_check.outcome == "not_verified"
    assert result.trace.self_check.reasons == [
        "no_context_provider",
        "verifier_requires_prompt_evidence",
    ]
    assert result.trace.self_check.retry_attempted is False

    run = result.loop_report.run
    assert run.context_provider == "none"
    assert run.metadata["context_provider"] == "none"
    assert "document_text" not in run.metadata["untrusted_inputs"]
    assert run.final_decision == LoopDecision.NOT_VERIFIED
    assert [step.phase for step in run.steps] == [
        LoopPhase.CONTEXT_SELECT,
        LoopPhase.DRAFT,
        LoopPhase.FORMAT_CHECK,
        LoopPhase.VERIFY,
        LoopPhase.FINAL,
    ]
    format_step = next(
        step for step in run.steps if step.phase == LoopPhase.FORMAT_CHECK
    )
    assert format_step.output_summary == "format_passed"
    assert format_step.metadata["reasons"] == ["clean_web_markdown"]
    verify_step = next(step for step in run.steps if step.phase == LoopPhase.VERIFY)
    assert verify_step.verification.outcome.value == "not_verified"
    assert verify_step.metadata["verifier_skipped"] is True
    assert qa.loop_session("direct").report_count == 1


def test_no_context_query_uses_same_thread_conversation_history():
    captured_prompts = []
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.llm = SimpleNamespace(
        invoke=lambda prompt: captured_prompts.append(prompt)
        or "Dynamic programming means solving a big problem by reusing smaller answers.",
        last_thinking=None,
    )

    result = qa.query_with_trace(
        "Please explain it in layman terms.",
        session_id="thread_memory",
        context_provider="none",
        conversation_history=[
            {
                "role": "user",
                "content": "Do you know what dynamic programming is?",
            },
            {"role": "assistant", "content": "Yes."},
        ],
    )

    assert "Dynamic programming means" in result.answer
    assert result.trace.question == "Please explain it in layman terms."
    prompt = captured_prompts[0]
    assert "Recent same-thread conversation" in prompt
    assert "Do you know what dynamic programming is?" in prompt
    assert "Assistant: Yes." in prompt
    assert "Current question: Please explain it in layman terms." in prompt
    assert "give a fuller step-by-step explanation" in prompt
    assert "model knowledge and thread memory as not verified evidence" in prompt
    assert "Use clean Markdown for readability" in prompt
    assert "each item on its own line" in prompt
    assert "Do not include internal verification labels" in prompt
    assert "Be concise" not in prompt
    assert result.loop_report.run.user_input == "Please explain it in layman terms."
    assert result.loop_report.run.metadata["conversation_context_turns"] == 2


def test_no_context_query_uses_semantic_thread_memory():
    captured_prompts = []
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.llm = SimpleNamespace(
        invoke=lambda prompt: captured_prompts.append(prompt)
        or "Dynamic programming is like keeping notes for solved subproblems.",
        last_thinking=None,
    )

    result = qa.query_with_trace(
        "Please explain that algorithm.",
        session_id="thread_memory",
        context_provider="none",
        semantic_memory=[
            {
                "role": "user",
                "content": "Dynamic programming stores answers to subproblems.",
                "score": 0.9,
                "message_id": 7,
            }
        ],
        semantic_memory_status="retrieved",
    )

    prompt = captured_prompts[0]
    assert "Relevant same-thread memory" in prompt
    assert "Dynamic programming stores answers to subproblems." in prompt
    assert "Current question: Please explain that algorithm." in prompt
    assert result.trace.question == "Please explain that algorithm."
    assert result.loop_report.run.user_input == "Please explain that algorithm."
    assert result.loop_report.run.metadata["semantic_memory_turns"] == 1
    assert result.loop_report.run.metadata["semantic_memory_status"] == "retrieved"
    memory_step = next(
        step
        for step in result.loop_report.run.steps
        if step.name == "Retrieve thread memory"
    )
    assert memory_step.output_summary == "1 semantic memories"
    assert memory_step.metadata == {
        "semantic_memory_count": 1,
        "semantic_memory_status": "retrieved",
    }


def test_semantic_memory_status_is_sanitized_in_public_loop_metadata():
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.llm = SimpleNamespace(
        invoke=lambda _prompt: "A short answer.",
        last_thinking=None,
    )

    result = qa.query_with_trace(
        "Say something.",
        context_provider="none",
        semantic_memory_status="SECRET_STATUS",
    )

    assert result.loop_report.run.metadata["semantic_memory_status"] == "unavailable"
    memory_step = next(
        step
        for step in result.loop_report.run.steps
        if step.name == "Retrieve thread memory"
    )
    assert memory_step.metadata["semantic_memory_status"] == "unavailable"


def test_no_context_query_applies_loop_recipe_to_prompt_and_report():
    captured_prompts = []
    recipe = {
        "recipe_id": "recipe_explainer",
        "name": "Plain-language explainer",
        "goal": "Explain technical ideas in simple language.",
        "instructions": "Avoid jargon unless you define it.",
        "success_criteria": ["Uses an analogy.", "Names uncertainty."],
        "stop_condition": "Stop after the answer is clear.",
        "context_provider": "none",
        "model_profile": "quality",
        "verifier": "default",
    }
    qa = DocumentQA(fast_mode=False, llm_backend="mock")
    qa.llm = SimpleNamespace(
        invoke=lambda prompt: captured_prompts.append(prompt)
        or "Dynamic programming is like keeping a notebook of solved subproblems.",
        last_thinking=None,
    )

    result = qa.query_with_trace(
        "Explain dynamic programming.",
        session_id="recipe_runtime",
        loop_recipe=recipe,
    )

    prompt = captured_prompts[0]
    assert "Loop recipe guidance" in prompt
    assert "Plain-language explainer" in prompt
    assert "Avoid jargon unless you define it." in prompt
    assert "Uses an analogy." in prompt
    run = result.loop_report.run
    assert run.metadata["recipe_id"] == "recipe_explainer"
    assert run.metadata["recipe_name"] == "Plain-language explainer"
    recipe_step = run.steps[0]
    assert recipe_step.phase == LoopPhase.INPUT
    assert recipe_step.name == "Apply loop recipe"
    assert recipe_step.metadata["success_criteria_count"] == 2


def test_no_context_query_strips_inline_citation_markers():
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.llm = SimpleNamespace(
        invoke=lambda _prompt: "The answer is definitely grounded [1].",
        last_thinking=None,
    )

    result = qa.query_with_trace(
        "What is the answer?",
        session_id="direct_citation",
        context_provider="none",
    )

    assert result.answer == "The answer is definitely grounded."
    assert "[1]" not in result.answer
    assert result.trace.citations == []
    assert result.trace.retrieved_chunk_count == 0
    assert result.trace.self_check.outcome == "not_verified"

    draft_step = next(
        step for step in result.loop_report.run.steps if step.phase == LoopPhase.DRAFT
    )
    assert "[1]" not in draft_step.output_summary
    assert draft_step.metadata["inline_citation_ids"] == []
    assert draft_step.metadata["removed_inline_citation_ids"] == [1]
    assert result.loop_report.run.final_decision == LoopDecision.NOT_VERIFIED


def test_no_context_query_preserves_attached_bracket_expressions():
    cases = [
        "The answer is definitely grounded[1].",
        "That fact[1] is useful.",
        "This is true[1].",
        "Use fact[1] carefully.",
        "Use source[0] and references[1] in the loop.",
        "Use `fact[1]` in the code sample.",
    ]

    for raw_answer in cases:
        qa = DocumentQA(fast_mode=True, llm_backend="mock")
        qa.llm = SimpleNamespace(invoke=lambda _prompt, answer=raw_answer: answer)

        result = qa.query_with_trace("What is the answer?", context_provider="none")

        assert result.answer == raw_answer
        assert result.trace.citations == []
        draft_step = next(
            step for step in result.loop_report.run.steps if step.phase == LoopPhase.DRAFT
        )
        assert draft_step.output_summary == raw_answer
        assert draft_step.metadata["inline_citation_ids"] == []
        assert draft_step.metadata["removed_inline_citation_ids"] == []


def test_no_context_query_preserves_code_indices():
    cases = [
        "Use arr[0] to access the first item, and arr[1] for the second.",
        "Call item[0] from the list.",
        "Read list[0] before list[1].",
        "Read `foo[1]` inside the code sample.",
        "Use users[0] to access the first user and scores[1] for the next score.",
        "Read records[0] before names[1].",
    ]

    for answer in cases:
        qa = DocumentQA(fast_mode=True, llm_backend="mock")
        qa.llm = SimpleNamespace(invoke=lambda _prompt, value=answer: value)

        result = qa.query_with_trace(
            "How do I access indexed values?",
            context_provider="none",
        )

        assert result.answer == answer
        assert result.trace.citations == []
        assert result.trace.self_check.outcome == "not_verified"
        draft_step = next(
            step for step in result.loop_report.run.steps if step.phase == LoopPhase.DRAFT
        )
        assert draft_step.output_summary == result.answer
        assert draft_step.metadata["inline_citation_ids"] == []
        assert draft_step.metadata["removed_inline_citation_ids"] == []


def test_no_context_format_check_retries_compact_markdown_list():
    prompts = []
    answers = [
        (
            "1. **Trigger:** Pressure builds. 2. **Outcome:** The plan "
            "escalates."
        ),
        "1. **Trigger:** Pressure builds.\n2. **Outcome:** The plan escalates.",
    ]

    def invoke(prompt):
        prompts.append(prompt)
        return answers[len(prompts) - 1]

    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.llm = SimpleNamespace(invoke=invoke, last_thinking=None)

    result = qa.query_with_trace(
        "Explain the sequence.",
        session_id="format_direct",
        context_provider="none",
    )

    assert result.answer == answers[1]
    assert len(prompts) == 2
    assert "Format retry instruction" in prompts[1]
    assert "compact_ordered_list" in prompts[1]
    phases = [step.phase for step in result.loop_report.run.steps]
    assert phases == [
        LoopPhase.CONTEXT_SELECT,
        LoopPhase.DRAFT,
        LoopPhase.FORMAT_CHECK,
        LoopPhase.RETRY,
        LoopPhase.DRAFT,
        LoopPhase.FORMAT_CHECK,
        LoopPhase.VERIFY,
        LoopPhase.FINAL,
    ]
    format_steps = [
        step
        for step in result.loop_report.run.steps
        if step.phase == LoopPhase.FORMAT_CHECK
    ]
    assert [step.output_summary for step in format_steps] == [
        "needs_retry",
        "format_passed",
    ]
    assert format_steps[0].metadata["reasons"] == ["compact_ordered_list"]
    assert format_steps[1].retry_count == 1


def test_no_context_format_check_retries_unlabeled_compact_numbered_list():
    prompts = []
    answers = [
        "1. install dependencies. 2. run tests.",
        "1. install dependencies.\n2. run tests.",
    ]

    def invoke(prompt):
        prompts.append(prompt)
        return answers[len(prompts) - 1]

    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.llm = SimpleNamespace(invoke=invoke, last_thinking=None)

    result = qa.query_with_trace(
        "Give me steps.",
        session_id="format_plain_list",
        context_provider="none",
    )

    assert result.answer == answers[1]
    assert len(prompts) == 2
    assert "Format retry instruction" in prompts[1]
    assert "compact_ordered_list" in prompts[1]
    format_steps = [
        step
        for step in result.loop_report.run.steps
        if step.phase == LoopPhase.FORMAT_CHECK
    ]
    assert [step.output_summary for step in format_steps] == [
        "needs_retry",
        "format_passed",
    ]
    assert format_steps[0].metadata["reasons"] == ["compact_ordered_list"]


def test_no_context_format_check_retries_compact_numbered_list_after_intro():
    prompts = []
    answers = [
        "Here are the steps:\n1. install dependencies. 2. run tests.",
        "Here are the steps:\n1. install dependencies.\n2. run tests.",
    ]

    def invoke(prompt):
        prompts.append(prompt)
        return answers[len(prompts) - 1]

    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.llm = SimpleNamespace(invoke=invoke, last_thinking=None)

    result = qa.query_with_trace(
        "Give me steps.",
        session_id="format_intro_plain_list",
        context_provider="none",
    )

    assert result.answer == answers[1]
    assert len(prompts) == 2
    assert "compact_ordered_list" in prompts[1]
    format_steps = [
        step
        for step in result.loop_report.run.steps
        if step.phase == LoopPhase.FORMAT_CHECK
    ]
    assert [step.output_summary for step in format_steps] == [
        "needs_retry",
        "format_passed",
    ]
    assert format_steps[0].metadata["reasons"] == ["compact_ordered_list"]


def test_no_context_format_check_does_not_retry_version_prose():
    cases = [
        ("What changed in Java 8?", "Use Java 8. It introduced lambdas."),
        (
            "Show literal step text.",
            "Keep `1. install dependencies. 2. run tests.` inline.",
        ),
    ]

    for question, answer in cases:
        qa = DocumentQA(fast_mode=True, llm_backend="mock")
        qa.llm = SimpleNamespace(
            invoke=lambda _prompt, value=answer: value,
            last_thinking=None,
        )

        result = qa.query_with_trace(question, context_provider="none")

        assert result.answer == answer
        assert LoopPhase.RETRY not in [
            step.phase for step in result.loop_report.run.steps
        ]
        format_step = next(
            step
            for step in result.loop_report.run.steps
            if step.phase == LoopPhase.FORMAT_CHECK
        )
        assert format_step.output_summary == "format_passed"


def test_no_context_format_check_sanitizes_internal_label_after_retry():
    prompts = []
    bad_answer = (
        "- **Answer:** I do not have a specific name. This response is based "
        "on model knowledge and is marked **not_verified**."
    )

    def invoke(prompt):
        prompts.append(prompt)
        return bad_answer

    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.llm = SimpleNamespace(invoke=invoke, last_thinking=None)

    result = qa.query_with_trace(
        "Do you have name?",
        session_id="format_internal_label_sanitized",
        context_provider="none",
    )

    assert result.answer == "I do not have a specific name."
    assert result.trace.error_message is None
    assert result.trace.self_check.outcome == "not_verified"
    assert len(prompts) == 2
    assert result.loop_report.run.final_decision == LoopDecision.NOT_VERIFIED
    sanitize_step = next(
        step
        for step in result.loop_report.run.steps
        if step.name == "Sanitize answer format"
    )
    assert sanitize_step.output_summary == "format_sanitized"
    assert sanitize_step.metadata["reasons"] == ["internal_verification_label"]
    assert "not_verified" not in result.answer
    assert "**Answer:**" not in result.answer


def test_no_context_format_sanitizer_preserves_code_literals():
    prompts = []
    bad_answer = (
        "Use `status == 'not_verified'` in tests. "
        "This response is marked **not_verified**."
    )

    def invoke(prompt):
        prompts.append(prompt)
        return bad_answer

    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.llm = SimpleNamespace(invoke=invoke, last_thinking=None)

    result = qa.query_with_trace(
        "How should I compare the status literal?",
        session_id="format_internal_label_preserve_code",
        context_provider="none",
    )

    assert result.answer == "Use `status == 'not_verified'` in tests."
    assert result.trace.error_message is None
    assert result.trace.self_check.outcome == "not_verified"
    assert result.loop_report.run.final_decision == LoopDecision.NOT_VERIFIED
    assert len(prompts) == 2


def test_no_context_format_check_fails_closed_when_retry_still_bad():
    prompts = []
    bad_answer = (
        "1. **Trigger:** Pressure builds. 2. **Outcome:** The plan escalates. "
        "not_verified"
    )

    def invoke(prompt):
        prompts.append(prompt)
        return bad_answer

    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.llm = SimpleNamespace(invoke=invoke, last_thinking=None)

    result = qa.query_with_trace(
        "Explain the sequence.",
        session_id="format_failed",
        context_provider="none",
    )

    assert result.answer == answer_loop_module.FORMAT_CHECK_FAILURE_ANSWER
    assert result.trace.error_message == "format_check_failed"
    assert result.trace.self_check is None
    assert result.trace.citations == []
    assert len(prompts) == 2

    run = result.loop_report.run
    assert run.final_decision == LoopDecision.ERROR
    assert run.error_message == "format_check_failed"
    assert LoopPhase.VERIFY not in [step.phase for step in run.steps]
    format_steps = [
        step for step in run.steps if step.phase == LoopPhase.FORMAT_CHECK
    ]
    assert [step.output_summary for step in format_steps] == [
        "needs_retry",
        "needs_retry",
    ]
    assert format_steps[1].metadata["retry_attempted"] is True
    error_step = next(step for step in run.steps if step.phase == LoopPhase.ERROR)
    assert error_step.name == "Format check failed"
    assert error_step.metadata["reasons"] == [
        "internal_verification_label",
        "compact_ordered_list",
    ]


def test_no_context_empty_draft_fails_closed():
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.llm = SimpleNamespace(invoke=lambda _prompt: "   ", last_thinking=None)

    result = qa.query_with_trace(
        "Say something.",
        session_id="empty_direct",
        context_provider="none",
    )

    assert result.answer == (
        "The model returned an empty answer. Please try again or check your LLM backend."
    )
    assert result.trace.error_message == "empty_direct_answer"
    assert result.trace.self_check is None
    assert result.trace.citations == []

    run = result.loop_report.run
    assert run.context_provider == "none"
    assert run.final_decision == LoopDecision.ERROR
    assert run.error_message == "empty_direct_answer"
    assert [step.phase for step in run.steps] == [
        LoopPhase.CONTEXT_SELECT,
        LoopPhase.DRAFT,
        LoopPhase.FINAL,
    ]
    draft_step = next(step for step in run.steps if step.phase == LoopPhase.DRAFT)
    assert draft_step.decision == LoopDecision.ERROR
    assert draft_step.error_message == "empty_direct_answer"
    assert qa.loop_session("empty_direct").report_count == 1


def test_query_before_document_fails_on_unavailable_llm_not_missing_document(
    monkeypatch,
):
    qa = DocumentQA(fast_mode=True, llm_backend="ollama")

    def fail_initialize_llm():
        raise RuntimeError("ollama unavailable")

    monkeypatch.setattr(qa, "_initialize_llm", fail_initialize_llm)

    result = qa.query_with_trace(
        "What is this?",
        session_id="no_llm",
        context_provider="none",
    )

    assert result.answer == (
        "Language model could not be initialized. Please check your LLM backend setup."
    )
    assert "upload" not in result.answer.lower()
    assert result.trace.error_message == "llm_initialization_failed"
    assert result.trace.citations == []
    assert result.trace.self_check is None
    assert result.loop_report.run.context_provider == "none"
    assert result.loop_report.run.final_decision == LoopDecision.ERROR
    assert result.loop_report.run.error_message == "llm_initialization_failed"
    assert qa.loop_session("no_llm").report_count == 1


def test_process_text_document_with_mock_llm_and_fake_embeddings(tmp_path):
    document = tmp_path / "phoenix.txt"
    document.write_text(
        "Project Phoenix is a document QA assistant. "
        "The launch date is June 2026.",
        encoding="utf-8",
    )
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.embeddings = FakeEmbeddings()

    qa.process_document(str(document))

    assert qa.current_document_name == "phoenix.txt"
    assert qa.vector_store is not None
    assert qa.retrieval_chain is not None
    assert qa.active_llm_backend == "mock"
    assert qa.query("Which file did I upload?") == "The indexed file is `phoenix.txt`."
    mock_answer = qa.query("What is Project Phoenix?", context_provider="document")
    assert "Project Phoenix" in mock_answer
    assert "[1]" in mock_answer

    status = qa.status()
    assert status.profile_label == "FAST"
    assert status.active_backend == "mock"
    assert status.active_model_label == "MockLLM (explicit demo)"
    assert status.embeddings_device == "cpu"
    assert status.document_name == "phoenix.txt"
    assert status.ready_for_queries is True
    assert status.mock_mode is True
    assert status.processing_report is not None
    assert status.processing_report.attempted_document_name == "phoenix.txt"
    assert status.processing_report.active_document_name == "phoenix.txt"
    assert status.processing_report.success is True
    assert status.processing_report.phase == "complete"
    assert status.processing_report.file_extension == ".txt"
    assert status.processing_report.chunk_count > 0
    assert status.processing_report.truncated is False
    assert status.processing_report.max_chunk_limit == qa.max_document_chunks
    assert status.processing_report.text_encoding_mode == "auto"
    assert status.processing_report.backend == "mock"
    assert status.processing_report.model_label == "MockLLM (explicit demo)"
    assert status.processing_report.error_message is None


def test_query_with_trace_returns_retrieved_citations(tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)

    result = qa.query_with_trace("What is Project Phoenix?", context_provider="document")

    assert "Project Phoenix" in result.answer
    assert "[1]" in result.answer
    assert result.trace.question == "What is Project Phoenix?"
    assert result.trace.document_name == "phoenix.txt"
    assert result.trace.backend == "mock"
    assert result.trace.model_label == "MockLLM (explicit demo)"
    assert result.trace.retrieved_chunk_count > 0
    assert result.trace.error_message is None
    assert result.trace.self_check.outcome == "not_verified"
    assert result.trace.self_check.reasons == [
        "mechanical_checks_passed",
        "verifier_unavailable_mock_backend",
    ]
    assert result.trace.self_check.retry_attempted is False
    assert len(result.trace.citations) >= 1
    citation = result.trace.citations[0]
    assert citation.citation_id == 1
    assert citation.source_name == "phoenix.txt"
    assert citation.page is None
    assert citation.chunk_index == 0
    assert "Project Phoenix" in citation.excerpt
    assert qa.chat_history[-1]["citations"][0]["source_name"] == "phoenix.txt"


def test_query_with_trace_uses_explicit_web_search_context():
    captured_prompts = []

    class FakeWebSearchClient:
        def __init__(self):
            self.calls = []

        def search(self, query, *, max_results=5):
            self.calls.append((query, max_results))
            return [
                web_search_module.WebSearchHit(
                    title="Python",
                    url="https://www.python.org/doc/",
                    snippet="Python is a programming language with readable syntax.",
                )
            ]

    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.web_search_client = FakeWebSearchClient()
    qa.llm = SimpleNamespace(
        invoke=lambda prompt: captured_prompts.append(prompt)
        or "Python is a programming language [1].",
        last_thinking=None,
    )

    result = qa.query_with_trace(
        "What is Python?",
        session_id="web_context",
        context_provider="web",
    )

    assert result.answer == "Python is a programming language [1]."
    assert qa.web_search_client.calls == [("What is Python?", qa.web_search_max_results)]
    assert captured_prompts
    assert "Web search context" in captured_prompts[0]
    assert "Python is a programming language with readable syntax." in captured_prompts[0]
    assert result.trace.retrieved_chunk_count == 1
    assert len(result.trace.citations) == 1
    citation = result.trace.citations[0]
    assert citation.source_name == "Python — https://www.python.org/doc/"
    assert citation.excerpt == "Python is a programming language with readable syntax."
    assert result.trace.self_check.outcome == "not_verified"

    run = result.loop_report.run
    assert run.context_provider == "web"
    assert run.metadata["requested_context_provider"] == "web"
    assert run.metadata["context_provider"] == "web"
    assert run.metadata["context_provider_name"] == "DuckDuckGo web snippets"
    assert "web_search_results" in run.metadata["untrusted_inputs"]
    assert "document_text" not in run.metadata["untrusted_inputs"]
    retrieve_step = next(step for step in run.steps if step.phase == LoopPhase.RETRIEVE)
    assert retrieve_step.metadata["retrieved_chunk_count"] == 1
    assert retrieve_step.metadata["citation_ids"] == [1]


def test_query_with_trace_web_search_works_with_real_mock_llm():
    class FakeWebSearchClient:
        def __init__(self):
            self.calls = []

        def search(self, query, *, max_results=5):
            self.calls.append((query, max_results))
            return [
                web_search_module.WebSearchHit(
                    title="Python",
                    url="https://www.python.org/",
                    snippet="Python is a programming language with readable syntax.",
                )
            ]

    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.web_search_client = FakeWebSearchClient()

    result = qa.query_with_trace("What is Python?", context_provider="web")

    assert result.answer == "Python is a programming language with readable syntax. [1]"
    assert qa.web_search_client.calls == [("What is Python?", qa.web_search_max_results)]
    assert result.trace.error_message is None
    assert result.trace.retrieved_chunk_count == 1
    assert result.trace.self_check.outcome == "not_verified"
    assert result.loop_report.run.context_provider == "web"
    assert result.loop_report.run.final_decision == LoopDecision.NOT_VERIFIED


def test_web_verifier_failure_retries_with_snippet_bounded_answer():
    class FakeWebSearchClient:
        def search(self, query, *, max_results=5):
            return [
                web_search_module.WebSearchHit(
                    title="Loop engineering",
                    url="https://example.test/loop-engineering",
                    snippet=(
                        "Loop engineering means designing AI agent loops rather "
                        "than one-off prompts."
                    ),
                )
            ]

    class RetryVerifierLLM:
        def __init__(self):
            self.calls = []

        def invoke(self, prompt):
            self.calls.append(prompt)
            if prompt.startswith("You are a strict citation verifier"):
                if (
                    "Loop engineering means designing AI agent loops rather than "
                    "one-off prompts [1]."
                ) in prompt:
                    return json.dumps(
                        {"outcome": "supported", "reason": "snippet supports answer"}
                    )
                return json.dumps(
                    {"outcome": "unsupported", "reason": "draft overreaches"}
                )
            if "Web evidence retry instruction:" in prompt:
                return (
                    "Loop engineering means designing AI agent loops rather than "
                    "one-off prompts [1]."
                )
            return (
                "Loop engineering designs AI agent loops, tools, governance, and "
                "production workflows [1]."
            )

    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.web_search_client = FakeWebSearchClient()
    qa.llm = RetryVerifierLLM()
    qa.active_llm_backend = "openai-compatible"
    qa.loaded_model_label = "Fake verifier gateway"

    result = qa.query_with_trace(
        "What is loop engineering?",
        context_provider="web",
    )

    assert result.answer == (
        "Loop engineering means designing AI agent loops rather than "
        "one-off prompts [1]."
    )
    assert result.trace.self_check.outcome == "supported"
    assert result.trace.self_check.retry_attempted is True
    verifier_calls = [
        call for call in qa.llm.calls if call.startswith("You are a strict")
    ]
    assert len(verifier_calls) == 2
    retry_steps = [
        step for step in result.loop_report.run.steps if step.phase == LoopPhase.RETRY
    ]
    assert retry_steps
    assert retry_steps[-1].output_summary == "retrying after web verifier failure"
    assert result.loop_report.run.final_decision == LoopDecision.SUPPORTED


@pytest.mark.parametrize("verifier_outcome", ["unsupported", "insufficient"])
def test_web_verifier_retry_still_fails_closed_when_retry_fails(verifier_outcome):
    class FakeWebSearchClient:
        def search(self, query, *, max_results=5):
            return [
                web_search_module.WebSearchHit(
                    title="Loop engineering",
                    url="https://example.test/loop-engineering",
                    snippet=(
                        "Loop engineering means designing AI agent loops rather "
                        "than one-off prompts."
                    ),
                )
            ]

    class StillFailingVerifierLLM:
        def __init__(self):
            self.calls = []
            self.last_thinking = None

        def invoke(self, prompt):
            self.calls.append(prompt)
            if prompt.startswith("You are a strict citation verifier"):
                return json.dumps(
                    {"outcome": verifier_outcome, "reason": "still not supported"}
                )
            self.last_thinking = "SECRET_WEB_DRAFT_THINKING"
            return "Loop engineering also guarantees production autonomy [1]."

    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.web_search_client = FakeWebSearchClient()
    qa.llm = StillFailingVerifierLLM()
    qa.active_llm_backend = "openai-compatible"
    qa.loaded_model_label = "Fake verifier gateway"

    result = qa.query_with_trace(
        "What is loop engineering?",
        context_provider="web",
    )

    assert result.answer == answer_loop_module.SELF_CHECK_REFUSAL_ANSWER
    assert result.trace.self_check.outcome == "needs_refusal"
    assert result.trace.self_check.reasons == [f"llm_verifier_{verifier_outcome}"]
    assert result.trace.self_check.retry_attempted is True
    verifier_calls = [
        call for call in qa.llm.calls if call.startswith("You are a strict")
    ]
    assert len(verifier_calls) == 2
    retry_steps = [
        step for step in result.loop_report.run.steps if step.phase == LoopPhase.RETRY
    ]
    assert len(retry_steps) == 1
    assert result.loop_report.run.final_decision == LoopDecision.REFUSE
    public_payload = web_contract_module.query_response_dict(result)
    public_json = json.dumps(public_payload)
    assert "SECRET_WEB_DRAFT_THINKING" not in public_json
    assert "production autonomy" not in public_json


def test_smart_web_verifier_failure_falls_back_to_unverified_direct_answer():
    class FakeWebSearchClient:
        def search(self, query, *, max_results=5):
            return [
                web_search_module.WebSearchHit(
                    title="Loop engineering",
                    url="https://example.test/loop-engineering",
                    snippet="Loop engineering is discussed in AI agent workflows.",
                )
            ]

    class FallbackLLM:
        last_thinking = None

        def __init__(self):
            self.calls = []

        def invoke(self, prompt):
            self.calls.append(prompt)
            if prompt.startswith("You are a strict citation verifier"):
                return json.dumps(
                    {"outcome": "insufficient", "reason": "snippet too thin"}
                )
            if prompt.startswith("You are AI Loop Engine running without"):
                return (
                    "Loop engineering means designing and inspecting the repeated "
                    "AI workflow around context, drafting, checking, retrying, and "
                    "refusing when needed."
                )
            return "Loop engineering is a complete agent architecture pattern [1]."

    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.web_search_client = FakeWebSearchClient()
    qa.llm = FallbackLLM()
    qa.active_llm_backend = "openai-compatible"
    qa.loaded_model_label = "Fake verifier gateway"

    result = qa.query_with_trace(
        "What is loop engineering when talking about agents?",
        session_id="smart_web_fallback",
    )

    assert result.answer.startswith("Loop engineering means designing")
    assert result.trace.citations == []
    assert result.trace.self_check.outcome == "not_verified"
    assert result.trace.self_check.reasons == [
        "no_context_provider",
        "verifier_requires_prompt_evidence",
        "smart_web_evidence_fallback",
    ]
    assert result.loop_report.run.context_provider == "none"
    assert result.loop_report.run.final_decision == LoopDecision.NOT_VERIFIED
    assert result.loop_report.run.metadata["attempted_context_provider"] == "web"
    assert result.loop_report.run.metadata["evidence_fallback"] is True
    assert result.loop_report.run.metadata["evidence_fallback_reason"] == (
        "web_evidence_not_verified"
    )
    response_payload = web_contract_module.query_response_dict(result)
    assert response_payload["summary"]["context_provider"] == "none"
    assert response_payload["summary"]["attempted_context_provider"] == "web"
    assert response_payload["summary"]["evidence_fallback"] is True
    assert response_payload["summary"]["evidence_fallback_reason"] == (
        "web_evidence_not_verified"
    )
    fallback_step = next(
        step
        for step in result.loop_report.run.steps
        if step.name == "Fallback to direct model knowledge"
    )
    assert fallback_step.metadata["source_self_check_outcome"] == "needs_refusal"
    assert fallback_step.metadata["source_self_check_reasons"] == [
        "llm_verifier_insufficient"
    ]
    assert any("Web evidence retry instruction:" in call for call in qa.llm.calls)


def test_smart_web_mechanical_failure_does_not_fallback_to_direct_answer():
    class FakeWebSearchClient:
        def search(self, query, *, max_results=5):
            return [
                web_search_module.WebSearchHit(
                    title="Loop engineering",
                    url="https://example.test/loop-engineering",
                    snippet=(
                        "Loop engineering means designing AI agent loops rather "
                        "than one-off prompts."
                    ),
                )
            ]

    class MissingCitationLLM:
        last_thinking = None

        def invoke(self, prompt):
            if prompt.startswith("You are AI Loop Engine running without"):
                raise AssertionError(
                    "mechanical citation failure must not fallback to direct answer"
                )
            return "Loop engineering means designing AI agent loops."

    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.web_search_client = FakeWebSearchClient()
    qa.llm = MissingCitationLLM()
    qa.active_llm_backend = "openai-compatible"
    qa.loaded_model_label = "Fake verifier gateway"

    result = qa.query_with_trace(
        "What is loop engineering?",
        session_id="smart_web_mechanical_failure",
    )

    assert result.answer == answer_loop_module.SELF_CHECK_REFUSAL_ANSWER
    assert result.trace.self_check.outcome == "needs_refusal"
    assert "missing_inline_citation" in result.trace.self_check.reasons
    assert result.loop_report.run.context_provider == "web"
    assert result.loop_report.run.final_decision == LoopDecision.REFUSE
    assert all(
        step.name != "Fallback to direct model knowledge"
        for step in result.loop_report.run.steps
    )


def test_smart_web_verifier_failure_with_broken_direct_fallback_returns_safe_error(
    caplog,
):
    class FakeWebSearchClient:
        def search(self, query, *, max_results=5):
            return [
                web_search_module.WebSearchHit(
                    title="Loop engineering",
                    url="https://example.test/loop-engineering",
                    snippet="Loop engineering is discussed in AI agent workflows.",
                )
            ]

    class BrokenFallbackLLM:
        last_thinking = None

        def __init__(self):
            self.verifier_calls = 0

        def invoke(self, prompt):
            if prompt.startswith("You are a strict citation verifier"):
                self.verifier_calls += 1
                return json.dumps(
                    {"outcome": "insufficient", "reason": "snippet too thin"}
                )
            if prompt.startswith("You are AI Loop Engine running without"):
                raise RuntimeError("DIRECT_MODEL_SECRET_FAILURE")
            return "SECRET_REJECTED_WEB_DRAFT is not supported enough [1]."

    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.web_search_client = FakeWebSearchClient()
    qa.llm = BrokenFallbackLLM()
    qa.active_llm_backend = "openai-compatible"
    qa.loaded_model_label = "Fake verifier gateway"
    caplog.set_level(logging.WARNING, logger=ai_loop_runtime_module.LOGGER.name)

    result = qa.query_with_trace(
        "What is loop engineering when talking about agents?",
        session_id="smart_web_broken_verifier_fallback",
    )
    public_payload = web_contract_module.query_response_dict(result)
    public_json = json.dumps(public_payload)

    assert result.answer == (
        "I hit an internal error while processing your question. Please try again."
    )
    assert result.trace.error_message == "direct_fallback_failed"
    assert result.trace.citations == []
    assert result.trace.self_check is None
    assert result.loop_report.run.final_decision == LoopDecision.ERROR
    assert result.loop_report.run.error_message == "direct_fallback_failed"
    assert result.loop_report.run.context_provider == "none"
    assert result.loop_report.run.metadata["context_provider"] == "none"
    assert result.loop_report.run.metadata["attempted_context_provider"] == "web"
    assert result.loop_report.run.metadata["evidence_fallback"] is True
    assert result.loop_report.run.metadata["fallback_error"] == "direct_fallback_failed"
    assert public_payload["summary"]["context_provider"] == "none"
    assert public_payload["summary"]["attempted_context_provider"] == "web"
    assert any(
        step.name == "Smart Evidence fallback"
        and step.error_message == "direct_fallback_failed"
        and step.metadata["source_self_check_reasons"] == [
            "llm_verifier_insufficient"
        ]
        for step in result.loop_report.run.steps
    )
    assert all(
        step.output_summary == "[redacted: failed evidence fallback]"
        for step in result.loop_report.run.steps
        if step.phase == LoopPhase.DRAFT
    )
    assert "DIRECT_MODEL_SECRET_FAILURE" not in public_json
    assert "SECRET_REJECTED_WEB_DRAFT" not in public_json
    assert "snippet too thin" not in public_json
    assert "DIRECT_MODEL_SECRET_FAILURE" not in caplog.text


def test_explicit_web_verifier_failure_does_not_fallback_to_direct_answer():
    class FakeWebSearchClient:
        def search(self, query, *, max_results=5):
            return [
                web_search_module.WebSearchHit(
                    title="Loop engineering",
                    url="https://example.test/loop-engineering",
                    snippet="Loop engineering is discussed in AI agent workflows.",
                )
            ]

    class RefusingVerifierLLM:
        last_thinking = None

        def invoke(self, prompt):
            if prompt.startswith("You are a strict citation verifier"):
                return json.dumps(
                    {"outcome": "insufficient", "reason": "snippet too thin"}
                )
            if prompt.startswith("You are AI Loop Engine running without"):
                raise AssertionError("explicit web must not fallback to direct answer")
            return "Loop engineering is a complete agent architecture pattern [1]."

    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.web_search_client = FakeWebSearchClient()
    qa.llm = RefusingVerifierLLM()
    qa.active_llm_backend = "openai-compatible"
    qa.loaded_model_label = "Fake verifier gateway"

    result = qa.query_with_trace(
        "What is loop engineering when talking about agents?",
        context_provider="web",
    )

    assert result.answer == answer_loop_module.SELF_CHECK_REFUSAL_ANSWER
    assert result.trace.self_check.outcome == "needs_refusal"
    assert result.loop_report.run.context_provider == "web"
    assert result.loop_report.run.final_decision == LoopDecision.REFUSE
    assert all(
        step.name != "Fallback to direct model knowledge"
        for step in result.loop_report.run.steps
    )


def test_smart_web_search_failure_falls_back_to_unverified_direct_answer():
    class FailingWebSearchClient:
        def search(self, query, *, max_results=5):
            raise web_search_module.WebSearchError("provider_down")

    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.web_search_client = FailingWebSearchClient()
    qa.llm = SimpleNamespace(
        invoke=lambda prompt: (
            "I can explain loop engineering from model knowledge, but this "
            "answer is not verified by web evidence."
            if prompt.startswith("You are AI Loop Engine running without")
            else "unexpected"
        ),
        last_thinking=None,
    )

    result = qa.query_with_trace(
        "What is loop engineering when talking about agents?",
        session_id="smart_web_error_fallback",
    )

    assert result.answer.startswith("I can explain loop engineering")
    assert result.trace.error_message is None
    assert result.trace.self_check.outcome == "not_verified"
    assert result.loop_report.run.context_provider == "none"
    assert result.loop_report.run.final_decision == LoopDecision.NOT_VERIFIED
    assert result.loop_report.run.metadata["attempted_context_provider"] == "web"
    assert result.loop_report.run.metadata["evidence_fallback_reason"] == (
        "web_search_failed"
    )
    assert any(
        step.name == "Web search retrieval" and step.decision == LoopDecision.ERROR
        for step in result.loop_report.run.steps
    )
    assert any(
        step.name == "Fallback to direct model knowledge"
        for step in result.loop_report.run.steps
    )


def test_smart_web_search_failure_with_broken_direct_fallback_returns_safe_error(
    caplog,
):
    class FailingWebSearchClient:
        def search(self, query, *, max_results=5):
            raise web_search_module.WebSearchError("provider_down")

    class BrokenFallbackLLM:
        last_thinking = None

        def invoke(self, prompt):
            if prompt.startswith("You are AI Loop Engine running without"):
                raise RuntimeError("DIRECT_MODEL_SECRET_FAILURE")
            return "unexpected"

    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.web_search_client = FailingWebSearchClient()
    qa.llm = BrokenFallbackLLM()
    caplog.set_level(logging.WARNING, logger=ai_loop_runtime_module.LOGGER.name)

    result = qa.query_with_trace(
        "What is loop engineering when talking about agents?",
        session_id="smart_web_broken_fallback",
    )
    public_payload = web_contract_module.query_response_dict(result)
    public_json = json.dumps(public_payload)

    assert result.answer == (
        "I hit an internal error while processing your question. Please try again."
    )
    assert result.trace.error_message == "direct_fallback_failed"
    assert result.trace.citations == []
    assert result.trace.self_check is None
    assert result.loop_report.run.final_decision == LoopDecision.ERROR
    assert result.loop_report.run.error_message == "direct_fallback_failed"
    assert result.loop_report.run.context_provider == "none"
    assert result.loop_report.run.metadata["context_provider"] == "none"
    assert result.loop_report.run.metadata["attempted_context_provider"] == "web"
    assert result.loop_report.run.metadata["evidence_fallback"] is True
    assert result.loop_report.run.metadata["fallback_error"] == "direct_fallback_failed"
    assert public_payload["summary"]["context_provider"] == "none"
    assert public_payload["summary"]["attempted_context_provider"] == "web"
    assert any(
        step.name == "Smart Evidence fallback"
        and step.error_message == "direct_fallback_failed"
        for step in result.loop_report.run.steps
    )
    assert "DIRECT_MODEL_SECRET_FAILURE" not in public_json
    assert "provider_down" not in public_json
    assert "DIRECT_MODEL_SECRET_FAILURE" not in caplog.text


@pytest.mark.parametrize(
    ("trigger", "fallback_answer", "expected_error", "secret"),
    [
        (
            "verifier_failure",
            "",
            "empty_direct_answer",
            "SECRET_REJECTED_WEB_DRAFT",
        ),
        (
            "web_search_failure",
            "",
            "empty_direct_answer",
            "SECRET_EMPTY_FALLBACK_THINKING",
        ),
        (
            "verifier_failure",
            "1. SECRET_FALLBACK_FORMAT_DRAFT. 2. run tests.",
            "format_check_failed",
            "SECRET_FALLBACK_FORMAT_DRAFT",
        ),
        (
            "web_search_failure",
            "1. SECRET_PROVIDER_FORMAT_DRAFT. 2. run tests.",
            "format_check_failed",
            "SECRET_PROVIDER_FORMAT_DRAFT",
        ),
    ],
)
def test_smart_web_terminal_fallback_errors_redact_draft_outputs(
    trigger,
    fallback_answer,
    expected_error,
    secret,
):
    class FakeWebSearchClient:
        def search(self, query, *, max_results=5):
            if trigger == "web_search_failure":
                raise web_search_module.WebSearchError("provider_down")
            return [
                web_search_module.WebSearchHit(
                    title="Loop engineering",
                    url="https://example.test/loop-engineering",
                    snippet="Loop engineering is discussed in AI agent workflows.",
                )
            ]

    class TerminalFallbackErrorLLM:
        def __init__(self):
            self.last_thinking = None

        def invoke(self, prompt):
            if prompt.startswith("You are a strict citation verifier"):
                return json.dumps(
                    {"outcome": "insufficient", "reason": "snippet too thin"}
                )
            if prompt.startswith("You are AI Loop Engine running without"):
                if expected_error == "empty_direct_answer":
                    self.last_thinking = "SECRET_EMPTY_FALLBACK_THINKING"
                return fallback_answer
            return "SECRET_REJECTED_WEB_DRAFT is not supported enough [1]."

    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.web_search_client = FakeWebSearchClient()
    qa.llm = TerminalFallbackErrorLLM()
    qa.active_llm_backend = "openai-compatible"
    qa.loaded_model_label = "Fake verifier gateway"

    result = qa.query_with_trace(
        "What is loop engineering when talking about agents?",
        session_id=f"smart_web_{trigger}_{expected_error}",
    )
    public_payload = web_contract_module.query_response_dict(result)
    public_json = json.dumps(public_payload)

    assert result.loop_report.run.final_decision == LoopDecision.ERROR
    assert result.loop_report.run.error_message == expected_error
    assert result.trace.error_message == expected_error
    assert result.trace.citations == []
    assert result.trace.self_check is None
    assert result.loop_report.run.metadata["attempted_context_provider"] == "web"
    assert result.loop_report.run.metadata["evidence_fallback"] is True
    assert all(
        step.output_summary == "[redacted: failed evidence fallback]"
        for step in result.loop_report.run.steps
        if step.phase == LoopPhase.DRAFT
    )
    assert secret not in public_json
    assert "SECRET_REJECTED_WEB_DRAFT" not in public_json
    assert "SECRET_EMPTY_FALLBACK_THINKING" not in public_json
    assert "provider_down" not in public_json
    assert "snippet too thin" not in public_json


def test_web_search_query_does_not_send_thread_memory_to_provider():
    class RecordingWebSearchClient:
        def __init__(self):
            self.queries = []

        def search(self, query, *, max_results=5):
            self.queries.append(query)
            return [
                web_search_module.WebSearchHit(
                    title="Python",
                    url="https://www.python.org/",
                    snippet="Python is a programming language.",
                )
            ]

    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.web_search_client = RecordingWebSearchClient()
    qa.llm = SimpleNamespace(
        invoke=lambda _prompt: "Python is a programming language [1].",
        last_thinking=None,
    )

    result = qa.query_with_trace(
        "What is Python?",
        context_provider="web",
        conversation_history=[
            {"role": "user", "content": "SECRET_CONVO should stay local."},
            {"role": "assistant", "content": "Previous answer."},
        ],
        semantic_memory=[
            {"role": "user", "content": "SECRET_MEMORY should stay local."},
        ],
        semantic_memory_status="retrieved",
    )

    assert result.answer == "Python is a programming language [1]."
    assert qa.web_search_client.queries == ["What is Python?"]
    assert "SECRET_CONVO" not in qa.web_search_client.queries[0]
    assert "SECRET_MEMORY" not in qa.web_search_client.queries[0]


def test_web_search_trace_omits_unrelated_active_document(tmp_path):
    document = tmp_path / "old-doc.txt"
    document.write_text("Project Phoenix is local document evidence.", encoding="utf-8")

    class FakeWebSearchClient:
        def search(self, query, *, max_results=5):
            return [
                web_search_module.WebSearchHit(
                    title="Python",
                    url="https://www.python.org/",
                    snippet="Python is a programming language.",
                )
            ]

    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    assert qa.status().document_name == "old-doc.txt"
    qa.web_search_client = FakeWebSearchClient()
    qa.llm = SimpleNamespace(
        invoke=lambda _prompt: "Python is a programming language [1].",
        last_thinking=None,
    )

    result = qa.query_with_trace("What is Python?", context_provider="web")
    public_report = result.loop_report.to_public_dict()
    answer_payload = web_contract_module.answer_trace_dict(result)
    summary_payload = web_contract_module.loop_summary_dict(result)

    assert result.loop_report.run.context_provider == "web"
    assert result.trace.document_name is None
    assert result.loop_report.run.metadata["document_name"] is None
    context_step = next(
        step for step in result.loop_report.run.steps if step.phase == LoopPhase.CONTEXT_SELECT
    )
    assert context_step.metadata["document_name"] is None
    assert answer_payload["document"] is None
    assert summary_payload["document"] is None
    assert "old-doc.txt" not in json.dumps(public_report)


def test_smart_context_provider_uses_web_search_without_document():
    class FakeWebSearchClient:
        def __init__(self):
            self.calls = []

        def search(self, query, *, max_results=5):
            self.calls.append((query, max_results))
            return [
                web_search_module.WebSearchHit(
                    title="Python",
                    url="https://www.python.org/",
                    snippet="Python is a programming language.",
                )
            ]

    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.web_search_client = FakeWebSearchClient()

    result = qa.query_with_trace("What is Python?", session_id="smart_web")

    assert qa.web_search_client.calls == [("What is Python?", qa.web_search_max_results)]
    assert result.answer == "Python is a programming language. [1]"
    assert result.loop_report.run.context_provider == "web"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"


def test_legacy_auto_context_provider_aliases_smart_web_without_document():
    class FakeWebSearchClient:
        def search(self, query, *, max_results=5):
            return [
                web_search_module.WebSearchHit(
                    title="Python",
                    url="https://www.python.org/",
                    snippet="Python is a programming language.",
                )
            ]

    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.web_search_client = FakeWebSearchClient()

    result = qa.query_with_trace(
        "What is Python?",
        session_id="auto_web",
        context_provider="auto",
    )

    assert result.loop_report.run.context_provider == "web"
    assert result.loop_report.run.metadata["requested_context_provider"] == "auto"


def test_smart_context_provider_uses_active_file_when_available(tmp_path):
    class FailingWebSearchClient:
        def search(self, query, *, max_results=5):
            raise AssertionError("smart context should use the indexed file")

    document = tmp_path / "phoenix.txt"
    document.write_text(
        "Project Phoenix is an indexed file about a June 2026 launch.",
        encoding="utf-8",
    )
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    qa.web_search_client = FailingWebSearchClient()

    result = qa.query_with_trace(
        "What does this file say about Project Phoenix?",
        session_id="smart_file",
    )

    assert result.loop_report.run.context_provider == "document"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.loop_report.run.metadata["document_name"] == "phoenix.txt"


def test_smart_context_provider_uses_active_file_for_short_fact_question(tmp_path):
    class FailingWebSearchClient:
        def search(self, query, *, max_results=5):
            raise AssertionError("doc fact question must not be sent to web search")

    document = tmp_path / "strategy.txt"
    document.write_text(
        "The launch date is June 2026. The owner is Nadia.",
        encoding="utf-8",
    )
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    qa.web_search_client = FailingWebSearchClient()

    result = qa.query_with_trace(
        "What is the launch date?",
        session_id="smart_short_file_fact",
    )

    assert "June 2026" in result.answer
    assert result.loop_report.run.context_provider == "document"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.loop_report.run.metadata["document_name"] == "strategy.txt"
    assert result.trace.document_name == "strategy.txt"
    assert result.trace.citations


def test_smart_context_provider_uses_active_file_for_late_chunk_fact_question(tmp_path):
    class FailingWebSearchClient:
        def search(self, query, *, max_results=5):
            raise AssertionError("late doc fact question must not be sent to web search")

    class FactEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return [self._embed(text) for text in texts]

        def embed_query(self, text):
            return self._embed(text)

        def _embed(self, text):
            lower = text.lower()
            return [
                float("launch" in lower),
                float("date" in lower),
                float("june" in lower),
            ]

    filler_sections = [
        (
            f"Background section {index}. "
            "This unrelated planning note discusses staffing, onboarding, and tooling."
        )
        for index in range(20)
    ]
    document = tmp_path / "strategy.txt"
    document.write_text(
        "\n\n".join(filler_sections)
        + "\n\nThe launch date is June 2026. The owner is Nadia.",
        encoding="utf-8",
    )
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.embeddings = FactEmbeddings()
    qa.profile["splitter_chunk_size"] = 90
    qa.profile["splitter_chunk_overlap"] = 0
    qa.process_document(str(document))
    qa.web_search_client = FailingWebSearchClient()

    assert len(qa.vector_store.documents) > 12

    result = qa.query_with_trace(
        "What is the launch date?",
        session_id="smart_late_file_fact",
    )

    assert "June 2026" in result.answer
    assert result.loop_report.run.context_provider == "document"
    assert result.loop_report.run.metadata["document_name"] == "strategy.txt"
    assert result.trace.citations


def test_smart_context_provider_ignores_active_file_without_file_intent(tmp_path):
    class FailingWebSearchClient:
        def search(self, query, *, max_results=5):
            raise AssertionError("non-lookup local task must not be sent to web search")

    document = tmp_path / "resume.txt"
    document.write_text("This resume is for an AI engineer in Hong Kong.", encoding="utf-8")
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    qa.web_search_client = FailingWebSearchClient()
    qa.llm = SimpleNamespace(
        invoke=lambda _prompt: "Here is a short joke.",
        last_thinking=None,
    )

    result = qa.query_with_trace("Tell me a joke.", session_id="smart_joke")

    assert result.answer == "Here is a short joke."
    assert result.loop_report.run.context_provider == "none"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.loop_report.run.metadata["document_name"] is None
    assert result.trace.citations == []


def test_smart_context_provider_uses_web_for_general_lookup_with_active_file(tmp_path):
    class FakeWebSearchClient:
        def __init__(self):
            self.calls = []

        def search(self, query, *, max_results=5):
            self.calls.append((query, max_results))
            return [
                web_search_module.WebSearchHit(
                    title="Jackie Chan",
                    url="https://example.com/jackie-chan",
                    snippet="Jackie Chan is an actor and martial artist.",
                )
            ]

    document = tmp_path / "resume.txt"
    document.write_text("This resume is for an AI engineer in Hong Kong.", encoding="utf-8")
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    qa.web_search_client = FakeWebSearchClient()

    result = qa.query_with_trace("Who is Jackie Chan?", session_id="smart_general_web")

    assert qa.web_search_client.calls == [
        ("Who is Jackie Chan?", qa.web_search_max_results)
    ]
    assert result.loop_report.run.context_provider == "web"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.loop_report.run.metadata["document_name"] is None


def test_smart_context_provider_uses_file_for_local_entity_lookup(tmp_path):
    class FailingWebSearchClient:
        def search(self, query, *, max_results=5):
            raise AssertionError("local file entity must not be sent to web search")

    document = tmp_path / "phoenix.txt"
    document.write_text(
        "Project Phoenix is a private launch plan indexed from a local file.",
        encoding="utf-8",
    )
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    qa.web_search_client = FailingWebSearchClient()

    result = qa.query_with_trace(
        "What is Project Phoenix?",
        session_id="smart_local_entity_file",
    )

    assert result.loop_report.run.context_provider == "document"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.loop_report.run.metadata["document_name"] == "phoenix.txt"


def test_smart_context_provider_private_entity_question_uses_file(tmp_path):
    class FailingWebSearchClient:
        def search(self, query, *, max_results=5):
            raise AssertionError("private local entity question must not use web search")

    document = tmp_path / "roadmap.txt"
    document.write_text(
        "Project Phoenix is a private launch plan indexed from a local file.",
        encoding="utf-8",
    )
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    qa.web_search_client = FailingWebSearchClient()

    result = qa.query_with_trace(
        "What is private Project Phoenix?",
        session_id="smart_private_entity_file",
    )

    assert result.loop_report.run.context_provider == "document"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.loop_report.run.metadata["document_name"] == "roadmap.txt"


def test_smart_context_provider_private_marker_before_entity_uses_file(tmp_path):
    class FailingWebSearchClient:
        def search(self, query, *, max_results=5):
            raise AssertionError("private marker before local entity must not use web")

    document = tmp_path / "roadmap.txt"
    document.write_text(
        "The private Project Phoenix is a launch plan.",
        encoding="utf-8",
    )
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    qa.web_search_client = FailingWebSearchClient()

    result = qa.query_with_trace(
        "What is Project Phoenix?",
        session_id="smart_private_marker_before_entity",
    )

    assert result.loop_report.run.context_provider == "document"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.loop_report.run.metadata["document_name"] == "roadmap.txt"


@pytest.mark.parametrize(
    ("document_text", "question"),
    [
        ("This private note mentions Project Phoenix.", "What is Project Phoenix?"),
        ("This private note mentions project Phoenix.", "What is Phoenix?"),
        ("This private note mentions the project Phoenix.", "What is Phoenix?"),
        ("This private note mentions private project Phoenix.", "What is Phoenix?"),
        (
            "This private note mentions the private project Phoenix.",
            "What is Phoenix?",
        ),
        (
            "This private note includes Project Phoenix in the roadmap.",
            "What is Project Phoenix?",
        ),
        ("This private note includes roadmap Nova.", "What is Nova?"),
        (
            "This private note references Ångström Project.",
            "What is Ångström Project?",
        ),
        (
            "This private note references Project X.",
            "What is Project X?",
        ),
        (
            "This private note references Project Phoenix for the vendor roadmap.",
            "What is Project Phoenix?",
        ),
        (
            "This private note references Project Phoenix for comparison in the roadmap.",
            "What is Project Phoenix?",
        ),
        (
            "This internal memo references Project Phoenix for comparison with Atlas.",
            "What is Project Phoenix?",
        ),
        (
            "This private note references Project Phoenix launch.",
            "What is Project Phoenix?",
        ),
        (
            "This private note references Project Phoenix release.",
            "What is Project Phoenix?",
        ),
        (
            "This private note references Project Phoenix's launch.",
            "What is Project Phoenix?",
        ),
        (
            "This private note references Project Phoenix’s launch.",
            "What is Project Phoenix?",
        ),
        (
            "This private note references Project X's launch.",
            "What is Project X?",
        ),
        (
            "This private note references Project Phoenix launches tomorrow.",
            "What is Project Phoenix?",
        ),
        (
            "This private note references Project Phoenix releases tomorrow.",
            "What is Project Phoenix?",
        ),
        (
            "This private note references Project Phoenix launched yesterday.",
            "What is Project Phoenix?",
        ),
        (
            "This private note references Project Phoenix released yesterday.",
            "What is Project Phoenix?",
        ),
        (
            "This private note references Project Phoenix launching soon.",
            "What is Project Phoenix?",
        ),
        (
            "This private note references Project Phoenix releasing soon.",
            "What is Project Phoenix?",
        ),
        (
            "The project is known internally as Project Phoenix.",
            "What is Project Phoenix?",
        ),
        (
            "The project is known internally as Project X.",
            "What is Project X?",
        ),
        (
            "The project is known internally as Project\nPhoenix.",
            "What is Project Phoenix?",
        ),
        (
            "The project is known internally as Project Phoenix for launch.",
            "What is Project Phoenix?",
        ),
        (
            "The project is known internally as Project Phoenix Launch.",
            "What is Project Phoenix?",
        ),
        (
            "The project is known internally as Private Project Phoenix.",
            "What is Project Phoenix?",
        ),
        (
            "The project is known internally as Alpha Project Phoenix.",
            "What is Project Phoenix?",
        ),
        (
            "The project is known internally as Private Alpha Beta Project Phoenix.",
            "What is Project Phoenix?",
        ),
        (
            "The project is known internally as Private Alpha Beta Project Phoenix Launch.",
            "What is Project Phoenix?",
        ),
        (
            "The project is known internally as Project-Phoenix.",
            "What is Project Phoenix?",
        ),
        (
            "The project is known internally as Project - Phoenix.",
            "What is Project Phoenix?",
        ),
        (
            "The project is known internally as Project–Phoenix.",
            "What is Project Phoenix?",
        ),
        (
            "The project is known internally as Project — Phoenix.",
            "What is Project Phoenix?",
        ),
        (
            "The project is known internally as Project—Phoenix.",
            "What is Project Phoenix?",
        ),
        (
            "The project is known internally as Project‑Phoenix.",
            "What is Project Phoenix?",
        ),
        (
            "The project is known internally as Project_Phoenix.",
            "What is Project Phoenix?",
        ),
        (
            "The project is known internally as Project/Phoenix.",
            "What is Project Phoenix?",
        ),
        (
            "The project is known internally as Project: Phoenix.",
            "What is Project Phoenix?",
        ),
        (
            "The project is known internally as Project : Phoenix.",
            "What is Project Phoenix?",
        ),
        (
            "The project is known internally as Project = Phoenix.",
            "What is Project Phoenix?",
        ),
        (
            "The project is known internally as (Project Phoenix).",
            "What is Project Phoenix?",
        ),
        (
            "The project is known internally as [Project Phoenix].",
            "What is Project Phoenix?",
        ),
        (
            "The project is known internally as ‑ Project Phoenix.",
            "What is Project Phoenix?",
        ),
        (
            "The project is known internally as − Project Phoenix.",
            "What is Project Phoenix?",
        ),
        (
            "This private note says aka Project Phoenix.",
            "What is Project Phoenix?",
        ),
        (
            "This private note says aka Project\nPhoenix.",
            "What is Project Phoenix?",
        ),
        (
            "This private note says aka (Project Phoenix).",
            "What is Project Phoenix?",
        ),
        (
            "This private note says aka Project_Phoenix.",
            "What is Project Phoenix?",
        ),
        (
            "This private note says aka Project/Phoenix.",
            "What is Project Phoenix?",
        ),
        (
            "This private note says aka Project–Phoenix.",
            "What is Project Phoenix?",
        ),
        (
            "The internal alias is Project Phoenix.",
            "What is Project Phoenix?",
        ),
        (
            "Internal alias: Project Phoenix.",
            "What is Project Phoenix?",
        ),
        (
            "This private alias is Project X.",
            "What is Project X?",
        ),
        (
            "The local alias is Phoenix.",
            "What is Phoenix?",
        ),
        (
            "This private note says codename Project Phoenix.",
            "What is Project Phoenix?",
        ),
        (
            "This private note says codename [Project Phoenix].",
            "What is Project Phoenix?",
        ),
        (
            "This private note says project named Project Phoenix.",
            "What is Project Phoenix?",
        ),
        (
            "This private note says project named (Project Phoenix).",
            "What is Project Phoenix?",
        ),
        (
            "This private note says project called Project Phoenix.",
            "What is Project Phoenix?",
        ),
        (
            "This private note references Project\nPhoenix for launch.",
            "What is Project Phoenix?",
        ),
        (
            "This private note references Phoenix for launch.",
            "What is Phoenix?",
        ),
        (
            "This private note references Phoenix Launches Tomorrow.",
            "What is Phoenix?",
        ),
        (
            "This private note references Phoenix Released Yesterday.",
            "What is Phoenix?",
        ),
        (
            "This private note references Python internal release.",
            "What is Python internal release?",
        ),
        (
            "This private note mentions Alpha Beta.",
            "What is Alpha Beta?",
        ),
        (
            "This private note references Project-Phoenix for launch.",
            "What is Project Phoenix?",
        ),
        (
            "This private note references Project_Phoenix for launch.",
            "What is Project Phoenix?",
        ),
        (
            "This private note references Project/Phoenix for launch.",
            "What is Project Phoenix?",
        ),
    ],
)
def test_smart_context_provider_private_project_mentions_use_file(
    tmp_path,
    document_text,
    question,
):
    class FailingWebSearchClient:
        def search(self, query, *, max_results=5):
            raise AssertionError("private project mention must not use web search")

    document = tmp_path / "roadmap.txt"
    document.write_text(document_text, encoding="utf-8")
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    qa.web_search_client = FailingWebSearchClient()

    result = qa.query_with_trace(
        question,
        session_id="smart_private_project_mention_file",
    )

    assert result.loop_report.run.context_provider == "document"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.loop_report.run.metadata["document_name"] == "roadmap.txt"


def test_smart_context_provider_private_reference_after_sample_window_uses_file(tmp_path):
    class FailingWebSearchClient:
        def search(self, query, *, max_results=5):
            raise AssertionError("late private reference must not leak to web search")

    filler = " ".join(f"filler{i}" for i in range(3000))
    document = tmp_path / "late-reference.txt"
    document.write_text(
        f"{filler}\n\nThis private note references Project Phoenix for launch.",
        encoding="utf-8",
    )
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    qa.web_search_client = FailingWebSearchClient()

    result = qa.query_with_trace(
        "What is Project Phoenix?",
        session_id="smart_late_private_reference_file",
    )

    assert result.loop_report.run.context_provider == "document"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.loop_report.run.metadata["document_name"] == "late-reference.txt"


@pytest.mark.parametrize(
    "marker_phrase",
    [
        "confidential-note",
        "internal-memo",
        "internal—memo",
        "local-note",
        "private-note",
        "private–note",
        "proprietary-note",
        "secret-note",
        "secret‑note",
        "unreleased-note",
        "uploaded-note",
    ],
)
def test_smart_context_provider_hyphenated_local_markers_use_file(
    tmp_path,
    marker_phrase,
):
    class FailingWebSearchClient:
        def search(self, query, *, max_results=5):
            raise AssertionError("hyphenated local marker must not leak to web search")

    document = tmp_path / "roadmap.txt"
    document.write_text(
        f"This {marker_phrase} references Project Phoenix for launch.",
        encoding="utf-8",
    )
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    qa.web_search_client = FailingWebSearchClient()

    result = qa.query_with_trace(
        "What is Project Phoenix?",
        session_id="smart_hyphenated_marker_file",
    )

    assert result.loop_report.run.context_provider == "document"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.loop_report.run.metadata["document_name"] == "roadmap.txt"


def test_smart_context_provider_uses_file_for_single_token_local_entity(tmp_path):
    class FailingWebSearchClient:
        def search(self, query, *, max_results=5):
            raise AssertionError("single-token local entity must not be sent to web search")

    document = tmp_path / "phoenix.txt"
    document.write_text(
        "Phoenix is a private launch plan indexed from a local file.",
        encoding="utf-8",
    )
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    qa.web_search_client = FailingWebSearchClient()

    result = qa.query_with_trace(
        "What is Phoenix?",
        session_id="smart_single_entity_file",
    )

    assert result.loop_report.run.context_provider == "document"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.loop_report.run.metadata["document_name"] == "phoenix.txt"


def test_smart_context_provider_uses_file_for_unicode_local_entity(tmp_path):
    class FailingWebSearchClient:
        def search(self, query, *, max_results=5):
            raise AssertionError("unicode local entity must not be sent to web search")

    document = tmp_path / "private-plan.txt"
    document.write_text(
        "Ångström Project is a private launch plan indexed from a local file.",
        encoding="utf-8",
    )
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    qa.web_search_client = FailingWebSearchClient()

    result = qa.query_with_trace(
        "What is Ångström Project?",
        session_id="smart_unicode_entity_file",
    )

    assert result.loop_report.run.context_provider == "document"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.loop_report.run.metadata["document_name"] == "private-plan.txt"


def test_smart_context_provider_uses_file_for_lowercase_private_codename(tmp_path):
    class FailingWebSearchClient:
        def search(self, query, *, max_results=5):
            raise AssertionError("private lowercase codename must not be sent to web search")

    document = tmp_path / "roadmap.txt"
    document.write_text(
        "The private codename is phoenix.",
        encoding="utf-8",
    )
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    qa.web_search_client = FailingWebSearchClient()

    result = qa.query_with_trace(
        "What is phoenix?",
        session_id="smart_lowercase_codename_file",
    )

    assert result.loop_report.run.context_provider == "document"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.loop_report.run.metadata["document_name"] == "roadmap.txt"


def test_smart_context_provider_uses_file_for_internal_lowercase_codename(tmp_path):
    class FailingWebSearchClient:
        def search(self, query, *, max_results=5):
            raise AssertionError("internal lowercase codename must not be sent to web search")

    document = tmp_path / "roadmap.txt"
    document.write_text(
        "Internal codename zxq is used for the project.",
        encoding="utf-8",
    )
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    qa.web_search_client = FailingWebSearchClient()

    result = qa.query_with_trace(
        "What is zxq?",
        session_id="smart_internal_codename_file",
    )

    assert result.loop_report.run.context_provider == "document"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.loop_report.run.metadata["document_name"] == "roadmap.txt"


@pytest.mark.parametrize(
    "document_text",
    [
        "Private codename: phoenix.",
        "Private codename - phoenix.",
        "Private codename “phoenix”.",
        "The private codename is known as phoenix.",
        "The private project is known internally as phoenix.",
        "The project is known internally as phoenix.",
    ],
)
def test_smart_context_provider_uses_file_for_punctuated_private_codename(
    tmp_path,
    document_text,
):
    class FailingWebSearchClient:
        def search(self, query, *, max_results=5):
            raise AssertionError("punctuated private codename must not go to web search")

    document = tmp_path / "roadmap.txt"
    document.write_text(document_text, encoding="utf-8")
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    qa.web_search_client = FailingWebSearchClient()

    result = qa.query_with_trace(
        "What is phoenix?",
        session_id="smart_punctuated_codename_file",
    )

    assert result.loop_report.run.context_provider == "document"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.loop_report.run.metadata["document_name"] == "roadmap.txt"


def test_smart_context_provider_bare_indexed_marker_still_uses_web(tmp_path):
    class FakeWebSearchClient:
        def __init__(self):
            self.calls = []

        def search(self, query, *, max_results=5):
            self.calls.append((query, max_results))
            return [
                web_search_module.WebSearchHit(
                    title="Python",
                    url="https://example.com/python",
                    snippet="Python is a programming language.",
                )
            ]

    document = tmp_path / "glossary.txt"
    document.write_text(
        "Python is indexed in this glossary as an example.",
        encoding="utf-8",
    )
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    qa.web_search_client = FakeWebSearchClient()

    result = qa.query_with_trace(
        "What is Python?",
        session_id="smart_bare_indexed_web",
    )

    assert qa.web_search_client.calls == [("What is Python?", qa.web_search_max_results)]
    assert result.loop_report.run.context_provider == "web"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.loop_report.run.metadata["document_name"] is None


def test_smart_context_provider_locally_installed_public_entity_uses_web(tmp_path):
    class FakeWebSearchClient:
        def __init__(self):
            self.calls = []

        def search(self, query, *, max_results=5):
            self.calls.append((query, max_results))
            return [
                web_search_module.WebSearchHit(
                    title="Python",
                    url="https://example.com/python",
                    snippet="Python is a programming language.",
                )
            ]

    document = tmp_path / "notes.txt"
    document.write_text(
        "Python is locally installed on this machine.",
        encoding="utf-8",
    )
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    qa.web_search_client = FakeWebSearchClient()

    result = qa.query_with_trace(
        "What is Python?",
        session_id="smart_locally_installed_web",
    )

    assert qa.web_search_client.calls == [("What is Python?", qa.web_search_max_results)]
    assert result.loop_report.run.context_provider == "web"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.loop_report.run.metadata["document_name"] is None


def test_smart_context_provider_locally_famous_current_entity_uses_web(tmp_path):
    class FakeWebSearchClient:
        def __init__(self):
            self.calls = []

        def search(self, query, *, max_results=5):
            self.calls.append((query, max_results))
            return [
                web_search_module.WebSearchHit(
                    title="Jackie Chan",
                    url="https://example.com/jackie-chan",
                    snippet="Jackie Chan lookup should use web evidence.",
                )
            ]

    document = tmp_path / "notes.txt"
    document.write_text(
        "Jackie Chan is locally famous in this note.",
        encoding="utf-8",
    )
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    qa.web_search_client = FakeWebSearchClient()

    result = qa.query_with_trace(
        "Who is Jackie Chan in 2026?",
        session_id="smart_locally_famous_web",
    )

    assert qa.web_search_client.calls == [
        ("Who is Jackie Chan in 2026?", qa.web_search_max_results)
    ]
    assert result.loop_report.run.context_provider == "web"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.loop_report.run.metadata["document_name"] is None


def test_smart_context_provider_can_prefer_web_for_current_questions_with_file(tmp_path):
    class FakeWebSearchClient:
        def __init__(self):
            self.calls = []

        def search(self, query, *, max_results=5):
            self.calls.append((query, max_results))
            return [
                web_search_module.WebSearchHit(
                    title="Python in 2026",
                    url="https://example.com/python-2026",
                    snippet="Python remains widely used in 2026.",
                )
            ]

    document = tmp_path / "phoenix.txt"
    document.write_text("Project Phoenix is local file evidence.", encoding="utf-8")
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    qa.web_search_client = FakeWebSearchClient()

    result = qa.query_with_trace(
        "Is Python still widely used in 2026?",
        session_id="smart_current_web",
    )

    assert qa.web_search_client.calls == [
        ("Is Python still widely used in 2026?", qa.web_search_max_results)
    ]
    assert result.loop_report.run.context_provider == "web"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.loop_report.run.metadata["document_name"] is None


def test_smart_context_provider_current_pronoun_question_uses_web_with_file(tmp_path):
    class FakeWebSearchClient:
        def __init__(self):
            self.calls = []

        def search(self, query, *, max_results=5):
            self.calls.append((query, max_results))
            return [
                web_search_module.WebSearchHit(
                    title="Local weather",
                    url="https://example.com/weather",
                    snippet="Current weather comes from web evidence.",
                )
            ]

    document = tmp_path / "resume.txt"
    document.write_text("This resume is for an AI engineer in Hong Kong.", encoding="utf-8")
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    qa.web_search_client = FakeWebSearchClient()

    result = qa.query_with_trace(
        "Is it raining today?",
        session_id="smart_current_pronoun_web",
    )

    assert qa.web_search_client.calls == [
        ("Is it raining today?", qa.web_search_max_results)
    ]
    assert result.loop_report.run.context_provider == "web"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.loop_report.run.metadata["document_name"] is None


def test_smart_context_provider_lookup_overlap_uses_web_with_file(tmp_path):
    class FakeWebSearchClient:
        def __init__(self):
            self.calls = []

        def search(self, query, *, max_results=5):
            self.calls.append((query, max_results))
            return [
                web_search_module.WebSearchHit(
                    title="Machine learning",
                    url="https://example.com/machine-learning",
                    snippet="Machine learning is a field of artificial intelligence.",
                )
            ]

    document = tmp_path / "resume.txt"
    document.write_text(
        "This resume mentions machine learning, recommender systems, and Python.",
        encoding="utf-8",
    )
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    qa.web_search_client = FakeWebSearchClient()

    result = qa.query_with_trace(
        "What is machine learning?",
        session_id="smart_lookup_overlap_web",
    )

    assert qa.web_search_client.calls == [
        ("What is machine learning?", qa.web_search_max_results)
    ]
    assert result.loop_report.run.context_provider == "web"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.loop_report.run.metadata["document_name"] is None


def test_smart_context_provider_public_entity_mention_still_uses_web(tmp_path):
    class FakeWebSearchClient:
        def __init__(self):
            self.calls = []

        def search(self, query, *, max_results=5):
            self.calls.append((query, max_results))
            return [
                web_search_module.WebSearchHit(
                    title="Python",
                    url="https://example.com/python",
                    snippet="Python is a programming language.",
                )
            ]

    document = tmp_path / "resume.txt"
    document.write_text(
        "This resume says the candidate uses Python and JavaScript.",
        encoding="utf-8",
    )
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    qa.web_search_client = FakeWebSearchClient()

    result = qa.query_with_trace("What is Python?", session_id="smart_public_python")

    assert qa.web_search_client.calls == [("What is Python?", qa.web_search_max_results)]
    assert result.loop_report.run.context_provider == "web"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.loop_report.run.metadata["document_name"] is None


def test_smart_context_provider_current_public_entity_mention_still_uses_web(tmp_path):
    class FakeWebSearchClient:
        def __init__(self):
            self.calls = []

        def search(self, query, *, max_results=5):
            self.calls.append((query, max_results))
            return [
                web_search_module.WebSearchHit(
                    title="OpenAI news",
                    url="https://example.com/openai-news",
                    snippet="OpenAI news should come from web evidence.",
                )
            ]

    document = tmp_path / "examples.txt"
    document.write_text(
        "These notes mention Python, OpenAI, and Jackie Chan as examples.",
        encoding="utf-8",
    )
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    qa.web_search_client = FakeWebSearchClient()

    result = qa.query_with_trace(
        "What is the latest OpenAI news?",
        session_id="smart_public_openai_news",
    )

    assert qa.web_search_client.calls == [
        ("What is the latest OpenAI news?", qa.web_search_max_results)
    ]
    assert result.loop_report.run.context_provider == "web"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.loop_report.run.metadata["document_name"] is None


def test_smart_context_provider_current_public_fact_lookup_still_uses_web(tmp_path):
    class FakeWebSearchClient:
        def __init__(self):
            self.calls = []

        def search(self, query, *, max_results=5):
            self.calls.append((query, max_results))
            return [
                web_search_module.WebSearchHit(
                    title="Python release date",
                    url="https://example.com/python-release-date",
                    snippet="The current Python release date should come from web evidence.",
                )
            ]

    document = tmp_path / "notes.txt"
    document.write_text(
        "These notes mention Python and an old release date as examples.",
        encoding="utf-8",
    )
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    qa.web_search_client = FakeWebSearchClient()

    result = qa.query_with_trace(
        "What is the current Python release date?",
        session_id="smart_current_public_fact_web",
    )

    assert qa.web_search_client.calls == [
        ("What is the current Python release date?", qa.web_search_max_results)
    ]
    assert result.loop_report.run.context_provider == "web"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.loop_report.run.metadata["document_name"] is None


@pytest.mark.parametrize(
    ("document_text", "question", "title"),
    [
        ("This private note mentions Python.", "What is Python?", "Python"),
        (
            "This internal memo mentions OpenAI as a vendor.",
            "What is OpenAI?",
            "OpenAI",
        ),
        (
            "This confidential list mentions Jackie Chan as an example.",
            "Who is Jackie Chan?",
            "Jackie Chan",
        ),
        (
            "This private note references Python release tomorrow.",
            "What is Python?",
            "Python",
        ),
        (
            "This private note references OpenAI launch tomorrow.",
            "What is OpenAI?",
            "OpenAI",
        ),
        (
            "This internal memo references React release tomorrow.",
            "What is React?",
            "React",
        ),
    ],
)
def test_smart_context_provider_public_entities_in_private_sentences_use_web(
    tmp_path,
    document_text,
    question,
    title,
):
    class FakeWebSearchClient:
        def __init__(self):
            self.calls = []

        def search(self, query, *, max_results=5):
            self.calls.append((query, max_results))
            return [
                web_search_module.WebSearchHit(
                    title=title,
                    url="https://example.com/public-entity",
                    snippet=f"{title} lookup should use web evidence.",
                )
            ]

    document = tmp_path / "notes.txt"
    document.write_text(document_text, encoding="utf-8")
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    qa.web_search_client = FakeWebSearchClient()

    result = qa.query_with_trace(
        question,
        session_id="smart_public_entity_private_sentence_web",
    )

    assert qa.web_search_client.calls == [(question, qa.web_search_max_results)]
    assert result.loop_report.run.context_provider == "web"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.loop_report.run.metadata["document_name"] is None


@pytest.mark.parametrize(
    ("document_text", "question", "title"),
    [
        ("Python is mentioned in this private note.", "What is Python?", "Python"),
        (
            "OpenAI is mentioned in this internal memo as a vendor.",
            "What is OpenAI?",
            "OpenAI",
        ),
        (
            "Jackie Chan is listed in this confidential example.",
            "Who is Jackie Chan?",
            "Jackie Chan",
        ),
    ],
)
def test_smart_context_provider_public_subject_mentions_in_private_text_use_web(
    tmp_path,
    document_text,
    question,
    title,
):
    class FakeWebSearchClient:
        def __init__(self):
            self.calls = []

        def search(self, query, *, max_results=5):
            self.calls.append((query, max_results))
            return [
                web_search_module.WebSearchHit(
                    title=title,
                    url="https://example.com/public-subject",
                    snippet=f"{title} lookup should use web evidence.",
                )
            ]

    document = tmp_path / "notes.txt"
    document.write_text(document_text, encoding="utf-8")
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    qa.web_search_client = FakeWebSearchClient()

    result = qa.query_with_trace(
        question,
        session_id="smart_public_subject_private_text_web",
    )

    assert qa.web_search_client.calls == [(question, qa.web_search_max_results)]
    assert result.loop_report.run.context_provider == "web"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.loop_report.run.metadata["document_name"] is None


@pytest.mark.parametrize(
    ("document_text", "question", "title"),
    [
        (
            "This private note mentions a movie called Fight Club.",
            "What is Fight Club?",
            "Fight Club",
        ),
        (
            "This internal memo references a library called React.",
            "What is React?",
            "React",
        ),
        (
            "This confidential list includes an actor named Jackie Chan.",
            "Who is Jackie Chan?",
            "Jackie Chan",
        ),
    ],
)
def test_smart_context_provider_public_called_named_entities_use_web(
    tmp_path,
    document_text,
    question,
    title,
):
    class FakeWebSearchClient:
        def __init__(self):
            self.calls = []

        def search(self, query, *, max_results=5):
            self.calls.append((query, max_results))
            return [
                web_search_module.WebSearchHit(
                    title=title,
                    url="https://example.com/public-called-named",
                    snippet=f"{title} lookup should use web evidence.",
                )
            ]

    document = tmp_path / "notes.txt"
    document.write_text(document_text, encoding="utf-8")
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    qa.web_search_client = FakeWebSearchClient()

    result = qa.query_with_trace(
        question,
        session_id="smart_public_called_named_web",
    )

    assert qa.web_search_client.calls == [(question, qa.web_search_max_results)]
    assert result.loop_report.run.context_provider == "web"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.loop_report.run.metadata["document_name"] is None


@pytest.mark.parametrize(
    ("document_text", "question", "title"),
    [
        (
            "The private movie Fight Club is mentioned in these notes.",
            "What is Fight Club?",
            "Fight Club",
        ),
        (
            "The internal library React is mentioned in this memo.",
            "What is React?",
            "React",
        ),
        (
            "The confidential actor Jackie Chan is listed in this example.",
            "Who is Jackie Chan?",
            "Jackie Chan",
        ),
    ],
)
def test_smart_context_provider_marker_prefixed_public_entities_use_web(
    tmp_path,
    document_text,
    question,
    title,
):
    class FakeWebSearchClient:
        def __init__(self):
            self.calls = []

        def search(self, query, *, max_results=5):
            self.calls.append((query, max_results))
            return [
                web_search_module.WebSearchHit(
                    title=title,
                    url="https://example.com/public-marker-prefixed",
                    snippet=f"{title} lookup should use web evidence.",
                )
            ]

    document = tmp_path / "notes.txt"
    document.write_text(document_text, encoding="utf-8")
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    qa.web_search_client = FakeWebSearchClient()

    result = qa.query_with_trace(
        question,
        session_id="smart_marker_prefixed_public_web",
    )

    assert qa.web_search_client.calls == [(question, qa.web_search_max_results)]
    assert result.loop_report.run.context_provider == "web"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.loop_report.run.metadata["document_name"] is None


@pytest.mark.parametrize(
    ("document_text", "question"),
    [
        (
            "This private note references Project Runway in an example.",
            "What is Project Runway?",
        ),
        (
            "This private note references Project Runway as an example.",
            "What is Project Runway?",
        ),
        (
            "This private note references Project Runway for comparison.",
            "What is Project Runway?",
        ),
        (
            "This private note references Project Runway in passing.",
            "What is Project Runway?",
        ),
        (
            "This private note references Project Runway for background.",
            "What is Project Runway?",
        ),
        (
            "This private note references Project Runway for research.",
            "What is Project Runway?",
        ),
        (
            "This private note references Project Runway.",
            "What is Project Runway?",
        ),
        (
            "This private note references Project Runway's finale as an example.",
            "What is Project Runway finale?",
        ),
    ],
)
def test_smart_context_provider_public_descriptor_led_references_use_web(
    tmp_path,
    document_text,
    question,
):
    class FakeWebSearchClient:
        def __init__(self):
            self.calls = []

        def search(self, query, *, max_results=5):
            self.calls.append((query, max_results))
            return [
                web_search_module.WebSearchHit(
                    title="Project Runway",
                    url="https://example.com/project-runway",
                    snippet="Project Runway lookup should use web evidence.",
                )
            ]

    document = tmp_path / "notes.txt"
    document.write_text(document_text, encoding="utf-8")
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    qa.web_search_client = FakeWebSearchClient()

    result = qa.query_with_trace(
        question,
        session_id="smart_public_descriptor_reference_web",
    )

    assert qa.web_search_client.calls == [
        (question, qa.web_search_max_results)
    ]
    assert result.loop_report.run.context_provider == "web"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.loop_report.run.metadata["document_name"] is None


@pytest.mark.parametrize(
    "document_text",
    [
        "This private note references private Project Runway.",
        "This private note references internal Project Runway.",
        "This private note references local Project Runway.",
        "This private note references Project Runway for launch.",
        "This private note references Project Runway launch.",
        "This private note references Project Runway for release.",
        "This private note references Project Runway release.",
        "This private note references Project Runway private launch.",
        "This private note references Project Runway internal release.",
        "This private note references Project Runway confidential roadmap.",
    ],
)
def test_smart_context_provider_public_exception_with_local_markers_uses_file(
    tmp_path,
    document_text,
):
    class FailingWebSearchClient:
        def search(self, query, *, max_results=5):
            raise AssertionError("explicit local Project Runway reference must not use web")

    document = tmp_path / "notes.txt"
    document.write_text(document_text, encoding="utf-8")
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    qa.web_search_client = FailingWebSearchClient()

    result = qa.query_with_trace(
        "What is Project Runway?",
        session_id="smart_public_exception_local_file",
    )

    assert result.loop_report.run.context_provider == "document"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.loop_report.run.metadata["document_name"] == "notes.txt"


def test_smart_context_provider_localization_does_not_mark_local_entity(tmp_path):
    class FakeWebSearchClient:
        def __init__(self):
            self.calls = []

        def search(self, query, *, max_results=5):
            self.calls.append((query, max_results))
            return [
                web_search_module.WebSearchHit(
                    title="Python",
                    url="https://example.com/python",
                    snippet="Python is a programming language.",
                )
            ]

    document = tmp_path / "notes.txt"
    document.write_text(
        "This note mentions Python localization work.",
        encoding="utf-8",
    )
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    qa.web_search_client = FakeWebSearchClient()

    result = qa.query_with_trace(
        "What is Python?",
        session_id="smart_localization_not_local",
    )

    assert qa.web_search_client.calls == [("What is Python?", qa.web_search_max_results)]
    assert result.loop_report.run.context_provider == "web"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.loop_report.run.metadata["document_name"] is None


def test_smart_context_provider_secretary_does_not_mark_secret_entity(tmp_path):
    class FakeWebSearchClient:
        def __init__(self):
            self.calls = []

        def search(self, query, *, max_results=5):
            self.calls.append((query, max_results))
            return [
                web_search_module.WebSearchHit(
                    title="Parker",
                    url="https://example.com/parker",
                    snippet="Parker lookup should use web evidence.",
                )
            ]

    document = tmp_path / "notes.txt"
    document.write_text(
        "This note mentions Secretary Parker in a public example.",
        encoding="utf-8",
    )
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    qa.web_search_client = FakeWebSearchClient()

    result = qa.query_with_trace(
        "Who is Parker?",
        session_id="smart_secretary_not_secret",
    )

    assert qa.web_search_client.calls == [("Who is Parker?", qa.web_search_max_results)]
    assert result.loop_report.run.context_provider == "web"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.loop_report.run.metadata["document_name"] is None


def test_smart_context_provider_file_intent_beats_direct_summary_hint(tmp_path):
    class FailingWebSearchClient:
        def search(self, query, *, max_results=5):
            raise AssertionError("file-intent summary must not be sent to web search")

    document = tmp_path / "strategy.txt"
    document.write_text("The strategy file says Project Phoenix launches in June 2026.", encoding="utf-8")
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    qa.web_search_client = FailingWebSearchClient()

    result = qa.query_with_trace(
        "Summarize this document.",
        session_id="smart_file_summary",
    )

    assert result.loop_report.run.context_provider == "document"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.loop_report.run.metadata["document_name"] == "strategy.txt"


def test_smart_context_provider_lookup_terms_are_not_overblocked_as_code():
    class FakeWebSearchClient:
        def __init__(self):
            self.calls = []

        def search(self, query, *, max_results=5):
            self.calls.append((query, max_results))
            return [
                web_search_module.WebSearchHit(
                    title="QR code",
                    url="https://example.com/qr-code",
                    snippet="A QR code is a two-dimensional barcode.",
                )
            ]

    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.web_search_client = FakeWebSearchClient()

    result = qa.query_with_trace("What is QR code?", session_id="smart_qr_code")

    assert qa.web_search_client.calls == [
        ("What is QR code?", qa.web_search_max_results)
    ]
    assert result.loop_report.run.context_provider == "web"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"


def test_smart_context_provider_keeps_private_local_tasks_off_web():
    class FailingWebSearchClient:
        def search(self, query, *, max_results=5):
            raise AssertionError("private local task must not be sent to web search")

    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.web_search_client = FailingWebSearchClient()
    qa.llm = SimpleNamespace(
        invoke=lambda _prompt: "Rewritten private draft.",
        last_thinking=None,
    )

    result = qa.query_with_trace(
        "Rewrite this private draft: SECRET_BOARD_PLAN",
        session_id="smart_private",
    )

    assert result.answer == "Rewritten private draft."
    assert result.loop_report.run.context_provider == "none"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.trace.citations == []


def test_smart_context_provider_keeps_private_local_tasks_off_active_file(tmp_path):
    class FailingWebSearchClient:
        def search(self, query, *, max_results=5):
            raise AssertionError("private local task must not be sent to web search")

    document = tmp_path / "resume.txt"
    document.write_text("This resume is for an AI engineer in Hong Kong.", encoding="utf-8")
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    qa.web_search_client = FailingWebSearchClient()
    qa.llm = SimpleNamespace(
        invoke=lambda _prompt: "Rewritten private draft.",
        last_thinking=None,
    )

    result = qa.query_with_trace(
        "Rewrite this private draft: SECRET_BOARD_PLAN",
        session_id="smart_private_with_file",
    )

    assert result.answer == "Rewritten private draft."
    assert result.loop_report.run.context_provider == "none"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.loop_report.run.metadata["document_name"] is None
    assert result.trace.citations == []


def test_smart_context_provider_keeps_draft_tasks_off_active_file(tmp_path):
    class FailingWebSearchClient:
        def search(self, query, *, max_results=5):
            raise AssertionError("draft task must not be sent to web search")

    document = tmp_path / "strategy.txt"
    document.write_text(
        "The launch email date is June 2026.",
        encoding="utf-8",
    )
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.process_document(str(document))
    qa.web_search_client = FailingWebSearchClient()
    qa.llm = SimpleNamespace(
        invoke=lambda _prompt: "Drafted launch email.",
        last_thinking=None,
    )

    result = qa.query_with_trace(
        "Show me a draft for the launch email.",
        session_id="smart_draft_with_file",
    )

    assert result.answer == "Drafted launch email."
    assert result.loop_report.run.context_provider == "none"
    assert result.loop_report.run.metadata["requested_context_provider"] == "smart"
    assert result.loop_report.run.metadata["document_name"] is None
    assert result.trace.citations == []


def test_recipe_context_provider_applies_when_query_provider_is_omitted():
    class FakeWebSearchClient:
        def __init__(self):
            self.calls = []

        def search(self, query, *, max_results=5):
            self.calls.append((query, max_results))
            return [
                web_search_module.WebSearchHit(
                    title="Python",
                    url="https://www.python.org/",
                    snippet="Python is a programming language.",
                )
            ]

    recipe = {
        "recipe_id": "recipe_web",
        "name": "Web evidence recipe",
        "goal": "Use web evidence.",
        "context_provider": "web",
    }
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.web_search_client = FakeWebSearchClient()

    result = qa.query_with_trace("What is Python?", loop_recipe=recipe)

    assert qa.web_search_client.calls == [("What is Python?", qa.web_search_max_results)]
    assert result.loop_report.run.context_provider == "web"
    assert result.loop_report.run.metadata["requested_context_provider"] == "web"
    assert result.answer == "Python is a programming language. [1]"


def test_web_search_provider_failure_reports_query_error():
    class FailingWebSearchClient:
        def search(self, query, *, max_results=5):
            raise web_search_module.WebSearchError("SECRET_PROVIDER_DETAIL")

    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.web_search_client = FailingWebSearchClient()

    result = qa.query_with_trace(
        "What changed today?",
        context_provider="web",
    )

    assert result.answer == (
        "Web search evidence is unavailable right now. Try again or ask "
        "without requiring web evidence."
    )
    assert result.trace.error_message == "web_search_failed"
    assert result.trace.citations == []
    assert result.loop_report.run.final_decision == LoopDecision.ERROR
    assert result.loop_report.run.error_message == "web_search_failed"
    assert "SECRET_PROVIDER_DETAIL" not in json.dumps(result.loop_report.to_public_dict())


def test_web_search_result_with_malformed_url_does_not_leak_or_fail():
    class MalformedUrlWebSearchClient:
        def search(self, query, *, max_results=5):
            return [
                web_search_module.WebSearchHit(
                    title="Example",
                    url="https://example.com:SECRET_PORT/path",
                    snippet="Example evidence is available.",
                )
            ]

    captured_prompts = []
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.web_search_client = MalformedUrlWebSearchClient()
    qa.llm = SimpleNamespace(
        invoke=lambda prompt: captured_prompts.append(prompt)
        or "Example evidence is available [1].",
        last_thinking=None,
    )

    result = qa.query_with_trace("What is available?", context_provider="web")
    public_payload = json.dumps(result.loop_report.to_public_dict())

    assert result.trace.error_message is None
    assert result.loop_report.run.final_decision == LoopDecision.NOT_VERIFIED
    assert result.trace.citations[0].source_name == "Example"
    assert "SECRET_PORT" not in captured_prompts[0]
    assert "SECRET_PORT" not in public_payload


def test_web_search_refusal_uses_provider_neutral_evidence_wording():
    class FakeWebSearchClient:
        def search(self, query, *, max_results=5):
            return [
                web_search_module.WebSearchHit(
                    title="Python",
                    url="https://www.python.org/",
                    snippet="Python is a programming language.",
                )
            ]

    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.web_search_client = FakeWebSearchClient()
    qa.llm = SimpleNamespace(
        invoke=lambda _prompt: "Python is useful.",
        last_thinking=None,
    )

    result = qa.query_with_trace("What is Python?", context_provider="web")
    response_payload = web_contract_module.query_response_dict(result)

    assert result.answer == answer_loop_module.SELF_CHECK_REFUSAL_ANSWER
    assert response_payload["answer"] == answer_loop_module.SELF_CHECK_REFUSAL_ANSWER
    assert response_payload["trace"]["answer"] == answer_loop_module.SELF_CHECK_REFUSAL_ANSWER
    assert "provided evidence" in result.answer
    assert "document" not in result.answer.lower()
    assert response_payload["summary"]["last_error"] is None
    assert result.loop_report.run.final_decision == LoopDecision.REFUSE
    assert response_payload["trace"]["loop_report"]["public_redaction"]["applied"] is True


def test_web_search_refusal_redacts_captured_model_thinking_publicly():
    class FakeWebSearchClient:
        def search(self, query, *, max_results=5):
            return [
                web_search_module.WebSearchHit(
                    title="Python",
                    url="https://www.python.org/",
                    snippet="Python is a programming language.",
                )
            ]

    class ThinkingLLM:
        last_thinking = "SECRET_WEB_THINKING"

        def invoke(self, _prompt):
            return "Python is useful."

    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.web_search_client = FakeWebSearchClient()
    qa.llm = ThinkingLLM()

    result = qa.query_with_trace("What is Python?", context_provider="web")
    response_payload = web_contract_module.query_response_dict(result)
    response_json = json.dumps(response_payload)

    assert result.answer == answer_loop_module.SELF_CHECK_REFUSAL_ANSWER
    assert result.trace.model_thinking == "SECRET_WEB_THINKING"
    assert response_payload["trace"]["model_thinking"]["available"] is False
    assert response_payload["trace"]["model_thinking"]["redacted"] is True
    assert response_payload["trace"]["model_thinking"]["content"] == (
        web_contract_module.MODEL_THINKING_REDACTION
    )
    assert "SECRET_WEB_THINKING" not in response_json


def test_invalid_context_provider_fails_closed():
    qa = DocumentQA(fast_mode=True, llm_backend="mock")

    result = qa.query_with_trace("What is Python?", context_provider="internettt")

    assert result.answer == "Invalid context provider. Choose smart, web, document, or none."
    assert result.trace.error_message == "invalid_context_provider"
    assert result.loop_report.run.final_decision == LoopDecision.BLOCK
    assert result.loop_report.run.error_message == "invalid_context_provider"
    assert result.loop_report.run.context_provider == "none"
    assert result.loop_report.run.steps[0].name == "Context provider validation"


def test_duckduckgo_instant_answer_parser_sanitizes_hits():
    payload = {
        "Heading": "Python",
        "AbstractText": "<b>Python</b> is a programming language.",
        "AbstractURL": "https://www.python.org/?secret=drop#frag",
        "RelatedTopics": [
            {
                "Text": "Python - Official website",
                "FirstURL": "javascript:alert(1)",
            },
            {
                "Text": "Python downloads",
                "FirstURL": "https://www.python.org/downloads/?token=drop",
            },
        ],
    }

    hits = web_search_module.parse_duckduckgo_instant_answer(payload, max_results=3)

    assert hits[0] == web_search_module.WebSearchHit(
        title="Python",
        url="https://www.python.org/",
        snippet="Python is a programming language.",
    )
    assert hits[1].title == "Web result"
    assert hits[1].url == ""
    assert hits[1].snippet == "Python - Official website"
    assert hits[2].url == "https://www.python.org/downloads/"


def test_duckduckgo_search_uses_html_fallback_when_instant_answer_is_empty():
    class FakeResponse:
        def __init__(self, body: bytes):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, _size: int):
            return self.body

    calls = []

    def fake_opener(request, *, timeout):
        calls.append(request.full_url)
        if request.full_url.startswith(web_search_module.DUCKDUCKGO_INSTANT_ANSWER_URL):
            return FakeResponse(b'{"RelatedTopics":[]}')
        assert request.full_url.startswith(web_search_module.DUCKDUCKGO_HTML_SEARCH_URL)
        html_body = b"""
        <html><body>
          <div class="result">
            <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpython%3Ftoken%3Ddrop">Python Result</a>
            <a class="result__snippet">Python is a programming language for clear code.</a>
          </div>
          <div class="result">
            <a class="result__a" href="https://example.org/docs#frag">Docs Result</a>
            <a class="result__snippet">Docs explain Python syntax.</a>
          </div>
        </body></html>
        """
        return FakeResponse(html_body)

    client = web_search_module.DuckDuckGoInstantAnswerSearch(opener=fake_opener)

    hits = client.search("Python", max_results=2)

    assert len(calls) == 2
    assert hits == [
        web_search_module.WebSearchHit(
            title="Python Result",
            url="https://example.com/python",
            snippet="Python is a programming language for clear code.",
        ),
        web_search_module.WebSearchHit(
            title="Docs Result",
            url="https://example.org/docs",
            snippet="Docs explain Python syntax.",
        ),
    ]


def test_duckduckgo_search_skips_html_fallback_when_instant_answer_has_enough_hits():
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, _size: int):
            return json.dumps(
                {
                    "Heading": "Python",
                    "AbstractText": "Python is a programming language.",
                    "AbstractURL": "https://www.python.org/",
                }
            ).encode("utf-8")

    calls = []

    def fake_opener(request, *, timeout):
        calls.append(request.full_url)
        if request.full_url.startswith(web_search_module.DUCKDUCKGO_HTML_SEARCH_URL):
            raise AssertionError("HTML fallback should not run when enough evidence exists")
        return FakeResponse()

    client = web_search_module.DuckDuckGoInstantAnswerSearch(opener=fake_opener)

    hits = client.search("Python", max_results=1)

    assert len(calls) == 1
    assert hits[0].snippet == "Python is a programming language."


def test_duckduckgo_html_parser_sanitizes_redirects_and_markup():
    html_text = """
    <html><body>
      <div class="result">
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fsafe%3Fsecret%3Ddrop">Example <b>Title</b></a>
        <a class="result__snippet">Example <b>snippet</b> with markup.</a>
      </div>
    </body></html>
    """

    hits = web_search_module.parse_duckduckgo_html_results(html_text, max_results=1)

    assert hits == [
        web_search_module.WebSearchHit(
            title="Example Title",
            url="https://example.com/safe",
            snippet="Example snippet with markup.",
        )
    ]


def test_duckduckgo_html_parser_rejects_lookalike_redirect_host():
    html_text = """
    <html><body>
      <div class="result">
        <a class="result__a" href="https://notduckduckgo.com/l/?uddg=https%3A%2F%2Fevil.example%2Fsecret">Lookalike</a>
        <a class="result__snippet">This should not be cited through a lookalike redirect.</a>
      </div>
    </body></html>
    """

    hits = web_search_module.parse_duckduckgo_html_results(html_text, max_results=1)

    assert hits == [
        web_search_module.WebSearchHit(
            title="Lookalike",
            url="https://notduckduckgo.com/l/",
            snippet="This should not be cited through a lookalike redirect.",
        )
    ]


def test_duckduckgo_html_parser_skips_results_without_safe_url():
    html_text = """
    <html><body>
      <div class="result">
        <a class="result__a" href="javascript:alert(1)">Unsafe Result</a>
        <a class="result__snippet">Unsafe results should not become citeable evidence.</a>
      </div>
    </body></html>
    """

    hits = web_search_module.parse_duckduckgo_html_results(html_text, max_results=1)

    assert hits == []


def test_query_trace_counts_only_prompt_included_chunks(tmp_path):
    document = tmp_path / "long-phoenix.txt"
    document.write_text(
        "\n\n".join(
            f"Project Phoenix evidence section {index}. The launch date is June 2026."
            for index in range(80)
        ),
        encoding="utf-8",
    )
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.embeddings = FakeEmbeddings()
    qa.process_document(str(document))
    assert len(qa.vector_store.documents) > 1
    qa.profile["context_chunks"] = 1

    result = qa.query_with_trace(
        "What is the launch date?",
        context_provider="document",
    )

    assert result.trace.retrieved_chunk_count == 1
    assert len(result.trace.citations) == 1


def test_query_with_trace_includes_loop_report_for_prompt_evidence(tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)

    result = qa.query_with_trace(
        "What is Project Phoenix?",
        session_id="session_a",
        context_provider="document",
    )

    report = result.loop_report
    assert report is not None
    run = report.run
    assert run.session_id == "session_a"
    assert run.user_input == "What is Project Phoenix?"
    assert run.context_provider == "document"
    assert run.metadata["context_provider"] == "document"
    assert run.metadata["context_provider_name"] == "phoenix.txt"
    assert run.backend == "mock"
    assert run.model_label == "MockLLM (explicit demo)"
    assert run.policy.allow_tool_calls is False
    assert "document_text" in run.metadata["untrusted_inputs"]
    assert run.final_decision == LoopDecision.NOT_VERIFIED
    assert run.final_answer == result.answer

    phases = [step.phase for step in run.steps]
    assert phases == [
        LoopPhase.CONTEXT_SELECT,
        LoopPhase.RETRIEVE,
        LoopPhase.DRAFT,
        LoopPhase.FORMAT_CHECK,
        LoopPhase.MECHANICAL_CHECK,
        LoopPhase.VERIFY,
        LoopPhase.FINAL,
    ]
    retrieve_step = next(step for step in run.steps if step.phase == LoopPhase.RETRIEVE)
    format_step = next(
        step for step in run.steps if step.phase == LoopPhase.FORMAT_CHECK
    )
    verify_step = next(step for step in run.steps if step.phase == LoopPhase.VERIFY)
    assert retrieve_step.metadata["retrieved_chunk_count"] == len(result.trace.citations)
    assert retrieve_step.metadata["citation_ids"] == [1]
    assert format_step.output_summary == "format_passed"
    assert verify_step.decision == LoopDecision.NOT_VERIFIED
    assert verify_step.verification.outcome.value == "not_verified"
    assert verify_step.verification.reasons == tuple(result.trace.self_check.reasons)


def test_query_records_loop_session_and_exports_jsonl(tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)

    first_result = qa.query_with_trace(
        "What is Project Phoenix?",
        session_id="alpha",
        context_provider="document",
    )
    second_result = qa.query_with_trace(
        "What is the launch date?",
        session_id="alpha",
        context_provider="document",
    )
    qa.query_with_trace(
        "What is the launch date?",
        session_id="beta",
        context_provider="document",
    )

    alpha_session = qa.loop_session("alpha")
    beta_session = qa.loop_session("beta")
    artifact_path = tmp_path / "alpha-session.jsonl"
    exported_path = qa.export_loop_session_jsonl(artifact_path, session_id="alpha")
    exported_lines = artifact_path.read_text(encoding="utf-8").splitlines()
    restored_reports = [
        LoopReport.from_dict(json.loads(line)) for line in exported_lines
    ]

    assert exported_path == str(artifact_path)
    assert alpha_session.report_count == 2
    assert beta_session.report_count == 1
    assert [report.run.run_id for report in alpha_session.reports] == [
        first_result.loop_report.run.run_id,
        second_result.loop_report.run.run_id,
    ]
    assert [report.run.session_id for report in restored_reports] == [
        "alpha",
        "alpha",
    ]
    assert restored_reports[0].run.user_input == "What is Project Phoenix?"
    assert restored_reports[1].run.final_decision == LoopDecision.NOT_VERIFIED


def test_loop_session_history_is_bounded(tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)
    qa.max_session_reports = 1

    first_result = qa.query_with_trace("What is Project Phoenix?", session_id="alpha")
    second_result = qa.query_with_trace("What is the launch date?", session_id="alpha")

    alpha_session = qa.loop_session("alpha")

    assert alpha_session.report_count == 1
    assert alpha_session.reports[0].run.run_id == second_result.loop_report.run.run_id
    assert alpha_session.reports[0].run.run_id != first_result.loop_report.run.run_id


def test_no_context_query_is_recorded_for_replay(tmp_path):
    qa = DocumentQA(fast_mode=True, llm_backend="mock")

    result = qa.query_with_trace(
        "What is this?",
        session_id="direct",
        context_provider="none",
    )

    session = qa.loop_session("direct")

    assert result.trace.error_message is None
    assert session.report_count == 1
    assert session.reports[0].run.context_provider == "none"
    assert session.reports[0].run.final_decision == LoopDecision.NOT_VERIFIED
    assert session.reports[0].run.error_message is None


def test_cleared_session_does_not_record_in_flight_query_result():
    class ClearSessionDuringDraftMiddleware:
        def __init__(self):
            self.qa = None
            self.cleared = False

        def before_step(self, run, step):
            if step.phase == LoopPhase.DRAFT and not self.cleared:
                self.cleared = True
                self.qa.clear_loop_session(run.session_id)
            return None

    middleware = ClearSessionDuringDraftMiddleware()
    qa = DocumentQA(
        fast_mode=True,
        llm_backend="mock",
        loop_middlewares=(middleware,),
    )
    middleware.qa = qa

    result = qa.query_with_trace("What is this?", session_id="stale")

    assert result.loop_report.run.session_id == "stale"
    assert middleware.cleared is True
    assert qa.loop_session("stale").report_count == 0
    assert qa.chat_history == []


def test_processed_document_is_wrapped_as_context_provider(tmp_path):
    qa, document = create_processed_mock_qa(tmp_path)

    active_state = qa._snapshot_active_document_state()

    assert isinstance(active_state.context_provider, DocumentContextProvider)
    assert active_state.context_provider.provider_type == "document"
    assert active_state.context_provider.display_name == document.name
    assert active_state.context_provider.vector_store is qa.vector_store
    assert active_state.context_provider.retrieval_chain is qa.retrieval_chain
    assert active_state.context_provider.ready is True


def test_loop_middleware_can_block_before_retrieval(tmp_path):
    class BlockRetrieveMiddleware:
        def before_step(self, run, step):
            if step.phase == LoopPhase.RETRIEVE:
                return GuardrailDecision(
                    decision=LoopDecision.BLOCK,
                    reason="retrieval_blocked",
                    metadata={"policy": "test"},
                )
            return None

    document = tmp_path / "phoenix.txt"
    document.write_text(
        "Project Phoenix launches in June 2026.",
        encoding="utf-8",
    )
    qa = DocumentQA(
        fast_mode=True,
        llm_backend="mock",
        loop_middlewares=(BlockRetrieveMiddleware(),),
    )
    qa.embeddings = FakeEmbeddings()
    qa.process_document(str(document))

    class CountingRetrievalChain:
        def __init__(self):
            self.calls = 0

        def invoke_with_trace(self, question, self_check_instruction=""):
            self.calls += 1
            return SimpleNamespace(
                answer="Project Phoenix launches in June 2026 [1].",
                retrieved_chunk_count=1,
                citations=[citation_for(document.name)],
                context="Project Phoenix launches in June 2026.",
            )

    retrieval_chain = CountingRetrievalChain()
    replace_retrieval_chain(qa, retrieval_chain)

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert retrieval_chain.calls == 0
    assert result.answer == "A loop guardrail blocked this query before it could complete."
    assert result.trace.error_message == "retrieval_blocked"
    assert result.loop_report.run.final_decision == LoopDecision.BLOCK
    phases = [step.phase for step in result.loop_report.run.steps]
    assert phases == [
        LoopPhase.CONTEXT_SELECT,
        LoopPhase.ERROR,
        LoopPhase.FINAL,
    ]
    guardrail_step = next(
        step for step in result.loop_report.run.steps if step.phase == LoopPhase.ERROR
    )
    assert guardrail_step.metadata["guardrail_decision"] == "block"
    assert guardrail_step.metadata["policy"] == "test"
    assert LoopPhase.RETRIEVE not in phases


def test_loop_middleware_can_block_after_retrieval_before_draft(tmp_path):
    class BlockAfterRetrieveMiddleware:
        def after_step(self, run, step):
            if step.phase == LoopPhase.RETRIEVE:
                return GuardrailDecision(
                    decision=LoopDecision.BLOCK,
                    reason="post_retrieval_blocked",
                )
            return None

    qa, _document = create_processed_mock_qa(tmp_path)
    qa.loop_middlewares = (BlockAfterRetrieveMiddleware(),)
    active_state = qa._snapshot_active_document_state()

    class CountingSplitRetrievalChain:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.draft_calls = 0

        def invoke_with_trace(self, question, self_check_instruction=""):
            return self.wrapped.invoke_with_trace(question, self_check_instruction)

        def retrieve_with_trace(self, question):
            return self.wrapped.retrieve_with_trace(question)

        def draft_with_trace(
            self, question, retrieved_context, self_check_instruction=""
        ):
            self.draft_calls += 1
            return self.wrapped.draft_with_trace(
                question,
                retrieved_context,
                self_check_instruction=self_check_instruction,
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            return self.wrapped.retry_with_trace(
                question,
                previous_result,
                self_check_instruction,
            )

    retrieval_chain = CountingSplitRetrievalChain(active_state.retrieval_chain)
    replace_retrieval_chain(qa, retrieval_chain)

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert retrieval_chain.draft_calls == 0
    assert result.answer == "A loop guardrail blocked this query before it could complete."
    assert result.trace.error_message == "post_retrieval_blocked"
    assert result.loop_report.run.final_decision == LoopDecision.BLOCK
    phases = [step.phase for step in result.loop_report.run.steps]
    assert phases == [
        LoopPhase.CONTEXT_SELECT,
        LoopPhase.RETRIEVE,
        LoopPhase.ERROR,
        LoopPhase.FINAL,
    ]
    assert LoopPhase.DRAFT not in phases


def test_loop_middleware_refusal_uses_guardrail_specific_answer(tmp_path):
    class RefuseBeforeRunMiddleware:
        def before_run(self, run):
            return GuardrailDecision(
                decision=LoopDecision.REFUSE,
                reason="policy_refused",
            )

    qa, _document = create_processed_mock_qa(tmp_path)
    qa.loop_middlewares = (RefuseBeforeRunMiddleware(),)

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == (
        "A loop guardrail refused this query before it could safely complete."
    )
    assert result.answer != (
        answer_loop_module.SELF_CHECK_REFUSAL_ANSWER
    )
    assert result.trace.error_message == "policy_refused"
    assert result.loop_report.run.final_decision == LoopDecision.REFUSE


def test_loop_middleware_retry_request_is_reported_as_unavailable_block(tmp_path):
    class RetryBeforeRunMiddleware:
        def before_run(self, run):
            return GuardrailDecision(
                decision=LoopDecision.RETRY,
                reason="policy_requested_retry",
            )

    qa, _document = create_processed_mock_qa(tmp_path)
    qa.loop_middlewares = (RetryBeforeRunMiddleware(),)

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == (
        "A loop guardrail requested a retry, but no safe retry path is available for this step."
    )
    assert result.trace.error_message == "guardrail_retry_unavailable"
    assert result.loop_report.run.final_decision == LoopDecision.BLOCK
    guardrail_step = next(
        step for step in result.loop_report.run.steps if step.name == "Guardrail decision"
    )
    assert guardrail_step.decision == LoopDecision.RETRY
    assert guardrail_step.metadata["guardrail_reason"] == "policy_requested_retry"


def test_after_run_guardrail_overrides_answer_without_stale_self_check(tmp_path):
    class RequireReviewAfterRunMiddleware:
        def after_run(self, run):
            if run.final_decision == LoopDecision.NOT_VERIFIED:
                return GuardrailDecision(
                    decision=LoopDecision.REQUIRES_REVIEW,
                    reason="review_mock_answer",
                )
            return None

    document = tmp_path / "phoenix.txt"
    document.write_text(
        "Project Phoenix launches in June 2026.",
        encoding="utf-8",
    )
    qa = DocumentQA(
        fast_mode=True,
        llm_backend="mock",
        loop_middlewares=(RequireReviewAfterRunMiddleware(),),
    )
    qa.embeddings = FakeEmbeddings()
    qa.process_document(str(document))

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == "This query requires human review before the loop can continue."
    assert result.trace.citations == []
    assert result.trace.self_check is None
    assert result.trace.error_message == "review_mock_answer"
    assert result.loop_report.run.final_decision == LoopDecision.REQUIRES_REVIEW
    assert result.loop_report.run.metadata["after_run_guardrail"] is True
    assert qa.chat_history[-1]["answer"] == result.answer
    assert qa.chat_history[-1]["self_check"] is None


def replace_retrieval_chain(qa, retrieval_chain):
    active_state = qa._snapshot_active_document_state()
    qa._commit_active_document_state(
        document_name=active_state.document_name,
        vector_store=active_state.vector_store,
        retrieval_chain=retrieval_chain,
        processing_report=qa.status().processing_report,
    )


def citation_for(document_name="phoenix.txt"):
    return AnswerCitation(
        citation_id=1,
        source_name=document_name,
        page=None,
        chunk_index=0,
        excerpt="Project Phoenix launches in June 2026.",
    )


class FakeVerifierLLM:
    def __init__(self, outcome="supported", raw_response=None, error=None):
        self.outcome = outcome
        self.raw_response = raw_response
        self.error = error
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        if self.error:
            raise self.error
        if self.raw_response is not None:
            return self.raw_response
        return json.dumps({"outcome": self.outcome, "reason": "test verifier"})


def enable_fake_verifier(qa, outcome="supported", raw_response=None, error=None):
    verifier = FakeVerifierLLM(
        outcome=outcome,
        raw_response=raw_response,
        error=error,
    )
    qa.active_llm_backend = "openai-compatible"
    qa.llm = verifier
    qa.loaded_model_label = "Fake verifier gateway"
    return verifier


def test_self_check_refuses_when_answer_has_no_prompt_evidence(tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)

    class NoEvidenceChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow.",
                retrieved_chunk_count=0,
                citations=[],
            )

    replace_retrieval_chain(qa, NoEvidenceChain())

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == (
        answer_loop_module.SELF_CHECK_REFUSAL_ANSWER
    )
    assert result.trace.self_check.outcome == "needs_refusal"
    assert result.trace.self_check.reasons == ["no_prompt_evidence"]
    assert result.trace.self_check.retry_attempted is False
    assert result.trace.citations == []


def test_self_check_retries_missing_inline_citation(tmp_path):
    qa, document = create_processed_mock_qa(tmp_path)

    class RetryCitationChain:
        def __init__(self):
            self.calls = []

        def invoke_with_trace(self, question, self_check_instruction=""):
            self.calls.append(self_check_instruction)
            answer = "Project Phoenix launches in June 2026."
            return SimpleNamespace(
                answer=answer,
                retrieved_chunk_count=1,
                citations=[citation_for(document.name)],
                context="original context",
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            self.calls.append(self_check_instruction)
            return SimpleNamespace(
                answer="Project Phoenix launches in June 2026 [1].",
                retrieved_chunk_count=previous_result.retrieved_chunk_count,
                citations=previous_result.citations,
                context=previous_result.context,
            )

    retrieval_chain = RetryCitationChain()
    replace_retrieval_chain(qa, retrieval_chain)

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert len(retrieval_chain.calls) == 2
    assert retrieval_chain.calls[0] == ""
    assert "missing_inline_citation" in retrieval_chain.calls[1]
    assert result.answer == "Project Phoenix launches in June 2026 [1]."
    assert result.trace.self_check.outcome == "not_verified"
    assert result.trace.self_check.reasons == [
        "mechanical_checks_passed",
        "verifier_unavailable_mock_backend",
    ]
    assert result.trace.self_check.retry_attempted is True
    assert qa.chat_history[-1]["self_check"]["retry_attempted"] is True
    assert result.loop_report.run.final_decision == LoopDecision.NOT_VERIFIED
    retry_steps = [
        step for step in result.loop_report.run.steps if step.phase == LoopPhase.RETRY
    ]
    assert len(retry_steps) == 1
    assert retry_steps[0].metadata["reasons"] == ["missing_inline_citation"]
    retry_draft_steps = [
        step
        for step in result.loop_report.run.steps
        if step.phase == LoopPhase.DRAFT and step.retry_count == 1
    ]
    assert len(retry_draft_steps) == 1


def test_document_format_check_retries_without_losing_citations(tmp_path):
    qa, document = create_processed_mock_qa(tmp_path)

    class FormatRetryChain:
        def __init__(self):
            self.calls = []

        def invoke_with_trace(self, question, self_check_instruction=""):
            self.calls.append(self_check_instruction)
            return SimpleNamespace(
                answer=(
                    "1. **Launch:** Project Phoenix launches in June 2026 [1]. "
                    "2. **Owner:** Alex owns the rollout [1]."
                ),
                retrieved_chunk_count=1,
                citations=[citation_for(document.name)],
                context="Project Phoenix launches in June 2026.",
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            self.calls.append(self_check_instruction)
            return SimpleNamespace(
                answer=(
                    "1. **Launch:** Project Phoenix launches in June 2026 [1].\n"
                    "2. **Owner:** Alex owns the rollout [1]."
                ),
                retrieved_chunk_count=previous_result.retrieved_chunk_count,
                citations=previous_result.citations,
                context=previous_result.context,
            )

    retrieval_chain = FormatRetryChain()
    replace_retrieval_chain(qa, retrieval_chain)

    result = qa.query_with_trace("Summarize Project Phoenix.")

    assert result.answer == (
        "1. **Launch:** Project Phoenix launches in June 2026 [1].\n"
        "2. **Owner:** Alex owns the rollout [1]."
    )
    assert len(retrieval_chain.calls) == 2
    assert retrieval_chain.calls[0] == ""
    assert "Format retry instruction" in retrieval_chain.calls[1]
    assert "compact_ordered_list" in retrieval_chain.calls[1]
    assert result.trace.citations == [citation_for(document.name)]
    assert result.trace.self_check.outcome == "not_verified"

    format_steps = [
        step
        for step in result.loop_report.run.steps
        if step.phase == LoopPhase.FORMAT_CHECK
    ]
    assert [step.output_summary for step in format_steps] == [
        "needs_retry",
        "format_passed",
    ]
    assert format_steps[0].metadata["reasons"] == ["compact_ordered_list"]
    assert format_steps[1].retry_count == 1
    retry_step = next(
        step for step in result.loop_report.run.steps if step.phase == LoopPhase.RETRY
    )
    assert retry_step.name == "Retry answer format"
    assert retry_step.metadata["reasons"] == ["compact_ordered_list"]


def test_document_format_check_retries_compact_numbered_list_after_intro(tmp_path):
    qa, document = create_processed_mock_qa(tmp_path)

    class IntroFormatRetryChain:
        def __init__(self):
            self.calls = []

        def invoke_with_trace(self, question, self_check_instruction=""):
            self.calls.append(self_check_instruction)
            return SimpleNamespace(
                answer=(
                    "Here are the steps:\n"
                    "1. cite the June launch [1]. "
                    "2. name Alex as owner [1]."
                ),
                retrieved_chunk_count=1,
                citations=[citation_for(document.name)],
                context="Project Phoenix launches in June 2026.",
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            self.calls.append(self_check_instruction)
            return SimpleNamespace(
                answer=(
                    "Here are the steps:\n"
                    "1. cite the June launch [1].\n"
                    "2. name Alex as owner [1]."
                ),
                retrieved_chunk_count=previous_result.retrieved_chunk_count,
                citations=previous_result.citations,
                context=previous_result.context,
            )

    retrieval_chain = IntroFormatRetryChain()
    replace_retrieval_chain(qa, retrieval_chain)

    result = qa.query_with_trace("Summarize Project Phoenix.")

    assert result.answer == (
        "Here are the steps:\n"
        "1. cite the June launch [1].\n"
        "2. name Alex as owner [1]."
    )
    assert len(retrieval_chain.calls) == 2
    assert "compact_ordered_list" in retrieval_chain.calls[1]
    assert result.trace.citations == [citation_for(document.name)]
    assert result.trace.self_check.outcome == "not_verified"
    format_steps = [
        step
        for step in result.loop_report.run.steps
        if step.phase == LoopPhase.FORMAT_CHECK
    ]
    assert [step.output_summary for step in format_steps] == [
        "needs_retry",
        "format_passed",
    ]


def test_document_format_check_fails_closed_when_retry_still_bad(tmp_path):
    qa, document = create_processed_mock_qa(tmp_path)

    class FailedFormatRetryChain:
        def __init__(self):
            self.calls = []

        def invoke_with_trace(self, question, self_check_instruction=""):
            self.calls.append(self_check_instruction)
            return SimpleNamespace(
                answer=(
                    "1. **Launch:** Project Phoenix launches in June 2026 [1]. "
                    "2. **Status:** not_verified [1]."
                ),
                retrieved_chunk_count=1,
                citations=[citation_for(document.name)],
                context="Project Phoenix launches in June 2026.",
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            self.calls.append(self_check_instruction)
            return SimpleNamespace(
                answer=(
                    "1. **Launch:** Project Phoenix launches in June 2026 [1]. "
                    "2. **Status:** not_verified [1]."
                ),
                retrieved_chunk_count=previous_result.retrieved_chunk_count,
                citations=previous_result.citations,
                context=previous_result.context,
            )

    retrieval_chain = FailedFormatRetryChain()
    replace_retrieval_chain(qa, retrieval_chain)

    result = qa.query_with_trace("Summarize Project Phoenix.")

    assert result.answer == answer_loop_module.FORMAT_CHECK_FAILURE_ANSWER
    assert result.trace.error_message == "format_check_failed"
    assert result.trace.self_check is None
    assert result.trace.citations == []
    assert len(retrieval_chain.calls) == 2
    assert "Format retry instruction" in retrieval_chain.calls[1]

    run = result.loop_report.run
    assert run.final_decision == LoopDecision.ERROR
    assert LoopPhase.MECHANICAL_CHECK not in [step.phase for step in run.steps]
    assert LoopPhase.VERIFY not in [step.phase for step in run.steps]
    format_steps = [
        step for step in run.steps if step.phase == LoopPhase.FORMAT_CHECK
    ]
    assert [step.output_summary for step in format_steps] == [
        "needs_retry",
        "needs_retry",
    ]
    assert all(
        "internal_verification_label" in step.metadata["reasons"]
        for step in format_steps
    )


def test_self_check_refuses_when_retry_still_fails(tmp_path):
    qa, document = create_processed_mock_qa(tmp_path)

    class FailedRetryChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer="Project Phoenix launches in June 2026.",
                retrieved_chunk_count=1,
                citations=[citation_for(document.name)],
                context="original context",
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow.",
                retrieved_chunk_count=previous_result.retrieved_chunk_count,
                citations=previous_result.citations,
                context=previous_result.context,
            )

    replace_retrieval_chain(qa, FailedRetryChain())

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == (
        answer_loop_module.SELF_CHECK_REFUSAL_ANSWER
    )
    assert result.trace.self_check.outcome == "needs_refusal"
    assert "self_check_failed_closed" in result.trace.self_check.reasons
    assert "missing_inline_citation" in result.trace.self_check.reasons
    assert result.trace.self_check.retry_attempted is True


def test_self_check_rejects_cited_but_unsupported_answer(tmp_path):
    qa, document = create_processed_mock_qa(tmp_path)
    verifier = enable_fake_verifier(qa, outcome="unsupported")

    class ContradictedCitationChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=1,
                citations=[citation_for(document.name)],
                context="Project Phoenix launches in June 2026.",
                model_thinking="I may cite a nonexistent budget source.",
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=previous_result.retrieved_chunk_count,
                citations=previous_result.citations,
                context=previous_result.context,
                model_thinking="I am still citing source 999.",
            )

    replace_retrieval_chain(qa, ContradictedCitationChain())

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == (
        answer_loop_module.SELF_CHECK_REFUSAL_ANSWER
    )
    assert result.trace.self_check.outcome == "needs_refusal"
    assert result.trace.self_check.reasons == ["llm_verifier_unsupported"]
    assert result.trace.self_check.retry_attempted is False
    assert result.trace.model_thinking == "I may cite a nonexistent budget source."
    public_payload = web_contract_module.query_response_dict(result)
    assert public_payload["trace"]["model_thinking"]["redacted"] is True
    assert "nonexistent budget" not in json.dumps(public_payload)
    assert len(verifier.calls) == 1
    assert "Project Phoenix launches tomorrow [1]." in verifier.calls[0]
    assert "Project Phoenix launches in June 2026." in verifier.calls[0]
    assert result.loop_report.run.final_decision == LoopDecision.REFUSE
    phases = [step.phase for step in result.loop_report.run.steps]
    assert LoopPhase.VERIFY in phases
    assert LoopPhase.REFUSE in phases
    verify_step = next(
        step for step in result.loop_report.run.steps if step.phase == LoopPhase.VERIFY
    )
    assert verify_step.decision == LoopDecision.REFUSE
    assert verify_step.verification.outcome.value == "unsupported"


def test_llm_verifier_marks_real_backend_answer_supported(tmp_path):
    qa, document = create_processed_mock_qa(tmp_path)
    verifier = enable_fake_verifier(qa, outcome="supported")

    class SupportedCitationChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer="Project Phoenix launches in June 2026 [1].",
                retrieved_chunk_count=1,
                citations=[citation_for(document.name)],
                context="Project Phoenix launches in June 2026.",
                model_thinking="I used citation [1] to answer the launch question.",
            )

    replace_retrieval_chain(qa, SupportedCitationChain())

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == "Project Phoenix launches in June 2026 [1]."
    assert result.trace.backend == "openai-compatible"
    assert result.trace.self_check.outcome == "supported"
    assert result.trace.self_check.reasons == [
        "mechanical_checks_passed",
        "llm_verifier_supported",
    ]
    assert result.trace.model_thinking == (
        "I used citation [1] to answer the launch question."
    )
    assert len(verifier.calls) == 1
    assert "When does Project Phoenix launch?" in verifier.calls[0]
    assert "Project Phoenix launches in June 2026." in verifier.calls[0]


def test_loop_middleware_can_block_before_verifier_call(tmp_path):
    class BlockVerifyMiddleware:
        def before_step(self, run, step):
            if step.phase == LoopPhase.VERIFY:
                return GuardrailDecision(
                    decision=LoopDecision.BLOCK,
                    reason="verifier_blocked",
                )
            return None

    qa, document = create_processed_mock_qa(tmp_path)
    qa.loop_middlewares = (BlockVerifyMiddleware(),)
    verifier = enable_fake_verifier(qa, outcome="supported")

    class SupportedCitationChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer="Project Phoenix launches in June 2026 [1].",
                retrieved_chunk_count=1,
                citations=[citation_for(document.name)],
                context="Project Phoenix launches in June 2026.",
                model_thinking="I may cite a nonexistent budget source.",
            )

    replace_retrieval_chain(qa, SupportedCitationChain())

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert verifier.calls == []
    assert result.answer == "A loop guardrail blocked this query before it could complete."
    assert result.trace.error_message == "verifier_blocked"
    assert result.trace.model_thinking is None
    assert result.loop_report.run.final_decision == LoopDecision.BLOCK
    phases = [step.phase for step in result.loop_report.run.steps]
    assert LoopPhase.MECHANICAL_CHECK in phases
    assert LoopPhase.VERIFY not in phases


def test_self_check_retries_hallucinated_inline_citation_id(tmp_path):
    qa, document = create_processed_mock_qa(tmp_path)
    verifier = enable_fake_verifier(qa, outcome="supported")

    class HallucinatedCitationChain:
        def __init__(self):
            self.calls = []

        def invoke_with_trace(self, question, self_check_instruction=""):
            self.calls.append(self_check_instruction)
            return SimpleNamespace(
                answer=(
                    "Project Phoenix launches in June 2026 [1]. "
                    "Budget is $10 [999]."
                ),
                retrieved_chunk_count=1,
                citations=[citation_for(document.name)],
                context="Project Phoenix launches in June 2026.",
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            self.calls.append(self_check_instruction)
            return SimpleNamespace(
                answer="Project Phoenix launches in June 2026 [1].",
                retrieved_chunk_count=previous_result.retrieved_chunk_count,
                citations=previous_result.citations,
                context=previous_result.context,
                model_thinking="I removed the invalid [999] citation.",
            )

    retrieval_chain = HallucinatedCitationChain()
    replace_retrieval_chain(qa, retrieval_chain)

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == "Project Phoenix launches in June 2026 [1]."
    assert result.trace.self_check.outcome == "supported"
    assert result.trace.self_check.retry_attempted is True
    assert len(retrieval_chain.calls) == 2
    assert "invalid_inline_citation" in retrieval_chain.calls[1]
    assert result.trace.model_thinking == "I removed the invalid [999] citation."
    assert len(verifier.calls) == 1
    assert "[999]" not in verifier.calls[0]


def test_self_check_refuses_when_retry_keeps_hallucinated_inline_citation_id(tmp_path):
    qa, document = create_processed_mock_qa(tmp_path)
    verifier = enable_fake_verifier(qa, outcome="supported")

    class StillHallucinatedCitationChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer=(
                    "Project Phoenix launches in June 2026 [1]. "
                    "Budget is $10 [999]."
                ),
                retrieved_chunk_count=1,
                citations=[citation_for(document.name)],
                context="Project Phoenix launches in June 2026.",
                model_thinking="I may cite a nonexistent budget source.",
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            return SimpleNamespace(
                answer=(
                    "Project Phoenix launches in June 2026 [1]. "
                    "Budget is $10 [999]."
                ),
                retrieved_chunk_count=previous_result.retrieved_chunk_count,
                citations=previous_result.citations,
                context=previous_result.context,
                model_thinking="I am still citing source 999.",
            )

    replace_retrieval_chain(qa, StillHallucinatedCitationChain())

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == (
        answer_loop_module.SELF_CHECK_REFUSAL_ANSWER
    )
    assert result.trace.self_check.outcome == "needs_refusal"
    assert "self_check_failed_closed" in result.trace.self_check.reasons
    assert "invalid_inline_citation" in result.trace.self_check.reasons
    assert result.trace.self_check.retry_attempted is True
    assert result.trace.model_thinking == "I am still citing source 999."
    public_payload = web_contract_module.query_response_dict(result)
    assert public_payload["trace"]["model_thinking"]["redacted"] is True
    assert "source 999" not in json.dumps(public_payload)
    assert verifier.calls == []


@pytest.mark.parametrize(
    "verifier_kwargs, expected_reasons",
    [
        ({"outcome": "insufficient"}, ["llm_verifier_insufficient"]),
        ({"raw_response": "not-json"}, ["llm_verifier_parse_failed", "missing_json"]),
        ({"error": RuntimeError("verifier down")}, ["llm_verifier_error"]),
    ],
)
def test_llm_verifier_failures_refuse_answer(
    tmp_path, verifier_kwargs, expected_reasons
):
    qa, document = create_processed_mock_qa(tmp_path)
    enable_fake_verifier(qa, **verifier_kwargs)

    class SupportedCitationChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer="Project Phoenix launches in June 2026 [1].",
                retrieved_chunk_count=1,
                citations=[citation_for(document.name)],
                context="Project Phoenix launches in June 2026.",
            )

    replace_retrieval_chain(qa, SupportedCitationChain())

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == (
        answer_loop_module.SELF_CHECK_REFUSAL_ANSWER
    )
    assert result.trace.self_check.outcome == "needs_refusal"
    assert result.trace.self_check.reasons == expected_reasons


def test_self_check_rejects_inverted_relationship_claim(tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)
    enable_fake_verifier(qa, outcome="unsupported")
    citation = AnswerCitation(
        citation_id=1,
        source_name="acquisition.txt",
        page=None,
        chunk_index=0,
        excerpt="Alice acquired Bob in 2026.",
    )

    class InvertedRelationshipChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer="Bob acquired Alice in 2026 [1].",
                retrieved_chunk_count=1,
                citations=[citation],
                context=citation.excerpt,
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            return SimpleNamespace(
                answer="Bob acquired Alice in 2026 [1].",
                retrieved_chunk_count=previous_result.retrieved_chunk_count,
                citations=previous_result.citations,
                context=previous_result.context,
            )

    replace_retrieval_chain(qa, InvertedRelationshipChain())

    result = qa.query_with_trace("Who acquired whom?", context_provider="document")

    assert result.answer == (
        answer_loop_module.SELF_CHECK_REFUSAL_ANSWER
    )
    assert result.trace.self_check.outcome == "needs_refusal"
    assert result.trace.self_check.reasons == ["llm_verifier_unsupported"]


def test_self_check_rejects_denied_extractive_claim(tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)
    citation = AnswerCitation(
        citation_id=1,
        source_name="denial.txt",
        page=None,
        chunk_index=0,
        excerpt="Project Phoenix launches tomorrow is false.",
    )

    class DeniedClaimChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=1,
                citations=[citation],
                context=citation.excerpt,
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=previous_result.retrieved_chunk_count,
                citations=previous_result.citations,
                context=previous_result.context,
            )

    replace_retrieval_chain(qa, DeniedClaimChain())

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == (
        answer_loop_module.SELF_CHECK_REFUSAL_ANSWER
    )
    assert result.trace.self_check.outcome == "needs_refusal"
    assert "citation_text_does_not_support_answer" in result.trace.self_check.reasons


def test_self_check_rejects_later_refuted_repeated_claim(tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)
    citation = AnswerCitation(
        citation_id=1,
        source_name="later-denial.txt",
        page=None,
        chunk_index=0,
        excerpt=(
            "Project Phoenix launches tomorrow. "
            "Project Phoenix launches tomorrow is false."
        ),
    )

    class LaterDeniedClaimChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=1,
                citations=[citation],
                context=citation.excerpt,
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=previous_result.retrieved_chunk_count,
                citations=previous_result.citations,
                context=previous_result.context,
            )

    replace_retrieval_chain(qa, LaterDeniedClaimChain())

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == (
        answer_loop_module.SELF_CHECK_REFUSAL_ANSWER
    )
    assert result.trace.self_check.outcome == "needs_refusal"
    assert "citation_text_does_not_support_answer" in result.trace.self_check.reasons


@pytest.mark.parametrize(
    "excerpt",
    [
        "Project Phoenix launches tomorrow, but that is false.",
        "Project Phoenix launches tomorrow; however, that is false.",
        "Project Phoenix launches tomorrow, although that is not true.",
        "Project Phoenix launches tomorrow, which is incorrect.",
    ],
)
def test_self_check_rejects_connector_refuted_extractive_claim(tmp_path, excerpt):
    qa, _document = create_processed_mock_qa(tmp_path)
    citation = AnswerCitation(
        citation_id=1,
        source_name="connector-denial.txt",
        page=None,
        chunk_index=0,
        excerpt=excerpt,
    )

    class ConnectorDeniedClaimChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=1,
                citations=[citation],
                context=citation.excerpt,
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=previous_result.retrieved_chunk_count,
                citations=previous_result.citations,
                context=previous_result.context,
            )

    replace_retrieval_chain(qa, ConnectorDeniedClaimChain())

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == (
        answer_loop_module.SELF_CHECK_REFUSAL_ANSWER
    )
    assert result.trace.self_check.outcome == "needs_refusal"
    assert "citation_text_does_not_support_answer" in result.trace.self_check.reasons


@pytest.mark.parametrize(
    "excerpt",
    [
        "Project Phoenix launches tomorrow, but that claim is false.",
        "Project Phoenix launches tomorrow, but that claim has been denied.",
        "Project Phoenix launches tomorrow, but that claim has been rejected.",
        "Project Phoenix launches tomorrow, but that claim has been debunked.",
        "Project Phoenix launches tomorrow, but that claim is contradicted by the schedule.",
        "Project Phoenix launches tomorrow, but that claim is untrue.",
        "Project Phoenix launches tomorrow, but that claim is unsupported.",
        "Project Phoenix launches tomorrow, but that claim is not supported by the schedule.",
        "Project Phoenix launches tomorrow, but that claim is disputed.",
        "Project Phoenix launches tomorrow, but that claim is inaccurate.",
        "Project Phoenix launches tomorrow, but that claim is baseless.",
        "Project Phoenix launches tomorrow, but that claim is unfounded.",
        "Project Phoenix launches tomorrow, but that claim has been disproven.",
        "Project Phoenix launches tomorrow, but that claim has been disproved.",
        "Project Phoenix launches tomorrow, but the claim is false.",
        "Project Phoenix launches tomorrow, but the prior claim is false.",
        "Project Phoenix launches tomorrow, but the statement is false.",
        "Project Phoenix launches tomorrow, but this assertion has been refuted.",
        "Project Phoenix launches tomorrow, but this claim is not true.",
    ],
)
def test_self_check_rejects_connector_noun_phrase_refutation(tmp_path, excerpt):
    qa, _document = create_processed_mock_qa(tmp_path)
    citation = AnswerCitation(
        citation_id=1,
        source_name="connector-noun-denial.txt",
        page=None,
        chunk_index=0,
        excerpt=excerpt,
    )

    class ConnectorNounDeniedClaimChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=1,
                citations=[citation],
                context=citation.excerpt,
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=previous_result.retrieved_chunk_count,
                citations=previous_result.citations,
                context=previous_result.context,
            )

    replace_retrieval_chain(qa, ConnectorNounDeniedClaimChain())

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == (
        answer_loop_module.SELF_CHECK_REFUSAL_ANSWER
    )
    assert result.trace.self_check.outcome == "needs_refusal"
    assert "citation_text_does_not_support_answer" in result.trace.self_check.reasons


@pytest.mark.parametrize(
    "excerpt",
    [
        "Project Phoenix launches tomorrow, which is correct.",
        "Project Phoenix launches tomorrow, but that claim is correct.",
        "Project Phoenix launches tomorrow, but that claim has not been rejected.",
        "Project Phoenix launches tomorrow, but that claim has not been disproven.",
    ],
)
def test_self_check_does_not_treat_positive_connector_as_denial(tmp_path, excerpt):
    qa, _document = create_processed_mock_qa(tmp_path)
    citation = AnswerCitation(
        citation_id=1,
        source_name="positive-connector.txt",
        page=None,
        chunk_index=0,
        excerpt=excerpt,
    )

    class PositiveConnectorClaimChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=1,
                citations=[citation],
                context=citation.excerpt,
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=previous_result.retrieved_chunk_count,
                citations=previous_result.citations,
                context=previous_result.context,
            )

    replace_retrieval_chain(qa, PositiveConnectorClaimChain())

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == "Project Phoenix launches tomorrow [1]."
    assert result.trace.self_check.outcome == "not_verified"
    assert result.trace.self_check.reasons == [
        "mechanical_checks_passed",
        "verifier_unavailable_mock_backend",
    ]


def test_self_check_rejects_prefix_refuted_extractive_claim(tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)
    citation = AnswerCitation(
        citation_id=1,
        source_name="prefix-denial.txt",
        page=None,
        chunk_index=0,
        excerpt="It is false that Project Phoenix launches tomorrow.",
    )

    class PrefixDeniedClaimChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=1,
                citations=[citation],
                context=citation.excerpt,
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=previous_result.retrieved_chunk_count,
                citations=previous_result.citations,
                context=previous_result.context,
            )

    replace_retrieval_chain(qa, PrefixDeniedClaimChain())

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == (
        answer_loop_module.SELF_CHECK_REFUSAL_ANSWER
    )
    assert result.trace.self_check.outcome == "needs_refusal"
    assert "citation_text_does_not_support_answer" in result.trace.self_check.reasons


@pytest.mark.parametrize(
    "excerpt",
    [
        "Project Phoenix launches tomorrow? No.",
        "Project Phoenix launches tomorrow? No, it launches in June 2026.",
    ],
)
def test_self_check_rejects_qa_style_denied_extractive_claim(tmp_path, excerpt):
    qa, _document = create_processed_mock_qa(tmp_path)
    citation = AnswerCitation(
        citation_id=1,
        source_name="qa-denial.txt",
        page=None,
        chunk_index=0,
        excerpt=excerpt,
    )

    class QaDeniedClaimChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=1,
                citations=[citation],
                context=citation.excerpt,
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=previous_result.retrieved_chunk_count,
                citations=previous_result.citations,
                context=previous_result.context,
            )

    replace_retrieval_chain(qa, QaDeniedClaimChain())

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == (
        answer_loop_module.SELF_CHECK_REFUSAL_ANSWER
    )
    assert result.trace.self_check.outcome == "needs_refusal"
    assert "citation_text_does_not_support_answer" in result.trace.self_check.reasons


def test_self_check_does_not_treat_no_later_qualifier_as_denial(tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)
    citation = AnswerCitation(
        citation_id=1,
        source_name="schedule.txt",
        page=None,
        chunk_index=0,
        excerpt="Project Phoenix launches tomorrow no later than noon.",
    )

    class QualifiedClaimChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=1,
                citations=[citation],
                context=citation.excerpt,
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=previous_result.retrieved_chunk_count,
                citations=previous_result.citations,
                context=previous_result.context,
            )

    replace_retrieval_chain(qa, QualifiedClaimChain())

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == "Project Phoenix launches tomorrow [1]."
    assert result.trace.self_check.outcome == "not_verified"
    assert result.trace.self_check.reasons == [
        "mechanical_checks_passed",
        "verifier_unavailable_mock_backend",
    ]


def test_self_check_rejects_false_premise_question_token_laundering(tmp_path):
    qa, document = create_processed_mock_qa(tmp_path)
    enable_fake_verifier(qa, outcome="unsupported")

    class FalsePremiseChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=1,
                citations=[citation_for(document.name)],
                context="Project Phoenix launches in June 2026.",
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=previous_result.retrieved_chunk_count,
                citations=previous_result.citations,
                context=previous_result.context,
            )

    replace_retrieval_chain(qa, FalsePremiseChain())

    result = qa.query_with_trace("Does Project Phoenix launch tomorrow?")

    assert result.answer == (
        answer_loop_module.SELF_CHECK_REFUSAL_ANSWER
    )
    assert result.trace.self_check.outcome == "needs_refusal"
    assert result.trace.self_check.reasons == ["llm_verifier_unsupported"]


def test_self_check_rejects_unchecked_unicode_claim(tmp_path):
    qa, document = create_processed_mock_qa(tmp_path)
    enable_fake_verifier(qa, outcome="unsupported")

    class UnicodeClaimChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer="Привет [1].",
                retrieved_chunk_count=1,
                citations=[citation_for(document.name)],
                context="Project Phoenix launches in June 2026.",
            )

    replace_retrieval_chain(qa, UnicodeClaimChain())

    result = qa.query_with_trace("What greeting is in the document?")

    assert result.answer == (
        answer_loop_module.SELF_CHECK_REFUSAL_ANSWER
    )
    assert result.trace.self_check.outcome == "needs_refusal"
    assert result.trace.self_check.reasons == ["llm_verifier_unsupported"]


def test_self_check_retry_reuses_original_prompt_evidence(tmp_path):
    qa, document = create_processed_mock_qa(tmp_path)

    class DriftingEvidenceChain:
        def __init__(self):
            self.invoke_calls = 0
            self.retry_context = None

        def invoke_with_trace(self, question, self_check_instruction=""):
            self.invoke_calls += 1
            if self.invoke_calls == 1:
                return SimpleNamespace(
                    answer="Project Phoenix launches in June 2026.",
                    retrieved_chunk_count=1,
                    citations=[citation_for(document.name)],
                    context="original prompt evidence",
                )
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=1,
                citations=[citation_for("drift.txt")],
                context="drifted prompt evidence",
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            self.retry_context = previous_result.context
            return SimpleNamespace(
                answer="Project Phoenix launches in June 2026 [1].",
                retrieved_chunk_count=previous_result.retrieved_chunk_count,
                citations=previous_result.citations,
                context=previous_result.context,
            )

    retrieval_chain = DriftingEvidenceChain()
    replace_retrieval_chain(qa, retrieval_chain)

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert retrieval_chain.invoke_calls == 1
    assert retrieval_chain.retry_context == "original prompt evidence"
    assert result.trace.citations[0].source_name == document.name
    assert result.trace.self_check.outcome == "not_verified"
    assert result.trace.self_check.reasons == [
        "mechanical_checks_passed",
        "verifier_unavailable_mock_backend",
    ]
    assert result.trace.self_check.retry_attempted is True


def test_document_chunks_record_citation_metadata(tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)

    assert qa.vector_store.documents[0].metadata["chunk_index"] == 0


def test_status_reports_configured_backend_before_initialization(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_COMPAT_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDINGS_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_COMPAT_EMBED_MODEL", raising=False)
    monkeypatch.setenv("OLLAMA_THINK_LEVEL", "max")
    qa = DocumentQA(
        device="cpu",
        fast_mode=False,
        llm_backend="openai-compatible",
        model_id="gpt-oss:20b",
    )

    status = qa.status()

    assert status.profile_label == "QUALITY"
    assert status.configured_backend == "openai-compatible"
    assert status.active_backend == "openai-compatible"
    assert status.active_model_label == "OpenAI-compatible (gpt-oss:20b)"
    assert status.embeddings_model == ""
    assert status.embeddings_device == "external"
    assert status.ready_for_queries is False
    assert status.mock_mode is False
    assert status.processing_report is None


def test_max_output_tokens_defaults_to_quality_and_fast_profiles(monkeypatch):
    monkeypatch.delenv("MAX_OUTPUT_TOKENS", raising=False)

    quality = DocumentQA(fast_mode=False, llm_backend="mock")
    fast = DocumentQA(fast_mode=True, llm_backend="mock")

    assert quality.profile["max_new_tokens"] == 1024
    assert quality.status().max_output_tokens == 1024
    assert fast.profile["max_new_tokens"] == 384
    assert fast.status().max_output_tokens == 384


def test_max_output_tokens_env_overrides_profile(monkeypatch):
    monkeypatch.setenv("MAX_OUTPUT_TOKENS", "2048")

    qa = DocumentQA(fast_mode=True, llm_backend="mock")

    assert qa.profile["max_new_tokens"] == 2048
    assert qa.max_output_tokens == 2048
    assert qa.status().max_output_tokens == 2048


def test_max_output_tokens_env_is_clamped(monkeypatch):
    monkeypatch.setenv("MAX_OUTPUT_TOKENS", "999999")
    high = DocumentQA(fast_mode=False, llm_backend="mock")
    monkeypatch.setenv("MAX_OUTPUT_TOKENS", "1")
    low = DocumentQA(fast_mode=False, llm_backend="mock")

    assert high.max_output_tokens == 8192
    assert low.max_output_tokens == 64


def test_mock_backend_ignores_ollama_think_level_env(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "mock")
    monkeypatch.setenv("LLM_MODEL", "gpt-oss:20b")
    monkeypatch.setenv("OLLAMA_THINK_LEVEL", "max")

    qa = DocumentQA(fast_mode=True)

    assert qa.llm_backend == "mock"
    assert qa.status().active_backend == "mock"


def test_auto_status_before_initialization_reports_local_first_plan(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDINGS_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_EMBED_MODEL", raising=False)
    qa = DocumentQA(fast_mode=True, llm_backend="auto")

    status = qa.status()

    assert status.configured_backend == "auto"
    assert status.active_backend == "auto"
    assert status.active_model_label == "Auto (Ollama nemotron-3-nano:4b)"
    assert status.embeddings_model == DEFAULT_OLLAMA_EMBEDDINGS_MODEL
    assert status.embeddings_device == "external"
    assert "Qwen" not in status.active_model_label


def test_embeddings_default_to_cpu_on_mps_device():
    qa = DocumentQA(
        device="mps",
        fast_mode=True,
        llm_backend="mock",
    )

    assert qa.device == "mps"
    assert qa.embeddings_device == "cpu"
    assert qa.status().embeddings_device == "cpu"


def test_embeddings_device_can_be_explicitly_cpu_on_accelerated_device():
    qa = DocumentQA(
        device="mps",
        embeddings_device="cpu",
        fast_mode=True,
        llm_backend="mock",
    )

    assert qa.embeddings_device == "cpu"


def test_initialize_embeddings_uses_local_cpu_embeddings():
    qa = DocumentQA(
        device="mps",
        fast_mode=True,
        llm_backend="mock",
    )

    qa._initialize_embeddings()

    assert qa.embeddings_model == "local-hashing-384"
    assert qa.embeddings_device == "cpu"
    assert isinstance(qa.embeddings, LocalHashingEmbeddings)
    assert len(qa.embeddings.embed_query("Project Phoenix")) == 384


def test_generic_model_env_configures_ollama_chat_and_embeddings(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "custom-chat:4b")
    monkeypatch.setenv("OLLAMA_MODEL", "ignored-chat:4b")
    monkeypatch.setenv("EMBEDDINGS_MODEL", "custom-embed")
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", "ignored-embed")

    qa = DocumentQA(fast_mode=True, llm_backend="ollama")

    assert qa.ollama_model == "custom-chat:4b"
    assert qa.embeddings_model == "custom-embed"
    assert qa.embeddings_device == "external"
    assert qa.status().active_model_label == "Ollama (custom-chat:4b)"


@pytest.mark.parametrize(
    "base_url, expected",
    [
        ("http://localhost:11434/", "http://localhost:11434"),
        ("http://127.0.0.1:11434", "http://127.0.0.1:11434"),
        ("http://[::1]:11434", "http://[::1]:11434"),
    ],
)
def test_normalize_ollama_base_url_accepts_loopback_forms(base_url, expected):
    assert normalize_ollama_base_url(base_url) == expected


@pytest.mark.parametrize(
    "base_url",
    [
        "not-a-url",
        "file:///tmp/ollama",
        "http://user:pass@localhost:11434",
        "http://localhost:bad",
        "http://gateway.example:11434",
        "http://localhost:11434/api",
        "http://localhost:11434?token=x",
        "http://localhost:11434#fragment",
    ],
)
def test_normalize_ollama_base_url_rejects_unsafe_forms(base_url):
    with pytest.raises(RuntimeError, match="OLLAMA_BASE_URL"):
        normalize_ollama_base_url(base_url)


@pytest.mark.parametrize(
    "base_url, expected",
    [
        ("http://user:SECRET_PASSWORD@localhost:11434", "http://localhost:11434"),
        (
            "http://localhost:11434?api_key=SECRET_QUERY_TOKEN",
            "http://localhost:11434",
        ),
        ("http://localhost:11434#SECRET_FRAGMENT", "http://localhost:11434"),
        ("http://localhost:11434/SECRET_PATH", "http://localhost:11434"),
        ("not-a-url", "<invalid>"),
        ("", "<unset>"),
    ],
)
def test_safe_ollama_base_url_for_error_redacts_secrets(base_url, expected):
    assert safe_ollama_base_url_for_error(base_url) == expected


@pytest.mark.parametrize(
    "unsafe_base_url, secret",
    [
        ("http://user:SECRET_PASSWORD@localhost:11434", "SECRET_PASSWORD"),
        (
            "http://localhost:11434?api_key=SECRET_QUERY_TOKEN",
            "SECRET_QUERY_TOKEN",
        ),
        ("http://localhost:11434#SECRET_FRAGMENT", "SECRET_FRAGMENT"),
        ("http://localhost:11434/SECRET_PATH", "SECRET_PATH"),
    ],
)
def test_ollama_initialization_error_redacts_unsafe_base_url(
    monkeypatch, unsafe_base_url, secret
):
    monkeypatch.setenv("OLLAMA_BASE_URL", unsafe_base_url)
    qa = DocumentQA(fast_mode=True, llm_backend="ollama")

    with pytest.raises(RuntimeError) as exc_info:
        qa._initialize_llm()

    chain_messages = exception_chain_messages(exc_info.value)
    assert chain_messages
    assert all(secret not in message for message in chain_messages)
    assert all(unsafe_base_url not in message for message in chain_messages)
    assert any("http://localhost:11434" in message for message in chain_messages)


def test_ollama_backend_rejects_remote_base_url_before_request(monkeypatch):
    request_attempted = False

    def forbidden_open(request, timeout):
        nonlocal request_attempted
        request_attempted = True
        raise AssertionError("remote Ollama request should not be sent")

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gateway.example:11434")
    monkeypatch.setattr(document_qa_module.OLLAMA_NO_PROXY_OPENER, "open", forbidden_open)
    qa = DocumentQA(fast_mode=True, llm_backend="ollama")

    with pytest.raises(RuntimeError) as exc_info:
        qa._initialize_llm()

    assert request_attempted is False
    assert "OLLAMA_BASE_URL" in str(exc_info.value)
    assert "openai-compatible" in str(exc_info.value)


def test_rejects_removed_embedding_model_names():
    with pytest.raises(RuntimeError, match="Unsupported embeddings_model"):
        DocumentQA(
            embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
            fast_mode=True,
            llm_backend="mock",
        )


def test_mock_backend_rejects_unsupported_embeddings_model_env(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "mock")
    monkeypatch.setenv(
        "EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    with pytest.raises(RuntimeError, match="Unsupported embeddings_model"):
        DocumentQA(fast_mode=True)


def test_default_local_hashing_embeddings_retrieve_common_launch_paraphrase(
    tmp_path,
):
    document = tmp_path / "phoenix-plan.txt"
    document.write_text(
        "Budget approval notes say the finance review is complete.\n\n"
        "Project Phoenix launch date is June 2026. The release goes to customers then.\n\n"
        "Support staffing notes describe the help desk rotation.",
        encoding="utf-8",
    )
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.profile["splitter_chunk_size"] = 90
    qa.profile["splitter_chunk_overlap"] = 0
    qa.profile["retrieval_k"] = 1
    qa.profile["retrieval_fetch_k"] = 3

    qa.process_document(str(document))
    result = qa.query_with_trace("When does it go live?")

    assert len(result.trace.citations) == 1
    assert "June 2026" in result.trace.citations[0].excerpt
    assert "June 2026" in result.answer


def test_ollama_embeddings_use_api_embed(monkeypatch):
    requests = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps({"embeddings": [[1.0, 0.0, 0.0]]}).encode("utf-8")

    def fake_open_ollama_request_no_proxy(request, *, timeout):
        requests.append(
            {
                "url": request.full_url,
                "payload": json.loads(request.data.decode("utf-8")),
                "timeout": timeout,
            }
        )
        return FakeResponse()

    monkeypatch.setenv("EMBEDDINGS_MODEL", "embeddinggemma")
    monkeypatch.setattr(
        document_qa_module,
        "open_ollama_request_no_proxy",
        fake_open_ollama_request_no_proxy,
    )
    qa = DocumentQA(fast_mode=True, llm_backend="ollama")

    qa._initialize_embeddings()
    vector = qa.embeddings.embed_query("Project Phoenix")

    assert isinstance(qa.embeddings, OllamaEmbeddings)
    assert vector == [1.0, 0.0, 0.0]
    assert requests[0] == {
        "url": "http://localhost:11434/api/embed",
        "payload": {
            "model": "embeddinggemma",
            "input": "ok",
            "truncate": True,
        },
        "timeout": 120,
    }
    assert requests[1]["payload"]["input"] == "Project Phoenix"


def test_ollama_embeddings_reject_remote_base_url_before_request(monkeypatch):
    request_attempted = False

    def forbidden_open(request, timeout):
        nonlocal request_attempted
        request_attempted = True
        raise AssertionError("remote Ollama embeddings request should not be sent")

    monkeypatch.setattr(document_qa_module.OLLAMA_NO_PROXY_OPENER, "open", forbidden_open)

    with pytest.raises(RuntimeError, match="OLLAMA_BASE_URL"):
        OllamaEmbeddings(
            model="embeddinggemma",
            base_url="http://gateway.example:11434",
        )

    assert request_attempted is False


def test_ollama_embeddings_http_error_body_is_not_reflected(monkeypatch):
    class SecretHttpError(urllib.error.HTTPError):
        def __init__(self):
            super().__init__(
                url="http://localhost:11434/api/embed",
                code=500,
                msg="SECRET_STATUS_TEXT",
                hdrs={},
                fp=None,
            )

        def read(self):
            return b"SECRET_BODY_TOKEN"

    def fake_open_ollama_request_no_proxy(request, *, timeout):
        raise SecretHttpError()

    monkeypatch.setattr(
        document_qa_module,
        "open_ollama_request_no_proxy",
        fake_open_ollama_request_no_proxy,
    )
    embeddings = OllamaEmbeddings(model="embeddinggemma")

    with pytest.raises(RuntimeError) as exc_info:
        embeddings.validate_model_available()

    message = str(exc_info.value)
    assert "HTTP 500" in message
    assert "SECRET_BODY_TOKEN" not in message
    assert "SECRET_STATUS_TEXT" not in message


def test_ollama_embeddings_url_error_reason_is_not_reflected(monkeypatch):
    def fake_open_ollama_request_no_proxy(request, *, timeout):
        raise urllib.error.URLError("proxy failed with SECRET_REASON_TOKEN")

    monkeypatch.setattr(
        document_qa_module,
        "open_ollama_request_no_proxy",
        fake_open_ollama_request_no_proxy,
    )
    embeddings = OllamaEmbeddings(model="embeddinggemma")

    with pytest.raises(RuntimeError) as exc_info:
        embeddings.validate_model_available()

    message = str(exc_info.value)
    assert "http://localhost:11434/api/embed" in message
    assert "SECRET_REASON_TOKEN" not in message
    assert "proxy failed" not in message


@pytest.mark.parametrize(
    "payload, expected_error",
    [
        ({"embeddings": []}, "lacked vectors"),
        ({"embeddings": [[1.0], [2.0]]}, "unexpected length"),
        ({"embeddings": [["not-a-number"]]}, "non-numeric"),
        ({"embeddings": [[float("nan")]]}, "non-finite"),
        ({"embeddings": [[float("inf")]]}, "non-finite"),
    ],
)
def test_ollama_embeddings_validation_rejects_invalid_scalar_response(
    monkeypatch, payload, expected_error
):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(
        document_qa_module,
        "open_ollama_request_no_proxy",
        lambda request, *, timeout: FakeResponse(),
    )
    embeddings = OllamaEmbeddings(model="embeddinggemma")

    with pytest.raises(RuntimeError, match=expected_error):
        embeddings.validate_model_available()


def test_ollama_embeddings_reject_inconsistent_document_dimensions(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps({"embeddings": [[1.0, 0.0], [1.0]]}).encode(
                "utf-8"
            )

    monkeypatch.setattr(
        document_qa_module,
        "open_ollama_request_no_proxy",
        lambda request, *, timeout: FakeResponse(),
    )
    embeddings = OllamaEmbeddings(model="embeddinggemma")

    with pytest.raises(RuntimeError, match="inconsistent dimensions"):
        embeddings.embed_documents(["alpha", "beta"])


def test_openai_compatible_embeddings_use_embeddings_endpoint(monkeypatch):
    requests = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(
                {
                    "data": [
                        {"index": 0, "embedding": [0.25, 0.75]},
                    ]
                }
            ).encode("utf-8")

    def fake_open_openai_compatible_request(request, *, timeout):
        requests.append(
            {
                "url": request.full_url,
                "headers": {
                    key.lower(): value for key, value in request.header_items()
                },
                "payload": json.loads(request.data.decode("utf-8")),
                "timeout": timeout,
            }
        )
        return FakeResponse()

    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "secret-token")
    monkeypatch.setenv("EMBEDDINGS_MODEL", "text-embedding-local")
    monkeypatch.setenv("OPENAI_COMPAT_TIMEOUT", "9")
    monkeypatch.setattr(
        document_qa_module,
        "open_openai_compatible_request",
        fake_open_openai_compatible_request,
    )
    qa = DocumentQA(fast_mode=True, llm_backend="openai-compatible")

    qa._initialize_embeddings()
    vector = qa.embeddings.embed_query("Project Phoenix")

    assert isinstance(qa.embeddings, OpenAICompatibleEmbeddings)
    assert vector == [0.25, 0.75]
    assert requests[0]["url"] == "http://localhost:8000/v1/embeddings"
    assert requests[0]["headers"]["authorization"] == "Bearer secret-token"
    assert requests[0]["payload"] == {
        "model": "text-embedding-local",
        "input": "ok",
    }
    assert requests[0]["timeout"] == 9
    assert requests[1]["payload"]["input"] == "Project Phoenix"


def test_openai_compatible_embeddings_direct_constructor_rejects_remote_http(
    monkeypatch,
):
    request_attempted = False

    def fake_open_openai_compatible_request(request, *, timeout):
        nonlocal request_attempted
        request_attempted = True
        raise AssertionError("remote HTTP embeddings request should not be sent")

    monkeypatch.setattr(
        document_qa_module,
        "open_openai_compatible_request",
        fake_open_openai_compatible_request,
    )

    with pytest.raises(RuntimeError) as exc_info:
        OpenAICompatibleEmbeddings(
            model="text-embedding-local",
            base_url="http://gateway.example/v1",
            api_key="SECRET_TOKEN",
        )

    chain_messages = exception_chain_messages(exc_info.value)
    assert request_attempted is False
    assert any("HTTPS" in message for message in chain_messages)
    assert all("SECRET_TOKEN" not in message for message in chain_messages)


def test_openai_compatible_embeddings_preserve_response_index_order(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(
                {
                    "data": [
                        {"index": 1, "embedding": [0.0, 1.0]},
                        {"index": 0, "embedding": [1.0, 0.0]},
                    ]
                }
            ).encode("utf-8")

    monkeypatch.setattr(
        document_qa_module,
        "open_openai_compatible_request",
        lambda request, *, timeout: FakeResponse(),
    )
    embeddings = OpenAICompatibleEmbeddings(
        model="text-embedding-local",
        base_url="http://localhost:8000/v1",
    )

    assert embeddings.embed_documents(["alpha", "beta"]) == [
        [1.0, 0.0],
        [0.0, 1.0],
    ]


def test_openai_compatible_embeddings_url_error_reason_is_not_reflected(
    monkeypatch,
):
    def fake_open_openai_compatible_request(request, *, timeout):
        raise urllib.error.URLError(
            "proxy failed with https://gateway.example/v1/SECRET_PATH and "
            "SECRET_REASON_TOKEN"
        )

    monkeypatch.setattr(
        document_qa_module,
        "open_openai_compatible_request",
        fake_open_openai_compatible_request,
    )
    embeddings = OpenAICompatibleEmbeddings(
        model="text-embedding-local",
        base_url="https://gateway.example/v1/SECRET_PATH",
    )

    with pytest.raises(RuntimeError) as exc_info:
        embeddings.validate_model_available()

    chain_messages = exception_chain_messages(exc_info.value)
    assert chain_messages
    assert all("SECRET_PATH" not in message for message in chain_messages)
    assert all("SECRET_REASON_TOKEN" not in message for message in chain_messages)
    assert all("proxy failed" not in message for message in chain_messages)
    assert any("https://gateway.example/embeddings" in message for message in chain_messages)


@pytest.mark.parametrize(
    "payload, expected_error",
    [
        ({"data": []}, "unexpected length"),
        (
            {
                "data": [
                    {"index": 0, "embedding": [1.0]},
                    {"index": 1, "embedding": [2.0]},
                ]
            },
            "unexpected length",
        ),
        (
            {"data": [{"index": 0, "embedding": ["not-a-number"]}]},
            "non-numeric",
        ),
        (
            {"data": [{"index": 0, "embedding": [float("nan")]}]},
            "non-finite",
        ),
        (
            {"data": [{"index": 0, "embedding": [float("inf")]}]},
            "non-finite",
        ),
    ],
)
def test_openai_compatible_embeddings_validation_rejects_invalid_scalar_response(
    monkeypatch, payload, expected_error
):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(
        document_qa_module,
        "open_openai_compatible_request",
        lambda request, *, timeout: FakeResponse(),
    )
    embeddings = OpenAICompatibleEmbeddings(
        model="text-embedding-local",
        base_url="http://localhost:8000/v1",
    )

    with pytest.raises(RuntimeError, match=expected_error):
        embeddings.validate_model_available()


@pytest.mark.parametrize(
    "data",
    [
        [
            {"index": 0, "embedding": [1.0, 0.0]},
            {"index": 0, "embedding": [0.0, 1.0]},
        ],
        [
            {"index": 0, "embedding": [1.0, 0.0]},
            {"embedding": [0.0, 1.0]},
        ],
        [
            {"index": 0, "embedding": [1.0, 0.0]},
            {"index": 2, "embedding": [0.0, 1.0]},
        ],
        [
            {"index": -1, "embedding": [1.0, 0.0]},
            {"index": 1, "embedding": [0.0, 1.0]},
        ],
        [
            {"index": False, "embedding": [1.0, 0.0]},
            {"index": 1, "embedding": [0.0, 1.0]},
        ],
    ],
)
def test_openai_compatible_embeddings_reject_invalid_document_indices(
    monkeypatch, data
):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps({"data": data}).encode("utf-8")

    monkeypatch.setattr(
        document_qa_module,
        "open_openai_compatible_request",
        lambda request, *, timeout: FakeResponse(),
    )
    embeddings = OpenAICompatibleEmbeddings(
        model="text-embedding-local",
        base_url="http://localhost:8000/v1",
    )

    with pytest.raises(RuntimeError, match="invalid indices"):
        embeddings.embed_documents(["alpha", "beta"])


def test_openai_compatible_embeddings_reject_inconsistent_document_dimensions(
    monkeypatch,
):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(
                {
                    "data": [
                        {"index": 0, "embedding": [1.0, 0.0]},
                        {"index": 1, "embedding": [1.0]},
                    ]
                }
            ).encode("utf-8")

    monkeypatch.setattr(
        document_qa_module,
        "open_openai_compatible_request",
        lambda request, *, timeout: FakeResponse(),
    )
    embeddings = OpenAICompatibleEmbeddings(
        model="text-embedding-local",
        base_url="http://localhost:8000/v1",
    )

    with pytest.raises(RuntimeError, match="inconsistent dimensions"):
        embeddings.embed_documents(["alpha", "beta"])


def test_openai_compatible_embeddings_without_model_fail_closed(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.delenv("EMBEDDINGS_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_COMPAT_EMBED_MODEL", raising=False)
    qa = DocumentQA(fast_mode=True, llm_backend="openai-compatible")

    qa._initialize_embeddings()

    assert qa.embeddings is None
    assert "EMBEDDINGS_MODEL" in qa.embeddings_error


def test_failed_unsupported_replacement_keeps_previous_document_queryable(tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)
    previous_vector_store = qa.vector_store
    previous_retrieval_chain = qa.retrieval_chain
    replacement = tmp_path / "replacement.csv"
    replacement.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Unsupported file type: .csv"):
        qa.process_document(str(replacement))

    assert qa.current_document_name == "phoenix.txt"
    assert qa.vector_store is previous_vector_store
    assert qa.retrieval_chain is previous_retrieval_chain
    assert qa.query("Which file did I upload?") == "The indexed file is `phoenix.txt`."
    report = qa.status().processing_report
    assert report.success is False
    assert report.phase == "validate"
    assert report.attempted_document_name == "replacement.csv"
    assert report.active_document_name == "phoenix.txt"
    assert report.file_extension == ".csv"
    assert report.chunk_count == 0
    assert "Unsupported file type: .csv" in report.error_message


def test_query_uses_active_state_snapshot_not_legacy_mixed_state(tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)

    qa.current_document_name = "replacement.txt"
    qa.vector_store = None
    qa.retrieval_chain = None

    assert qa.status().document_name == "phoenix.txt"
    assert qa.query("Which file did I upload?") == "The indexed file is `phoenix.txt`."


def test_concurrent_replacement_uploads_are_serialized(monkeypatch, tmp_path):
    first = tmp_path / "first.txt"
    first.write_text("First document for Project Phoenix.", encoding="utf-8")
    second = tmp_path / "second.txt"
    second.write_text("Second document for Project Phoenix.", encoding="utf-8")
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    qa.embeddings = FakeEmbeddings()
    original_process_document = DocumentQA.process_document
    original_load_documents = DocumentQA._load_documents
    second_upload_attempted = threading.Event()
    first_load_started = threading.Event()
    second_load_started = threading.Event()
    release_first_load = threading.Event()
    errors = []

    def tracked_process_document(self, document_path, text_encoding=None):
        if document_path == str(second):
            second_upload_attempted.set()
        return original_process_document(
            self, document_path, text_encoding=text_encoding
        )

    def delayed_first_load(self, document_path, file_extension, text_encoding=None):
        if document_path == str(first):
            first_load_started.set()
            assert release_first_load.wait(timeout=5)
        elif document_path == str(second):
            second_load_started.set()
        return original_load_documents(
            self, document_path, file_extension, text_encoding=text_encoding
        )

    def process(path):
        try:
            qa.process_document(str(path))
        except Exception as exc:
            errors.append(exc)

    monkeypatch.setattr(DocumentQA, "process_document", tracked_process_document)
    monkeypatch.setattr(DocumentQA, "_load_documents", delayed_first_load)

    first_thread = threading.Thread(target=process, args=(first,))
    first_thread.start()
    assert first_load_started.wait(timeout=5)

    second_thread = threading.Thread(target=process, args=(second,))
    second_thread.start()
    assert second_upload_attempted.wait(timeout=5)
    assert not second_load_started.wait(timeout=0.2)
    release_first_load.set()

    first_thread.join(timeout=5)
    second_thread.join(timeout=5)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert second_load_started.is_set()
    assert errors == []
    assert qa.status().document_name == "second.txt"
    assert qa.query("Which file did I upload?") == "The indexed file is `second.txt`."


def test_failed_ambiguous_text_replacement_keeps_previous_document_queryable(tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)
    previous_vector_store = qa.vector_store
    previous_retrieval_chain = qa.retrieval_chain
    replacement = tmp_path / "ambiguous.txt"
    replacement.write_bytes("Привет Phoenix".encode("cp1251"))

    with pytest.raises(RuntimeError, match="Could not decode text document"):
        qa.process_document(str(replacement))

    assert qa.current_document_name == "phoenix.txt"
    assert qa.vector_store is previous_vector_store
    assert qa.retrieval_chain is previous_retrieval_chain
    assert qa.query("Which file did I upload?") == "The indexed file is `phoenix.txt`."
    report = qa.status().processing_report
    assert report.success is False
    assert report.phase == "load"
    assert report.attempted_document_name == "ambiguous.txt"
    assert report.active_document_name == "phoenix.txt"
    assert report.file_extension == ".txt"
    assert report.text_encoding_mode == "auto"
    assert "Could not decode text document" in report.error_message


def test_failed_embedding_initialization_keeps_previous_document_queryable(
    monkeypatch, tmp_path
):
    qa, _document = create_processed_mock_qa(tmp_path)
    previous_vector_store = qa.vector_store
    previous_retrieval_chain = qa.retrieval_chain
    qa.embeddings = None
    replacement = tmp_path / "replacement.txt"
    replacement.write_text("Replacement document should not commit.", encoding="utf-8")

    def fail_initialize_embeddings(self):
        self.embeddings_error = "boom"
        self.embeddings = None

    monkeypatch.setattr(DocumentQA, "_initialize_embeddings", fail_initialize_embeddings)

    with pytest.raises(RuntimeError, match="Embedding backend is unavailable"):
        qa.process_document(str(replacement))

    assert qa.current_document_name == "phoenix.txt"
    assert qa.vector_store is previous_vector_store
    assert qa.retrieval_chain is previous_retrieval_chain
    assert qa.query("Which file did I upload?") == "The indexed file is `phoenix.txt`."
    report = qa.status().processing_report
    assert report.success is False
    assert report.phase == "initialize_embeddings"
    assert report.attempted_document_name == "replacement.txt"
    assert report.active_document_name == "phoenix.txt"
    assert "Embedding backend is unavailable" in report.error_message
    assert "boom" in report.error_message


def test_empty_replacement_keeps_previous_document_queryable(tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)
    previous_vector_store = qa.vector_store
    previous_retrieval_chain = qa.retrieval_chain
    replacement = tmp_path / "empty.txt"
    replacement.write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="No text chunks were generated"):
        qa.process_document(str(replacement))

    assert qa.current_document_name == "phoenix.txt"
    assert qa.vector_store is previous_vector_store
    assert qa.retrieval_chain is previous_retrieval_chain
    assert qa.query("Which file did I upload?") == "The indexed file is `phoenix.txt`."
    report = qa.status().processing_report
    assert report.success is False
    assert report.phase == "split"
    assert report.attempted_document_name == "empty.txt"
    assert report.active_document_name == "phoenix.txt"
    assert report.chunk_count == 0
    assert "No text chunks were generated" in report.error_message


def test_index_failure_keeps_previous_document_queryable(monkeypatch, tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)
    previous_vector_store = qa.vector_store
    previous_retrieval_chain = qa.retrieval_chain
    replacement = tmp_path / "replacement.txt"
    replacement.write_text("Replacement document should not commit.", encoding="utf-8")

    def fail_from_documents(documents, embedding):
        raise ValueError("index boom")

    monkeypatch.setattr(FaissVectorStore, "from_documents", fail_from_documents)

    with pytest.raises(RuntimeError, match="index boom"):
        qa.process_document(str(replacement))

    assert qa.current_document_name == "phoenix.txt"
    assert qa.vector_store is previous_vector_store
    assert qa.retrieval_chain is previous_retrieval_chain
    assert qa.query("Which file did I upload?") == "The indexed file is `phoenix.txt`."
    report = qa.status().processing_report
    assert report.success is False
    assert report.phase == "index"
    assert report.attempted_document_name == "replacement.txt"
    assert report.active_document_name == "phoenix.txt"
    assert report.chunk_count > 0
    assert "index boom" in report.error_message


def test_chain_failure_keeps_previous_document_queryable(monkeypatch, tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)
    previous_vector_store = qa.vector_store
    previous_retrieval_chain = qa.retrieval_chain
    replacement = tmp_path / "replacement.txt"
    replacement.write_text("Replacement document should not commit.", encoding="utf-8")

    def fail_build_retrieval_chain(self, vector_store, document_name):
        raise RuntimeError("chain boom")

    monkeypatch.setattr(
        DocumentQA, "_build_retrieval_chain", fail_build_retrieval_chain
    )

    with pytest.raises(RuntimeError, match="chain boom"):
        qa.process_document(str(replacement))

    assert qa.current_document_name == "phoenix.txt"
    assert qa.vector_store is previous_vector_store
    assert qa.retrieval_chain is previous_retrieval_chain
    assert qa.query("Which file did I upload?") == "The indexed file is `phoenix.txt`."
    report = qa.status().processing_report
    assert report.success is False
    assert report.phase == "chain"
    assert report.attempted_document_name == "replacement.txt"
    assert report.active_document_name == "phoenix.txt"
    assert report.chunk_count > 0
    assert "chain boom" in report.error_message


def test_successful_upload_report_records_truncation(tmp_path):
    document = tmp_path / "long.txt"
    document.write_text(("Project Phoenix launch notes.\n" * 500), encoding="utf-8")
    qa = DocumentQA(
        fast_mode=True,
        llm_backend="mock",
        max_document_chunks=1,
    )
    qa.embeddings = FakeEmbeddings()

    qa.process_document(str(document))

    report = qa.status().processing_report
    assert report.success is True
    assert report.phase == "complete"
    assert report.chunk_count == 1
    assert report.truncated is True
    assert report.max_chunk_limit == 1


def test_text_loader_sets_source_metadata(tmp_path):
    document = tmp_path / "notes.md"
    document.write_text("# Notes\nProject Phoenix launch notes.", encoding="utf-8")
    qa = DocumentQA(fast_mode=True, llm_backend="mock")

    loaded_documents = qa._load_documents(str(document), ".md")

    assert len(loaded_documents) == 1
    assert loaded_documents[0].page_content.startswith("# Notes")
    assert loaded_documents[0].metadata["source"] == str(document)


def test_load_documents_uses_decode_text_file_override(monkeypatch, tmp_path):
    document = tmp_path / "notes.txt"
    document.write_bytes(b"not used by override")
    qa = DocumentQA(fast_mode=True, llm_backend="mock")

    def fake_decode_text_file(self, document_path, text_encoding=None):
        assert document_path == str(document)
        assert text_encoding == "auto"
        return "decoded through compatibility seam"

    monkeypatch.setattr(DocumentQA, "_decode_text_file", fake_decode_text_file)

    loaded_documents = qa._load_documents(
        str(document),
        ".txt",
        text_encoding="auto",
    )

    assert loaded_documents[0].page_content == "decoded through compatibility seam"
    assert loaded_documents[0].metadata["source"] == str(document)


def test_decode_text_file_uses_decode_supported_text_override(tmp_path):
    document = tmp_path / "notes.txt"
    document.write_bytes(b"override target")
    seen_encodings = []

    class CustomDecodeQA(DocumentQA):
        def _decode_supported_text(self, raw_content, encoding):
            assert raw_content == b"override target"
            seen_encodings.append(encoding)
            return "decoded through supported override"

    qa = CustomDecodeQA(fast_mode=True, llm_backend="mock")

    assert (
        qa._decode_text_file(str(document), text_encoding="utf-8-or-western")
        == "decoded through supported override"
    )
    assert seen_encodings == ["utf-8"]


@pytest.mark.parametrize(
    ("encoding", "text"),
    [
        ("latin-1", "Café"),
        ("latin-1", "Crème brûlée"),
        ("latin-1", "Voilà"),
        ("latin-1", "Résumé"),
        ("latin-1", "naïve façade"),
        ("latin-1", "piñata"),
        ("latin-1", "paño Phoenix"),
        ("latin-1", "mañana Phoenix"),
        ("latin-1", "Málaga"),
        ("latin-1", "Córdoba"),
        ("latin-1", "García"),
        ("latin-1", "María"),
        ("latin-1", "Hélène"),
        ("latin-1", "Québec"),
        ("latin-1", "Montréal"),
        ("latin-1", "Montréal Québec"),
        ("latin-1", "München"),
        ("latin-1", "Zürich"),
        ("latin-1", "Düsseldorf"),
        ("latin-1", "Göteborg"),
        ("latin-1", "Köln"),
        ("latin-1", "Jürgen"),
        ("latin-1", "Bücher"),
        ("latin-1", "Søren"),
        ("latin-1", "København"),
        ("latin-1", "Å"),
        ("latin-1", "Ångström"),
        ("latin-1", "Tromsø"),
        ("latin-1", "smørrebrød"),
        ("latin-1", "A Coruña"),
        ("latin-1", "açaí"),
        ("latin-1", "Ação"),
        ("latin-1", "coração"),
        ("latin-1", "João"),
        ("latin-1", "mãe"),
        ("latin-1", "smörgåsbord"),
        ("latin-1", "El Niño"),
        ("latin-1", "São Paulo"),
        ("latin-1", "François"),
        ("latin-1", "über"),
        ("latin-1", "garçon"),
        ("latin-1", "Area 10 m²"),
        ("latin-1", "Volume 5 cm³"),
        ("latin-1", "Temp 20 °C"),
        ("latin-1", "Price £10"),
        ("latin-1", "Half ½ cup"),
        ("latin-1", "Café Phoenix résumé"),
        ("cp1252", "Café Phoenix — résumé"),
        ("cp1252", "Price €10 — Phoenix"),
        ("cp1252", "¿Cómo estás? Phoenix"),
        ("cp1252", "Trademark ™ Phoenix"),
    ],
)
def test_text_loader_detects_common_non_utf8_encodings(tmp_path, encoding, text):
    document = tmp_path / "legacy.txt"
    document.write_bytes(text.encode(encoding))
    qa = DocumentQA(fast_mode=True, llm_backend="mock")

    loaded_documents = qa._load_documents(
        str(document), ".txt", text_encoding=encoding
    )

    assert loaded_documents[0].page_content == text
    assert "\ufffd" not in loaded_documents[0].page_content


@pytest.mark.parametrize(
    ("encoding", "text"),
    [
        ("utf-8", "Café Phoenix"),
        ("utf-8", "Résumé Phoenix"),
        ("utf-8", "München Phoenix"),
        ("utf-8", "Price €10 — Phoenix"),
        ("utf-8-sig", "Café Phoenix"),
        ("latin-1", "Göteborg Phoenix"),
        ("cp1252", "Price €10 — Phoenix"),
    ],
)
def test_text_loader_utf8_or_western_mode_preserves_text(tmp_path, encoding, text):
    document = tmp_path / "default_text.txt"
    document.write_bytes(text.encode(encoding))
    qa = DocumentQA(fast_mode=True, llm_backend="mock")

    loaded_documents = qa._load_documents(
        str(document), ".txt", text_encoding="utf-8-or-western"
    )

    assert loaded_documents[0].page_content == text
    assert "Ã" not in loaded_documents[0].page_content
    assert "â" not in loaded_documents[0].page_content


@pytest.mark.parametrize(
    ("encoding", "text"),
    [
        ("utf-16-le", "Project Phoenix"),
        ("utf-16-be", "Project Phoenix"),
        ("utf-32-le", "Project Phoenix"),
        ("utf-32-be", "Project Phoenix"),
        ("utf-32", "Project Phoenix"),
    ],
)
def test_text_loader_detects_utf_family_encodings(tmp_path, encoding, text):
    document = tmp_path / "unicode.txt"
    document.write_bytes(text.encode(encoding))
    qa = DocumentQA(fast_mode=True, llm_backend="mock")

    loaded_documents = qa._load_documents(str(document), ".txt")

    assert loaded_documents[0].page_content == text
    assert "\x00" not in loaded_documents[0].page_content


def test_text_loader_uses_confident_detector_for_legacy_encoding(tmp_path):
    text = "Zażółć gęślą jaźń"
    document = tmp_path / "polish.txt"
    document.write_bytes(text.encode("cp1250"))
    qa = DocumentQA(fast_mode=True, llm_backend="mock")

    loaded_documents = qa._load_documents(str(document), ".txt")

    assert loaded_documents[0].page_content == text


@pytest.mark.parametrize(
    ("encoding", "text"),
    [
        ("cp1250", "Dvořák Phoenix"),
        ("cp1250", "město Phoenix"),
        ("cp1251", "Привет Phoenix"),
        ("cp1254", "İstanbul Phoenix"),
        ("cp1257", "māja Phoenix"),
        ("cp1257", "Rīga Phoenix"),
    ],
)
def test_text_loader_uses_explicit_legacy_encoding(tmp_path, encoding, text):
    document = tmp_path / "legacy.txt"
    document.write_bytes(text.encode(encoding))
    qa = DocumentQA(fast_mode=True, llm_backend="mock")

    loaded_documents = qa._load_documents(
        str(document), ".txt", text_encoding=encoding
    )

    assert loaded_documents[0].page_content == text


def test_text_loader_reports_invalid_explicit_encoding(tmp_path):
    document = tmp_path / "legacy.txt"
    document.write_bytes("Phoenix".encode("utf-8"))
    qa = DocumentQA(fast_mode=True, llm_backend="mock")

    with pytest.raises(ValueError, match="Could not decode text document"):
        qa._load_documents(str(document), ".txt", text_encoding="not-a-codec")


@pytest.mark.parametrize(
    ("encoding", "text", "mojibake"),
    [
        ("cp1251", "Привет Phoenix", "Ïðèâåò Phoenix"),
        ("cp1251", "аб Phoenix", "àá Phoenix"),
        ("cp1251", "я и ты Phoenix", "ÿ è òû Phoenix"),
        ("cp1251", "Привет — Phoenix", "Ïðèâåò — Phoenix"),
        ("cp1251", "Привет – Phoenix", "Ïðèâåò – Phoenix"),
        ("cp1251", "Привет… Phoenix", "Ïðèâåò… Phoenix"),
        ("cp1251", "Ј Phoenix", "£ Phoenix"),
        ("cp1251", "Јован Phoenix", "£îâàí Phoenix"),
        ("cp1251", "Ђ Phoenix", "€ Phoenix"),
        ("cp1251", "№ Phoenix", "¹ Phoenix"),
        ("gb18030", "项目 Phoenix", "ÏîÄ¿ Phoenix"),
        ("cp1250", "Ł10 Phoenix", "£10 Phoenix"),
        ("cp1250", "Łódź Phoenix", "£ódŸ Phoenix"),
        ("cp1254", "İstanbul Phoenix", "Ýstanbul Phoenix"),
        ("cp1257", "māja Phoenix", "mâja Phoenix"),
        ("cp1257", "Rīga Phoenix", "Rîga Phoenix"),
        ("cp1250", "Dvořák Phoenix", "Dvoøák Phoenix"),
        ("cp1250", "město Phoenix", "mìsto Phoenix"),
        ("cp1250", "książka Phoenix", "ksi¹¿ka Phoenix"),
        ("cp1250", "Dąb Phoenix", "D¹b Phoenix"),
        ("cp1250", "zażalenie Phoenix", "za¿alenie Phoenix"),
        ("iso-8859-2", "Łódź Phoenix", "£ód¼ Phoenix"),
        ("big5", "項目 Phoenix", "¶µ¥Ø Phoenix"),
    ],
)
def test_text_loader_rejects_ambiguous_legacy_text_instead_of_mojibake(
    tmp_path, encoding, text, mojibake
):
    document = tmp_path / "ambiguous.txt"
    document.write_bytes(text.encode(encoding))
    qa = DocumentQA(fast_mode=True, llm_backend="mock")

    with pytest.raises(ValueError, match="Could not decode text document"):
        qa._load_documents(str(document), ".txt")

    assert document.read_bytes().decode("cp1252") == mojibake


def test_faiss_vector_store_returns_relevant_documents():
    documents = [
        Document(page_content="Project Phoenix launch notes"),
        Document(page_content="Unrelated accounting memo"),
    ]
    vector_store = FaissVectorStore.from_documents(documents, FakeEmbeddings())

    results = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": 1, "fetch_k": 2}
    ).invoke("Phoenix project")

    assert results == [documents[0]]


def test_rejects_unsupported_file_type_before_model_initialization(tmp_path):
    document = tmp_path / "data.csv"
    document.write_text("a,b\n1,2\n", encoding="utf-8")
    qa = DocumentQA(fast_mode=True, llm_backend="mock")

    with pytest.raises(RuntimeError, match="Unsupported file type: .csv"):
        qa.process_document(str(document))

    assert qa.llm is None


def test_rejects_oversized_document_before_model_initialization(tmp_path):
    document = tmp_path / "too-large.txt"
    document.write_text("too large", encoding="utf-8")
    qa = DocumentQA(
        fast_mode=True, llm_backend="mock", max_document_bytes=1
    )

    with pytest.raises(RuntimeError, match="File is too large"):
        qa.process_document(str(document))

    assert qa.llm is None


def test_auto_backend_uses_ollama_when_available(monkeypatch):
    def fake_ollama_loader(self, model_id):
        return MockLLM()

    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("OLLAMA_MODEL", "nemotron-3-nano:4b")
    monkeypatch.setattr(DocumentQA, "_load_ollama_model", fake_ollama_loader)
    qa = DocumentQA(device="cpu", llm_backend="auto")

    qa._initialize_llm()

    assert qa.active_llm_backend == "ollama"
    assert qa.loaded_model_id == "nemotron-3-nano:4b"
    assert qa.loaded_model_label == "Ollama (nemotron-3-nano:4b)"


def test_invalid_llm_backend_env_fails_closed_before_loading_model(monkeypatch):
    load_attempted = False

    def forbidden_ollama_loader(self, model_id):
        nonlocal load_attempted
        load_attempted = True
        raise AssertionError("invalid backend must not select Ollama")

    monkeypatch.setenv("LLM_BACKEND", "mockk")
    monkeypatch.setattr(DocumentQA, "_load_ollama_model", forbidden_ollama_loader)

    with pytest.raises(RuntimeError, match="Unsupported LLM_BACKEND='mockk'"):
        DocumentQA(device="cpu")

    assert load_attempted is False


@pytest.mark.parametrize("backend", ["endpoint", "local"])
def test_removed_hf_backend_names_fail_closed_before_loading_model(
    monkeypatch, backend
):
    load_attempted = False

    def forbidden_ollama_loader(self, model_id):
        nonlocal load_attempted
        load_attempted = True
        raise AssertionError("removed backend must not select Ollama")

    monkeypatch.setattr(DocumentQA, "_load_ollama_model", forbidden_ollama_loader)

    with pytest.raises(RuntimeError, match="Unsupported LLM_BACKEND"):
        DocumentQA(device="cpu", llm_backend=backend)

    assert load_attempted is False


def test_auto_backend_uses_ollama_on_cuda_when_available(monkeypatch):
    def fake_ollama_loader(self, model_id):
        return MockLLM()

    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.setattr(DocumentQA, "_load_ollama_model", fake_ollama_loader)
    qa = DocumentQA(device="cuda", llm_backend="auto")

    qa._initialize_llm()

    assert qa.active_llm_backend == "ollama"
    assert qa.loaded_model_id == "nemotron-3-nano:4b"


def test_auto_backend_uses_ollama_on_mps_when_available(monkeypatch):
    def fake_ollama_loader(self, model_id):
        return MockLLM()

    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.setattr(DocumentQA, "_load_ollama_model", fake_ollama_loader)
    qa = DocumentQA(device="mps", llm_backend="auto")

    qa._initialize_llm()

    assert qa.active_llm_backend == "ollama"
    assert qa.loaded_model_id == "nemotron-3-nano:4b"


def test_auto_backend_fails_closed_when_ollama_unavailable(monkeypatch):
    def failing_ollama_loader(self, model_id):
        raise ValueError("ollama unavailable")

    monkeypatch.setattr(DocumentQA, "_load_ollama_model", failing_ollama_loader)
    qa = DocumentQA(device="mps", llm_backend="auto")

    with pytest.raises(RuntimeError, match="Unable to initialize auto-selected"):
        qa._initialize_llm()

    assert qa.active_llm_backend is None
    assert qa.llm is None


def test_explicit_ollama_backend_uses_ollama(monkeypatch):
    loaded_models = []

    def fake_ollama_loader(self, model_id):
        loaded_models.append(model_id)
        return MockLLM()

    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("OLLAMA_MODEL", "nemotron-3-nano:4b")
    monkeypatch.setattr(DocumentQA, "_load_ollama_model", fake_ollama_loader)
    qa = DocumentQA(device="cpu", llm_backend="ollama")

    qa._initialize_llm()

    assert loaded_models == ["nemotron-3-nano:4b"]
    assert qa.active_llm_backend == "ollama"
    assert qa.loaded_model_id == "nemotron-3-nano:4b"
    assert qa.loaded_model_label == "Ollama (nemotron-3-nano:4b)"
    assert qa.status().active_model_label == "Ollama (nemotron-3-nano:4b)"


def test_explicit_ollama_backend_does_not_mock(monkeypatch):
    def fake_ollama_loader(self, model_id):
        return MockLLM()

    monkeypatch.setattr(DocumentQA, "_load_ollama_model", fake_ollama_loader)
    qa = DocumentQA(device="cpu", llm_backend="ollama")

    qa._initialize_llm()

    assert qa.active_llm_backend == "ollama"


def test_explicit_ollama_backend_failure_fails_closed(monkeypatch):
    def failing_ollama_loader(self, model_id):
        raise ValueError("ollama down")

    monkeypatch.setattr(DocumentQA, "_load_ollama_model", failing_ollama_loader)
    qa = DocumentQA(device="cpu", llm_backend="ollama")

    with pytest.raises(RuntimeError, match="Unable to initialize ollama LLM"):
        qa._initialize_llm()

    assert qa.active_llm_backend is None
    assert qa.llm is None


def test_explicit_openai_compatible_backend_uses_chat_completions(monkeypatch):
    requests = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "ok"}}]}
            ).encode("utf-8")

    def fake_open_openai_compatible_request(request, *, timeout):
        requests.append(
            {
                "url": request.full_url,
                "headers": {
                    key.lower(): value for key, value in request.header_items()
                },
                "payload": json.loads(request.data.decode("utf-8")),
                "timeout": timeout,
            }
        )
        return FakeResponse()

    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "local-chat")
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "secret-token")
    monkeypatch.setenv("OPENAI_COMPAT_TIMEOUT", "7")
    monkeypatch.setattr(
        document_qa_module,
        "open_openai_compatible_request",
        fake_open_openai_compatible_request,
    )
    qa = DocumentQA(device="cpu", llm_backend="openai-compatible")

    qa._initialize_llm()
    answer = qa.llm._call("Question?", stop=["END"])

    assert answer == "ok"
    assert qa.active_llm_backend == "openai-compatible"
    assert qa.loaded_model_id == "local-chat"
    assert qa.loaded_model_label == "OpenAI-compatible (local-chat)"
    assert len(requests) == 2
    assert requests[0]["url"] == "http://localhost:8000/v1/chat/completions"
    assert requests[0]["headers"]["authorization"] == "Bearer secret-token"
    assert requests[0]["timeout"] == 7
    assert requests[0]["payload"]["model"] == "local-chat"
    assert requests[0]["payload"]["messages"] == [
        {"role": "user", "content": "Respond with ok."}
    ]
    assert requests[0]["payload"]["max_tokens"] == 1
    assert requests[1]["payload"]["messages"] == [
        {"role": "user", "content": "Question?"}
    ]
    assert requests[1]["payload"]["stop"] == ["END"]


def test_openai_compatible_llm_direct_constructor_rejects_remote_http(
    monkeypatch,
):
    request_attempted = False

    def fake_open_openai_compatible_request(request, *, timeout):
        nonlocal request_attempted
        request_attempted = True
        raise AssertionError("remote HTTP chat request should not be sent")

    monkeypatch.setattr(
        document_qa_module,
        "open_openai_compatible_request",
        fake_open_openai_compatible_request,
    )
    with pytest.raises(RuntimeError) as exc_info:
        OpenAICompatibleLLM(
            model="local-chat",
            base_url="http://gateway.example/v1",
            api_key="SECRET_TOKEN",
        )

    chain_messages = exception_chain_messages(exc_info.value)
    assert request_attempted is False
    assert any("HTTPS" in message for message in chain_messages)
    assert all("SECRET_TOKEN" not in message for message in chain_messages)


def test_openai_compatible_does_not_use_generic_openai_api_key(monkeypatch):
    requests = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "ok"}}]}
            ).encode("utf-8")

    def fake_open_openai_compatible_request(request, *, timeout):
        requests.append(
            {key.lower(): value for key, value in request.header_items()}
        )
        return FakeResponse()

    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "local-chat")
    monkeypatch.delenv("OPENAI_COMPAT_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "should-not-be-used")
    monkeypatch.setattr(
        document_qa_module,
        "open_openai_compatible_request",
        fake_open_openai_compatible_request,
    )
    qa = DocumentQA(device="cpu", llm_backend="openai-compatible")

    qa._initialize_llm()

    assert "authorization" not in requests[0]


def test_openai_compatible_http_error_body_is_not_reflected(monkeypatch):
    class SecretHttpError(urllib.error.HTTPError):
        def __init__(self):
            super().__init__(
                url="http://localhost:8000/v1/chat/completions",
                code=401,
                msg="Unauthorized SECRET_STATUS_TEXT",
                hdrs={},
                fp=None,
            )

        def read(self):
            return b"debug echo SECRET_API_KEY from proxy"

    def fake_open_openai_compatible_request(request, *, timeout):
        raise SecretHttpError()

    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "local-chat")
    monkeypatch.setattr(
        document_qa_module,
        "open_openai_compatible_request",
        fake_open_openai_compatible_request,
    )
    qa = DocumentQA(device="cpu", llm_backend="openai-compatible")

    with pytest.raises(RuntimeError) as exc_info:
        qa._initialize_llm()

    message = str(exc_info.value)
    inner_error = exc_info.value.__cause__
    assert inner_error is not None
    cause_message = str(inner_error)
    assert "HTTP 401" in message
    assert "HTTP 401" in cause_message
    assert "SECRET_API_KEY" not in message
    assert "SECRET_API_KEY" not in cause_message
    assert "SECRET_STATUS_TEXT" not in message
    assert "SECRET_STATUS_TEXT" not in cause_message
    assert "debug echo" not in message
    assert "debug echo" not in cause_message
    assert inner_error.__cause__ is None


def test_openai_compatible_url_error_reason_is_not_reflected(monkeypatch):
    def fake_open_openai_compatible_request(request, *, timeout):
        raise urllib.error.URLError("proxy failed with SECRET_REASON_TOKEN")

    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "local-chat")
    monkeypatch.setattr(
        document_qa_module,
        "open_openai_compatible_request",
        fake_open_openai_compatible_request,
    )
    qa = DocumentQA(device="cpu", llm_backend="openai-compatible")

    with pytest.raises(RuntimeError) as exc_info:
        qa._initialize_llm()

    message = str(exc_info.value)
    inner_error = exc_info.value.__cause__
    assert inner_error is not None
    cause_message = str(inner_error)
    assert "http://localhost:8000/v1/chat/completions" in message
    assert "http://localhost:8000/v1/chat/completions" in cause_message
    assert "SECRET_REASON_TOKEN" not in message
    assert "SECRET_REASON_TOKEN" not in cause_message
    assert "proxy failed" not in message
    assert "proxy failed" not in cause_message
    assert inner_error.__cause__ is None
    assert inner_error.__context__ is None
    assert all(
        "SECRET_REASON_TOKEN" not in chain_message
        for chain_message in exception_chain_messages(exc_info.value)
    )
    assert all(
        "proxy failed" not in chain_message
        for chain_message in exception_chain_messages(exc_info.value)
    )


def test_openai_compatible_initialization_error_redacts_path_secret(monkeypatch):
    def fake_open_openai_compatible_request(request, *, timeout):
        raise urllib.error.URLError("proxy failed")

    unsafe_base_url = "https://gateway.example/v1/SECRET_PATH"
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", unsafe_base_url)
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "local-chat")
    monkeypatch.setattr(
        document_qa_module,
        "open_openai_compatible_request",
        fake_open_openai_compatible_request,
    )
    qa = DocumentQA(device="cpu", llm_backend="openai-compatible")

    with pytest.raises(RuntimeError) as exc_info:
        qa._initialize_llm()

    chain_messages = exception_chain_messages(exc_info.value)
    assert chain_messages
    assert all("SECRET_PATH" not in message for message in chain_messages)
    assert all(unsafe_base_url not in message for message in chain_messages)
    assert any("https://gateway.example" in message for message in chain_messages)
    assert all("proxy failed" not in message for message in chain_messages)


def test_openai_compatible_initialization_logs_redact_path_secret(
    monkeypatch, caplog
):
    def fake_open_openai_compatible_request(request, *, timeout):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                if request.full_url.endswith("/chat/completions"):
                    return json.dumps(
                        {"choices": [{"message": {"content": "ok"}}]}
                    ).encode("utf-8")
                return json.dumps(
                    {"data": [{"index": 0, "embedding": [1.0, 0.0]}]}
                ).encode("utf-8")

        return FakeResponse()

    unsafe_base_url = "https://gateway.example/v1/SECRET_PATH"
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", unsafe_base_url)
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "local-chat")
    monkeypatch.setenv("EMBEDDINGS_MODEL", "text-embedding-local")
    monkeypatch.setattr(
        document_qa_module,
        "open_openai_compatible_request",
        fake_open_openai_compatible_request,
    )
    caplog.set_level(logging.INFO, logger=document_qa_module.LOGGER.name)
    qa = DocumentQA(device="cpu", llm_backend="openai-compatible")

    qa._initialize_llm()
    qa._initialize_embeddings()

    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "https://gateway.example" in log_text
    assert "SECRET_PATH" not in log_text
    assert unsafe_base_url not in log_text


def test_openai_compatible_backend_alias_is_supported(monkeypatch):
    def fake_openai_compatible_loader(self, model_id):
        return MockLLM()

    monkeypatch.setattr(
        DocumentQA, "_load_openai_compatible_model", fake_openai_compatible_loader
    )
    qa = DocumentQA(device="cpu", llm_backend="openai_compatible")

    qa._initialize_llm()

    assert qa.llm_backend == "openai-compatible"
    assert qa.active_llm_backend == "openai-compatible"


def test_explicit_openai_compatible_without_base_url_fails_closed(monkeypatch):
    monkeypatch.delenv("OPENAI_COMPAT_BASE_URL", raising=False)
    qa = DocumentQA(device="cpu", llm_backend="openai-compatible")

    with pytest.raises(RuntimeError, match="OPENAI_COMPAT_BASE_URL"):
        qa._initialize_llm()

    assert qa.active_llm_backend is None
    assert qa.llm is None


def test_explicit_openai_compatible_without_model_fails_closed(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_COMPAT_MODEL", raising=False)
    qa = DocumentQA(device="cpu", llm_backend="openai-compatible")

    with pytest.raises(RuntimeError, match="OPENAI_COMPAT_MODEL"):
        qa._initialize_llm()

    assert qa.active_llm_backend is None
    assert qa.llm is None


@pytest.mark.parametrize(
    "base_url, expected",
    [
        ("http://localhost:8000/v1/", "http://localhost:8000/v1"),
        ("http://127.0.0.1:8000/v1", "http://127.0.0.1:8000/v1"),
        ("http://[::1]:8000/v1", "http://[::1]:8000/v1"),
        ("https://gateway.example/v1", "https://gateway.example/v1"),
    ],
)
def test_normalize_openai_compatible_base_url_accepts_http_forms(
    base_url, expected
):
    assert normalize_openai_compatible_base_url(base_url) == expected


@pytest.mark.parametrize(
    "base_url",
    [
        "not-a-url",
        "file:///tmp/model",
        "http://user:pass@localhost:8000/v1",
        "http://localhost:bad/v1",
        "http://gateway.example/v1",
        "http://localhost:8000/v1?token=x",
        "http://localhost:8000/v1#fragment",
    ],
)
def test_normalize_openai_compatible_base_url_rejects_unsafe_forms(base_url):
    with pytest.raises(RuntimeError, match="OPENAI_COMPAT_BASE_URL"):
        normalize_openai_compatible_base_url(base_url)


@pytest.mark.parametrize(
    "base_url, expected",
    [
        (
            "https://user:SECRET_PASSWORD@gateway.example/v1",
            "https://gateway.example/v1",
        ),
        (
            "https://gateway.example/v1?api_key=SECRET_QUERY_TOKEN",
            "https://gateway.example/v1",
        ),
        (
            "https://gateway.example:8443/v1#SECRET_FRAGMENT",
            "https://gateway.example:8443/v1",
        ),
        (
            "https://gateway.example/v1/SECRET_PATH",
            "https://gateway.example",
        ),
        ("not-a-url", "<invalid>"),
        ("", "<unset>"),
    ],
)
def test_safe_openai_compatible_base_url_for_error_redacts_secrets(
    base_url, expected
):
    assert safe_openai_compatible_base_url_for_error(base_url) == expected


def exception_chain_messages(error):
    messages = []
    seen = set()
    stack = [error]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        messages.append(str(current))
        stack.extend([current.__cause__, current.__context__])
    return messages


@pytest.mark.parametrize(
    "unsafe_base_url, secret",
    [
        ("https://user:SECRET_PASSWORD@gateway.example/v1", "SECRET_PASSWORD"),
        (
            "https://gateway.example/v1?api_key=SECRET_QUERY_TOKEN",
            "SECRET_QUERY_TOKEN",
        ),
        ("https://gateway.example/v1#SECRET_FRAGMENT", "SECRET_FRAGMENT"),
    ],
)
def test_openai_compatible_initialization_error_redacts_unsafe_url(
    monkeypatch, unsafe_base_url, secret
):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", unsafe_base_url)
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "local-chat")
    qa = DocumentQA(device="cpu", llm_backend="openai-compatible")

    with pytest.raises(RuntimeError) as exc_info:
        qa._initialize_llm()

    message = str(exc_info.value)
    cause_message = str(exc_info.value.__cause__)
    assert secret not in message
    assert secret not in cause_message
    assert unsafe_base_url not in message
    assert unsafe_base_url not in cause_message
    assert "gateway.example" in message


def test_openai_compatible_invalid_port_error_drops_parser_cause(monkeypatch):
    monkeypatch.setenv(
        "OPENAI_COMPAT_BASE_URL", "http://localhost:SECRET_PORT/v1"
    )
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "local-chat")
    qa = DocumentQA(device="cpu", llm_backend="openai-compatible")

    with pytest.raises(RuntimeError) as exc_info:
        qa._initialize_llm()

    chain_messages = exception_chain_messages(exc_info.value)
    assert chain_messages
    assert all("SECRET_PORT" not in message for message in chain_messages)
    assert all(
        "http://localhost:SECRET_PORT/v1" not in message
        for message in chain_messages
    )
    assert exc_info.value.__cause__ is not None
    assert exc_info.value.__cause__.__cause__ is None
    assert exc_info.value.__cause__.__context__ is None


def test_openai_compatible_rejects_remote_http_before_sending_api_key(
    monkeypatch,
):
    request_attempted = False

    def fake_open_openai_compatible_request(request, *, timeout):
        nonlocal request_attempted
        request_attempted = True
        raise AssertionError("remote HTTP request should not be sent")

    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://gateway.example/v1")
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "local-chat")
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "SECRET_API_KEY")
    monkeypatch.setattr(
        document_qa_module,
        "open_openai_compatible_request",
        fake_open_openai_compatible_request,
    )
    qa = DocumentQA(device="cpu", llm_backend="openai-compatible")

    with pytest.raises(RuntimeError) as exc_info:
        qa._initialize_llm()

    chain_messages = exception_chain_messages(exc_info.value)
    assert not request_attempted
    assert any("HTTPS" in message for message in chain_messages)
    assert all("SECRET_API_KEY" not in message for message in chain_messages)


def test_openai_compatible_loopback_http_bypasses_proxy_urlopen(monkeypatch):
    requests = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "ok"}}]}
            ).encode("utf-8")

    def fake_no_proxy_open(request, timeout):
        requests.append(
            {
                "url": request.full_url,
                "headers": {
                    key.lower(): value for key, value in request.header_items()
                },
                "timeout": timeout,
            }
        )
        return FakeResponse()

    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8443")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: (_ for _ in ()).throw(
            AssertionError(
                "OpenAI-compatible loopback HTTP must bypass proxy-aware urlopen"
            )
        ),
    )
    monkeypatch.setattr(
        document_qa_module.OPENAI_COMPAT_NO_PROXY_OPENER,
        "open",
        fake_no_proxy_open,
    )
    llm = OpenAICompatibleLLM(
        model="local-chat",
        base_url="http://localhost:8000/v1",
        api_key="SECRET_PROXY_TOKEN",
        timeout=7,
    )

    llm.validate_model_available()

    assert requests == [
        {
            "url": "http://localhost:8000/v1/chat/completions",
            "headers": {
                "content-type": "application/json",
                "authorization": "Bearer SECRET_PROXY_TOKEN",
            },
            "timeout": 7,
        }
    ]


def test_openai_compatible_initializes_with_explicit_backend(monkeypatch):
    def fake_openai_compatible_loader(self, model_id):
        return MockLLM()

    monkeypatch.setattr(
        DocumentQA, "_load_openai_compatible_model", fake_openai_compatible_loader
    )
    qa = DocumentQA(device="cpu", llm_backend="openai-compatible")

    qa._initialize_llm()

    assert qa.active_llm_backend == "openai-compatible"
    assert isinstance(qa.llm, MockLLM)


def test_ollama_llm_validates_model_and_generates(monkeypatch):
    requests = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

    def fake_open_ollama_request_no_proxy(request, *, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        requests.append((request.full_url, payload, timeout))
        if request.full_url.endswith("/api/show"):
            return FakeResponse({"model_info": {}})
        if request.full_url.endswith("/api/generate"):
            return FakeResponse(
                {
                    "response": (
                        "<think>SECRET_GENERATE_THINKING</think>\n"
                        "Project Phoenix answer [1]."
                    )
                }
            )
        raise AssertionError(f"Unexpected Ollama URL: {request.full_url}")

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: (_ for _ in ()).throw(
            AssertionError("OllamaLLM must bypass proxy-aware urlopen")
        ),
    )
    monkeypatch.setattr(
        "src.DocumentQA.open_ollama_request_no_proxy",
        fake_open_ollama_request_no_proxy,
    )
    llm = OllamaLLM(
        model=DEFAULT_OLLAMA_MODEL,
        base_url="http://localhost:11434/",
        timeout=7,
        options={"temperature": 0, "num_predict": 160},
    )

    llm.validate_model_available()
    answer = llm.invoke("Answer with citation.", stop=["END"])

    assert answer == "Project Phoenix answer [1]."
    assert llm.supports_thinking is False
    assert llm.last_thinking is None
    assert llm._strip_thinking_text("<think>unfinished reasoning") == ""
    assert "SECRET_GENERATE_THINKING" not in answer
    assert requests[0] == (
        "http://localhost:11434/api/show",
        {"model": DEFAULT_OLLAMA_MODEL},
        7,
    )
    assert requests[1][0] == "http://localhost:11434/api/generate"
    assert requests[1][1]["model"] == DEFAULT_OLLAMA_MODEL
    assert requests[1][1]["prompt"] == "Answer with citation."
    assert requests[1][1]["stream"] is False
    assert requests[1][1]["think"] is False
    assert requests[1][1]["options"] == {
        "temperature": 0,
        "num_predict": 160,
        "stop": ["END"],
    }


def test_ollama_llm_uses_chat_thinking_when_model_supports_it(monkeypatch):
    requests = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

    def fake_open_ollama_request_no_proxy(request, *, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        requests.append((request.full_url, payload, timeout))
        if request.full_url.endswith("/api/show"):
            return FakeResponse({"capabilities": ["completion", "thinking"]})
        if request.full_url.endswith("/api/chat"):
            return FakeResponse(
                {
                    "message": {
                        "content": "Project Phoenix launches in June 2026 [1].",
                        "thinking": "I checked the cited Phoenix launch excerpt.",
                    }
                }
            )
        raise AssertionError(f"Unexpected Ollama URL: {request.full_url}")

    monkeypatch.setattr(
        "src.DocumentQA.open_ollama_request_no_proxy",
        fake_open_ollama_request_no_proxy,
    )
    llm = OllamaLLM(
        model=DEFAULT_OLLAMA_MODEL,
        base_url="http://localhost:11434",
        timeout=9,
        options={"temperature": 0},
    )

    llm.validate_model_available()
    answer = llm.invoke("Answer with citation.")

    assert answer == "Project Phoenix launches in June 2026 [1]."
    assert llm.supports_thinking is True
    assert llm.last_thinking == "I checked the cited Phoenix launch excerpt."
    assert requests[1][0] == "http://localhost:11434/api/chat"
    assert requests[1][1] == {
        "model": DEFAULT_OLLAMA_MODEL,
        "messages": [{"role": "user", "content": "Answer with citation."}],
        "stream": False,
        "think": True,
        "options": {"temperature": 0},
    }


def test_ollama_llm_respects_explicit_thinking_token_budget(monkeypatch):
    requests = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

    def fake_open_ollama_request_no_proxy(request, *, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        requests.append((request.full_url, payload, timeout))
        if request.full_url.endswith("/api/show"):
            return FakeResponse({"capabilities": ["completion", "thinking"]})
        if request.full_url.endswith("/api/chat"):
            return FakeResponse(
                {
                    "message": {
                        "content": "Complete answer after thinking [1].",
                        "thinking": "Reasoning that would otherwise consume budget.",
                    },
                    "done": True,
                    "done_reason": "stop",
                }
            )
        raise AssertionError(f"Unexpected Ollama URL: {request.full_url}")

    monkeypatch.setattr(
        "src.DocumentQA.open_ollama_request_no_proxy",
        fake_open_ollama_request_no_proxy,
    )
    llm = OllamaLLM(
        model=DEFAULT_OLLAMA_MODEL,
        options={"temperature": 0, "num_predict": 160},
    )

    llm.validate_model_available()
    answer = llm.invoke("Answer with citation.")

    assert answer == "Complete answer after thinking [1]."
    assert requests[1][1]["options"] == {
        "temperature": 0,
        "num_predict": 160,
    }


def test_ollama_llm_rejects_length_truncated_chat_response(monkeypatch):
    requests = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

    def fake_open_ollama_request_no_proxy(request, *, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        requests.append((request.full_url, payload, timeout))
        if request.full_url.endswith("/api/show"):
            return FakeResponse({"capabilities": ["completion", "thinking"]})
        if request.full_url.endswith("/api/chat"):
            return FakeResponse(
                {
                    "message": {
                        "content": "This answer is visibly cut off because",
                        "thinking": "Long hidden thinking consumed the budget.",
                    },
                    "done": True,
                    "done_reason": "length",
                }
            )
        raise AssertionError(f"Unexpected Ollama URL: {request.full_url}")

    monkeypatch.setattr(
        "src.DocumentQA.open_ollama_request_no_proxy",
        fake_open_ollama_request_no_proxy,
    )
    llm = OllamaLLM(model=DEFAULT_OLLAMA_MODEL)

    llm.validate_model_available()
    with pytest.raises(RuntimeError, match="stopped generation"):
        llm.invoke("Answer with citation.")

    assert llm.last_thinking is None
    assert requests[1][0] == "http://localhost:11434/api/chat"


def test_ollama_llm_uses_level_for_gpt_oss_thinking(monkeypatch):
    requests = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

    def fake_open_ollama_request_no_proxy(request, *, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        requests.append((request.full_url, payload, timeout))
        if request.full_url.endswith("/api/show"):
            return FakeResponse({"capabilities": ["completion", "thinking"]})
        if request.full_url.endswith("/api/chat"):
            return FakeResponse(
                {
                    "message": {
                        "content": "GPT-OSS answer [1].",
                        "thinking": "GPT-OSS medium trace.",
                    }
                }
            )
        raise AssertionError(f"Unexpected Ollama URL: {request.full_url}")

    monkeypatch.setattr(
        "src.DocumentQA.open_ollama_request_no_proxy",
        fake_open_ollama_request_no_proxy,
    )
    llm = OllamaLLM(model="gpt-oss:20b")

    llm.validate_model_available()
    answer = llm.invoke("Answer with citation.")

    assert answer == "GPT-OSS answer [1]."
    assert llm.think_level == "medium"
    assert llm.last_thinking == "GPT-OSS medium trace."
    assert requests[1][0] == "http://localhost:11434/api/chat"
    assert requests[1][1]["model"] == "gpt-oss:20b"
    assert requests[1][1]["think"] == "medium"


def test_ollama_llm_respects_explicit_think_level(monkeypatch):
    requests = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

    def fake_open_ollama_request_no_proxy(request, *, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        requests.append((request.full_url, payload, timeout))
        if request.full_url.endswith("/api/show"):
            return FakeResponse({"capabilities": ["completion", "thinking"]})
        if request.full_url.endswith("/api/chat"):
            return FakeResponse(
                {
                    "message": {
                        "content": "High effort answer [1].",
                        "thinking": "High effort trace.",
                    }
                }
            )
        raise AssertionError(f"Unexpected Ollama URL: {request.full_url}")

    monkeypatch.setattr(
        "src.DocumentQA.open_ollama_request_no_proxy",
        fake_open_ollama_request_no_proxy,
    )
    llm = OllamaLLM(model=DEFAULT_OLLAMA_MODEL, think_level="high")

    llm.validate_model_available()
    answer = llm.invoke("Answer with citation.")

    assert answer == "High effort answer [1]."
    assert requests[1][1]["think"] == "high"


def test_ollama_think_level_rejects_gpt_oss_max(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    monkeypatch.setenv("LLM_MODEL", "gpt-oss:20b")
    monkeypatch.setenv("OLLAMA_THINK_LEVEL", "max")
    monkeypatch.setattr(
        "src.DocumentQA.open_ollama_request_no_proxy",
        lambda request, *, timeout: (_ for _ in ()).throw(
            AssertionError("Invalid GPT-OSS think level must fail before Ollama I/O")
        ),
    )
    qa = DocumentQA(fast_mode=True)

    with pytest.raises(RuntimeError, match="OLLAMA_THINK_LEVEL=max"):
        qa._load_ollama_model("gpt-oss:20b")


def test_ollama_llm_revalidates_mutated_think_level_before_request(monkeypatch):
    calls = 0

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps({"capabilities": ["completion", "thinking"]}).encode(
                "utf-8"
            )

    def fake_open_ollama_request_no_proxy(request, *, timeout):
        nonlocal calls
        calls += 1
        if request.full_url.endswith("/api/show"):
            return FakeResponse()
        raise AssertionError("Invalid GPT-OSS think level must fail before chat request")

    monkeypatch.setattr(
        "src.DocumentQA.open_ollama_request_no_proxy",
        fake_open_ollama_request_no_proxy,
    )
    llm = OllamaLLM(model="gpt-oss:20b")
    llm.validate_model_available()
    llm.think_level = "max"

    with pytest.raises(RuntimeError, match="OLLAMA_THINK_LEVEL=max"):
        llm.invoke("Answer with citation.")

    assert calls == 1


def test_ollama_llm_respects_disabled_model_thinking(monkeypatch):
    requests = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

    def fake_open_ollama_request_no_proxy(request, *, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        requests.append((request.full_url, payload, timeout))
        if request.full_url.endswith("/api/show"):
            return FakeResponse({"capabilities": ["completion", "thinking"]})
        if request.full_url.endswith("/api/generate"):
            return FakeResponse(
                {"response": "<think>hidden scratch</think>Plain answer [1]."}
            )
        raise AssertionError(f"Unexpected Ollama URL: {request.full_url}")

    monkeypatch.setattr(
        "src.DocumentQA.open_ollama_request_no_proxy",
        fake_open_ollama_request_no_proxy,
    )
    llm = OllamaLLM(
        model=DEFAULT_OLLAMA_MODEL,
        enable_thinking=False,
    )

    llm.validate_model_available()
    answer = llm.invoke("Answer plainly.")

    assert answer == "Plain answer [1]."
    assert requests[1][0] == "http://localhost:11434/api/generate"
    assert requests[1][1]["think"] is False
    assert llm.last_thinking is None


def test_ollama_llm_http_error_body_is_not_reflected(monkeypatch):
    class SecretHttpError(urllib.error.HTTPError):
        def __init__(self):
            super().__init__(
                url="http://localhost:11434/api/show",
                code=500,
                msg="SECRET_STATUS_TEXT",
                hdrs={},
                fp=None,
            )

        def read(self):
            return b"SECRET_BODY_TOKEN"

    def fake_open_ollama_request_no_proxy(request, *, timeout):
        raise SecretHttpError()

    monkeypatch.setattr(
        "src.DocumentQA.open_ollama_request_no_proxy",
        fake_open_ollama_request_no_proxy,
    )
    llm = OllamaLLM(model=DEFAULT_OLLAMA_MODEL)

    with pytest.raises(RuntimeError) as exc_info:
        llm.validate_model_available()

    message = str(exc_info.value)
    assert "HTTP 500" in message
    assert "SECRET_BODY_TOKEN" not in message
    assert "SECRET_STATUS_TEXT" not in message


def test_ollama_llm_url_error_reason_is_not_reflected(monkeypatch):
    def fake_open_ollama_request_no_proxy(request, *, timeout):
        raise urllib.error.URLError("proxy failed with SECRET_REASON_TOKEN")

    monkeypatch.setattr(
        "src.DocumentQA.open_ollama_request_no_proxy",
        fake_open_ollama_request_no_proxy,
    )
    llm = OllamaLLM(model=DEFAULT_OLLAMA_MODEL)

    with pytest.raises(RuntimeError) as exc_info:
        llm.validate_model_available()

    message = str(exc_info.value)
    assert "http://localhost:11434/api/show" in message
    assert "SECRET_REASON_TOKEN" not in message
    assert "proxy failed" not in message
