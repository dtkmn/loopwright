import argparse
import json
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from langchain_core.language_models.llms import LLM
from pydantic import Field

from src.ai_loop_engine import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    LLM_MODEL_ENV_VAR,
    OLLAMA_BASE_URL_ENV_VAR,
    OLLAMA_MODEL_ENV_VAR,
    SELF_CHECK_REFUSAL_ANSWER,
    AILoopEngine,
    QueryResult,
)
from src.golden_eval import (
    GOLDEN_DOCUMENT_TEXT,
    GOLDEN_EVAL_CASES,
    GoldenEvalCase,
    GoldenEvalEmbeddings,
)
from src.ollama_model_eval import (
    build_ollama_qa,
    normalize_local_ollama_base_url,
    unload_ollama_model,
)


PROVIDER_FREE_MODEL = "provider-free-golden"


@dataclass(frozen=True)
class LoopEvalCaseResult:
    case_id: str
    question: str
    passed: bool
    answer: str
    expected_terms: List[str]
    expect_refusal: bool
    self_check_outcome: Optional[str]
    self_check_reasons: List[str]
    citation_count: int
    invalid_inline_citations: List[str]
    final_decision: Optional[str]
    phases: List[str]
    loop_report: Optional[Dict]
    error_message: Optional[str] = None


@dataclass(frozen=True)
class LoopEvalRunResult:
    mode: str
    model: str
    passed: bool
    passed_cases: int
    total_cases: int
    case_results: List[LoopEvalCaseResult]
    initialization_error: Optional[str] = None


QaFactory = Callable[[str, str, int], AILoopEngine]


class ProviderFreeGoldenLLM(LLM):
    calls: List[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "provider-free-golden"

    def _call(self, prompt: str, stop=None, **kwargs) -> str:
        self.calls.append(prompt)
        if "strict citation verifier" in prompt:
            return self._verifier_response(prompt)
        return self._answer_response(prompt)

    def _answer_response(self, prompt: str) -> str:
        question = self._question_from_prompt(prompt)
        if "budget" in question:
            return "The approved Project Phoenix budget is $42 million [1]."
        if "owner" in question or "owns" in question:
            return "Alex Rivera [1]."
        if "venue" in question or "where" in question:
            return "Project Phoenix launches from Lunar Base Alpha [1]."
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


def build_provider_free_qa(model: str, base_url: str, timeout: int) -> AILoopEngine:
    qa = AILoopEngine(fast_mode=True, llm_backend="ollama")
    # The fake eval injects its verifier before any query, so AILoopEngine never
    # initializes a live Ollama client. Keep the active label honest in artifacts.
    qa.llm = ProviderFreeGoldenLLM()
    qa.active_llm_backend = "provider-free"
    qa.loaded_model_id = model or PROVIDER_FREE_MODEL
    qa.loaded_model_label = f"Provider-free golden model ({qa.loaded_model_id})"
    qa.embeddings = GoldenEvalEmbeddings()
    return qa


def selected_cases(case_ids: Optional[Sequence[str]]) -> Sequence[GoldenEvalCase]:
    if not case_ids:
        return GOLDEN_EVAL_CASES
    requested_case_ids = set(case_ids)
    cases = [case for case in GOLDEN_EVAL_CASES if case.case_id in requested_case_ids]
    missing_case_ids = sorted(requested_case_ids - {case.case_id for case in cases})
    if missing_case_ids:
        valid_case_ids = ", ".join(case.case_id for case in GOLDEN_EVAL_CASES)
        raise ValueError(
            f"Unknown golden case id(s): {', '.join(missing_case_ids)}. "
            f"Valid case ids: {valid_case_ids}"
        )
    return cases


def _non_empty_model_tags(values: Sequence[str]) -> List[str]:
    return [value.strip() for value in values if value and value.strip()]


def models_from_args(args) -> List[str]:
    if args.models:
        return _non_empty_model_tags(args.models)
    if args.mode == "fake":
        return [PROVIDER_FREE_MODEL]
    env_models = os.getenv("OLLAMA_EVAL_MODELS", "").strip()
    if env_models:
        parsed_env_models = _non_empty_model_tags(env_models.split(","))
        if parsed_env_models:
            return parsed_env_models
    env_model = (
        os.getenv(LLM_MODEL_ENV_VAR, "").strip()
        or os.getenv(OLLAMA_MODEL_ENV_VAR, "").strip()
    )
    if env_model:
        return [env_model]
    return [DEFAULT_OLLAMA_MODEL]


def _phase_values(result: QueryResult) -> List[str]:
    if not result.loop_report:
        return []
    return [step.phase.value for step in result.loop_report.run.steps]


def _final_decision(result: QueryResult) -> Optional[str]:
    if not result.loop_report or not result.loop_report.run.final_decision:
        return None
    return result.loop_report.run.final_decision.value


def _loop_report_dict(result: QueryResult) -> Optional[Dict]:
    return result.loop_report.to_dict() if result.loop_report else None


def _invalid_inline_citations(result: QueryResult) -> List[str]:
    valid_citation_ids = {
        str(citation.citation_id) for citation in result.trace.citations
    }
    inline_citation_ids = re.findall(r"\[(\d+)\]", result.answer)
    return [
        citation_id
        for citation_id in inline_citation_ids
        if citation_id not in valid_citation_ids
    ]


def score_case(case: GoldenEvalCase, result: QueryResult) -> LoopEvalCaseResult:
    self_check = result.trace.self_check
    self_check_outcome = self_check.outcome if self_check else None
    self_check_reasons = list(self_check.reasons) if self_check else []
    invalid_inline_citations = _invalid_inline_citations(result)
    phases = _phase_values(result)
    final_decision = _final_decision(result)
    answer_lower = result.answer.lower()
    citation_count = len(result.trace.citations)

    if invalid_inline_citations and "invalid_inline_citation" not in self_check_reasons:
        self_check_reasons.append("invalid_inline_citation")

    has_core_loop = all(
        phase in phases
        for phase in ("context_select", "retrieve", "draft", "mechanical_check", "final")
    )
    has_verifier = "verify" in phases

    if case.expect_refusal:
        passed = (
            result.answer == SELF_CHECK_REFUSAL_ANSWER
            and final_decision == "refuse"
            and "refuse" in phases
            and has_core_loop
            and has_verifier
            and self_check_outcome != "supported"
        )
    else:
        expected_terms_present = all(
            expected_term.lower() in answer_lower
            for expected_term in case.expected_terms
        )
        inline_citation_present = any(
            f"[{citation.citation_id}]" in result.answer
            for citation in result.trace.citations
        )
        passed = (
            expected_terms_present
            and inline_citation_present
            and not invalid_inline_citations
            and citation_count > 0
            and self_check_outcome == "supported"
            and final_decision == "supported"
            and has_core_loop
            and has_verifier
        )

    return LoopEvalCaseResult(
        case_id=case.case_id,
        question=case.question,
        passed=passed,
        answer=result.answer,
        expected_terms=list(case.expected_terms),
        expect_refusal=case.expect_refusal,
        self_check_outcome=self_check_outcome,
        self_check_reasons=self_check_reasons,
        citation_count=citation_count,
        invalid_inline_citations=invalid_inline_citations,
        final_decision=final_decision,
        phases=phases,
        loop_report=_loop_report_dict(result),
    )


def failed_case(case: GoldenEvalCase, error: Exception) -> LoopEvalCaseResult:
    return LoopEvalCaseResult(
        case_id=case.case_id,
        question=case.question,
        passed=False,
        answer="",
        expected_terms=list(case.expected_terms),
        expect_refusal=case.expect_refusal,
        self_check_outcome=None,
        self_check_reasons=[],
        citation_count=0,
        invalid_inline_citations=[],
        final_decision=None,
        phases=[],
        loop_report=None,
        error_message=str(error),
    )


def evaluate_model(
    model: str,
    *,
    mode: str,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    timeout: int = 60,
    qa_factory: Optional[QaFactory] = None,
    cases: Sequence[GoldenEvalCase] = GOLDEN_EVAL_CASES,
    unload_after: bool = True,
) -> LoopEvalRunResult:
    if mode not in {"fake", "ollama"}:
        raise ValueError("mode must be fake or ollama")
    if mode == "ollama":
        base_url = normalize_local_ollama_base_url(base_url)
    qa_factory = qa_factory or (
        build_provider_free_qa if mode == "fake" else build_ollama_qa
    )

    case_results: List[LoopEvalCaseResult] = []
    initialization_error = None
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            document_path = Path(tmp_dir) / "project_phoenix_brief.md"
            document_path.write_text(GOLDEN_DOCUMENT_TEXT, encoding="utf-8")
            try:
                qa = qa_factory(model, base_url, timeout)
                qa.process_document(str(document_path))
            except Exception as exc:
                initialization_error = str(exc)
                return LoopEvalRunResult(
                    mode=mode,
                    model=model,
                    passed=False,
                    passed_cases=0,
                    total_cases=len(cases),
                    case_results=[],
                    initialization_error=initialization_error,
                )

            for case in cases:
                try:
                    case_results.append(
                        score_case(
                            case,
                            qa.query_with_trace(
                                case.question,
                                context_provider="document",
                            ),
                        )
                    )
                except Exception as exc:
                    case_results.append(failed_case(case, exc))
    finally:
        if mode == "ollama" and unload_after:
            unload_ollama_model(model, base_url=base_url, timeout=min(timeout, 15))

    passed_cases = sum(1 for result in case_results if result.passed)
    return LoopEvalRunResult(
        mode=mode,
        model=model,
        passed=passed_cases == len(case_results),
        passed_cases=passed_cases,
        total_cases=len(case_results),
        case_results=case_results,
    )


def results_to_dict(results: Sequence[LoopEvalRunResult]) -> Dict:
    return {
        "schema_version": "loop-eval/v1",
        "results": [asdict(result) for result in results],
    }


def results_to_json(results: Sequence[LoopEvalRunResult]) -> str:
    return json.dumps(results_to_dict(results), indent=2)


def write_artifact(path: str, results: Sequence[LoopEvalRunResult]) -> None:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(results_to_json(results) + "\n", encoding="utf-8")


def format_text_results(results: Sequence[LoopEvalRunResult]) -> str:
    lines = []
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(
            f"{status} {result.mode}/{result.model}: "
            f"{result.passed_cases}/{result.total_cases} cases"
        )
        if result.initialization_error:
            lines.append(f"  initialization_error: {result.initialization_error}")
            continue
        for case_result in result.case_results:
            case_status = "PASS" if case_result.passed else "FAIL"
            lines.append(
                "  "
                f"{case_status} {case_result.case_id}: "
                f"decision={case_result.final_decision}, "
                f"self_check={case_result.self_check_outcome}, "
                f"phases={','.join(case_result.phases)}"
            )
            if case_result.error_message:
                lines.append(f"    error: {case_result.error_message}")
            if not case_result.passed:
                answer = case_result.answer.replace("\n", " ").strip()
                lines.append(f"    answer: {answer}")
                if case_result.self_check_reasons:
                    lines.append(
                        "    reasons: " + ", ".join(case_result.self_check_reasons)
                    )
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description=(
            "Run Loopwright golden evals and optionally write loop-report "
            "JSON artifacts."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("fake", "ollama"),
        default="fake",
        help="fake is provider-free and CI-safe; ollama uses a local Ollama server.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help=(
            "Model labels/tags. Fake mode defaults to provider-free-golden. "
            "Ollama mode defaults to OLLAMA_EVAL_MODELS, LLM_MODEL, "
            "OLLAMA_MODEL, or the app default."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv(OLLAMA_BASE_URL_ENV_VAR, DEFAULT_OLLAMA_BASE_URL),
        help="Ollama base URL for --mode ollama. Must be loopback.",
    )
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--case",
        dest="case_ids",
        action="append",
        help="Run only a selected golden case id. Can be repeated.",
    )
    parser.add_argument(
        "--all-cases",
        action="store_true",
        help=(
            "Run the full golden set in Ollama mode. Without --case or "
            "--all-cases, Ollama mode runs only launch_date to keep local "
            "smoke tests small."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print JSON results.")
    parser.add_argument(
        "--artifact",
        help="Write the full JSON artifact, including loop reports, to this path.",
    )
    parser.add_argument(
        "--allow-multi-model",
        action="store_true",
        help=(
            "Allow multiple models in one command. Ollama mode disables this by "
            "default to avoid local memory pressure."
        ),
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help=(
            "Exit 0 for failing eval scores. Argument validation and safety "
            "guard failures still return nonzero."
        ),
    )
    parser.add_argument(
        "--no-unload",
        action="store_true",
        help="Do not ask Ollama to unload each model after an Ollama eval run.",
    )
    return parser.parse_args(argv)


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    qa_factory: Optional[QaFactory] = None,
    output_stream=None,
) -> int:
    args = parse_args(argv)
    output_stream = output_stream or sys.stdout
    case_ids = args.case_ids
    if args.mode == "ollama" and not case_ids and not args.all_cases:
        case_ids = ("launch_date",)
    try:
        cases = selected_cases(case_ids)
    except ValueError as exc:
        print(str(exc), file=output_stream)
        return 2

    try:
        base_url = (
            normalize_local_ollama_base_url(args.base_url)
            if args.mode == "ollama"
            else args.base_url
        )
    except ValueError as exc:
        print(str(exc), file=output_stream)
        return 2

    models = models_from_args(args)
    if not models:
        print("No model tags selected for loop eval.", file=output_stream)
        return 2
    if args.mode == "ollama" and len(models) > 1 and not args.allow_multi_model:
        print(
            "Multiple Ollama models in one eval are disabled by default to avoid "
            "local memory pressure. Run one model per command, or pass "
            "--allow-multi-model only on a machine with enough free unified memory.",
            file=output_stream,
        )
        return 2

    results = [
        evaluate_model(
            model,
            mode=args.mode,
            base_url=base_url,
            timeout=args.timeout,
            qa_factory=qa_factory,
            cases=cases,
            unload_after=not args.no_unload,
        )
        for model in models
    ]
    if args.artifact:
        write_artifact(args.artifact, results)
    if args.json:
        print(results_to_json(results), file=output_stream)
    else:
        print(format_text_results(results), file=output_stream)
    if args.no_fail:
        return 0
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
