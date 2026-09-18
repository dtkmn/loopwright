import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from src import app as web_app
from src.answer_loop import SELF_CHECK_REFUSAL_ANSWER
from src.DocumentQA import (
    AnswerCitation,
    AnswerSelfCheck,
    AnswerTrace,
    DocumentQA,
    DocumentProcessingError,
    DocumentProcessingReport,
    DocumentQAStatus,
    QueryResult,
)
from src.loop_engine import (
    ANSWER_CANDIDATE_SHA256_METADATA_KEY,
    DEFAULT_LOOP_RECIPE_ID,
    EVIDENCE_SET_SHA256_METADATA_KEY,
    EvidenceReference,
    LoopDecision,
    LoopPhase,
    LoopRecipe,
    LoopReport,
    LoopRun,
    LoopStep,
    LoopTerminalReason,
    PUBLIC_REDACTION_REASON,
    PUBLIC_REDACTION_TEXT,
    VerificationOutcome,
    VerificationResult,
    answer_candidate_sha256,
    evidence_set_sha256,
    utc_now,
)
from src.thread_store import ThreadStore
from src.web_contract import (
    MODEL_THINKING_SHA256_METADATA_KEY,
    answer_trace_dict,
    loop_summary_dict,
    loop_timeline_dict,
    public_loop_payload_from_report,
    query_response_dict,
    runtime_status_dict,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_THREAD_ENV_VARS = {
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "TOKENIZERS_PARALLELISM",
}


def visible_not_verified_steps(
    *,
    answer,
    evidence,
    started_at,
    prefix,
    backend="mock",
    model_label="MockLLM (explicit demo)",
):
    answer_digest = answer_candidate_sha256(answer)
    evidence_digest = evidence_set_sha256(evidence)
    binding = {
        ANSWER_CANDIDATE_SHA256_METADATA_KEY: answer_digest,
        EVIDENCE_SET_SHA256_METADATA_KEY: evidence_digest,
    }
    return (
        LoopStep(
            step_id=f"step_{prefix}_draft",
            phase=LoopPhase.DRAFT,
            decision=LoopDecision.CONTINUE,
            started_at=started_at,
            ended_at=started_at,
            backend=backend,
            model_label=model_label,
            metadata=binding,
        ),
        LoopStep(
            step_id=f"step_{prefix}_verify",
            phase=LoopPhase.VERIFY,
            decision=LoopDecision.NOT_VERIFIED,
            started_at=started_at,
            ended_at=started_at,
            backend=backend,
            model_label=model_label,
            verification=VerificationResult(
                outcome=VerificationOutcome.NOT_VERIFIED,
            ),
            metadata=binding,
        ),
        LoopStep(
            step_id=f"step_{prefix}_final",
            phase=LoopPhase.FINAL,
            decision=LoopDecision.NOT_VERIFIED,
            started_at=started_at,
            ended_at=started_at,
            backend=backend,
            model_label=model_label,
            metadata=binding,
        ),
    )


class FakeQA:
    fast_mode = True
    loaded_model_id = None
    loaded_model_label = None
    active_llm_backend = "mock"
    llm_backend = "mock"
    current_document_name = None
    latest_processing_report = None

    def process_document(self, document_path, text_encoding=None):
        self.document_path = document_path
        self.text_encoding = text_encoding
        self.uploaded_text = Path(document_path).read_text(encoding="utf-8")
        self.current_document_name = Path(document_path).name
        active_backend = self.active_llm_backend or self.llm_backend
        active_model_label = (
            self.loaded_model_label
            or self.loaded_model_id
            or ("MockLLM (explicit demo)" if active_backend == "mock" else "unknown")
        )
        self.latest_processing_report = DocumentProcessingReport(
            attempted_document_name=self.current_document_name,
            active_document_name=self.current_document_name,
            success=True,
            phase="complete",
            file_extension=".txt",
            chunk_count=1,
            truncated=False,
            max_chunk_limit=2000,
            text_encoding_mode=text_encoding or "auto",
            backend=active_backend,
            model_label=active_model_label,
            error_message=None,
        )
        return self.status()

    def status(self):
        self.status_calls = getattr(self, "status_calls", 0) + 1
        active_backend = self.active_llm_backend or self.llm_backend
        active_model_label = (
            self.loaded_model_label
            or self.loaded_model_id
            or ("MockLLM (explicit demo)" if active_backend == "mock" else "unknown")
        )
        return DocumentQAStatus(
            profile_label="FAST" if self.fast_mode else "QUALITY",
            max_output_tokens=384 if self.fast_mode else 1024,
            configured_backend=self.llm_backend,
            active_backend=active_backend,
            active_model_label=active_model_label,
            loaded_model_id=self.loaded_model_id,
            loaded_model_label=self.loaded_model_label,
            embeddings_model="fake-embeddings",
            embeddings_device="cpu",
            device="cpu",
            document_name=self.current_document_name,
            ready_for_queries=bool(self.current_document_name),
            processing_report=self.latest_processing_report,
        )

    def query_with_trace(
        self,
        message,
        session_id="default",
        conversation_history=None,
        semantic_memory=None,
        semantic_memory_status="not_requested",
        loop_recipe=None,
        context_provider=None,
    ):
        self.last_query_session_id = session_id
        self.last_conversation_history = list(conversation_history or [])
        self.last_semantic_memory = list(semantic_memory or [])
        self.last_semantic_memory_status = semantic_memory_status
        self.last_loop_recipe = dict(loop_recipe or {})
        self.last_context_provider = context_provider
        active_backend = self.active_llm_backend or self.llm_backend
        active_model_label = (
            self.loaded_model_label
            or self.loaded_model_id
            or ("MockLLM (explicit demo)" if active_backend == "mock" else "unknown")
        )
        answer = "Project Phoenix is described in the indexed file."
        model_thinking = "I matched Project Phoenix against citation [1]."
        evidence = (
            EvidenceReference.from_source(
                citation_id=1,
                provider="document",
                source_identity=self.current_document_name or "demo.txt",
                page=None,
                chunk_index=0,
                excerpt="Project Phoenix is a loop workbench.",
            ),
        )
        run_started_at = utc_now()
        steps = []
        if self.last_loop_recipe:
            steps.append(
                LoopStep(
                    phase=LoopPhase.INPUT,
                    decision=LoopDecision.CONTINUE,
                    name="Apply loop recipe",
                    output_summary=self.last_loop_recipe.get("name"),
                    metadata={
                        "recipe_id": self.last_loop_recipe.get("recipe_id"),
                        "recipe_name": self.last_loop_recipe.get("name"),
                    },
                    started_at=run_started_at,
                    ended_at=run_started_at,
                )
            )
        if semantic_memory_status != "not_requested":
            steps.append(
                LoopStep(
                    phase=LoopPhase.CONTEXT_SELECT,
                    decision=LoopDecision.CONTINUE,
                    name="Retrieve thread memory",
                    output_summary=(
                        f"{len(self.last_semantic_memory)} semantic memories"
                        if self.last_semantic_memory
                        else semantic_memory_status
                    ),
                    metadata={
                        "semantic_memory_count": len(self.last_semantic_memory),
                        "semantic_memory_status": semantic_memory_status,
                    },
                    started_at=run_started_at,
                    ended_at=run_started_at,
                )
            )
        steps.extend(
            [
                LoopStep(
                    phase=LoopPhase.RETRIEVE,
                    decision=LoopDecision.CONTINUE,
                    name="Retrieve prompt evidence",
                    output_summary="1 prompt chunks",
                    metadata={"retrieved_chunk_count": 1, "citation_ids": [1]},
                    started_at=run_started_at,
                    ended_at=run_started_at,
                ),
                LoopStep(
                    phase=LoopPhase.DRAFT,
                    decision=LoopDecision.CONTINUE,
                    name="Draft answer",
                    backend=active_backend,
                    model_label=active_model_label,
                    metadata={
                        ANSWER_CANDIDATE_SHA256_METADATA_KEY: (
                            answer_candidate_sha256(answer)
                        ),
                        EVIDENCE_SET_SHA256_METADATA_KEY: evidence_set_sha256(
                            evidence
                        ),
                    },
                    started_at=run_started_at,
                    ended_at=run_started_at,
                ),
                LoopStep(
                    phase=LoopPhase.MECHANICAL_CHECK,
                    decision=LoopDecision.CONTINUE,
                    name="Mechanical checks",
                    output_summary="mechanical_checks_passed",
                    started_at=run_started_at,
                    ended_at=run_started_at,
                ),
                LoopStep(
                    phase=LoopPhase.VERIFY,
                    decision=LoopDecision.NOT_VERIFIED,
                    name="Answer verifier",
                    backend=active_backend,
                    model_label=active_model_label,
                    verification=VerificationResult(
                        outcome=VerificationOutcome.NOT_VERIFIED,
                    ),
                    metadata={
                        ANSWER_CANDIDATE_SHA256_METADATA_KEY: (
                            answer_candidate_sha256(answer)
                        ),
                        EVIDENCE_SET_SHA256_METADATA_KEY: evidence_set_sha256(
                            evidence
                        ),
                    },
                    started_at=run_started_at,
                    ended_at=run_started_at,
                ),
            ]
        )
        steps.append(
            LoopStep(
                phase=LoopPhase.FINAL,
                decision=LoopDecision.NOT_VERIFIED,
                name="Final answer",
                backend=active_backend,
                model_label=active_model_label,
                metadata={
                    ANSWER_CANDIDATE_SHA256_METADATA_KEY: (
                        answer_candidate_sha256(answer)
                    ),
                    EVIDENCE_SET_SHA256_METADATA_KEY: evidence_set_sha256(evidence),
                },
                started_at=run_started_at,
                ended_at=run_started_at,
            )
        )
        loop_report = LoopReport(
            run=LoopRun(
                run_id="run_fake",
                session_id=session_id,
                started_at=run_started_at,
                completed_at=run_started_at,
                user_input=message,
                context_provider="document",
                backend=active_backend,
                model_label=active_model_label,
                steps=tuple(steps),
                evidence=evidence,
                final_decision=LoopDecision.NOT_VERIFIED,
                final_answer=answer,
                metadata={
                    "conversation_context_turns": len(self.last_conversation_history),
                    "semantic_memory_turns": len(self.last_semantic_memory),
                    "semantic_memory_status": semantic_memory_status,
                    "recipe_id": self.last_loop_recipe.get("recipe_id"),
                    "recipe_name": self.last_loop_recipe.get("name"),
                    MODEL_THINKING_SHA256_METADATA_KEY: hashlib.sha256(
                        model_thinking.encode("utf-8")
                    ).hexdigest(),
                },
            )
        )
        return QueryResult(
            answer=answer,
            trace=AnswerTrace(
                question=message,
                document_name=self.current_document_name,
                backend=active_backend,
                model_label=active_model_label,
                retrieved_chunk_count=1,
                citations=[
                    AnswerCitation(
                        citation_id=1,
                        source_name=self.current_document_name or "demo.txt",
                        page=None,
                        chunk_index=0,
                        excerpt="Project Phoenix is a loop workbench.",
                    )
                ],
                self_check=AnswerSelfCheck(
                    outcome="not_verified",
                    reasons=[
                        "mechanical_checks_passed",
                        "verifier_unavailable_mock_backend",
                    ],
                    retry_attempted=False,
                ),
                model_thinking=model_thinking,
            ),
            loop_report=loop_report,
        )

    def memory_embedding_model_label(self):
        return "fake-memory"

    def embed_memory_texts(self, texts):
        vectors = []
        for text in texts:
            lowered = str(text).lower()
            if "dynamic programming" in lowered or "algorithm" in lowered:
                vectors.append([1.0, 0.0])
            elif "phoenix" in lowered:
                vectors.append([0.8, 0.2])
            else:
                vectors.append([0.0, 1.0])
        self.embedded_memory_texts = list(getattr(self, "embedded_memory_texts", []))
        self.embedded_memory_texts.extend(str(text) for text in texts)
        return "fake-memory", vectors

    def clear_loop_session(self, session_id="default"):
        self.cleared_loop_session_id = session_id


def processed_report(
    *,
    document_name="good.txt",
    success=True,
    phase="complete",
    error_message=None,
):
    return DocumentProcessingReport(
        attempted_document_name=document_name,
        active_document_name=document_name,
        success=success,
        phase=phase,
        file_extension=".txt",
        chunk_count=1 if success else 0,
        truncated=False,
        max_chunk_limit=2000,
        text_encoding_mode="auto",
        backend="mock",
        model_label="MockLLM (explicit demo)",
        error_message=error_message,
    )


def test_app_bootstraps_native_defaults_before_fastapi_import():
    env = os.environ.copy()
    for name in NATIVE_THREAD_ENV_VARS:
        env.pop(name, None)

    code = """
import builtins
import json
import os

real_import = builtins.__import__

def tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "fastapi" or name.startswith("fastapi."):
        print(json.dumps({
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
            "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
        }, sort_keys=True))
        raise SystemExit(0)
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = tracking_import
import src.app
raise SystemExit("src.app did not import fastapi")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert json.loads(result.stdout) == {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }


def test_app_loads_env_file_before_native_defaults_and_fastapi_import(tmp_path):
    env = os.environ.copy()
    env.pop("AI_LOOP_DISABLE_ENV_FILE", None)
    env.pop("FAST_MODE", None)
    env.pop("LLM_BACKEND", None)
    for name in NATIVE_THREAD_ENV_VARS:
        env.pop(name, None)
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else str(REPO_ROOT)
    )
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "OMP_NUM_THREADS=7",
                "LLM_BACKEND=mock",
                "FAST_MODE=true",
            ]
        ),
        encoding="utf-8",
    )

    code = """
import builtins
import json
import os

real_import = builtins.__import__

def tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "fastapi" or name.startswith("fastapi."):
        print(json.dumps({
            "FAST_MODE": os.environ.get("FAST_MODE"),
            "LLM_BACKEND": os.environ.get("LLM_BACKEND"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
            "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
        }, sort_keys=True))
        raise SystemExit(0)
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = tracking_import
import src.app
raise SystemExit("src.app did not import fastapi")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert json.loads(result.stdout) == {
        "FAST_MODE": "true",
        "LLM_BACKEND": "mock",
        "OMP_NUM_THREADS": "7",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }


def test_app_import_keeps_engine_lazy():
    assert web_app.qa_system is None


def test_app_main_defaults_to_loopback(monkeypatch):
    observed = {}

    def fake_run(app, *, host, port, log_level):
        observed.update(
            {"app": app, "host": host, "port": port, "log_level": log_level}
        )

    monkeypatch.delenv("WEB_HOST", raising=False)
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.delenv("WEB_PORT", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.setattr(web_app.uvicorn, "run", fake_run)

    web_app.main()

    assert observed["host"] == "127.0.0.1"
    assert observed["port"] == 7860


def test_app_main_allows_explicit_non_loopback_host(monkeypatch):
    observed = {}

    def fake_run(app, *, host, port, log_level):
        observed.update({"host": host, "port": port, "log_level": log_level})

    monkeypatch.setenv("WEB_HOST", "0.0.0.0")
    monkeypatch.setenv("WEB_PORT", "8799")
    monkeypatch.setattr(web_app.uvicorn, "run", fake_run)

    web_app.main()

    assert observed["host"] == "0.0.0.0"
    assert observed["port"] == 8799


def test_thread_store_path_prefers_loopwright_env(monkeypatch, tmp_path):
    new_path = tmp_path / "loopwright.sqlite3"
    old_path = tmp_path / "legacy.sqlite3"
    monkeypatch.setenv("LOOPWRIGHT_THREAD_DB_PATH", str(new_path))
    monkeypatch.setenv("AI_LOOP_THREAD_DB_PATH", str(old_path))

    assert web_app.default_thread_store_path() == new_path


def test_thread_store_path_accepts_legacy_ai_loop_env(monkeypatch, tmp_path):
    legacy_path = tmp_path / "legacy.sqlite3"
    monkeypatch.delenv("LOOPWRIGHT_THREAD_DB_PATH", raising=False)
    monkeypatch.setenv("AI_LOOP_THREAD_DB_PATH", str(legacy_path))

    assert web_app.default_thread_store_path() == legacy_path


def test_thread_store_path_uses_legacy_default_when_new_default_missing(
    monkeypatch,
    tmp_path,
):
    monkeypatch.delenv("LOOPWRIGHT_THREAD_DB_PATH", raising=False)
    monkeypatch.delenv("AI_LOOP_THREAD_DB_PATH", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    legacy_default = tmp_path / ".ai-loop-engine" / "threads.sqlite3"
    legacy_default.parent.mkdir()
    legacy_default.write_bytes(b"legacy sqlite placeholder")

    assert web_app.default_thread_store_path() == legacy_default


def test_thread_store_path_prefers_loopwright_default_when_present(
    monkeypatch,
    tmp_path,
):
    monkeypatch.delenv("LOOPWRIGHT_THREAD_DB_PATH", raising=False)
    monkeypatch.delenv("AI_LOOP_THREAD_DB_PATH", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    legacy_default = tmp_path / ".ai-loop-engine" / "threads.sqlite3"
    new_default = tmp_path / ".loopwright" / "threads.sqlite3"
    legacy_default.parent.mkdir()
    new_default.parent.mkdir()
    legacy_default.write_bytes(b"legacy sqlite placeholder")
    new_default.write_bytes(b"new sqlite placeholder")

    assert web_app.default_thread_store_path() == new_default


def test_static_frontend_is_served():
    client = TestClient(web_app.create_app(FakeQA()))

    response = client.get("/")
    script = client.get("/assets/app.js")
    styles = client.get("/assets/styles.css")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert "Loopwright" in response.text
    assert "Model status" in response.text
    assert "backend-pill" in response.text
    assert "model-pill" in response.text
    assert "Threads" in response.text
    assert "Loop Recipe" in response.text
    assert "Durable Runs" in response.text
    assert "Attach File" in response.text
    assert "file-scope" in response.text
    assert "No file attached to this session" in response.text
    assert "Model Thinking" in response.text
    assert "active-thread-memory" in response.text
    assert "memory-status" in response.text
    assert "Ask</button>" in response.text
    assert "Run Loop" not in response.text
    assert "query-context-control" not in response.text
    assert "/assets/app.js" in response.text
    assert response.text.index('id="query-input"') < response.text.index(
        'id="upload-button"'
    )
    assert response.text.index('id="upload-button"') < response.text.index(
        'id="query-button"'
    )
    assert script.status_code == 200
    assert script.headers["cache-control"] == "no-store"
    assert "No messages yet." in script.text
    assert "No file attached to this session" in script.text
    assert "renderMessageThinking" in script.text
    assert "session_id" in script.text
    assert "switchThread" in script.text
    assert "recipe_id" in script.text
    assert "renderRuns" in script.text
    assert "inspectDurableRun" in script.text
    assert "durable_public_report" in script.text
    assert "PUBLIC_REPORT_PROJECTION_SCHEMA" in script.text
    assert "storedPublicLoopPayload" not in script.text
    assert "runMemoryLabel" in script.text
    assert "normalizeMessageMarkdownStructure" in script.text
    assert "result.trace?.model_thinking" in script.text
    assert "message-thinking" in script.text
    assert "innerHTML" not in script.text
    assert styles.status_code == 200
    assert styles.headers["cache-control"] == "no-store"
    assert ".thread-button" in styles.text
    assert ".memory-status" in styles.text
    assert ".message-content strong" in styles.text
    assert ".message-thinking" in styles.text
    assert ".message-code-block" in styles.text
    assert ".run-inspector-status" in styles.text
    assert '.run-row[data-state="quarantined"]' in styles.text



def run_frontend_node(script: str) -> None:
    if shutil.which("node") is None:
        pytest.skip("node is required for static frontend execution test")

    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "APP_JS_PATH": str(REPO_ROOT / "src" / "web_static" / "app.js"),
            "FRONTEND_HARNESS_URL": (
                REPO_ROOT / "tests" / "frontend_harness.mjs"
            ).as_uri(),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""


def test_public_display_label_limit_counts_unicode_code_points():
    started_at = utc_now()
    answer = "Visible answer"
    accepted_label = "🚀" * 300 + "\ufeff\u200b\u200d\u2066"
    accepted_report = LoopReport(
        run=LoopRun(
            run_id="run_unicode_model_label",
            user_input="question",
            context_provider="none",
            backend="mock",
            model_label=accepted_label,
            started_at=started_at,
            completed_at=started_at,
            steps=visible_not_verified_steps(
                answer=answer,
                evidence=(),
                started_at=started_at,
                prefix="unicode_model_label",
                model_label=accepted_label,
            ),
            final_decision=LoopDecision.NOT_VERIFIED,
            terminal_reason=LoopTerminalReason.NOT_VERIFIED,
            final_answer=answer,
        )
    )

    public_run = accepted_report.to_public_dict()["run"]

    assert public_run["model_label"] == accepted_label
    assert public_run["steps"][0]["model_label"] == accepted_label

    for rejected_label in (
        "🚀" * 513,
        "\ufeff",
        "\ufeff\u00a0\u0085",
        "\u200b",
        "\u200d",
        "\u2066",
        "\u200b\u200d\u2066\ufeff",
    ):
        rejected_report = replace(
            accepted_report,
            run=replace(
                accepted_report.run,
                model_label=rejected_label,
                steps=tuple(
                    replace(step, model_label=rejected_label)
                    for step in accepted_report.run.steps
                ),
            ),
        )
        with pytest.raises(ValueError, match="display label"):
            rejected_report.to_public_dict()

    with pytest.raises(ValueError, match="model_label must not be empty"):
        replace(accepted_report.run, model_label="\u0085")


def test_static_frontend_inspects_a_durable_public_run_with_citations():
    run_frontend_node(
        r'''
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
const {
  createDom,
  createThreadPayload,
  deferred,
  findNode,
  importFreshApp,
  jsonResponse,
  nodeText,
  tick,
} = await import(process.env.FRONTEND_HARNESS_URL);

const dom = createDom();
const runDetail = deferred();
const fetchedUrls = [];
const evidenceId = `evidence_${"a".repeat(64)}`;
const astralLabelWithinLimit =
  "🚀".repeat(300) + "\uFEFF\u200B\u200D\u2066";
const astralLabelOverLimit = "🚀".repeat(513);
const projectionSource = readFileSync("src/public_projection.py", "utf8");
const appSource = readFileSync(process.env.APP_JS_PATH, "utf8");
const tokenListBetween = (source, startMarker, endMarker) => {
  const start = source.indexOf(startMarker);
  assert.notEqual(start, -1, `missing range start marker: ${startMarker}`);
  const contentStart = start + startMarker.length;
  const end = source.indexOf(endMarker, contentStart);
  assert.notEqual(end, -1, `missing range end marker: ${endMarker}`);
  return source.slice(contentStart, end).trim().split(/\s+/);
};
const serverMarkRangeTokens = tokenListBetween(
  projectionSource,
  'for token in """\n',
  '\n""".split()',
);
const browserMarkRangeTokens = tokenListBetween(
  appSource,
  'const PUBLIC_DISPLAY_MARK_CODE_POINT_RANGES = Object.freeze(\n  `\n',
  '\n  `.trim()',
);
assert.equal(serverMarkRangeTokens.length, 310);
assert.deepEqual(
  browserMarkRangeTokens,
  serverMarkRangeTokens,
  "browser and server must pin the same Unicode 15.0.0 mark ranges",
);
assert.ok(projectionSource.includes('_PUBLIC_DISPLAY_MARK_UNICODE_VERSION = "15.0.0"'));
assert.ok(appSource.includes('PUBLIC_DISPLAY_MARK_UNICODE_VERSION = "15.0.0"'));
const markBounds = serverMarkRangeTokens.map((token) => {
  const [start, end = start] = token.split("-");
  return [Number.parseInt(start, 16), Number.parseInt(end, 16)];
});
const markRangeStarts = markBounds
  .map(([start]) => String.fromCodePoint(start))
  .join("");
const markRangeEnds = markBounds
  .map(([, end]) => String.fromCodePoint(end))
  .join("");
const allC1Controls = Array.from(
  { length: 0x20 },
  (_, index) => String.fromCodePoint(0x80 + index),
).join("");
const allVariationSelectors = [
  ...Array.from(
    { length: 0x10 },
    (_, index) => String.fromCodePoint(0xfe00 + index),
  ),
  ...Array.from(
    { length: 0xf0 },
    (_, index) => String.fromCodePoint(0xe0100 + index),
  ),
].join("");
const visibleBaseWithEveryMarkRange = `VisibleModel${markRangeStarts}`;
const recipes = [{
  recipe_id: "recipe_general_loop",
  name: "General assistant loop",
  goal: "Answer clearly.",
  context_provider: "smart",
  model_profile: "quality",
  verifier: "default",
  instructions: "",
  success_criteria: [],
  stop_condition: "",
  is_default: true,
}];
const report = {
  schema_version: "loop-report/v1",
  projection_schema_version: "loop-public-report/v1",
  public: true,
  public_redaction: { applied: false, reason: null },
  run: {
    run_id: "run_history",
    session_id: "thread_history",
    context_provider: "document",
    conversation_context_count: 2,
    semantic_memory_count: 1,
    semantic_memory_status: "retrieved",
    backend: "openai-compatible",
    model_label: astralLabelWithinLimit,
    policy: {
      max_retries: 1,
      require_citations: true,
      require_verifier_for_supported: true,
      allow_mock_supported: false,
      allow_tool_calls: false,
      require_human_review_for_tools: true,
    },
    started_at: "2026-06-26T00:00:00.000Z",
    completed_at: "2026-06-26T00:00:01.000Z",
    steps: [{
      step_id: "step_format",
      phase: "format_check",
      decision: "continue",
      started_at: "2026-06-26T00:00:00.000Z",
      ended_at: "2026-06-26T00:00:00.250Z",
      duration_ms: 250,
      backend: null,
      model_label: null,
      retry_count: 0,
      error_present: false,
      verification: null,
      human_review_required: false,
    }, {
      step_id: "step_mechanical",
      phase: "mechanical_check",
      decision: "retry",
      started_at: "2026-06-26T00:00:00.250Z",
      ended_at: "2026-06-26T00:00:00.500Z",
      duration_ms: 250,
      backend: null,
      model_label: null,
      retry_count: 0,
      error_present: false,
      verification: null,
      human_review_required: false,
    }, {
      step_id: "step_history",
      phase: "verify",
      decision: "supported",
      started_at: "2026-06-26T00:00:00.500Z",
      ended_at: "2026-06-26T00:00:01.000Z",
      duration_ms: 500,
      backend: "openai-compatible",
      model_label: astralLabelWithinLimit,
      retry_count: 0,
      error_present: false,
      verification: {
        outcome: "supported",
        verifier_backend: "openai-compatible",
        verifier_model_label: visibleBaseWithEveryMarkRange,
        same_model_as_drafter: true,
      },
      human_review_required: false,
    }],
    final_decision: "supported",
    terminal_reason: "completed",
    final_answer: "The evidence supports Project Phoenix.",
    error_present: false,
    evidence: [{
      evidence_id: evidenceId,
      citation_id: 1,
      provider: "document",
      locator: { page: null, chunk_index: 0 },
    }],
  },
};
const oversizedVerifierLabel = astralLabelOverLimit;
const hostileReportCases = [
  ["run_bad_run_backend", (candidate) => {
    candidate.run.backend = "secret_backend";
  }],
  ["run_bad_run_model", (candidate) => {
    candidate.run.model_label = "SECRET_MODEL_URL=https://attacker.invalid/key";
  }],
  ["run_bad_step_backend", (candidate) => {
    candidate.run.steps[2].backend = "secret_step_backend";
  }],
  ["run_bad_step_model", (candidate) => {
    candidate.run.steps[2].model_label = "StepModel\nSECRET_CONTROL_LABEL";
  }],
  ["run_bad_verifier_backend", (candidate) => {
    candidate.run.steps[2].verification.verifier_backend = "secret_verifier_backend";
  }],
  ["run_bad_verifier_model", (candidate) => {
    candidate.run.steps[2].verification.verifier_model_label = oversizedVerifierLabel;
  }],
  ["run_blank_bom_model", (candidate) => {
    candidate.run.model_label = "\uFEFF";
  }],
  ["run_blank_nel_step_model", (candidate) => {
    candidate.run.steps[2].model_label = "\u0085";
  }],
  ["run_blank_mixed_verifier_model", (candidate) => {
    candidate.run.steps[2].verification.verifier_model_label = "\uFEFF\u00A0\u0085";
  }],
  ["run_blank_zero_width_model", (candidate) => {
    candidate.run.model_label = "\u200B";
  }],
  ["run_blank_joiner_step_model", (candidate) => {
    candidate.run.steps[2].model_label = "\u200D";
  }],
  ["run_blank_bidi_verifier_model", (candidate) => {
    candidate.run.steps[2].verification.verifier_model_label = "\u2066";
  }],
  ["run_blank_mixed_format_model", (candidate) => {
    candidate.run.model_label = "\u200B\u200D\u2066\uFEFF";
  }],
  ["run_all_c1_controls_model", (candidate) => {
    candidate.run.model_label = allC1Controls;
  }],
  ["run_visible_c1_control_model", (candidate) => {
    candidate.run.model_label = `Visible${String.fromCodePoint(0x80)}`;
  }],
  ["run_mark_range_starts_model", (candidate) => {
    candidate.run.model_label = markRangeStarts;
  }],
  ["run_mark_range_ends_step_model", (candidate) => {
    candidate.run.steps[2].model_label = markRangeEnds;
  }],
  ["run_variation_selectors_verifier_model", (candidate) => {
    candidate.run.steps[2].verification.verifier_model_label = allVariationSelectors;
  }],
].map(([runId, mutate]) => {
  const candidate = JSON.parse(JSON.stringify(report));
  candidate.run.run_id = runId;
  mutate(candidate);
  return { runId, report: candidate };
});
const storedLoopPayload = {
  timeline: { rows: [], final_decision: "supported" },
  summary: { final_decision: "supported" },
  trace: {
    question: "FOREIGN sibling question",
    answer: "FOREIGN sibling answer",
    document: "foreign-sibling.txt",
    backend: "foreign-backend",
    model: "ForeignModel",
    retrieved_chunk_count: 1,
    citations: [{
      id: 99,
      source: "foreign-sibling.txt",
      page: null,
      chunk: 1,
      excerpt: "FOREIGN sibling citation",
    }],
    self_check: {
      outcome: "supported",
      reasons: ["citation_matches_claim"],
      retry_attempted: false,
    },
    model_thinking: {
      available: false,
      redacted: false,
      content: null,
      label: "Model Thinking (unverified)",
    },
    loop_report: report,
  },
  raw_report: { secret: "raw report must never render" },
};
const latest = {
  timeline: { rows: [], final_decision: "not_verified" },
  summary: { run_id: "run_latest", final_decision: "not_verified" },
  trace: {},
};
const historicalRun = {
  run_id: "run_history",
  thread_id: "thread_history",
  projection_status: "available",
  projection_schema_version: "loop-public-report/v1",
  final_decision: "supported",
  terminal_reason: "completed",
  context_provider: "document",
  backend: "openai-compatible",
  model: "GatewayModel",
  step_count: 1,
  started_at: "2026-06-26T00:00:00.000Z",
  completed_at: "2026-06-26T00:00:01.000Z",
  created_at: "2026-06-26T00:00:01.000Z",
};
const extraFieldRun = { ...historicalRun, run_id: "run_extra_field" };
const hostileRuns = hostileReportCases.map(({ runId }) => ({
  ...historicalRun,
  run_id: runId,
}));
const thread = createThreadPayload(
  "thread_history",
  [{
    role: "assistant",
    content: "The evidence supports Project Phoenix.",
    loop_payload: storedLoopPayload,
  }],
  latest,
  {
    loopRuns: [historicalRun, extraFieldRun, ...hostileRuns],
    loopRunCount: 2 + hostileRuns.length,
  },
);

globalThis.fetch = async (url, options = {}) => {
  fetchedUrls.push(url);
  const method = String(options.method || "GET").toUpperCase();
  if (url === "/api/status") {
    return jsonResponse({ backend: "mock", model: "MockLLM" });
  }
  if (url === "/api/recipes") {
    return jsonResponse({ default_recipe_id: "recipe_general_loop", recipes });
  }
  if (url.startsWith("/api/recipes/") && method === "GET") {
    return jsonResponse(recipes[0]);
  }
  if (url === "/api/threads" && method === "GET") {
    return jsonResponse({ threads: [thread] });
  }
  if (url === "/api/threads/thread_history" && method === "GET") {
    return jsonResponse(thread);
  }
  if (url === "/api/threads/thread_history/runs/run_history" && method === "GET") {
    return runDetail.promise;
  }
  if (url === "/api/threads/thread_history/runs/run_extra_field" && method === "GET") {
    return jsonResponse({
      public: true,
      thread_id: "thread_history",
      run_id: "run_extra_field",
      report: {
        ...report,
        unexpected_secret_payload: "SECRET_EXTRA_REPORT_FIELD",
        run: { ...report.run, run_id: "run_extra_field" },
      },
    });
  }
  const hostileCase = hostileReportCases.find(
    ({ runId }) => url === `/api/threads/thread_history/runs/${runId}`,
  );
  if (hostileCase && method === "GET") {
    return jsonResponse({
      public: true,
      thread_id: "thread_history",
      run_id: hostileCase.runId,
      report: hostileCase.report,
    });
  }
  throw new Error(`unexpected fetch ${url}`);
};

await importFreshApp();
await tick();
assert.ok(dom["summary-json"].textContent.includes("run_latest"));
const row = findNode(
  dom["run-list"],
  (node) => node.className === "run-row" && nodeText(node).includes("run_history"),
);
assert.ok(row, "stored run should be selectable");
assert.equal(row.tagName, "BUTTON");

const inspection = row.dispatch("click");
await tick();
let status = findNode(
  dom["run-list"],
  (node) => node.className === "run-inspector-status",
);
assert.equal(status.dataset.state, "loading");
assert.ok(status.textContent.includes("Loading stored public run run_history"));
const pendingRow = findNode(
  dom["run-list"],
  (node) => node.className === "run-row" && nodeText(node).includes("run_history"),
);
assert.equal(pendingRow.dataset.selected, "false");
assert.equal(pendingRow.ariaBusy, "true");
assert.ok(dom["summary-json"].textContent.includes("run_latest"));
runDetail.resolve(jsonResponse({
  public: true,
  thread_id: "thread_history",
  run_id: "run_history",
  final_decision: "block",
  backend: "FOREIGN envelope backend",
  model: "FOREIGN envelope model",
  report,
}));
await inspection;

status = findNode(
  dom["run-list"],
  (node) => node.className === "run-inspector-status",
);
assert.equal(status.dataset.state, "loaded");
assert.equal(status.textContent, "Viewing stored public run run_history.");
const selectedRow = findNode(
  dom["run-list"],
  (node) => node.className === "run-row" && nodeText(node).includes("run_history"),
);
assert.equal(selectedRow.dataset.selected, "true");
assert.equal(dom["final-decision"].textContent, "supported");
assert.ok(nodeText(dom.timeline).includes("Verify"));
const summary = JSON.parse(dom["summary-json"].textContent);
assert.equal(summary.source, "durable_public_report");
assert.equal(summary.public, true);
assert.equal(summary.projection_schema_version, "loop-public-report/v1");
assert.equal(summary.run_id, "run_history");
assert.equal(summary.terminal_reason, "completed");
assert.equal(summary.conversation_context_count, 2);
assert.equal(summary.semantic_memory_count, 1);
assert.equal(summary.semantic_memory_status, "retrieved");
assert.equal(summary.format_check, "passed");
assert.equal(summary.mechanical_check, "retry");
const trace = JSON.parse(dom["trace-json"].textContent);
assert.equal(trace.public, true);
assert.equal(trace.question, null);
assert.equal(trace.answer, "The evidence supports Project Phoenix.");
assert.deepEqual(trace.citations, [{
  evidence_id: evidenceId,
  citation_id: 1,
  provider: "document",
  locator: { page: null, chunk_index: 0 },
}]);
assert.equal(trace.loop_report.run.run_id, "run_history");
assert.equal(trace.loop_report.run.model_label, astralLabelWithinLimit);
assert.equal(trace.loop_report.run.steps[2].model_label, astralLabelWithinLimit);
assert.equal(
  trace.loop_report.run.steps[2].verification.verifier_model_label,
  visibleBaseWithEveryMarkRange,
);
assert.equal(dom["trace-json"].textContent.includes("raw report must never render"), false);
assert.equal(dom["trace-json"].textContent.includes("FOREIGN sibling"), false);
assert.equal(dom["trace-json"].textContent.includes("foreign-sibling.txt"), false);
assert.equal(dom["summary-json"].textContent.includes("FOREIGN envelope"), false);
const extraFieldRow = findNode(
  dom["run-list"],
  (node) => node.className === "run-row" && nodeText(node).includes("run_extra_field"),
);
await extraFieldRow.dispatch("click");
status = findNode(
  dom["run-list"],
  (node) => node.className === "run-inspector-status",
);
assert.equal(status.dataset.state, "error");
assert.ok(status.textContent.includes("unavailable or malformed"));
assert.equal(dom["trace-json"].textContent.includes("SECRET_EXTRA_REPORT_FIELD"), false);
assert.equal(dom["summary-json"].textContent.includes("run_history"), true);
for (const { runId } of hostileReportCases) {
  const hostileRow = findNode(
    dom["run-list"],
    (node) => node.className === "run-row" && nodeText(node).includes(runId),
  );
  assert.ok(hostileRow, `hostile provenance row should exist for ${runId}`);
  await hostileRow.dispatch("click");
  status = findNode(
    dom["run-list"],
    (node) => node.className === "run-inspector-status",
  );
  assert.equal(status.dataset.state, "error");
  assert.ok(status.textContent.includes("unavailable or malformed"));
  assert.equal(dom["summary-json"].textContent.includes("run_history"), true);
  const renderedPublicText = [
    dom["summary-json"].textContent,
    dom["trace-json"].textContent,
    nodeText(dom.timeline),
    nodeText(dom["memory-status"]),
  ].join("\n");
  assert.equal(renderedPublicText.includes("secret_backend"), false);
  assert.equal(renderedPublicText.includes("secret_step_backend"), false);
  assert.equal(renderedPublicText.includes("secret_verifier_backend"), false);
  assert.equal(renderedPublicText.includes("SECRET_"), false);
  assert.equal(renderedPublicText.includes(oversizedVerifierLabel), false);
}
assert.deepEqual(
  fetchedUrls.filter((url) => url.includes("/runs/")),
  [
    "/api/threads/thread_history/runs/run_history",
    "/api/threads/thread_history/runs/run_extra_field",
    ...hostileReportCases.map(
      ({ runId }) => `/api/threads/thread_history/runs/${runId}`,
    ),
  ],
);
'''
    )


def test_static_frontend_durable_run_inspector_fails_closed():
    run_frontend_node(
        r'''
import assert from "node:assert/strict";
const {
  createDom,
  createThreadPayload,
  errorResponse,
  findNode,
  importFreshApp,
  jsonResponse,
  nodeText,
  tick,
} = await import(process.env.FRONTEND_HARNESS_URL);

const dom = createDom();
const recipes = [{
  recipe_id: "recipe_general_loop",
  name: "General assistant loop",
  goal: "Answer clearly.",
  is_default: true,
}];
const loopRuns = [
  "run_missing",
  "run_malformed",
  "run_raw",
  "run_identity",
  "run_nested_identity",
  "run_bad_evidence",
  "run_cross",
  "run_legacy",
].map((runId) => ({
  run_id: runId,
  thread_id: "thread_fail_closed",
  projection_status: "available",
  projection_schema_version: "loop-public-report/v1",
  final_decision: "not_verified",
  terminal_reason: "not_verified",
  context_provider: "none",
  backend: "mock",
  model: "MockLLM",
  step_count: 1,
  started_at: "2026-06-26T00:00:00.000Z",
  completed_at: "2026-06-26T00:00:01.000Z",
  created_at: "2026-06-26T00:00:01.000Z",
}));
const latest = {
  timeline: { rows: [], final_decision: "not_verified" },
  summary: { marker: "latest remains visible" },
  trace: { marker: "latest trace remains visible" },
};
const crossRunStoredReport = {
  schema_version: "loop-report/v1",
  projection_schema_version: "loop-public-report/v1",
  run: {
    run_id: "run_cross",
    session_id: "thread_fail_closed",
    context_provider: "document",
    backend: "openai-compatible",
    model_label: "OtherRunModel",
    steps: [],
    evidence: [],
    final_decision: "supported",
    metadata: {},
  },
};
const thread = createThreadPayload(
  "thread_fail_closed",
  [{
    role: "assistant",
    content: "Stored payload from another report.",
    loop_payload: {
      trace: {
        question: "cross-run secret question",
        answer: "cross-run secret answer",
        citations: [{ id: 1, source: "wrong.txt", excerpt: "cross-run secret citation" }],
        loop_report: crossRunStoredReport,
      },
    },
  }],
  latest,
  { loopRuns, loopRunCount: loopRuns.length },
);

globalThis.fetch = async (url, options = {}) => {
  const method = String(options.method || "GET").toUpperCase();
  if (url === "/api/status") {
    return jsonResponse({ backend: "mock", model: "MockLLM" });
  }
  if (url === "/api/recipes") {
    return jsonResponse({ default_recipe_id: "recipe_general_loop", recipes });
  }
  if (url.startsWith("/api/recipes/") && method === "GET") {
    return jsonResponse(recipes[0]);
  }
  if (url === "/api/threads" && method === "GET") {
    return jsonResponse({ threads: [thread] });
  }
  if (url === "/api/threads/thread_fail_closed" && method === "GET") {
    return jsonResponse(thread);
  }
  if (url.endsWith("/runs/run_missing")) {
    return errorResponse(404, { detail: "Loop run not found." });
  }
  if (url.endsWith("/runs/run_malformed")) {
    return jsonResponse({
      public: true,
      thread_id: "thread_fail_closed",
      run_id: "run_malformed",
      report: { run: { run_id: "run_malformed", steps: "not-an-array" } },
    });
  }
  if (url.endsWith("/runs/run_raw")) {
    return jsonResponse({
      public: false,
      thread_id: "thread_fail_closed",
      run_id: "run_raw",
      report: {
        projection_schema_version: "loop-public-report/v1",
        run: {
          run_id: "run_raw",
          session_id: "thread_fail_closed",
          steps: [],
          evidence: [],
          user_input: "raw secret prompt",
          final_answer: "raw secret answer",
        },
      },
    });
  }
  if (url.endsWith("/runs/run_identity")) {
    return jsonResponse({
      public: true,
      thread_id: "thread_other",
      run_id: "run_identity",
      report: {
        schema_version: "loop-report/v1",
        projection_schema_version: "loop-public-report/v1",
        run: {
          run_id: "run_identity",
          session_id: "thread_fail_closed",
          steps: [],
          evidence: [],
          final_decision: "not_verified",
          metadata: {},
        },
      },
    });
  }
  if (url.endsWith("/runs/run_nested_identity")) {
    return jsonResponse({
      public: true,
      thread_id: "thread_fail_closed",
      run_id: "run_nested_identity",
      report: {
        schema_version: "loop-report/v1",
        projection_schema_version: "loop-public-report/v1",
        run: {
          run_id: "run_foreign",
          session_id: "thread_other",
          user_input: "nested identity secret",
          final_answer: "nested identity secret",
          steps: [],
          evidence: [],
          final_decision: "not_verified",
          metadata: {},
        },
      },
    });
  }
  if (url.endsWith("/runs/run_bad_evidence")) {
    return jsonResponse({
      public: true,
      thread_id: "thread_fail_closed",
      run_id: "run_bad_evidence",
      report: {
        schema_version: "loop-report/v1",
        projection_schema_version: "loop-public-report/v1",
        run: {
          run_id: "run_bad_evidence",
          session_id: "thread_fail_closed",
          steps: [],
          evidence: [{ source: "secret.txt", excerpt: "malformed evidence secret" }],
          final_decision: "not_verified",
          metadata: {},
        },
      },
    });
  }
  if (url.endsWith("/runs/run_cross")) {
    return jsonResponse({
      public: true,
      thread_id: "thread_fail_closed",
      run_id: "run_cross",
      report: {
        schema_version: "loop-report/v1",
        projection_schema_version: "loop-public-report/v1",
        public: true,
        public_redaction: { applied: false, reason: null },
        run: {
          run_id: "run_cross",
          session_id: "thread_fail_closed",
          context_provider: "document",
          conversation_context_count: null,
          semantic_memory_count: null,
          semantic_memory_status: null,
          backend: "openai-compatible",
          model_label: "ActualRunModel",
          policy: {
            max_retries: 1,
            require_citations: false,
            require_verifier_for_supported: true,
            allow_mock_supported: false,
            allow_tool_calls: false,
            require_human_review_for_tools: true,
          },
          started_at: "2026-06-26T00:00:00.000Z",
          completed_at: "2026-06-26T00:00:01.000Z",
          steps: [],
          evidence: [],
          final_decision: "not_verified",
          terminal_reason: "not_verified",
          final_answer: "Unverified answer.",
          error_present: false,
        },
      },
    });
  }
  if (url.endsWith("/runs/run_legacy")) {
    return jsonResponse({
      public: true,
      thread_id: "thread_fail_closed",
      run_id: "run_legacy",
      report: {
        schema_version: "loop-report/v1",
        run: {
          run_id: "run_legacy",
          session_id: "thread_fail_closed",
          context_provider: "none",
          backend: "mock",
          model_label: "MockLLM",
          steps: [],
          final_decision: "not_verified",
          metadata: {},
        },
      },
    });
  }
  throw new Error(`unexpected fetch ${url}`);
};

await importFreshApp();
await tick();
const clickRun = async (runId) => {
  const row = findNode(
    dom["run-list"],
    (node) => node.className === "run-row" && nodeText(node).includes(runId),
  );
  assert.ok(row, `stored row should exist for ${runId}`);
  await row.dispatch("click");
  return findNode(
    dom["run-list"],
    (node) => node.className === "run-inspector-status",
  );
};
const assertNoSelectedRun = () => {
  const runRows = dom["run-list"].children.filter(
    (node) => node.className === "run-row",
  );
  assert.ok(runRows.length > 0);
  for (const row of runRows) {
    assert.equal(row.dataset.selected, "false");
    assert.equal(row.ariaPressed, "false");
  }
};

let status = await clickRun("run_missing");
assert.equal(status.dataset.state, "error");
assert.equal(status.textContent, "Could not inspect run_missing: Loop run not found.");
assert.ok(dom["summary-json"].textContent.includes("latest remains visible"));
assertNoSelectedRun();

status = await clickRun("run_malformed");
assert.equal(status.dataset.state, "error");
assert.ok(status.textContent.includes("unavailable or malformed"));
assert.ok(dom["summary-json"].textContent.includes("latest remains visible"));
assertNoSelectedRun();

status = await clickRun("run_raw");
assert.equal(status.dataset.state, "error");
assert.ok(status.textContent.includes("unavailable or malformed"));
assert.equal(dom["summary-json"].textContent.includes("raw secret"), false);
assert.equal(dom["trace-json"].textContent.includes("raw secret"), false);
assertNoSelectedRun();

status = await clickRun("run_identity");
assert.equal(status.dataset.state, "error");
assert.ok(status.textContent.includes("unavailable or malformed"));
assertNoSelectedRun();

status = await clickRun("run_nested_identity");
assert.equal(status.dataset.state, "error");
assert.ok(status.textContent.includes("unavailable or malformed"));
assert.equal(dom["trace-json"].textContent.includes("nested identity secret"), false);
assertNoSelectedRun();

status = await clickRun("run_bad_evidence");
assert.equal(status.dataset.state, "error");
assert.ok(status.textContent.includes("unavailable or malformed"));
assert.equal(dom["trace-json"].textContent.includes("malformed evidence secret"), false);
assertNoSelectedRun();

status = await clickRun("run_cross");
assert.equal(status.dataset.state, "loaded");
assert.ok(status.textContent.includes("No public citation identities are available"));
assert.equal(dom["trace-json"].textContent.includes("cross-run secret"), false);
const crossRunTrace = JSON.parse(dom["trace-json"].textContent);
assert.deepEqual(crossRunTrace.citations, []);
assert.ok(
  nodeText(dom["memory-status"]).includes("last run memory use unknown"),
  "missing historical provenance must remain unknown rather than becoming zero",
);

status = await clickRun("run_legacy");
assert.equal(status.dataset.state, "error");
assert.ok(status.textContent.includes("unavailable or malformed"));
assert.equal(dom["summary-json"].textContent.includes("run_cross"), true);
let retainedCrossRow = findNode(
  dom["run-list"],
  (node) => node.className === "run-row" && nodeText(node).includes("run_cross"),
);
assert.equal(retainedCrossRow.dataset.selected, "true");

status = await clickRun("run_malformed");
assert.equal(status.dataset.state, "error");
assert.equal(dom["summary-json"].textContent.includes("run_cross"), true);
retainedCrossRow = findNode(
  dom["run-list"],
  (node) => node.className === "run-row" && nodeText(node).includes("run_cross"),
);
const failedMalformedRow = findNode(
  dom["run-list"],
  (node) => node.className === "run-row" && nodeText(node).includes("run_malformed"),
);
assert.equal(retainedCrossRow.dataset.selected, "true");
assert.equal(failedMalformedRow.dataset.selected, "false");
'''
    )


def test_static_frontend_quarantined_run_summaries_are_bounded_and_disabled():
    run_frontend_node(
        r'''
import assert from "node:assert/strict";
const {
  createDom,
  createThreadPayload,
  findNode,
  importFreshApp,
  jsonResponse,
  nodeText,
  tick,
} = await import(process.env.FRONTEND_HARNESS_URL);

const dom = createDom();
const threadId = "thread_quarantined_runs";
const createdAt = "2026-06-26T00:00:01.000Z";
const oversizedReason = `SECRET_OVERSIZED_REASON_${"x".repeat(12000)}`;
const oversizedModel = `SECRET_OVERSIZED_MODEL_${"x".repeat(12000)}`;
const availableBase = {
  run_id: "run_available_template",
  thread_id: threadId,
  projection_status: "available",
  projection_schema_version: "loop-public-report/v1",
  final_decision: "not_verified",
  terminal_reason: "not_verified",
  context_provider: "none",
  backend: "mock",
  model: "MockLLM",
  step_count: 1,
  started_at: "2026-06-26T00:00:00.000Z",
  completed_at: createdAt,
  created_at: createdAt,
};
const loopRuns = [
  {
    run_id: "run_invalid_report",
    thread_id: threadId,
    created_at: createdAt,
    projection_status: "quarantined",
    quarantine_reason: "stored_loop_report_invalid",
  },
  {
    run_id: "run_legacy_unbound",
    thread_id: threadId,
    created_at: createdAt,
    projection_status: "quarantined",
    quarantine_reason: "legacy_visible_answer_unbound",
  },
  {
    run_id: "run_oversized_reason",
    thread_id: threadId,
    created_at: createdAt,
    projection_status: "quarantined",
    quarantine_reason: oversizedReason,
  },
  {
    run_id: "run_wrong_reason_type",
    thread_id: threadId,
    created_at: createdAt,
    projection_status: "quarantined",
    quarantine_reason: { secret: "SECRET_OBJECT_REASON" },
  },
  {
    ...availableBase,
    run_id: "run_available_with_reason",
    quarantine_reason: "legacy_visible_answer_unbound",
  },
  {
    run_id: "run_wrong_status_type",
    thread_id: threadId,
    created_at: createdAt,
    projection_status: { secret: "SECRET_STATUS_OBJECT" },
    quarantine_reason: "stored_loop_report_invalid",
  },
  {
    run_id: "run_missing_status",
    thread_id: threadId,
    created_at: createdAt,
    quarantine_reason: "stored_loop_report_invalid",
  },
  {
    ...availableBase,
    run_id: "run_array_decision",
    final_decision: ["SECRET_ARRAY_DECISION"],
  },
  {
    ...availableBase,
    run_id: "run_object_context",
    context_provider: { secret: "SECRET_CONTEXT_OBJECT" },
  },
  {
    ...availableBase,
    run_id: "run_bad_backend_enum",
    backend: "SECRET_FORGED_BACKEND",
  },
  {
    ...availableBase,
    run_id: "run_object_backend",
    backend: { secret: "SECRET_BACKEND_OBJECT" },
  },
  {
    ...availableBase,
    run_id: "run_oversized_model",
    model: oversizedModel,
  },
  {
    ...availableBase,
    run_id: "run_url_model",
    model: "https://SECRET_MODEL_GATEWAY.example",
  },
  {
    ...availableBase,
    run_id: "run_blank_model",
    model: "   ",
  },
  {
    ...availableBase,
    run_id: "run_control_model",
    model: "MockLLM\nSECRET_CONTROL_MODEL",
  },
  {
    ...availableBase,
    run_id: "run_object_steps",
    step_count: { secret: "SECRET_STEP_OBJECT" },
  },
  {
    ...availableBase,
    run_id: "run_wrong_thread",
    thread_id: "thread_SECRET_other",
  },
  {
    ...availableBase,
    run_id: "run_array_schema",
    projection_schema_version: ["SECRET_SCHEMA_ARRAY"],
  },
  {
    ...availableBase,
    run_id: "run_bad_timestamp",
    started_at: { secret: "SECRET_TIMESTAMP_OBJECT" },
  },
  {
    ...availableBase,
    run_id: "run_extra_field",
    forged_summary: "SECRET_EXTRA_SUMMARY_FIELD",
  },
];
const thread = createThreadPayload(
  "thread_quarantined_runs",
  [],
  null,
  { loopRuns, loopRunCount: loopRuns.length },
);
const recipes = [{
  recipe_id: "recipe_general_loop",
  name: "General assistant loop",
  goal: "Answer clearly.",
  is_default: true,
}];
const runDetailFetches = [];

globalThis.fetch = async (url, options = {}) => {
  const method = String(options.method || "GET").toUpperCase();
  if (url === "/api/status") {
    return jsonResponse({ backend: "mock", model: "MockLLM" });
  }
  if (url === "/api/recipes") {
    return jsonResponse({ default_recipe_id: "recipe_general_loop", recipes });
  }
  if (url.startsWith("/api/recipes/") && method === "GET") {
    return jsonResponse(recipes[0]);
  }
  if (url === "/api/threads" && method === "GET") {
    return jsonResponse({ threads: [thread] });
  }
  if (url === "/api/threads/thread_quarantined_runs" && method === "GET") {
    return jsonResponse(thread);
  }
  if (url.includes("/runs/")) {
    runDetailFetches.push(url);
    throw new Error("quarantined row must not request detail");
  }
  throw new Error(`unexpected fetch ${url}`);
};

await importFreshApp();
await tick();
for (const { run_id: runId } of loopRuns) {
  const row = findNode(
    dom["run-list"],
    (node) => node.className === "run-row" && nodeText(node).includes(runId),
  );
  assert.ok(row, `quarantined row should remain visible for ${runId}`);
  assert.equal(row.disabled, true);
  assert.equal(row.dataset.state, "quarantined");
  assert.equal(row.dataset.selected, "false");
  assert.equal(row.ariaPressed, "false");
  assert.equal(row.ariaBusy, "false");
  assert.ok(nodeText(row).includes("Quarantined · Stored loop run"));
  await row.dispatch("click");
}
assert.deepEqual(runDetailFetches, []);
const listText = nodeText(dom["run-list"]);
assert.ok(listText.includes("Stored report failed validation."));
assert.ok(listText.includes("Legacy visible answer lacks safe answer binding."));
assert.ok(listText.includes("Stored runs are quarantined and cannot be inspected."));
assert.equal(listText.includes("SECRET_"), false);
assert.equal(listText.includes(oversizedReason), false);
'''
    )


def test_static_frontend_durable_run_inspector_preserves_terminal_redaction():
    run_frontend_node(
        r'''
import assert from "node:assert/strict";
const {
  createDom,
  createThreadPayload,
  findNode,
  importFreshApp,
  jsonResponse,
  nodeText,
  tick,
} = await import(process.env.FRONTEND_HARNESS_URL);

const dom = createDom();
const recipes = [{
  recipe_id: "recipe_general_loop",
  name: "General assistant loop",
  goal: "Answer clearly.",
  is_default: true,
}];
const report = {
  schema_version: "loop-report/v1",
  projection_schema_version: "loop-public-report/v1",
  public: true,
  public_redaction: { applied: true, reason: "terminal_public_redaction" },
  run: {
    run_id: "run_blocked",
    session_id: "thread_blocked",
    context_provider: "document",
    conversation_context_count: null,
    semantic_memory_count: null,
    semantic_memory_status: null,
    backend: null,
    model_label: null,
    policy: {
      max_retries: 1,
      require_citations: true,
      require_verifier_for_supported: true,
      allow_mock_supported: false,
      allow_tool_calls: false,
      require_human_review_for_tools: true,
    },
    started_at: "2026-06-26T00:00:00.000Z",
    completed_at: "2026-06-26T00:00:01.000Z",
    steps: [{
      step_id: "step_blocked",
      phase: "verify",
      decision: "block",
      started_at: "2026-06-26T00:00:00.000Z",
      ended_at: "2026-06-26T00:00:01.000Z",
      duration_ms: 1000,
      backend: null,
      model_label: null,
      retry_count: 0,
      error_present: false,
      verification: null,
      human_review_required: false,
    }],
    final_decision: "block",
    terminal_reason: "blocked",
    final_answer: null,
    error_present: false,
    evidence: [],
  },
};
const storedLoopPayload = {
  trace: {
    question: "terminal secret trace question",
    answer: "terminal secret trace answer",
    citations: [{
      id: 1,
      source: "terminal-secret.txt",
      excerpt: "terminal secret trace citation",
    }],
    model_thinking: {
      available: true,
      redacted: false,
      content: "terminal secret trace thinking",
      label: "Model Thinking (unverified)",
      note: "Model thinking was redacted after a terminal decision.",
    },
    loop_report: report,
  },
  raw_report: {
    user_input: "terminal secret prompt",
    final_answer: "terminal secret answer",
    model_thinking: "terminal secret thinking",
  },
};
const blockedRun = {
  run_id: "run_blocked",
  thread_id: "thread_blocked",
  projection_status: "available",
  projection_schema_version: "loop-public-report/v1",
  final_decision: "block",
  terminal_reason: "blocked",
  context_provider: "document",
  backend: null,
  model: null,
  step_count: 1,
  started_at: "2026-06-26T00:00:00.000Z",
  completed_at: "2026-06-26T00:00:01.000Z",
  created_at: "2026-06-26T00:00:01.000Z",
};
const thread = createThreadPayload(
  "thread_blocked",
  [{ role: "assistant", content: "Request blocked.", loop_payload: storedLoopPayload }],
  null,
  { loopRuns: [blockedRun], loopRunCount: 1 },
);

globalThis.fetch = async (url, options = {}) => {
  const method = String(options.method || "GET").toUpperCase();
  if (url === "/api/status") {
    return jsonResponse({ backend: "mock", model: "MockLLM" });
  }
  if (url === "/api/recipes") {
    return jsonResponse({ default_recipe_id: "recipe_general_loop", recipes });
  }
  if (url.startsWith("/api/recipes/") && method === "GET") {
    return jsonResponse(recipes[0]);
  }
  if (url === "/api/threads" && method === "GET") {
    return jsonResponse({ threads: [thread] });
  }
  if (url === "/api/threads/thread_blocked" && method === "GET") {
    return jsonResponse(thread);
  }
  if (url === "/api/threads/thread_blocked/runs/run_blocked" && method === "GET") {
    return jsonResponse({
      public: true,
      thread_id: "thread_blocked",
      run_id: "run_blocked",
      report,
    });
  }
  throw new Error(`unexpected fetch ${url}`);
};

await importFreshApp();
await tick();
const row = findNode(
  dom["run-list"],
  (node) => node.className === "run-row" && nodeText(node).includes("run_blocked"),
);
await row.dispatch("click");
const traceText = dom["trace-json"].textContent;
const trace = JSON.parse(traceText);
assert.equal(trace.question, null);
assert.equal(trace.answer, "[redacted: terminal decision]");
assert.equal(trace.error, "terminal_public_redaction");
assert.deepEqual(trace.citations, []);
assert.equal(traceText.includes("terminal secret prompt"), false);
assert.equal(traceText.includes("terminal secret answer"), false);
assert.equal(traceText.includes("terminal secret thinking"), false);
assert.equal(traceText.includes("terminal secret trace question"), false);
assert.equal(traceText.includes("terminal secret trace answer"), false);
assert.equal(traceText.includes("terminal secret trace citation"), false);
assert.equal(traceText.includes("terminal secret trace thinking"), false);
assert.equal(dom["thinking-state"].textContent, "redacted");
assert.equal(dom["thinking-content"].textContent.includes("terminal secret"), false);
const summary = JSON.parse(dom["summary-json"].textContent);
assert.equal(summary.public_redaction.applied, true);
assert.equal(summary.final_decision, "block");
assert.equal(summary.last_error, "terminal_public_redaction");
'''
    )


def test_static_frontend_rejects_exact_shape_terminal_redaction_contradictions():
    run_frontend_node(
        r'''
import assert from "node:assert/strict";
const {
  createDom,
  createThreadPayload,
  findNode,
  importFreshApp,
  jsonResponse,
  nodeText,
  tick,
} = await import(process.env.FRONTEND_HARNESS_URL);

const dom = createDom();
const recipes = [{
  recipe_id: "recipe_general_loop",
  name: "General assistant loop",
  goal: "Answer clearly.",
  is_default: true,
}];
const baseReport = {
  schema_version: "loop-report/v1",
  projection_schema_version: "loop-public-report/v1",
  public: true,
  public_redaction: { applied: true, reason: "terminal_public_redaction" },
  run: {
    run_id: "run_base",
    session_id: "thread_redaction_contract",
    context_provider: "document",
    conversation_context_count: null,
    semantic_memory_count: null,
    semantic_memory_status: null,
    backend: null,
    model_label: null,
    policy: {
      max_retries: 1,
      require_citations: true,
      require_verifier_for_supported: true,
      allow_mock_supported: false,
      allow_tool_calls: false,
      require_human_review_for_tools: true,
    },
    started_at: "2026-06-26T00:00:00.000Z",
    completed_at: "2026-06-26T00:00:01.000Z",
    steps: [{
      step_id: "step_blocked",
      phase: "verify",
      decision: "block",
      started_at: "2026-06-26T00:00:00.000Z",
      ended_at: "2026-06-26T00:00:01.000Z",
      duration_ms: 1000,
      backend: null,
      model_label: null,
      retry_count: 0,
      error_present: false,
      verification: null,
      human_review_required: false,
    }],
    evidence: [],
    final_decision: "block",
    terminal_reason: "blocked",
    final_answer: null,
    error_present: false,
  },
};
const cloneReport = (runId) => {
  const report = JSON.parse(JSON.stringify(baseReport));
  report.run.run_id = runId;
  return report;
};
const variants = [
  ["run_secret_answer", (report) => {
    report.run.final_answer = "TERMINAL_SECRET_ANSWER";
  }],
  ["run_secret_evidence", (report) => {
    report.run.evidence = [{
      evidence_id: `evidence_${"a".repeat(64)}`,
      citation_id: 1,
      provider: "document",
      locator: { page: null, chunk_index: 0 },
    }];
  }],
  ["run_secret_provenance", (report) => {
    report.run.backend = "ollama";
    report.run.model_label = "TERMINAL_SECRET_MODEL";
  }],
  ["run_secret_step", (report) => {
    report.run.steps[0].backend = "ollama";
    report.run.steps[0].model_label = "TERMINAL_SECRET_STEP_MODEL";
    report.run.steps[0].verification = {
      outcome: "unsupported",
      verifier_backend: "ollama",
      verifier_model_label: "TERMINAL_SECRET_VERIFIER",
      same_model_as_drafter: true,
    };
  }],
  ["run_unredacted_block", (report) => {
    report.public_redaction = { applied: false, reason: null };
  }],
  ["run_partial_memory", (report) => {
    report.run.conversation_context_count = 0;
  }],
  ["run_false_memory_status", (report) => {
    report.run.conversation_context_count = 0;
    report.run.semantic_memory_count = 1;
    report.run.semantic_memory_status = "not_requested";
  }],
  ["run_empty_retrieved_memory", (report) => {
    report.run.conversation_context_count = 0;
    report.run.semantic_memory_count = 0;
    report.run.semantic_memory_status = "retrieved";
  }],
  ["run_conflicting_terminal", (report) => {
    report.run.steps[0].phase = "refuse";
    report.run.steps[0].decision = "block";
  }],
  ["run_bad_terminal_reason", (report) => {
    report.run.terminal_reason = "completed";
  }],
].map(([runId, mutate]) => {
  const report = cloneReport(runId);
  mutate(report);
  return { runId, report };
});
const loopRuns = variants.map(({ runId }) => ({
  run_id: runId,
  thread_id: "thread_redaction_contract",
  projection_status: "available",
  projection_schema_version: "loop-public-report/v1",
  final_decision: "block",
  terminal_reason: "blocked",
  context_provider: "document",
  backend: null,
  model: null,
  step_count: 1,
  started_at: "2026-06-26T00:00:00.000Z",
  completed_at: "2026-06-26T00:00:01.000Z",
  created_at: "2026-06-26T00:00:01.000Z",
}));
const latest = {
  timeline: { rows: [], final_decision: "not_verified" },
  summary: { marker: "latest remains visible" },
  trace: { marker: "latest trace remains visible" },
};
const thread = createThreadPayload(
  "thread_redaction_contract",
  [],
  latest,
  { loopRuns, loopRunCount: loopRuns.length },
);

globalThis.fetch = async (url, options = {}) => {
  const method = String(options.method || "GET").toUpperCase();
  if (url === "/api/status") {
    return jsonResponse({ backend: "mock", model: "MockLLM" });
  }
  if (url === "/api/recipes") {
    return jsonResponse({ default_recipe_id: "recipe_general_loop", recipes });
  }
  if (url.startsWith("/api/recipes/") && method === "GET") {
    return jsonResponse(recipes[0]);
  }
  if (url === "/api/threads" && method === "GET") {
    return jsonResponse({ threads: [thread] });
  }
  if (url === "/api/threads/thread_redaction_contract" && method === "GET") {
    return jsonResponse(thread);
  }
  const variant = variants.find(({ runId }) => url.endsWith(`/runs/${runId}`));
  if (variant && method === "GET") {
    return jsonResponse({
      public: true,
      thread_id: "thread_redaction_contract",
      run_id: variant.runId,
      report: variant.report,
    });
  }
  throw new Error(`unexpected fetch ${url}`);
};

await importFreshApp();
await tick();
for (const { runId } of variants) {
  const row = findNode(
    dom["run-list"],
    (node) => node.className === "run-row" && nodeText(node).includes(runId),
  );
  assert.ok(row, `stored row should exist for ${runId}`);
  await row.dispatch("click");
  const status = findNode(
    dom["run-list"],
    (node) => node.className === "run-inspector-status",
  );
  assert.equal(status.dataset.state, "error");
  assert.ok(status.textContent.includes("unavailable or malformed"));
  assert.equal(dom["summary-json"].textContent.includes("latest remains visible"), true);
  assert.equal(dom["trace-json"].textContent.includes("TERMINAL_SECRET"), false);
  const selectedRows = dom["run-list"].children.filter(
    (node) => node.className === "run-row" && node.dataset.selected === "true",
  );
  assert.deepEqual(selectedRows, []);
}
'''
    )


def test_static_frontend_durable_run_inspector_does_not_recover_refusal_siblings():
    run_frontend_node(
        r'''
import assert from "node:assert/strict";
const {
  createDom,
  createThreadPayload,
  findNode,
  importFreshApp,
  jsonResponse,
  nodeText,
  tick,
} = await import(process.env.FRONTEND_HARNESS_URL);

const dom = createDom();
const recipes = [{
  recipe_id: "recipe_general_loop",
  name: "General assistant loop",
  goal: "Answer clearly.",
  is_default: true,
}];
const report = {
  schema_version: "loop-report/v1",
  projection_schema_version: "loop-public-report/v1",
  public: true,
  public_redaction: { applied: true, reason: "terminal_public_redaction" },
  run: {
    run_id: "run_evidence_refusal",
    session_id: "thread_evidence_refusal",
    context_provider: "document",
    conversation_context_count: null,
    semantic_memory_count: null,
    semantic_memory_status: null,
    backend: null,
    model_label: null,
    policy: {
      max_retries: 1,
      require_citations: true,
      require_verifier_for_supported: true,
      allow_mock_supported: false,
      allow_tool_calls: false,
      require_human_review_for_tools: true,
    },
    started_at: "2026-06-26T00:00:00.000Z",
    completed_at: "2026-06-26T00:00:01.000Z",
    steps: [],
    final_decision: "refuse",
    terminal_reason: "verification_failed",
    final_answer: null,
    error_present: true,
    evidence: [],
  },
};
const publicQuestion = "What does the document prove?";
const publicAnswer = "I cannot verify that claim from the available evidence.";
const publicExcerpt = "Project Phoenix is still under review.";
const loopPayload = {
  trace: {
    question: publicQuestion,
    answer: publicAnswer,
    citations: [{
      id: 1,
      source: "phoenix.txt",
      chunk: 1,
      excerpt: publicExcerpt,
    }],
    self_check: {
      outcome: "unsupported",
      reasons: ["evidence_does_not_support_claim"],
      retry_attempted: true,
    },
    model_thinking: {
      available: true,
      redacted: false,
      content: "refusal thinking must not render",
    },
    loop_report: report,
  },
};
const run = {
  run_id: "run_evidence_refusal",
  thread_id: "thread_evidence_refusal",
  projection_status: "available",
  projection_schema_version: "loop-public-report/v1",
  final_decision: "refuse",
  terminal_reason: "verification_failed",
  context_provider: "document",
  backend: null,
  model: null,
  step_count: 0,
  started_at: "2026-06-26T00:00:00.000Z",
  completed_at: "2026-06-26T00:00:01.000Z",
  created_at: "2026-06-26T00:00:01.000Z",
};
const thread = createThreadPayload(
  "thread_evidence_refusal",
  [{ role: "assistant", content: publicAnswer, loop_payload: loopPayload }],
  null,
  { loopRuns: [run], loopRunCount: 1 },
);

globalThis.fetch = async (url, options = {}) => {
  const method = String(options.method || "GET").toUpperCase();
  if (url === "/api/status") {
    return jsonResponse({ backend: "mock", model: "MockLLM" });
  }
  if (url === "/api/recipes") {
    return jsonResponse({ default_recipe_id: "recipe_general_loop", recipes });
  }
  if (url.startsWith("/api/recipes/") && method === "GET") {
    return jsonResponse(recipes[0]);
  }
  if (url === "/api/threads" && method === "GET") {
    return jsonResponse({ threads: [thread] });
  }
  if (url === "/api/threads/thread_evidence_refusal" && method === "GET") {
    return jsonResponse(thread);
  }
  if (
    url === "/api/threads/thread_evidence_refusal/runs/run_evidence_refusal" &&
    method === "GET"
  ) {
    return jsonResponse({
      public: true,
      thread_id: "thread_evidence_refusal",
      run_id: "run_evidence_refusal",
      report,
    });
  }
  throw new Error(`unexpected fetch ${url}`);
};

await importFreshApp();
await tick();
const row = findNode(
  dom["run-list"],
  (node) => node.className === "run-row" && nodeText(node).includes("run_evidence_refusal"),
);
await row.dispatch("click");
const trace = JSON.parse(dom["trace-json"].textContent);
assert.equal(trace.question, null);
assert.equal(
  trace.answer,
  "I could not find enough relevant information in the provided evidence to answer that.",
);
assert.deepEqual(trace.citations, []);
assert.equal(trace.citation_details, "Citation details redacted for this terminal run.");
assert.equal(trace.model_thinking.redacted, true);
assert.equal(trace.model_thinking.content, null);
assert.equal(trace.error, null);
assert.equal(dom["trace-json"].textContent.includes(publicQuestion), false);
assert.equal(dom["trace-json"].textContent.includes(publicAnswer), false);
assert.equal(dom["trace-json"].textContent.includes(publicExcerpt), false);
assert.equal(dom["trace-json"].textContent.includes("refusal thinking must not render"), false);
'''
    )


def test_static_frontend_ignores_stale_durable_run_detail_after_thread_switch():
    run_frontend_node(
        r'''
import assert from "node:assert/strict";
const {
  createDom,
  createThreadPayload,
  deferred,
  findNode,
  importFreshApp,
  jsonResponse,
  nodeText,
  tick,
} = await import(process.env.FRONTEND_HARNESS_URL);

const dom = createDom();
const alphaDetail = deferred();
const recipes = [{
  recipe_id: "recipe_general_loop",
  name: "General assistant loop",
  goal: "Answer clearly.",
  is_default: true,
}];
const alphaRun = {
  run_id: "run_alpha",
  thread_id: "thread_alpha",
  projection_status: "available",
  projection_schema_version: "loop-public-report/v1",
  final_decision: "not_verified",
  terminal_reason: "not_verified",
  context_provider: "none",
  backend: "openai-compatible",
  model: "AlphaModel",
  step_count: 0,
  started_at: "2026-06-26T00:00:00.000Z",
  completed_at: "2026-06-26T00:00:01.000Z",
  created_at: "2026-06-26T00:00:01.000Z",
};
const alpha = createThreadPayload(
  "thread_alpha",
  [],
  {
    timeline: { rows: [], final_decision: "not_verified" },
    summary: { marker: "alpha latest" },
    trace: {},
  },
  { loopRuns: [alphaRun], loopRunCount: 1 },
);
alpha.title = "Alpha thread";
const beta = createThreadPayload(
  "thread_beta",
  [],
  {
    timeline: { rows: [], final_decision: "not_verified" },
    summary: { marker: "beta latest" },
    trace: {},
  },
  { loopRuns: [], loopRunCount: 0 },
);
beta.title = "Beta thread";

globalThis.fetch = async (url, options = {}) => {
  const method = String(options.method || "GET").toUpperCase();
  if (url === "/api/status") {
    return jsonResponse({ backend: "mock", model: "MockLLM" });
  }
  if (url === "/api/recipes") {
    return jsonResponse({ default_recipe_id: "recipe_general_loop", recipes });
  }
  if (url.startsWith("/api/recipes/") && method === "GET") {
    return jsonResponse(recipes[0]);
  }
  if (url === "/api/threads" && method === "GET") {
    return jsonResponse({ threads: [alpha, beta] });
  }
  if (url === "/api/threads/thread_alpha" && method === "GET") {
    return jsonResponse(alpha);
  }
  if (url === "/api/threads/thread_beta" && method === "GET") {
    return jsonResponse(beta);
  }
  if (url === "/api/threads/thread_alpha/runs/run_alpha" && method === "GET") {
    return alphaDetail.promise;
  }
  throw new Error(`unexpected fetch ${url}`);
};

await importFreshApp();
await tick();
const alphaRow = findNode(
  dom["run-list"],
  (node) => node.className === "run-row" && nodeText(node).includes("run_alpha"),
);
const pendingInspection = alphaRow.dispatch("click");
await tick();
const betaButton = dom["thread-list"].children.find((node) =>
  nodeText(node).includes("Beta thread"),
);
assert.ok(betaButton);
await betaButton.dispatch("click");
assert.ok(dom["summary-json"].textContent.includes("beta latest"));

alphaDetail.resolve(jsonResponse({
  public: true,
  thread_id: "thread_alpha",
  run_id: "run_alpha",
  report: {
    schema_version: "loop-report/v1",
    projection_schema_version: "loop-public-report/v1",
    run: {
      run_id: "run_alpha",
      session_id: "thread_alpha",
      context_provider: "none",
      backend: "openai-compatible",
      model_label: "AlphaModel",
      steps: [],
      evidence: [],
      final_decision: "not_verified",
      metadata: {},
    },
  },
}));
await pendingInspection;
assert.equal(dom["active-thread-title"].textContent, "Beta thread");
assert.ok(dom["summary-json"].textContent.includes("beta latest"));
assert.equal(dom["summary-json"].textContent.includes("run_alpha"), false);
assert.equal(nodeText(dom["run-list"]).includes("run_alpha"), false);
'''
    )


def test_static_frontend_thread_switch_commits_atomically_and_ignores_stale_results():
    run_frontend_node(
        r'''
import assert from "node:assert/strict";
const {
  createDom,
  createThreadPayload,
  deferred,
  errorResponse,
  importFreshApp,
  jsonResponse,
  nodeText,
  tick,
} = await import(process.env.FRONTEND_HARNESS_URL);

const dom = createDom();
const recipe = {
  recipe_id: "recipe_general_loop",
  name: "General assistant loop",
  goal: "Answer clearly.",
  context_provider: "smart",
  model_profile: "quality",
  verifier: "default",
  instructions: "",
  success_criteria: [],
  stop_condition: "",
  is_default: true,
};
const emptyLatest = (marker) => ({
  timeline: { rows: [], final_decision: null, last_error: null },
  summary: { marker },
  trace: {},
});
const alpha = createThreadPayload(
  "thread_alpha",
  [{ role: "user", content: "alpha private context" }],
  emptyLatest("alpha latest"),
);
alpha.title = "Alpha";
const beta = createThreadPayload(
  "thread_beta",
  [{ role: "user", content: "beta private context" }],
  emptyLatest("beta latest"),
);
beta.title = "Beta";
const gamma = createThreadPayload(
  "thread_gamma",
  [{ role: "user", content: "gamma private context" }],
  emptyLatest("gamma latest"),
);
gamma.title = "Gamma";
const detailWithTitle = (thread, title) => {
  const detail = JSON.parse(JSON.stringify(thread));
  detail.title = title;
  return detail;
};
const betaSuccess = deferred();
const gammaFailure = deferred();
const gammaStale = deferred();
const alphaFresh = deferred();
const alphaAfterStatus = deferred();
const betaStale = deferred();
const betaFresh = deferred();
const staleBetaStatus = deferred();
let alphaGetCount = 0;
let betaGetCount = 0;
let gammaGetCount = 0;
let deferNextBetaStatus = false;
const querySessions = [];
const uploadSessions = [];
const clearSessions = [];
const deleteUrls = [];
const statusSessions = [];
const confirmations = [];

globalThis.confirm = (message) => {
  confirmations.push(message);
  return true;
};
globalThis.fetch = async (url, options = {}) => {
  const method = String(options.method || "GET").toUpperCase();
  if (url === "/api/recipes") {
    return jsonResponse({ default_recipe_id: recipe.recipe_id, recipes: [recipe] });
  }
  if (url === "/api/recipes/recipe_general_loop") {
    return jsonResponse(recipe);
  }
  if (url === "/api/threads" && method === "GET") {
    return jsonResponse({ threads: [alpha, beta, gamma] });
  }
  if (url === "/api/threads/thread_alpha" && method === "GET") {
    alphaGetCount += 1;
    if (alphaGetCount <= 2) {
      return jsonResponse(alpha);
    }
    return [alphaFresh.promise, alphaAfterStatus.promise][alphaGetCount - 3];
  }
  if (url === "/api/threads/thread_beta" && method === "GET") {
    betaGetCount += 1;
    return [
      betaSuccess.promise,
      betaStale.promise,
      betaFresh.promise,
    ][betaGetCount - 1];
  }
  if (url === "/api/threads/thread_gamma" && method === "GET") {
    gammaGetCount += 1;
    return [gammaFailure.promise, gammaStale.promise][gammaGetCount - 1];
  }
  if (url === "/api/status") {
    const statusSession = options.headers?.["x-ai-loop-session-id"] || null;
    statusSessions.push(statusSession);
    if (deferNextBetaStatus && statusSession === "thread_beta") {
      deferNextBetaStatus = false;
      return staleBetaStatus.promise;
    }
    return jsonResponse({ backend: "mock", model: "MockLLM" });
  }
  if (url === "/api/query" && method === "POST") {
    querySessions.push(JSON.parse(options.body).session_id);
    return errorResponse(503, { detail: "query captured" });
  }
  if (url === "/api/documents" && method === "POST") {
    uploadSessions.push(options.body.get("session_id"));
    return errorResponse(503, { detail: "upload captured" });
  }
  if (url === "/api/chat/clear" && method === "POST") {
    clearSessions.push(JSON.parse(options.body).session_id);
    return jsonResponse({
      timeline: { rows: [], final_decision: null, last_error: null },
      summary: {},
      trace: {},
    });
  }
  if (url === "/api/threads/thread_alpha" && method === "DELETE") {
    deleteUrls.push(url);
    return errorResponse(503, { detail: "delete captured" });
  }
  throw new Error(`unexpected fetch ${method} ${url}`);
};

await importFreshApp();
await tick();
querySessions.length = 0;
uploadSessions.length = 0;
clearSessions.length = 0;
deleteUrls.length = 0;
statusSessions.length = 0;
confirmations.length = 0;

const threadButton = (title) => {
  const button = dom["thread-list"].children.find((node) =>
    nodeText(node).includes(title),
  );
  assert.ok(button, `thread button should exist for ${title}`);
  return button;
};

await threadButton("Alpha").dispatch("click");
assert.equal(alphaGetCount, 1, "the selected thread must not start a stale refresh");

const pendingBetaSwitch = threadButton("Beta").dispatch("click");
await tick();
assert.equal(dom["active-thread-title"].textContent, "Alpha");
assert.equal(
  globalThis.localStorage.getItem("loopwright.active-thread.v1"),
  "thread_alpha",
);
assert.equal(dom["query-button"].disabled, false);
assert.equal(dom["clear-chat"].disabled, false);
assert.equal(dom["upload-button"].disabled, false);
assert.equal(dom["delete-thread"].disabled, false);

dom["query-input"].value = "Keep this query in Alpha";
await dom["query-form"].dispatch("submit");
dom["document-file"].files = [new Blob(["alpha"], { type: "text/plain" })];
await dom["document-file"].dispatch("change");
await dom["clear-chat"].dispatch("click");
await dom["delete-thread"].dispatch("click");
assert.deepEqual(querySessions, ["thread_alpha"]);
assert.deepEqual(uploadSessions, ["thread_alpha"]);
assert.deepEqual(clearSessions, ["thread_alpha"]);
assert.deepEqual(deleteUrls, ["/api/threads/thread_alpha"]);
assert.ok(confirmations[0].includes('Delete thread "Alpha"?'));
assert.ok(statusSessions.every((sessionId) => sessionId === "thread_alpha"));

betaSuccess.resolve(jsonResponse(detailWithTitle(beta, "Beta loaded")));
await pendingBetaSwitch;
assert.equal(dom["active-thread-title"].textContent, "Beta loaded");
assert.equal(
  globalThis.localStorage.getItem("loopwright.active-thread.v1"),
  "thread_beta",
);

const failingGammaSwitch = threadButton("Gamma").dispatch("click");
await tick();
assert.equal(dom["active-thread-title"].textContent, "Beta loaded");
gammaFailure.resolve(errorResponse(503, { detail: "gamma detail unavailable" }));
await failingGammaSwitch;
assert.equal(dom["active-thread-title"].textContent, "Beta loaded");
assert.equal(
  globalThis.localStorage.getItem("loopwright.active-thread.v1"),
  "thread_beta",
);
assert.equal(dom["upload-status"].textContent, "gamma detail unavailable");

const staleGammaSwitch = threadButton("Gamma").dispatch("click");
const freshAlphaSwitch = threadButton("Alpha").dispatch("click");
alphaFresh.resolve(jsonResponse(detailWithTitle(alpha, "Alpha fresh")));
await freshAlphaSwitch;
assert.equal(dom["active-thread-title"].textContent, "Alpha fresh");
dom["upload-status"].textContent = "newer selection remains authoritative";
gammaStale.resolve(errorResponse(503, { detail: "stale gamma failure" }));
await staleGammaSwitch;
assert.equal(dom["active-thread-title"].textContent, "Alpha fresh");
assert.equal(
  dom["upload-status"].textContent,
  "newer selection remains authoritative",
);

const staleBetaSwitch = threadButton("Beta").dispatch("click");
const freshBetaSwitch = threadButton("Beta").dispatch("click");
deferNextBetaStatus = true;
betaFresh.resolve(jsonResponse(detailWithTitle(beta, "Beta fresh")));
await tick();
assert.equal(dom["active-thread-title"].textContent, "Beta fresh");
dom["upload-status"].textContent = "fresh detail remains authoritative";
betaStale.resolve(jsonResponse(detailWithTitle(beta, "Beta stale")));
await staleBetaSwitch;
assert.equal(dom["active-thread-title"].textContent, "Beta fresh");
assert.equal(
  dom["upload-status"].textContent,
  "fresh detail remains authoritative",
);

const alphaAfterStatusSwitch = threadButton("Alpha").dispatch("click");
alphaAfterStatus.resolve(jsonResponse(detailWithTitle(alpha, "Alpha after status")));
await alphaAfterStatusSwitch;
assert.equal(dom["active-thread-title"].textContent, "Alpha after status");
dom["upload-status"].textContent = "newer status remains authoritative";
staleBetaStatus.resolve(errorResponse(503, { detail: "stale beta status failure" }));
await freshBetaSwitch;
assert.equal(dom["active-thread-title"].textContent, "Alpha after status");
assert.equal(
  dom["upload-status"].textContent,
  "newer status remains authoritative",
);
'''
    )


def test_static_frontend_uploads_files_to_active_thread():
    run_frontend_node(
        r'''
import assert from "node:assert/strict";
const {
  createDom,
  createThreadPayload,
  importFreshApp,
  jsonResponse,
  tick,
} = await import(process.env.FRONTEND_HARNESS_URL);

const dom = createDom();
const statusSessionIds = [];
const uploadSessionIds = [];
const uploadEncodingModes = [];
const recipes = [{
  recipe_id: "recipe_general_loop",
  name: "General assistant loop",
  goal: "Answer the request clearly.",
  context_provider: "smart",
  model_profile: "quality",
  verifier: "default",
  instructions: "",
  success_criteria: [],
  stop_condition: "",
  is_default: true,
}];
const serverThreads = [createThreadPayload("thread_alpha")];

globalThis.fetch = async (url, options = {}) => {
  const method = String(options.method || "GET").toUpperCase();
  if (url === "/api/config") {
    return jsonResponse({ text_encodings: [{ label: "Auto", value: "auto" }] });
  }
  if (url === "/api/status") {
    statusSessionIds.push(options.headers?.["x-ai-loop-session-id"] || null);
    return jsonResponse({
      backend: "mock",
      model: "MockLLM",
      ready_for_queries: false,
      query_mode: "direct",
      chunk_count: 0,
    });
  }
  if (url === "/api/recipes") {
    return jsonResponse({ default_recipe_id: "recipe_general_loop", recipes });
  }
  if (url.startsWith("/api/recipes/") && method === "GET") {
    return jsonResponse(recipes[0]);
  }
  if (url === "/api/threads" && method === "GET") {
    return jsonResponse({ threads: serverThreads });
  }
  if (url.startsWith("/api/threads/") && method === "GET") {
    return jsonResponse(serverThreads[0]);
  }
  if (url === "/api/documents" && method === "POST") {
    uploadSessionIds.push(options.body.get("session_id"));
    uploadEncodingModes.push(options.body.get("text_encoding"));
    return jsonResponse({
      message: "File indexed.",
      status: {
        backend: "mock",
        model: "MockLLM",
        ready_for_queries: true,
        active_document: "alpha.txt",
        query_mode: "retrieval",
        chunk_count: 1,
      },
    });
  }
  throw new Error(`unexpected fetch ${url}`);
};

await importFreshApp();
await tick();

dom["document-file"].files = [
  new Blob(["Project Phoenix"], { type: "text/plain" }),
];
await dom["document-file"].dispatch("change");

assert.deepEqual(statusSessionIds, ["thread_alpha"]);
assert.deepEqual(uploadSessionIds, ["thread_alpha"]);
assert.deepEqual(uploadEncodingModes, ["auto"]);
assert.equal(dom["backend-pill"].textContent, "mock");
assert.equal(dom["model-pill"].textContent, "MockLLM");
assert.equal(dom["ready-pill"].textContent, "file attached");
assert.equal(dom["file-scope"].textContent, "alpha.txt");
assert.equal(
  dom["upload-status"].textContent,
  "alpha.txt is attached to this session.",
);
'''
    )


def test_static_frontend_ignores_stale_upload_response_after_thread_switch():
    run_frontend_node(
        r'''
import assert from "node:assert/strict";
const {
  createDom,
  createThreadPayload,
  deferred,
  importFreshApp,
  jsonResponse,
  tick,
} = await import(process.env.FRONTEND_HARNESS_URL);

const dom = createDom();
const uploadResponse = deferred();
const statusSessionIds = [];
const uploadSessionIds = [];
const recipes = [{
  recipe_id: "recipe_general_loop",
  name: "General assistant loop",
  goal: "Answer the request clearly.",
  context_provider: "smart",
  model_profile: "quality",
  verifier: "default",
  instructions: "",
  success_criteria: [],
  stop_condition: "",
  is_default: true,
}];
const serverThreads = [
  createThreadPayload("thread_alpha"),
  createThreadPayload("thread_beta"),
];
serverThreads[0].title = "Alpha thread";
serverThreads[1].title = "Beta thread";

globalThis.fetch = async (url, options = {}) => {
  const method = String(options.method || "GET").toUpperCase();
  if (url === "/api/config") {
    return jsonResponse({ text_encodings: [{ label: "Auto", value: "auto" }] });
  }
  if (url === "/api/status") {
    const sessionId = options.headers?.["x-ai-loop-session-id"] || null;
    statusSessionIds.push(sessionId);
    return jsonResponse({
      backend: "mock",
      model: "MockLLM",
      ready_for_queries: false,
      active_document: null,
      query_mode: "direct",
      chunk_count: 0,
    });
  }
  if (url === "/api/recipes") {
    return jsonResponse({ default_recipe_id: "recipe_general_loop", recipes });
  }
  if (url.startsWith("/api/recipes/") && method === "GET") {
    return jsonResponse(recipes[0]);
  }
  if (url === "/api/threads" && method === "GET") {
    return jsonResponse({ threads: serverThreads });
  }
  if (url.startsWith("/api/threads/") && method === "GET") {
    const id = decodeURIComponent(url.slice("/api/threads/".length));
    return jsonResponse(serverThreads.find((thread) => thread.id === id));
  }
  if (url === "/api/documents" && method === "POST") {
    uploadSessionIds.push(options.body.get("session_id"));
    return uploadResponse.promise;
  }
  throw new Error(`unexpected fetch ${url}`);
};

await importFreshApp();
await tick();

dom["document-file"].files = [
  new Blob(["Project Phoenix"], { type: "text/plain" }),
];
const uploadPromise = dom["document-file"].dispatch("change");
await tick();

const betaButton = dom["thread-list"].children.find(
  (button) => button.dataset.active === "false",
);
await betaButton.click();
await tick();

uploadResponse.resolve(jsonResponse({
  message: "Alpha indexed.",
  status: {
    backend: "mock",
    model: "MockLLM",
    ready_for_queries: true,
    active_document: "alpha.txt",
    query_mode: "retrieval",
    chunk_count: 1,
  },
}));
await uploadPromise;
await tick();

assert.deepEqual(uploadSessionIds, ["thread_alpha"]);
assert.equal(statusSessionIds.at(-1), "thread_beta");
assert.equal(dom["file-scope"].textContent, "No file attached");
assert.equal(
  dom["upload-status"].textContent,
  "Active session changed; current file status refreshed.",
);
'''
    )


def test_static_frontend_renders_assistant_content_and_model_thinking():
    run_frontend_node(
        r'''
import assert from "node:assert/strict";
const {
  createDom,
  createThreadPayload,
  findNode,
  importFreshApp,
  jsonResponse,
  nodeText,
  tick,
} = await import(process.env.FRONTEND_HARNESS_URL);

const dom = createDom();
const queryBodies = [];
const serverThreads = [
  createThreadPayload("thread_initial"),
  createThreadPayload("thread_with_runs", [], null, {
    includeRuns: false,
    loopRunCount: 2,
  }),
];
const recipes = [{
  recipe_id: "recipe_general_loop",
  name: "General assistant loop",
  goal: "Answer the request clearly.",
  context_provider: "smart",
  model_profile: "quality",
  verifier: "default",
  instructions: "Use same-thread memory carefully.",
  success_criteria: ["Answer the request clearly."],
  stop_condition: "Stop when done.",
  is_default: true,
}];

function collectNodes(root, predicate, matches = []) {
  for (const child of root.children) {
    if (predicate(child)) {
      matches.push(child);
    }
    collectNodes(child, predicate, matches);
  }
  return matches;
}

globalThis.fetch = async (url, options = {}) => {
  const method = String(options.method || "GET").toUpperCase();
  if (url === "/api/config") {
    return jsonResponse({ text_encodings: [{ label: "Auto", value: "auto" }] });
  }
  if (url === "/api/status") {
    return jsonResponse({
      backend: "ollama",
      model: "thinking-model",
      ready_for_queries: false,
      query_mode: "direct",
      chunk_count: 0,
    });
  }
  if (url === "/api/recipes") {
    return jsonResponse({ default_recipe_id: "recipe_general_loop", recipes });
  }
  if (url.startsWith("/api/recipes/") && method === "GET") {
    return jsonResponse(recipes[0]);
  }
  if (url === "/api/threads" && method === "GET") {
    return jsonResponse({ threads: serverThreads });
  }
  if (url === "/api/threads/thread_initial/runs/run_frontend" && method === "GET") {
    return jsonResponse({
      public: true,
      thread_id: "thread_initial",
      run_id: "run_frontend",
        report: {
          schema_version: "loop-report/v1",
          projection_schema_version: "loop-public-report/v1",
          public: true,
          public_redaction: { applied: false, reason: null },
          run: {
            run_id: "run_frontend",
            session_id: "thread_initial",
            context_provider: "none",
            conversation_context_count: 1,
            semantic_memory_count: 2,
            semantic_memory_status: "retrieved",
            backend: "mock",
            model_label: "MockLLM",
            policy: {
              max_retries: 1,
              require_citations: true,
              require_verifier_for_supported: true,
              allow_mock_supported: false,
              allow_tool_calls: false,
              require_human_review_for_tools: true,
            },
            started_at: "2026-06-26T00:00:00.000Z",
            completed_at: "2026-06-26T00:00:01.000Z",
            steps: [],
            evidence: [],
            final_decision: "not_verified",
            terminal_reason: "not_verified",
            final_answer: "Stored answer.",
            error_present: false,
          },
        },
    });
  }
  if (url.startsWith("/api/threads/") && method === "GET") {
    const id = decodeURIComponent(url.slice("/api/threads/".length));
    const thread = serverThreads.find((item) => item.id === id);
    if (thread) {
      return jsonResponse(thread);
    }
  }
  if (url === "/api/query") {
    const request = JSON.parse(options.body);
    queryBodies.push(request);
    return jsonResponse({
      answer: [
        "**Step-by-step detail** 1. **Trigger:** Pressure builds. 2. **Outcome:** The plan escalates.",
        "",
        "Use Java 8. It introduced lambdas.",
        "",
        "1. First do X. 2. Then do Y.",
        "",
        "1. install dependencies. 2. run tests.",
        "",
        "Here is `dp[0]` safely:",
        "Keep `step 1. Start 2. Stop` inline.",
        "```java",
        "public class FibonacciDP {",
        "  public static int fib(int n) { return n; }",
        "}",
        "```",
        "- uses memoization",
        "<img src=x onerror=alert(1)>",
      ].join("\n"),
      run: {
        run_id: "run_frontend",
        thread_id: "thread_initial",
        projection_status: "available",
        projection_schema_version: "loop-public-report/v1",
        final_decision: "not_verified",
        terminal_reason: "not_verified",
        context_provider: "none",
        backend: "mock",
        model: "MockLLM",
        step_count: 4,
        started_at: "2026-06-26T00:00:00.000Z",
        completed_at: "2026-06-26T00:00:01.000Z",
        created_at: "2026-06-26T00:00:00.000Z",
      },
      timeline: { rows: [], final_decision: "not_verified" },
      summary: {
        conversation_context_count: 1,
        semantic_memory_count: 2,
        semantic_memory_status: "retrieved",
      },
      trace: {
        model_thinking: {
          available: true,
          redacted: false,
          label: "Model Thinking (unverified)",
          content: "Captured thinking from model.",
          note: "Model-emitted thinking is useful for debugging the loop.",
        },
      },
      thread: createThreadPayload(request.session_id, [], null, {
        memoryCount: 4,
        loopRunCount: 1,
      }),
    });
  }
  if (url === "/api/chat/clear") {
    return jsonResponse({ timeline: { rows: [], final_decision: null }, summary: {}, trace: {} });
  }
  throw new Error(`unexpected fetch ${url}`);
};

await importFreshApp();
await tick();
dom["query-context"].value = "web";
dom["query-input"].value = "What happened?";
await dom["query-form"].dispatch("submit");

assert.equal(queryBodies.length, 1);
assert.ok(queryBodies[0].session_id.startsWith("thread_"));
assert.equal(queryBodies[0].recipe_id, "recipe_general_loop");
assert.equal(queryBodies[0].context_provider, "web");
const assistantMessage = findNode(
  dom.messages,
  (node) => node.className === "message assistant",
);
assert.ok(assistantMessage, "assistant message should render");
const messageContent = findNode(
  assistantMessage,
  (node) => node.className === "message-content",
);
assert.ok(messageContent, "assistant message should include rich content");
const codeBlock = findNode(
  messageContent,
  (node) => node.className === "message-code-block",
);
assert.ok(codeBlock, "assistant markdown fence should become a code block");
const language = findNode(codeBlock, (node) => node.className === "message-code-language");
assert.equal(language.textContent, "java");
const answerCode = findNode(codeBlock, (node) => node.tagName === "CODE");
assert.ok(answerCode.textContent.includes("public class FibonacciDP"));
const inlineCode = findNode(
  messageContent,
  (node) => node.className === "message-inline-code",
);
assert.equal(inlineCode.textContent, "dp[0]");
const compactInlineCode = findNode(
  messageContent,
  (node) => node.tagName === "CODE" && node.textContent === "step 1. Start 2. Stop",
);
assert.ok(compactInlineCode, "compact numbered inline code should stay inline");
const boldText = findNode(messageContent, (node) => node.tagName === "STRONG");
assert.ok(boldText, "assistant markdown bold should become strong text");
assert.equal(boldText.textContent, "Step-by-step detail");
const orderedLists = collectNodes(messageContent, (node) => node.tagName === "OL");
assert.ok(orderedLists.length >= 2, "compact ordered markdown should become ordered lists");
const orderedText = orderedLists.map((node) => nodeText(node)).join(" ").replace(/\s+/g, " ");
assert.ok(orderedText.includes("Trigger: Pressure builds."));
assert.ok(orderedText.includes("Outcome: The plan escalates."));
assert.ok(orderedText.includes("First do X."));
assert.ok(orderedText.includes("Then do Y."));
assert.ok(orderedText.includes("install dependencies."));
assert.ok(orderedText.includes("run tests."));
assert.equal(
  orderedText.includes("It introduced lambdas."),
  false,
  "ordinary version prose should not become an ordered-list item",
);
const messageText = nodeText(messageContent).replace(/\s+/g, " ");
assert.ok(messageText.includes("Use Java 8. It introduced lambdas."));
assert.equal(
  nodeText(messageContent).includes("**"),
  false,
  "rendered markdown markers should not leak into normal text",
);
const list = findNode(messageContent, (node) => node.tagName === "UL");
assert.ok(list, "assistant markdown list should render as a list");
const image = findNode(messageContent, (node) => node.tagName === "IMG");
assert.equal(image, null, "model HTML must stay inert text");
const thinking = findNode(dom.messages, (node) => node.className === "message-thinking");
assert.ok(thinking, "assistant message should include thinking details");
const thinkingPre = findNode(thinking, (node) => node.tagName === "PRE");
assert.equal(thinkingPre.textContent, "Captured thinking from model.");
const run = findNode(dom["run-list"], (node) => node.className === "run-row");
assert.ok(run, "durable run summary should render after query");
assert.equal(run.disabled, false, "newly completed runs should be selectable immediately");
assert.ok(
  nodeText(dom["thread-list"]).includes("2 runs"),
  "thread summaries should preserve backend loop_run_count before details load",
);
assert.ok(
  nodeText(dom["thread-list"]).includes("0 memories"),
  "thread summaries should show indexed memory counts",
);
assert.ok(
  nodeText(dom["thread-list"]).includes("4 memories"),
  "thread summaries should update indexed memory counts after query",
);
assert.ok(
  nodeText(dom["active-thread-memory"]).includes("4 memories indexed"),
  "active thread header should show indexed memory count",
);
assert.ok(
  nodeText(dom["active-thread-memory"]).includes(
    "last run used 1 recent turn + 2 recalled memories",
  ),
  "active thread header should show last-run memory usage",
);
assert.ok(
  nodeText(dom["memory-status"]).includes("4 memories indexed"),
  "loop panel should show indexed memory count",
);
assert.ok(
  nodeText(dom["memory-status"]).includes(
    "last run used 1 recent turn + 2 recalled memories",
  ),
  "loop panel should show last-run memory usage",
);
await run.dispatch("click");
const runStatus = findNode(
  dom["run-list"],
  (node) => node.className === "run-inspector-status",
);
assert.equal(runStatus.dataset.state, "loaded");
assert.ok(runStatus.textContent.includes("run_frontend"));
await dom["clear-chat"].dispatch("click");
assert.ok(
  nodeText(dom["active-thread-memory"]).includes("0 memories indexed"),
  "clear should reset visible indexed memory count",
);
assert.ok(
  nodeText(dom["memory-status"]).includes("no completed run yet"),
  "clear should reset last-run memory status",
);
'''
    )


def test_static_frontend_omits_context_provider_for_automatic_recipe_default():
    run_frontend_node(
        r'''
import assert from "node:assert/strict";
const {
  createDom,
  createThreadPayload,
  importFreshApp,
  jsonResponse,
  tick,
} = await import(process.env.FRONTEND_HARNESS_URL);

const dom = createDom();
const queryBodies = [];
const recipes = [{
  recipe_id: "recipe_general_loop",
  name: "Web recipe",
  goal: "Use web evidence by default.",
  context_provider: "web",
  model_profile: "quality",
  verifier: "default",
  instructions: "Search when evidence is needed.",
  success_criteria: ["Uses the recipe context provider."],
  stop_condition: "Stop when answered.",
  is_default: true,
}];

globalThis.fetch = async (url, options = {}) => {
  const method = String(options.method || "GET").toUpperCase();
  if (url === "/api/config") {
    return jsonResponse({ title: "Loopwright" });
  }
  if (url === "/api/status") {
    return jsonResponse({
      backend: "mock",
      model: "MockLLM",
      ready_for_queries: false,
      query_mode: "direct",
      profile: "FAST",
    });
  }
  if (url === "/api/threads") {
    return jsonResponse({ threads: [createThreadPayload("thread_initial")] });
  }
  if (url === "/api/threads/thread_initial" && method === "GET") {
    return jsonResponse(createThreadPayload("thread_initial"));
  }
  if (url === "/api/recipes") {
    return jsonResponse({ default_recipe_id: "recipe_general_loop", recipes });
  }
  if (url.startsWith("/api/recipes/") && method === "GET") {
    return jsonResponse(recipes[0]);
  }
  if (url === "/api/query" && method === "POST") {
    const request = JSON.parse(options.body);
    queryBodies.push(request);
    return jsonResponse({
      answer: "Using recipe context.",
      summary: { context_provider: "web" },
      timeline: { rows: [], final_decision: "not_verified" },
      trace: {},
      thread: createThreadPayload(request.session_id),
    });
  }
  throw new Error(`unexpected fetch ${url}`);
};

await importFreshApp();
await tick();
assert.equal(dom["query-context"].value, "");
dom["query-input"].value = "What changed?";
await dom["query-form"].dispatch("submit");

assert.equal(queryBodies.length, 1);
assert.equal(queryBodies[0].recipe_id, "recipe_general_loop");
assert.equal(Object.hasOwn(queryBodies[0], "context_provider"), false);
'''
    )


def test_static_frontend_renders_inline_fences_and_hides_empty_thinking():
    run_frontend_node(
        r'''
import assert from "node:assert/strict";
const {
  createDom,
  createThreadPayload,
  findNode,
  importFreshApp,
  jsonResponse,
  nodeText,
  tick,
} = await import(process.env.FRONTEND_HARNESS_URL);

const dom = createDom();
const recipes = [{
  recipe_id: "recipe_general_loop",
  name: "General assistant loop",
  goal: "Answer the request clearly.",
  instructions: "Use same-thread memory carefully.",
  success_criteria: ["Answer the request clearly."],
  stop_condition: "Stop when done.",
  is_default: true,
}];
globalThis.fetch = async (url, options = {}) => {
  const method = String(options.method || "GET").toUpperCase();
  if (url === "/api/config") {
    return jsonResponse({ text_encodings: [{ label: "Auto", value: "auto" }] });
  }
  if (url === "/api/status") {
    return jsonResponse({
      backend: "ollama",
      model: "thinking-model",
      ready_for_queries: false,
      query_mode: "direct",
      chunk_count: 0,
    });
  }
  if (url === "/api/recipes") {
    return jsonResponse({ default_recipe_id: "recipe_general_loop", recipes });
  }
  if (url.startsWith("/api/recipes/") && method === "GET") {
    return jsonResponse(recipes[0]);
  }
  if (url === "/api/threads" && method === "GET") {
    return jsonResponse({ threads: [createThreadPayload("thread_inline")] });
  }
  if (url.startsWith("/api/threads/") && method === "GET") {
    return jsonResponse(createThreadPayload("thread_inline"));
  }
  if (url === "/api/query") {
    return jsonResponse({
      answer: "Inline fence: ```java public class InlineFence {}``` done.",
      timeline: { rows: [], final_decision: "not_verified" },
      summary: {},
      trace: {
        model_thinking: {
          available: false,
          redacted: false,
          label: "Model Thinking (unverified)",
          content: null,
          note: "Model-emitted thinking is useful for debugging the loop.",
        },
      },
    });
  }
  if (url === "/api/chat/clear") {
    return jsonResponse({ timeline: { rows: [], final_decision: null }, summary: {}, trace: {} });
  }
  throw new Error(`unexpected fetch ${url}`);
};

await importFreshApp();
await tick();
dom["query-input"].value = "What happened?";
await dom["query-form"].dispatch("submit");
const inlineAssistant = findNode(dom.messages, (node) => node.className === "message assistant");
const inlineCodeBlock = findNode(inlineAssistant, (node) => node.className === "message-code-block");
assert.ok(inlineCodeBlock, "inline fenced code should become a code block");
const inlineLanguage = findNode(inlineCodeBlock, (node) => node.className === "message-code-language");
assert.equal(inlineLanguage.textContent, "java");
const inlineCode = findNode(inlineCodeBlock, (node) => node.tagName === "CODE");
assert.equal(inlineCode.textContent, "public class InlineFence {}");
const emptyThinking = findNode(dom.messages, (node) => node.className === "message-thinking");
assert.equal(emptyThinking, null);
'''
    )


def test_static_frontend_threads_and_stale_response_guard():
    run_frontend_node(
        r'''
import assert from "node:assert/strict";
const {
  createDom,
  createThreadPayload,
  deferred,
  errorResponse,
  findNode,
  importFreshApp,
  jsonResponse,
  nodeText,
  tick,
} = await import(process.env.FRONTEND_HARNESS_URL);

const recipes = [{
  recipe_id: "recipe_general_loop",
  name: "General assistant loop",
  goal: "Answer the request clearly.",
  instructions: "Use same-thread memory carefully.",
  success_criteria: ["Answer the request clearly."],
  stop_condition: "Stop when done.",
  is_default: true,
}];

async function clickThreadByTitle(dom, title) {
  const button = dom["thread-list"].children.find((node) =>
    nodeText(node).includes(title),
  );
  assert.ok(button, `thread button should exist for ${title}`);
  await button.dispatch("click");
  await tick();
}

const switchDom = createDom();
const queryBodies = [];
const switchQuery = deferred();
const serverThreads = [
  createThreadPayload("thread_initial"),
  createThreadPayload("thread_other"),
];
globalThis.fetch = async (url, options = {}) => {
  const method = String(options.method || "GET").toUpperCase();
  if (url === "/api/config") {
    return jsonResponse({ text_encodings: [{ label: "Auto", value: "auto" }] });
  }
  if (url === "/api/status") {
    return jsonResponse({ backend: "ollama", model: "thinking-model", ready_for_queries: false, query_mode: "direct", chunk_count: 0 });
  }
  if (url === "/api/recipes") {
    return jsonResponse({ default_recipe_id: "recipe_general_loop", recipes });
  }
  if (url.startsWith("/api/recipes/") && method === "GET") {
    return jsonResponse(recipes[0]);
  }
  if (url === "/api/threads" && method === "GET") {
    return jsonResponse({ threads: serverThreads });
  }
  if (url === "/api/threads" && method === "POST") {
    const thread = createThreadPayload(`thread_created_${serverThreads.length}`);
    serverThreads.unshift(thread);
    return jsonResponse(thread);
  }
  if (url.startsWith("/api/threads/") && method === "GET") {
    const id = decodeURIComponent(url.slice("/api/threads/".length));
    return jsonResponse(serverThreads.find((item) => item.id === id) || serverThreads[0]);
  }
  if (url === "/api/query") {
    queryBodies.push(JSON.parse(options.body));
    if (queryBodies.length === 2) {
      await switchQuery.promise;
    }
    return jsonResponse({ answer: "Loop answer", timeline: { rows: [], final_decision: "not_verified" }, summary: {}, trace: { model_thinking: null } });
  }
  if (url === "/api/chat/clear") {
    return jsonResponse({ timeline: { rows: [], final_decision: null }, summary: {}, trace: {} });
  }
  throw new Error(`unexpected fetch ${url}`);
};
await importFreshApp();
await tick();
switchDom["query-input"].value = "First thread question";
await switchDom["query-form"].dispatch("submit");
const firstThreadId = queryBodies[0].session_id;
await switchDom["new-thread"].dispatch("click");
switchDom["query-input"].value = "Second thread question";
const pendingSwitchSubmit = switchDom["query-form"].dispatch("submit");
await tick();
assert.equal(queryBodies.length, 2);
assert.notEqual(queryBodies[1].session_id, firstThreadId);
const secondThreadId = queryBodies[1].session_id;
await clickThreadByTitle(switchDom, "First thread question");
const secondServerThread = serverThreads.find((thread) => thread.id === secondThreadId);
secondServerThread.messages = [
  { role: "user", content: "Second thread question" },
  { role: "assistant", content: "Loop answer" },
];
secondServerThread.message_count = 2;
await clickThreadByTitle(switchDom, "Second thread question");
switchQuery.resolve();
await pendingSwitchSubmit;
const switchedAnswers = switchDom.messages.children.filter(
  (node) =>
    node.className === "message assistant" &&
    nodeText(node).includes("Loop answer"),
);
assert.equal(
  switchedAnswers.length,
  1,
  "server-persisted in-flight answer should not be duplicated after switching",
);
assert.equal(switchDom["thread-list"].children.length, 3);

const repeatDom = createDom();
const repeatQuery = deferred();
const slowRepeatDetail = { enabled: false, gate: null };
const repeatThreads = [
  createThreadPayload("thread_repeat", [
    { role: "user", content: "Repeat question" },
    { role: "assistant", content: "Old answer" },
  ]),
  createThreadPayload("thread_repeat_other"),
];
repeatThreads[0].title = "Repeat thread";
repeatThreads[1].title = "Other thread";
globalThis.fetch = async (url, options = {}) => {
  const method = String(options.method || "GET").toUpperCase();
  if (url === "/api/config") {
    return jsonResponse({ text_encodings: [{ label: "Auto", value: "auto" }] });
  }
  if (url === "/api/status") {
    return jsonResponse({ backend: "ollama", model: "thinking-model", ready_for_queries: false, query_mode: "direct", chunk_count: 0 });
  }
  if (url === "/api/recipes") {
    return jsonResponse({ default_recipe_id: "recipe_general_loop", recipes });
  }
  if (url.startsWith("/api/recipes/") && method === "GET") {
    return jsonResponse(recipes[0]);
  }
  if (url === "/api/threads" && method === "GET") {
    return jsonResponse({ threads: repeatThreads });
  }
  if (url.startsWith("/api/threads/") && method === "GET") {
    const id = decodeURIComponent(url.slice("/api/threads/".length));
    if (id === "thread_repeat" && slowRepeatDetail.enabled) {
      await slowRepeatDetail.gate.promise;
    }
    return jsonResponse(repeatThreads.find((item) => item.id === id) || repeatThreads[0]);
  }
  if (url === "/api/query") {
    await repeatQuery.promise;
    return jsonResponse({ answer: "New repeat answer", timeline: { rows: [], final_decision: "not_verified" }, summary: {}, trace: { model_thinking: null } });
  }
  if (url === "/api/chat/clear") {
    return jsonResponse({ timeline: { rows: [], final_decision: null }, summary: {}, trace: {} });
  }
  throw new Error(`unexpected fetch ${url}`);
};
await importFreshApp();
await tick();
repeatDom["query-input"].value = "Repeat question";
const pendingRepeatSubmit = repeatDom["query-form"].dispatch("submit");
await tick();
await clickThreadByTitle(repeatDom, "Other thread");
await clickThreadByTitle(repeatDom, "Repeat thread");
assert.equal(repeatDom["final-decision"].textContent, "running");
const pendingRepeatUserTurns = repeatDom.messages.children.filter(
  (node) =>
    node.className === "message user" &&
    nodeText(node).includes("Repeat question"),
);
assert.equal(
  pendingRepeatUserTurns.length,
  2,
  "same-text pending user turn should survive thread reload",
);
slowRepeatDetail.enabled = true;
slowRepeatDetail.gate = deferred();
const slowRepeatSwitch = clickThreadByTitle(repeatDom, "Repeat thread");
await tick();
await clickThreadByTitle(repeatDom, "Other thread");
slowRepeatDetail.gate.resolve();
await slowRepeatSwitch;
slowRepeatDetail.enabled = false;
assert.equal(repeatDom["active-thread-title"].textContent, "Other thread");
assert.equal(
  repeatDom["final-decision"].textContent,
  "idle",
  "late thread-detail response should not repaint another thread's running loop",
);
repeatQuery.resolve();
await pendingRepeatSubmit;
repeatThreads[0].messages = [
  { role: "user", content: "Repeat question" },
  { role: "assistant", content: "Old answer" },
  { role: "user", content: "Repeat question" },
  { role: "assistant", content: "New repeat answer" },
];
repeatThreads[0].message_count = 4;
await clickThreadByTitle(repeatDom, "Repeat thread");
const repeatUserTurns = repeatDom.messages.children.filter(
  (node) =>
    node.className === "message user" &&
    nodeText(node).includes("Repeat question"),
);
assert.equal(repeatUserTurns.length, 2, "same-text follow-up user turn should survive thread reload");
assert.ok(nodeText(repeatDom.messages).includes("New repeat answer"));

const staleDom = createDom();
const staleQuery = deferred();
const staleServerThreads = [createThreadPayload("thread_stale")];
let staleClearRequests = 0;
globalThis.fetch = async (url, options = {}) => {
  const method = String(options.method || "GET").toUpperCase();
  if (url === "/api/config") {
    return jsonResponse({ text_encodings: [{ label: "Auto", value: "auto" }] });
  }
  if (url === "/api/status") {
    return jsonResponse({ backend: "ollama", model: "thinking-model", ready_for_queries: false, query_mode: "direct", chunk_count: 0 });
  }
  if (url === "/api/recipes") {
    return jsonResponse({ default_recipe_id: "recipe_general_loop", recipes });
  }
  if (url.startsWith("/api/recipes/") && method === "GET") {
    return jsonResponse(recipes[0]);
  }
  if (url === "/api/threads" && method === "GET") {
    return jsonResponse({ threads: staleServerThreads });
  }
  if (url.startsWith("/api/threads/") && method === "GET") {
    return jsonResponse(staleServerThreads[0]);
  }
  if (url === "/api/query") {
    await staleQuery.promise;
    return jsonResponse({ answer: "Stale answer should not reappear", timeline: { rows: [], final_decision: "not_verified" }, summary: {}, trace: { model_thinking: null } });
  }
  if (url === "/api/chat/clear") {
    staleClearRequests += 1;
    return jsonResponse({ timeline: { rows: [], final_decision: null }, summary: {}, trace: {} });
  }
  throw new Error(`unexpected fetch ${url}`);
};
await importFreshApp();
await tick();
staleDom["query-input"].value = "Question that will be cleared";
const pendingSubmit = staleDom["query-form"].dispatch("submit");
await tick();
const pendingAssistant = findNode(
  staleDom.messages,
  (node) => node.className === "message assistant pending",
);
assert.ok(pendingAssistant, "pending assistant message should render while query runs");
assert.ok(
  nodeText(pendingAssistant).includes("Thinking"),
  "pending assistant message should explain that the model is working",
);
assert.equal(staleDom["query-button"].disabled, true);
assert.equal(staleDom["query-input"].disabled, true);
assert.equal(staleDom["query-context"].disabled, true);
assert.equal(staleDom["final-decision"].textContent, "running");
assert.equal(staleDom["clear-chat"].disabled, true);
await staleDom["clear-chat"].dispatch("click");
assert.equal(staleClearRequests, 0, "clear must not overlap an active query");
assert.equal(staleDom["query-button"].disabled, true);
assert.equal(staleDom["query-input"].disabled, true);
assert.equal(staleDom["query-context"].disabled, true);
assert.equal(staleDom["final-decision"].textContent, "running");
staleQuery.resolve();
await pendingSubmit;
const completedAssistant = findNode(
  staleDom.messages,
  (node) =>
    node.className === "message assistant" &&
    nodeText(node).includes("Stale answer should not reappear"),
);
assert.ok(completedAssistant);
assert.equal(staleDom["query-button"].disabled, false);
assert.equal(staleDom["query-input"].disabled, false);
assert.equal(staleDom["query-context"].disabled, false);

const errorDom = createDom();
globalThis.fetch = async (url, options = {}) => {
  const method = String(options.method || "GET").toUpperCase();
  if (url === "/api/config") {
    return jsonResponse({ text_encodings: [{ label: "Auto", value: "auto" }] });
  }
  if (url === "/api/status") {
    return jsonResponse({ backend: "ollama", model: "thinking-model", ready_for_queries: false, query_mode: "direct", chunk_count: 0 });
  }
  if (url === "/api/recipes") {
    return jsonResponse({ default_recipe_id: "recipe_general_loop", recipes });
  }
  if (url.startsWith("/api/recipes/") && method === "GET") {
    return jsonResponse(recipes[0]);
  }
  if (url === "/api/threads" && method === "GET") {
    return jsonResponse({ threads: [createThreadPayload("thread_error")] });
  }
  if (url.startsWith("/api/threads/") && method === "GET") {
    return jsonResponse(createThreadPayload("thread_error"));
  }
  if (url === "/api/query") {
    return errorResponse(503, { detail: "backend unavailable" });
  }
  if (url === "/api/chat/clear") {
    return jsonResponse({ timeline: { rows: [], final_decision: null }, summary: {}, trace: {} });
  }
  throw new Error(`unexpected fetch ${url}`);
};
await importFreshApp();
await tick();
errorDom["query-input"].value = "Question that errors";
await errorDom["query-form"].dispatch("submit");
assert.equal(errorDom["final-decision"].textContent, "error");
assert.ok(nodeText(errorDom.messages).includes("backend unavailable"));
assert.ok(errorDom["trace-json"].textContent.includes("query_failed"));
assert.equal(errorDom["trace-json"].textContent.includes("backend unavailable"), false);
	'''
    )


def test_static_frontend_upload_in_other_thread_preserves_running_query():
    run_frontend_node(
        r'''
import assert from "node:assert/strict";
const {
  createDom,
  createThreadPayload,
  deferred,
  findNode,
  importFreshApp,
  jsonResponse,
  nodeText,
  tick,
} = await import(process.env.FRONTEND_HARNESS_URL);

const dom = createDom();
const threads = [
  createThreadPayload("thread_a"),
  createThreadPayload("thread_b"),
];
threads[0].title = "Thread A";
threads[1].title = "Thread B";
const recipe = {
  recipe_id: "recipe_general_loop",
  name: "General assistant loop",
  goal: "Answer clearly.",
  instructions: "Use same-thread memory carefully.",
  success_criteria: ["Answer clearly."],
  stop_condition: "Stop when done.",
  is_default: true,
};
const queryGate = deferred();
const queryBodies = [];
const uploads = [];
const status = {
  backend: "mock",
  model: "mock",
  ready_for_queries: false,
  query_mode: "direct",
  chunk_count: 0,
};

globalThis.fetch = async (url, options = {}) => {
  if (url === "/api/config") {
    return jsonResponse({ text_encodings: [{ label: "Auto", value: "auto" }] });
  }
  if (url === "/api/status") {
    return jsonResponse(status);
  }
  if (url === "/api/recipes") {
    return jsonResponse({ default_recipe_id: recipe.recipe_id, recipes: [recipe] });
  }
  if (url === `/api/recipes/${recipe.recipe_id}`) {
    return jsonResponse(recipe);
  }
  if (url === "/api/threads") {
    return jsonResponse({ threads });
  }
  if (url.startsWith("/api/threads/")) {
    const threadId = decodeURIComponent(url.slice("/api/threads/".length));
    return jsonResponse(threads.find((thread) => thread.id === threadId));
  }
  if (url === "/api/documents") {
    uploads.push(options.body.get("session_id"));
    return jsonResponse({ status: { ...status, active_document: "beta.txt" } });
  }
  if (url === "/api/query") {
    const body = JSON.parse(options.body);
    queryBodies.push(body);
    if (queryBodies.length === 1) {
      await queryGate.promise;
    }
    return jsonResponse({
      answer: "Completed answer from A",
      timeline: { rows: [], final_decision: "not_verified" },
      summary: {},
      trace: { model_thinking: null },
    });
  }
  throw new Error(`unexpected fetch ${url}`);
};

async function switchTo(title) {
  const button = dom["thread-list"].children.find((node) =>
    nodeText(node).includes(title),
  );
  assert.ok(button, `thread button should exist for ${title}`);
  await button.dispatch("click");
  await tick();
}

await importFreshApp();
await tick();
dom["query-input"].value = "Pending question from A";
const pendingQuery = dom["query-form"].dispatch("submit");
await tick();
try {
  assert.equal(queryBodies.length, 1);
  assert.equal(queryBodies[0].session_id, "thread_a");
  await switchTo("Thread B");
  assert.equal(dom["upload-button"].disabled, false);
  dom["document-file"].files = [new Blob(["Beta context"])];
  await dom["document-file"].dispatch("change");
  assert.deepEqual(uploads, ["thread_b"]);
  for (const id of ["query-button", "query-input", "query-context"]) {
    assert.equal(dom[id].disabled, true, `${id} must stay locked while A runs`);
  }

  // Dispatch directly to prove the handler also rejects a second query when
  // a submission bypasses disabled controls.
  dom["query-input"].value = "Blocked question from B";
  await dom["query-form"].dispatch("submit");
  assert.equal(queryBodies.length, 1, "B must not replace A's running query");
  assert.equal(nodeText(dom.messages).includes("Blocked question from B"), false);

  await switchTo("Pending question from A");
  assert.equal(dom["active-thread-title"].textContent, "Thread A");
  assert.ok(nodeText(dom.messages).includes("Pending question from A"));
  assert.ok(findNode(dom.messages, (node) =>
    node.className === "message assistant pending",
  ));
  assert.equal(dom["final-decision"].textContent, "running");
  assert.equal(dom["query-button"].disabled, true);
} finally {
  queryGate.resolve();
  await pendingQuery;
}

assert.ok(nodeText(dom.messages).includes("Completed answer from A"));
assert.equal(findNode(dom.messages, (node) =>
  node.className === "message assistant pending",
), null);
for (const id of ["query-button", "query-input", "query-context"]) {
  assert.equal(dom[id].disabled, false, `${id} should recover when A completes`);
}
'''
    )


def test_static_frontend_deletes_threads_from_server_and_ui():
    run_frontend_node(
        r'''
import assert from "node:assert/strict";
const {
  createDom,
  createThreadPayload,
  deferred,
  errorResponse,
  importFreshApp,
  jsonResponse,
  nodeText,
  tick,
} = await import(process.env.FRONTEND_HARNESS_URL);

const dom = createDom();
const recipes = [{
  recipe_id: "recipe_general_loop",
  name: "General assistant loop",
  goal: "Answer the request clearly.",
  instructions: "Use same-thread memory carefully.",
  success_criteria: ["Answer the request clearly."],
  stop_condition: "Stop when done.",
  is_default: true,
}];
let serverThreads = [
  createThreadPayload("thread_delete", [
    { role: "user", content: "Delete me question" },
    { role: "assistant", content: "Delete me answer" },
  ]),
  createThreadPayload("thread_keep", [
    { role: "user", content: "Keep me question" },
  ]),
];
serverThreads[0].title = "Delete me";
serverThreads[1].title = "Keep me";
const deleted = [];
const detailFetches = [];
const queryBodies = [];
const confirmations = [];
const deleteGate = deferred();
let createdCount = 0;
globalThis.confirm = (message) => {
  confirmations.push(message);
  return true;
};

globalThis.fetch = async (url, options = {}) => {
  const method = String(options.method || "GET").toUpperCase();
  if (url === "/api/config") {
    return jsonResponse({ text_encodings: [{ label: "Auto", value: "auto" }] });
  }
  if (url === "/api/status") {
    return jsonResponse({ backend: "ollama", model: "thinking-model", ready_for_queries: false, query_mode: "direct", chunk_count: 0 });
  }
  if (url === "/api/recipes") {
    return jsonResponse({ default_recipe_id: "recipe_general_loop", recipes });
  }
  if (url.startsWith("/api/recipes/") && method === "GET") {
    return jsonResponse(recipes[0]);
  }
  if (url === "/api/threads" && method === "GET") {
    return jsonResponse({ threads: serverThreads });
  }
  if (url === "/api/threads" && method === "POST") {
    createdCount += 1;
    const thread = createThreadPayload(`thread_created_${createdCount}`);
    serverThreads = [thread];
    return jsonResponse(thread);
  }
  if (url.startsWith("/api/threads/") && method === "GET") {
    const id = decodeURIComponent(url.slice("/api/threads/".length));
    detailFetches.push(id);
    const thread = serverThreads.find((item) => item.id === id);
    return thread
      ? jsonResponse(thread)
      : errorResponse(404, { detail: "Thread not found." });
  }
  if (url.startsWith("/api/threads/") && method === "DELETE") {
    const id = decodeURIComponent(url.slice("/api/threads/".length));
    deleted.push(id);
    serverThreads = serverThreads.filter((thread) => thread.id !== id);
    await deleteGate.promise;
    return jsonResponse({ deleted: true, thread_id: id });
  }
  if (url === "/api/query") {
    queryBodies.push(JSON.parse(options.body));
    return jsonResponse({ answer: "Should not be sent", timeline: { rows: [], final_decision: "not_verified" }, summary: {}, trace: { model_thinking: null } });
  }
  throw new Error(`unexpected fetch ${url}`);
};

await importFreshApp();
await tick();
assert.equal(dom["active-thread-title"].textContent, "Delete me");
assert.ok(nodeText(dom["thread-list"]).includes("Delete me"));
assert.ok(nodeText(dom["thread-list"]).includes("Keep me"));
assert.deepEqual(detailFetches, ["thread_delete"]);

const pendingDelete = dom["delete-thread"].dispatch("click");
await tick();
assert.deepEqual(deleted, ["thread_delete"]);
assert.equal(confirmations.length, 1);
assert.ok(confirmations[0].includes("Delete me"));
assert.equal(dom["query-button"].disabled, true);
dom["query-input"].value = "do not resurrect deleted thread";
await dom["query-form"].dispatch("submit");
assert.deepEqual(queryBodies, []);
deleteGate.resolve();
await pendingDelete;
await tick();
assert.equal(nodeText(dom["thread-list"]).includes("Delete me"), false);
assert.ok(nodeText(dom["thread-list"]).includes("Keep me"));
assert.equal(dom["active-thread-title"].textContent, "Keep me");
assert.equal(globalThis.localStorage.getItem("loopwright.active-thread.v1"), "thread_keep");
assert.deepEqual(detailFetches, ["thread_delete", "thread_keep"]);
assert.equal(dom["query-button"].disabled, false);

await dom["delete-thread"].dispatch("click");
await tick();
assert.deepEqual(deleted, ["thread_delete", "thread_keep"]);
assert.equal(createdCount, 1);
assert.equal(serverThreads.length, 1);
assert.equal(serverThreads[0].id, "thread_created_1");
assert.equal(dom["active-thread-title"].textContent, "New thread");
assert.equal(globalThis.localStorage.getItem("loopwright.active-thread.v1"), "thread_created_1");
assert.equal(dom["delete-thread"].disabled, false);
assert.equal(nodeText(dom["thread-list"]).includes("Keep me"), false);
'''
    )


def test_static_frontend_overlapping_thread_deletes_keep_query_disabled():
    run_frontend_node(
        r'''
import assert from "node:assert/strict";
const {
  createDom,
  createThreadPayload,
  deferred,
  errorResponse,
  importFreshApp,
  jsonResponse,
  nodeText,
  tick,
} = await import(process.env.FRONTEND_HARNESS_URL);

const dom = createDom();
const recipes = [{
  recipe_id: "recipe_general_loop",
  name: "General assistant loop",
  goal: "Answer the request clearly.",
  instructions: "Use same-thread memory carefully.",
  success_criteria: ["Answer the request clearly."],
  stop_condition: "Stop when done.",
  is_default: true,
}];
let serverThreads = [
  createThreadPayload("thread_a"),
  createThreadPayload("thread_b"),
];
serverThreads[0].title = "Thread A";
serverThreads[1].title = "Thread B";
const gates = {
  thread_a: deferred(),
  thread_b: deferred(),
};
const detailFetches = [];
const queryBodies = [];
let createdCount = 0;
globalThis.confirm = () => true;

async function clickThreadByTitle(title) {
  const button = dom["thread-list"].children.find((node) =>
    nodeText(node).includes(title),
  );
  assert.ok(button, `thread button should exist for ${title}`);
  await button.dispatch("click");
  await tick();
}

globalThis.fetch = async (url, options = {}) => {
  const method = String(options.method || "GET").toUpperCase();
  if (url === "/api/config") {
    return jsonResponse({ text_encodings: [{ label: "Auto", value: "auto" }] });
  }
  if (url === "/api/status") {
    return jsonResponse({ backend: "ollama", model: "thinking-model", ready_for_queries: false, query_mode: "direct", chunk_count: 0 });
  }
  if (url === "/api/recipes") {
    return jsonResponse({ default_recipe_id: "recipe_general_loop", recipes });
  }
  if (url.startsWith("/api/recipes/") && method === "GET") {
    return jsonResponse(recipes[0]);
  }
  if (url === "/api/threads" && method === "GET") {
    return jsonResponse({ threads: serverThreads });
  }
  if (url === "/api/threads" && method === "POST") {
    createdCount += 1;
    const thread = createThreadPayload(`thread_created_${createdCount}`);
    serverThreads = [thread];
    return jsonResponse(thread);
  }
  if (url.startsWith("/api/threads/") && method === "GET") {
    const id = decodeURIComponent(url.slice("/api/threads/".length));
    detailFetches.push(id);
    const thread = serverThreads.find((item) => item.id === id);
    return thread
      ? jsonResponse(thread)
      : errorResponse(404, { detail: "Thread not found." });
  }
  if (url.startsWith("/api/threads/") && method === "DELETE") {
    const id = decodeURIComponent(url.slice("/api/threads/".length));
    serverThreads = serverThreads.filter((thread) => thread.id !== id);
    await gates[id].promise;
    return jsonResponse({ deleted: true, thread_id: id });
  }
  if (url === "/api/query") {
    queryBodies.push(JSON.parse(options.body));
    return jsonResponse({ answer: "Should not be sent", timeline: { rows: [], final_decision: "not_verified" }, summary: {}, trace: { model_thinking: null } });
  }
  throw new Error(`unexpected fetch ${url}`);
};

await importFreshApp();
await tick();
assert.equal(dom["active-thread-title"].textContent, "Thread A");

const pendingDeleteA = dom["delete-thread"].dispatch("click");
await tick();
assert.equal(dom["query-button"].disabled, true);
await clickThreadByTitle("Thread B");
assert.equal(dom["active-thread-title"].textContent, "Thread B");

const pendingDeleteB = dom["delete-thread"].dispatch("click");
await tick();
dom["query-input"].value = "do not query B while deleting";
await dom["query-form"].dispatch("submit");
assert.deepEqual(queryBodies, []);
assert.equal(dom["query-button"].disabled, true);

gates.thread_a.resolve();
await pendingDeleteA;
await tick();
assert.equal(
  dom["query-button"].disabled,
  true,
  "finishing another delete must not re-enable the active pending-delete thread",
);
await dom["query-form"].dispatch("submit");
assert.deepEqual(queryBodies, []);
assert.equal(
  detailFetches.filter((id) => id === "thread_b").length,
  1,
  "finishing delete A should not refetch active thread B while B is pending deletion",
);

gates.thread_b.resolve();
await pendingDeleteB;
await tick();
assert.equal(createdCount, 1);
assert.equal(dom["active-thread-title"].textContent, "New thread");
assert.equal(dom["query-button"].disabled, false);
assert.deepEqual(queryBodies, []);
'''
    )


def test_static_frontend_delete_waits_for_same_thread_query_completion():
    run_frontend_node(
        r'''
import assert from "node:assert/strict";
const {
  createDom,
  createThreadPayload,
  deferred,
  errorResponse,
  findNode,
  importFreshApp,
  jsonResponse,
  nodeText,
  tick,
} = await import(process.env.FRONTEND_HARNESS_URL);

const dom = createDom();
const recipes = [{
  recipe_id: "recipe_general_loop",
  name: "General assistant loop",
  goal: "Answer the request clearly.",
  instructions: "Use same-thread memory carefully.",
  success_criteria: ["Answer the request clearly."],
  stop_condition: "Stop when done.",
  is_default: true,
}];
let serverThreads = [createThreadPayload("thread_delete")];
serverThreads[0].title = "Delete during query";
const queryGate = deferred();
const deleteGate = deferred();
const queryBodies = [];
const deleteRequests = [];
let createdCount = 0;
globalThis.confirm = () => true;

globalThis.fetch = async (url, options = {}) => {
  const method = String(options.method || "GET").toUpperCase();
  if (url === "/api/config") {
    return jsonResponse({ text_encodings: [{ label: "Auto", value: "auto" }] });
  }
  if (url === "/api/status") {
    return jsonResponse({ backend: "ollama", model: "thinking-model", ready_for_queries: false, query_mode: "direct", chunk_count: 0 });
  }
  if (url === "/api/recipes") {
    return jsonResponse({ default_recipe_id: "recipe_general_loop", recipes });
  }
  if (url.startsWith("/api/recipes/") && method === "GET") {
    return jsonResponse(recipes[0]);
  }
  if (url === "/api/threads" && method === "GET") {
    return jsonResponse({ threads: serverThreads });
  }
  if (url === "/api/threads" && method === "POST") {
    createdCount += 1;
    const thread = createThreadPayload(`thread_created_${createdCount}`);
    serverThreads = [thread];
    return jsonResponse(thread);
  }
  if (url.startsWith("/api/threads/") && method === "GET") {
    const id = decodeURIComponent(url.slice("/api/threads/".length));
    const thread = serverThreads.find((item) => item.id === id);
    return thread
      ? jsonResponse(thread)
      : errorResponse(404, { detail: "Thread not found." });
  }
  if (url === "/api/query") {
    queryBodies.push(JSON.parse(options.body));
    await queryGate.promise;
    return jsonResponse({
      answer: "Stale answer should not render",
      timeline: { rows: [], final_decision: "not_verified" },
      summary: {},
      trace: { model_thinking: null },
      thread: createThreadPayload("thread_delete", [
        { role: "user", content: "Slow question" },
        { role: "assistant", content: "Stale answer should not render" },
      ]),
    });
  }
  if (url.startsWith("/api/threads/") && method === "DELETE") {
    const id = decodeURIComponent(url.slice("/api/threads/".length));
    deleteRequests.push(id);
    serverThreads = serverThreads.filter((thread) => thread.id !== id);
    await deleteGate.promise;
    return jsonResponse({ deleted: true, thread_id: id });
  }
  throw new Error(`unexpected fetch ${url}`);
};

await importFreshApp();
await tick();
dom["query-input"].value = "Slow question";
const pendingQuery = dom["query-form"].dispatch("submit");
await tick();
assert.deepEqual(queryBodies.map((body) => body.session_id), ["thread_delete"]);
assert.equal(dom["query-button"].disabled, true);

assert.equal(dom["delete-thread"].disabled, true);
await dom["delete-thread"].dispatch("click");
await tick();
assert.equal(dom["query-button"].disabled, true);
assert.deepEqual(deleteRequests, [], "delete must not overlap its thread query");

queryGate.resolve();
await pendingQuery;
await tick();
assert.equal(
  dom["query-button"].disabled,
  false,
  "query completion should release destructive controls",
);
const completedAssistant = findNode(
  dom.messages,
  (node) =>
    node.className === "message assistant" &&
    nodeText(node).includes("Stale answer should not render"),
);
assert.ok(completedAssistant);

const pendingDelete = dom["delete-thread"].dispatch("click");
await tick();
assert.deepEqual(deleteRequests, ["thread_delete"]);
assert.equal(dom["query-button"].disabled, true);

deleteGate.resolve();
await pendingDelete;
await tick();
assert.equal(createdCount, 1);
assert.equal(dom["active-thread-title"].textContent, "New thread");
assert.equal(dom["query-button"].disabled, false);
'''
    )


def test_static_frontend_clear_failure_reconciles_and_excludes_thread_mutations():
    run_frontend_node(
        r'''
import assert from "node:assert/strict";
const {
  createDom,
  createThreadPayload,
  deferred,
  errorResponse,
  importFreshApp,
  jsonResponse,
  nodeText,
  tick,
} = await import(process.env.FRONTEND_HARNESS_URL);

const dom = createDom();
const thread = createThreadPayload("thread_clear", [
  { role: "user", content: "Durable question" },
  { role: "assistant", content: "Durable answer" },
]);
thread.title = "Clear safely";
const recipes = [{
  recipe_id: "recipe_general_loop",
  name: "General assistant loop",
  goal: "Answer clearly.",
  instructions: "Stay in the thread.",
  success_criteria: ["Answer clearly."],
  stop_condition: "Stop when done.",
  is_default: true,
}];
const firstClearGate = deferred();
const uploadGate = deferred();
let detailFetches = 0;
let failDetailRefresh = false;
let clearRequests = 0;
let uploadRequests = 0;
let deleteRequests = 0;
let queryRequests = 0;
let confirmations = 0;
globalThis.confirm = () => {
  confirmations += 1;
  return true;
};

globalThis.fetch = async (url, options = {}) => {
  const method = String(options.method || "GET").toUpperCase();
  if (url === "/api/config") {
    return jsonResponse({ text_encodings: [{ label: "Auto", value: "auto" }] });
  }
  if (url === "/api/status") {
    return jsonResponse({ backend: "mock", model: "MockLLM" });
  }
  if (url === "/api/recipes") {
    return jsonResponse({ default_recipe_id: "recipe_general_loop", recipes });
  }
  if (url.startsWith("/api/recipes/") && method === "GET") {
    return jsonResponse(recipes[0]);
  }
  if (url === "/api/threads" && method === "GET") {
    return jsonResponse({ threads: [thread] });
  }
  if (url === "/api/threads/thread_clear" && method === "GET") {
    detailFetches += 1;
    return failDetailRefresh
      ? errorResponse(503, { detail: "detail refresh unavailable" })
      : jsonResponse(thread);
  }
  if (url === "/api/chat/clear" && method === "POST") {
    clearRequests += 1;
    if (clearRequests === 1) {
      await firstClearGate.promise;
    }
    return errorResponse(503, { detail: "clear rejected" });
  }
  if (url === "/api/documents" && method === "POST") {
    uploadRequests += 1;
    await uploadGate.promise;
    return jsonResponse({ status: { backend: "mock", model: "MockLLM" } });
  }
  if (url === "/api/threads/thread_clear" && method === "DELETE") {
    deleteRequests += 1;
    return jsonResponse({ deleted: true, thread_id: "thread_clear" });
  }
  if (url === "/api/query" && method === "POST") {
    queryRequests += 1;
    return errorResponse(503, { detail: "query should be blocked" });
  }
  throw new Error(`unexpected fetch ${method} ${url}`);
};

await importFreshApp();
await tick();
assert.ok(nodeText(dom.messages).includes("Durable answer"));
assert.equal(detailFetches, 1);

const pendingClear = dom["clear-chat"].dispatch("click");
await tick();
assert.equal(nodeText(dom.messages).includes("Durable answer"), false);
assert.equal(dom["query-button"].disabled, true);
assert.equal(dom["clear-chat"].disabled, true);
assert.equal(dom["delete-thread"].disabled, true);
assert.equal(dom["upload-button"].disabled, true);

dom["query-input"].value = "must not overlap clear";
await dom["query-form"].dispatch("submit");
dom["document-file"].files = [new Blob(["blocked"], { type: "text/plain" })];
await dom["document-file"].dispatch("change");
await dom["delete-thread"].dispatch("click");
await dom["clear-chat"].dispatch("click");
assert.equal(queryRequests, 0);
assert.equal(uploadRequests, 0);
assert.equal(deleteRequests, 0);
assert.equal(confirmations, 0);
assert.equal(clearRequests, 1);

firstClearGate.resolve();
await pendingClear;
assert.equal(detailFetches, 2, "failed clear should refresh authoritative detail");
assert.ok(nodeText(dom.messages).includes("Durable answer"));
assert.ok(nodeText(dom["run-list"]).includes("refreshed the durable thread state"));
assert.equal(dom["query-button"].disabled, false);
assert.equal(dom["clear-chat"].disabled, false);
assert.equal(dom["delete-thread"].disabled, false);
assert.equal(dom["upload-button"].disabled, false);

failDetailRefresh = true;
await dom["clear-chat"].dispatch("click");
assert.equal(detailFetches, 3);
assert.ok(nodeText(dom.messages).includes("Durable answer"));
assert.ok(nodeText(dom["run-list"]).includes("restored the previous local view"));

dom["document-file"].files = [new Blob(["pending"], { type: "text/plain" })];
const pendingUpload = dom["document-file"].dispatch("change");
await tick();
assert.equal(uploadRequests, 1);
assert.equal(dom["query-button"].disabled, true);
assert.equal(dom["clear-chat"].disabled, true);
assert.equal(dom["delete-thread"].disabled, true);
await dom["clear-chat"].dispatch("click");
await dom["delete-thread"].dispatch("click");
dom["query-input"].value = "must not overlap upload";
await dom["query-form"].dispatch("submit");
assert.equal(clearRequests, 2);
assert.equal(deleteRequests, 0);
assert.equal(queryRequests, 0);

uploadGate.resolve();
await pendingUpload;
assert.equal(dom["query-button"].disabled, false);
assert.equal(dom["clear-chat"].disabled, false);
assert.equal(dom["delete-thread"].disabled, false);
assert.equal(dom["upload-button"].disabled, false);
'''
    )


def test_static_frontend_delete_failure_reconciliation_cannot_overwrite_switch():
    run_frontend_node(
        r'''
import assert from "node:assert/strict";
const {
  createDom,
  createThreadPayload,
  deferred,
  errorResponse,
  importFreshApp,
  jsonResponse,
  nodeText,
  tick,
} = await import(process.env.FRONTEND_HARNESS_URL);

const dom = createDom();
const threads = [
  createThreadPayload("thread_alpha", [
    { role: "user", content: "Alpha question" },
    { role: "assistant", content: "Alpha durable answer" },
  ]),
  createThreadPayload("thread_beta", [
    { role: "assistant", content: "Beta answer" },
  ]),
];
threads[0].title = "Alpha";
threads[1].title = "Beta";
const recipes = [{
  recipe_id: "recipe_general_loop",
  name: "General assistant loop",
  goal: "Answer clearly.",
  instructions: "Stay in the thread.",
  success_criteria: ["Answer clearly."],
  stop_condition: "Stop when done.",
  is_default: true,
}];
const deleteGate = deferred();
const detailFetches = [];
const deleteRequests = [];
globalThis.confirm = () => true;

globalThis.fetch = async (url, options = {}) => {
  const method = String(options.method || "GET").toUpperCase();
  if (url === "/api/config") {
    return jsonResponse({ text_encodings: [{ label: "Auto", value: "auto" }] });
  }
  if (url === "/api/status") {
    return jsonResponse({ backend: "mock", model: "MockLLM" });
  }
  if (url === "/api/recipes") {
    return jsonResponse({ default_recipe_id: "recipe_general_loop", recipes });
  }
  if (url.startsWith("/api/recipes/") && method === "GET") {
    return jsonResponse(recipes[0]);
  }
  if (url === "/api/threads" && method === "GET") {
    return jsonResponse({ threads });
  }
  if (url.startsWith("/api/threads/") && method === "GET") {
    const id = decodeURIComponent(url.slice("/api/threads/".length));
    detailFetches.push(id);
    return jsonResponse(threads.find((thread) => thread.id === id));
  }
  if (url === "/api/threads/thread_alpha" && method === "DELETE") {
    deleteRequests.push("thread_alpha");
    await deleteGate.promise;
    return errorResponse(503, { detail: "delete rejected" });
  }
  throw new Error(`unexpected fetch ${method} ${url}`);
};

await importFreshApp();
await tick();
const buttonFor = (title) => {
  const button = dom["thread-list"].children.find((node) =>
    nodeText(node).includes(title),
  );
  assert.ok(button);
  return button;
};

const pendingDelete = dom["delete-thread"].dispatch("click");
await tick();
assert.deepEqual(deleteRequests, ["thread_alpha"]);
assert.equal(buttonFor("Alpha").disabled, true);
await buttonFor("Beta").dispatch("click");
await tick();
assert.equal(dom["active-thread-title"].textContent, "Beta");
assert.equal(
  globalThis.localStorage.getItem("loopwright.active-thread.v1"),
  "thread_beta",
);

deleteGate.resolve();
await pendingDelete;
assert.deepEqual(detailFetches, ["thread_alpha", "thread_beta", "thread_alpha"]);
assert.equal(dom["active-thread-title"].textContent, "Beta");
assert.equal(
  globalThis.localStorage.getItem("loopwright.active-thread.v1"),
  "thread_beta",
);
assert.ok(nodeText(dom.messages).includes("Beta answer"));
assert.equal(dom["query-button"].disabled, false);

await buttonFor("Alpha").dispatch("click");
await tick();
assert.equal(dom["active-thread-title"].textContent, "Alpha");
assert.ok(nodeText(dom.messages).includes("Alpha durable answer"));
'''
    )


def test_static_frontend_delete_commit_keeps_replacement_identity_when_detail_fails():
    run_frontend_node(
        r'''
import assert from "node:assert/strict";
const {
  createDom,
  createThreadPayload,
  errorResponse,
  importFreshApp,
  jsonResponse,
  nodeText,
  tick,
} = await import(process.env.FRONTEND_HARNESS_URL);

const dom = createDom();
let threads = [
  createThreadPayload("thread_delete", [
    { role: "assistant", content: "Deleted answer" },
  ]),
  createThreadPayload("thread_keep", [
    { role: "assistant", content: "Summary not yet hydrated" },
  ]),
];
threads[0].title = "Delete me";
threads[1].title = "Keep me";
const recipes = [{
  recipe_id: "recipe_general_loop",
  name: "General assistant loop",
  goal: "Answer clearly.",
  instructions: "Stay in the thread.",
  success_criteria: ["Answer clearly."],
  stop_condition: "Stop when done.",
  is_default: true,
}];
globalThis.confirm = () => true;

globalThis.fetch = async (url, options = {}) => {
  const method = String(options.method || "GET").toUpperCase();
  if (url === "/api/config") {
    return jsonResponse({ text_encodings: [{ label: "Auto", value: "auto" }] });
  }
  if (url === "/api/status") {
    return jsonResponse({ backend: "mock", model: "MockLLM" });
  }
  if (url === "/api/recipes") {
    return jsonResponse({ default_recipe_id: "recipe_general_loop", recipes });
  }
  if (url.startsWith("/api/recipes/") && method === "GET") {
    return jsonResponse(recipes[0]);
  }
  if (url === "/api/threads" && method === "GET") {
    return jsonResponse({ threads });
  }
  if (url === "/api/threads/thread_delete" && method === "GET") {
    return jsonResponse(threads[0]);
  }
  if (url === "/api/threads/thread_keep" && method === "GET") {
    return errorResponse(503, { detail: "replacement detail unavailable" });
  }
  if (url === "/api/threads/thread_delete" && method === "DELETE") {
    threads = threads.filter((thread) => thread.id !== "thread_delete");
    return jsonResponse({ deleted: true, thread_id: "thread_delete" });
  }
  throw new Error(`unexpected fetch ${method} ${url}`);
};

await importFreshApp();
await tick();
await dom["delete-thread"].dispatch("click");
await tick();
assert.equal(dom["active-thread-title"].textContent, "Keep me");
assert.equal(
  globalThis.localStorage.getItem("loopwright.active-thread.v1"),
  "thread_keep",
);
assert.equal(nodeText(dom["thread-list"]).includes("Delete me"), false);
assert.ok(nodeText(dom["thread-list"]).includes("Keep me"));
assert.ok(dom["upload-status"].textContent.includes("Deleted thread"));
assert.ok(dom["upload-status"].textContent.includes("replacement detail unavailable"));
assert.equal(dom["query-button"].disabled, false);
assert.equal(dom["clear-chat"].disabled, false);
assert.equal(dom["delete-thread"].disabled, false);
assert.equal(dom["upload-button"].disabled, false);
'''
    )


def test_static_frontend_running_query_allows_other_thread_delete_without_reenabling():
    run_frontend_node(
        r'''
import assert from "node:assert/strict";
const {
  createDom,
  createThreadPayload,
  deferred,
  errorResponse,
  importFreshApp,
  jsonResponse,
  nodeText,
  tick,
} = await import(process.env.FRONTEND_HARNESS_URL);

const dom = createDom();
const recipes = [{
  recipe_id: "recipe_general_loop",
  name: "General assistant loop",
  goal: "Answer the request clearly.",
  instructions: "Use same-thread memory carefully.",
  success_criteria: ["Answer the request clearly."],
  stop_condition: "Stop when done.",
  is_default: true,
}];
let serverThreads = [
  createThreadPayload("thread_a"),
  createThreadPayload("thread_b"),
];
serverThreads[0].title = "Thread A";
serverThreads[1].title = "Thread B";
const queryGate = deferred();
const gates = {
  thread_b: deferred(),
};
const deleteRequests = [];
let createdCount = 0;
globalThis.confirm = () => true;

async function clickThreadByTitle(title) {
  const button = dom["thread-list"].children.find((node) =>
    nodeText(node).includes(title),
  );
  assert.ok(button, `thread button should exist for ${title}`);
  await button.dispatch("click");
  await tick();
}

globalThis.fetch = async (url, options = {}) => {
  const method = String(options.method || "GET").toUpperCase();
  if (url === "/api/config") {
    return jsonResponse({ text_encodings: [{ label: "Auto", value: "auto" }] });
  }
  if (url === "/api/status") {
    return jsonResponse({ backend: "ollama", model: "thinking-model", ready_for_queries: false, query_mode: "direct", chunk_count: 0 });
  }
  if (url === "/api/recipes") {
    return jsonResponse({ default_recipe_id: "recipe_general_loop", recipes });
  }
  if (url.startsWith("/api/recipes/") && method === "GET") {
    return jsonResponse(recipes[0]);
  }
  if (url === "/api/threads" && method === "GET") {
    return jsonResponse({ threads: serverThreads });
  }
  if (url === "/api/threads" && method === "POST") {
    createdCount += 1;
    const thread = createThreadPayload(`thread_created_${createdCount}`);
    serverThreads = [thread];
    return jsonResponse(thread);
  }
  if (url.startsWith("/api/threads/") && method === "GET") {
    const id = decodeURIComponent(url.slice("/api/threads/".length));
    const thread = serverThreads.find((item) => item.id === id);
    return thread
      ? jsonResponse(thread)
      : errorResponse(404, { detail: "Thread not found." });
  }
  if (url === "/api/query") {
    await queryGate.promise;
    serverThreads[0].messages = [
      { role: "user", content: "Slow query on A" },
      { role: "assistant", content: "Old answer" },
    ];
    serverThreads[0].message_count = 2;
    return jsonResponse({ answer: "Old answer", timeline: { rows: [], final_decision: "not_verified" }, summary: {}, trace: { model_thinking: null } });
  }
  if (url.startsWith("/api/threads/") && method === "DELETE") {
    const id = decodeURIComponent(url.slice("/api/threads/".length));
    deleteRequests.push(id);
    serverThreads = serverThreads.filter((thread) => thread.id !== id);
    await gates[id].promise;
    return jsonResponse({ deleted: true, thread_id: id });
  }
  throw new Error(`unexpected fetch ${url}`);
};

await importFreshApp();
await tick();
dom["query-input"].value = "Slow query on A";
const pendingQuery = dom["query-form"].dispatch("submit");
await tick();
assert.equal(dom["query-button"].disabled, true);

const pendingDeleteA = dom["delete-thread"].dispatch("click");
await tick();
await pendingDeleteA;
assert.deepEqual(deleteRequests, [], "the active query must block deletion of A");
await clickThreadByTitle("Thread B");
const pendingDeleteB = dom["delete-thread"].dispatch("click");
await tick();
assert.deepEqual(deleteRequests, ["thread_b"]);
assert.equal(dom["active-thread-title"].textContent, "Thread B");
assert.equal(dom["query-button"].disabled, true);

queryGate.resolve();
await pendingQuery;
await tick();
assert.equal(
  dom["query-button"].disabled,
  true,
  "query A completion must not enable controls while active thread B is deleting",
);

gates.thread_b.resolve();
await pendingDeleteB;
await tick();
assert.equal(createdCount, 0);
assert.equal(dom["active-thread-title"].textContent, "Thread A");
assert.equal(dom["query-button"].disabled, false);
'''
    )


def test_static_frontend_recipe_save_and_detail_failure_guard():
    run_frontend_node(
        r'''
import assert from "node:assert/strict";
const {
  createDom,
  createThreadPayload,
  errorResponse,
  importFreshApp,
  jsonResponse,
  tick,
} = await import(process.env.FRONTEND_HARNESS_URL);

const dom = createDom();
const recipeBodies = [];
const recipes = [{
  recipe_id: "recipe_general_loop",
  name: "General assistant loop",
  goal: "Answer the request clearly.",
  context_provider: "smart",
  model_profile: "quality",
  verifier: "default",
  instructions: "Use same-thread memory carefully.",
  success_criteria: ["Answer the request clearly."],
  stop_condition: "Stop when done.",
  is_default: true,
}];
globalThis.fetch = async (url, options = {}) => {
  const method = String(options.method || "GET").toUpperCase();
  if (url === "/api/config") {
    return jsonResponse({ text_encodings: [{ label: "Auto", value: "auto" }] });
  }
  if (url === "/api/status") {
    return jsonResponse({ backend: "ollama", model: "thinking-model", ready_for_queries: false, query_mode: "direct", chunk_count: 0 });
  }
  if (url === "/api/recipes") {
    if (method === "POST") {
      const body = JSON.parse(options.body);
      recipeBodies.push(body);
      const recipe = { recipe_id: `recipe_created_${recipeBodies.length}`, ...body, is_default: false };
      recipes.push(recipe);
      return jsonResponse(recipe);
    }
    return jsonResponse({ default_recipe_id: "recipe_general_loop", recipes });
  }
  if (url.startsWith("/api/recipes/") && method === "GET") {
    return jsonResponse(recipes[0]);
  }
  if (url === "/api/threads" && method === "GET") {
    return jsonResponse({ threads: [createThreadPayload("thread_recipe_save")] });
  }
  if (url.startsWith("/api/threads/") && method === "GET") {
    return jsonResponse(createThreadPayload("thread_recipe_save"));
  }
  if (url === "/api/query") {
    return jsonResponse({ answer: "Loop answer", timeline: { rows: [], final_decision: "not_verified" }, summary: {}, trace: { model_thinking: null } });
  }
  if (url === "/api/chat/clear") {
    return jsonResponse({ timeline: { rows: [], final_decision: null }, summary: {}, trace: {} });
  }
  throw new Error(`unexpected fetch ${url}`);
};
await importFreshApp();
await tick();
dom["recipe-new"].dispatch("click");
dom["recipe-name"].value = "Debug loop";
dom["recipe-goal"].value = "Explain loop behavior.";
dom["recipe-instructions"].value = "Keep evidence and advice separate.";
dom["recipe-criteria"].value = "Names the loop step\nStates uncertainty";
dom["recipe-stop"].value = "Stop after a clear answer.";
await dom["recipe-save"].dispatch("click");
assert.equal(recipeBodies.length, 1);
assert.deepEqual(recipeBodies[0].success_criteria, ["Names the loop step", "States uncertainty"]);
assert.equal(recipeBodies[0].goal, "Explain loop behavior.");
assert.equal(dom["recipe-select"].value, "recipe_created_1");

const summaryOnlyDom = createDom();
const summaryOnlyPatchBodies = [];
globalThis.fetch = async (url, options = {}) => {
  const method = String(options.method || "GET").toUpperCase();
  if (url === "/api/config") {
    return jsonResponse({ text_encodings: [{ label: "Auto", value: "auto" }] });
  }
  if (url === "/api/status") {
    return jsonResponse({ backend: "ollama", model: "thinking-model", ready_for_queries: false, query_mode: "direct", chunk_count: 0 });
  }
  if (url === "/api/recipes") {
    return jsonResponse({ default_recipe_id: "recipe_custom", recipes: [{ recipe_id: "recipe_custom", name: "Custom summary", goal: "Summary-only goal.", is_default: false }] });
  }
  if (url === "/api/recipes/recipe_custom" && method === "GET") {
    return errorResponse(503, { detail: "Recipe detail unavailable." });
  }
  if (url === "/api/recipes/recipe_custom" && method === "PATCH") {
    summaryOnlyPatchBodies.push(JSON.parse(options.body));
    return jsonResponse({ recipe_id: "recipe_custom", name: "Should not save", goal: "Should not save.", instructions: "", success_criteria: [], stop_condition: "", is_default: false });
  }
  if (url === "/api/threads" && method === "GET") {
    return jsonResponse({ threads: [createThreadPayload("thread_summary_only")] });
  }
  if (url.startsWith("/api/threads/") && method === "GET") {
    return jsonResponse(createThreadPayload("thread_summary_only"));
  }
  throw new Error(`unexpected fetch ${url}`);
};
await importFreshApp();
await tick();
assert.equal(summaryOnlyDom["recipe-save"].disabled, true);
assert.equal(summaryOnlyDom["recipe-name"].readOnly, true);
summaryOnlyDom["recipe-name"].value = "Wipe hidden fields";
await summaryOnlyDom["recipe-save"].dispatch("click");
assert.equal(summaryOnlyPatchBodies.length, 0);
assert.equal(summaryOnlyDom["recipe-save"].disabled, true);
assert.ok(summaryOnlyDom["recipe-status"].textContent.includes("detail unavailable"));
'''
    )


def test_static_frontend_recipe_import_export_delete_controls():
    run_frontend_node(
        r'''
import assert from "node:assert/strict";
const {
  createDom,
  createThreadPayload,
  importFreshApp,
  jsonResponse,
  tick,
} = await import(process.env.FRONTEND_HARNESS_URL);

const dom = createDom();
globalThis.localStorage.setItem("ai-loop-engine.active-recipe.v1", "recipe_custom");
let deleteConfirmed = false;
globalThis.confirm = (message) => {
  deleteConfirmed = message.includes("Custom reviewer");
  return true;
};
const requests = { deleted: [], imported: [], exported: [] };
const recipes = [
  { recipe_id: "recipe_general_loop", name: "General assistant loop", goal: "Answer the request clearly.", instructions: "Default instructions.", success_criteria: ["Default passes."], stop_condition: "Stop when done.", is_default: true },
  { recipe_id: "recipe_custom", name: "Custom reviewer", goal: "Find weak assumptions.", instructions: "Be sharp.", success_criteria: ["Names risk."], stop_condition: "Stop after verdict.", context_provider: "smart", model_profile: "quality", verifier: "default", metadata: { source: "test" }, is_default: false },
];
globalThis.fetch = async (url, options = {}) => {
  const method = String(options.method || "GET").toUpperCase();
  if (url === "/api/config") {
    return jsonResponse({ text_encodings: [{ label: "Auto", value: "auto" }] });
  }
  if (url === "/api/status") {
    return jsonResponse({ backend: "ollama", model: "thinking-model", ready_for_queries: false, query_mode: "direct", chunk_count: 0 });
  }
  if (url === "/api/recipes") {
    if (method === "POST") {
      const body = JSON.parse(options.body);
      requests.imported.push(body);
      const imported = { ...body, is_default: false, context_provider: body.context_provider || "smart", model_profile: body.model_profile || "quality", verifier: body.verifier || "default", metadata: body.metadata || {} };
      recipes.push(imported);
      return jsonResponse(imported);
    }
    return jsonResponse({ default_recipe_id: "recipe_general_loop", recipes: recipes.map((recipe) => ({ recipe_id: recipe.recipe_id, name: recipe.name, goal: recipe.goal, context_provider: recipe.context_provider || "smart", model_profile: recipe.model_profile || "quality", verifier: recipe.verifier || "default", is_default: recipe.is_default })) });
  }
  if (url === "/api/recipes/recipe_custom/export" && method === "GET") {
    requests.exported.push(url);
    return jsonResponse({ ...recipes.find((recipe) => recipe.recipe_id === "recipe_custom"), exported_from: "Loopwright" });
  }
  if (url === "/api/recipes/recipe_custom" && method === "DELETE") {
    requests.deleted.push(url);
    const index = recipes.findIndex((recipe) => recipe.recipe_id === "recipe_custom");
    if (index >= 0) {
      recipes.splice(index, 1);
    }
    return jsonResponse({ deleted: true, recipe_id: "recipe_custom" });
  }
  if (url.startsWith("/api/recipes/") && method === "GET") {
    const id = decodeURIComponent(url.slice("/api/recipes/".length));
    const recipe = recipes.find((item) => item.recipe_id === id);
    if (recipe) {
      return jsonResponse(recipe);
    }
  }
  if (url === "/api/threads" && method === "GET") {
    return jsonResponse({ threads: [createThreadPayload("thread_recipe_flow")] });
  }
  if (url.startsWith("/api/threads/") && method === "GET") {
    return jsonResponse(createThreadPayload("thread_recipe_flow"));
  }
  throw new Error(`unexpected fetch ${url}`);
};
await importFreshApp();
await tick();
assert.equal(dom["recipe-select"].value, "recipe_custom");
assert.equal(dom["recipe-instructions"].value, "Be sharp.");
await dom["recipe-export"].dispatch("click");
assert.deepEqual(requests.exported, ["/api/recipes/recipe_custom/export"]);
assert.equal(globalThis.__downloads.length, 1);
assert.equal(globalThis.__downloads[0].download, "recipe_custom.json");
assert.equal(globalThis.__downloads[0].href, "blob:test-1");
assert.deepEqual(globalThis.__revokedObjectUrls, ["blob:test-1"]);
const exportedRecipe = JSON.parse(await globalThis.__objectUrls[0].blob.text());
assert.equal(exportedRecipe.recipe_id, "recipe_custom");
assert.equal(exportedRecipe.instructions, "Be sharp.");
assert.deepEqual(exportedRecipe.success_criteria, ["Names risk."]);
assert.deepEqual(exportedRecipe.metadata, { source: "test" });
assert.equal(exportedRecipe.exported_from, "Loopwright");
assert.ok(dom["recipe-status"].textContent.includes("Exported Custom reviewer"));
await dom["recipe-delete"].dispatch("click");
assert.equal(deleteConfirmed, true);
assert.deepEqual(requests.deleted, ["/api/recipes/recipe_custom"]);
assert.equal(dom["recipe-select"].value, "recipe_general_loop");
assert.ok(dom["recipe-status"].textContent.includes("Deleted Custom reviewer"));
dom["recipe-import"].files = [{
  async text() {
    return JSON.stringify({ recipe_id: "recipe_imported", name: "Imported recipe", goal: "Use imported loop guidance.", instructions: "Preserve imported instructions.", success_criteria: ["Imported criterion."], stop_condition: "Stop after imported answer.", metadata: { imported: true } });
  },
}];
await dom["recipe-import"].dispatch("change");
assert.equal(requests.imported.length, 1);
assert.equal(requests.imported[0].recipe_id, "recipe_imported");
assert.deepEqual(requests.imported[0].success_criteria, ["Imported criterion."]);
assert.equal(dom["recipe-select"].value, "recipe_imported");
assert.equal(dom["recipe-instructions"].value, "Preserve imported instructions.");
assert.ok(dom["recipe-status"].textContent.includes("Imported Imported recipe"));
assert.equal(dom["recipe-import"].value, "");
'''
    )

def test_config_and_status_endpoints_return_runtime_contract():
    fake_qa = FakeQA()
    client = TestClient(web_app.create_app(fake_qa))

    config = client.get("/api/config").json()
    status = client.get("/api/status").json()

    assert config["title"] == "Loopwright"
    assert {"label": "Auto", "value": "auto"} in config["text_encodings"]
    assert status["active_document"] is None
    assert status["backend"] == "mock"
    assert status["ready_for_queries"] is False
    assert status["readiness_scope"] == "retrieval_pipeline"
    assert status["direct_query_available"] is True
    assert status["query_mode"] == "direct"
    assert status["context_optional"] is True


def test_upload_document_indexes_context_and_reports_status():
    fake_qa = FakeQA()
    client = TestClient(web_app.create_app(fake_qa))

    response = client.post(
        "/api/documents",
        data={"text_encoding": "cp1251"},
        files={"file": ("demo.txt", b"Project Phoenix", "text/plain")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "processed in mock mode" in payload["message"]
    assert payload["status"]["active_document"] == "demo.txt"
    assert payload["status"]["text_encoding_mode"] == "cp1251"
    assert fake_qa.text_encoding == "cp1251"
    assert fake_qa.uploaded_text == "Project Phoenix"


def test_upload_document_is_scoped_to_thread_runtime():
    engines = {}
    store = ThreadStore.in_memory()
    store.create_thread(thread_id="thread_alpha")
    store.create_thread(thread_id="thread_beta")

    def engine_factory(session_id):
        engine = FakeQA()
        engines[session_id] = engine
        return engine

    client = TestClient(
        web_app.create_app(
            engine_factory=engine_factory,
            thread_store=store,
        )
    )

    response = client.post(
        "/api/documents",
        data={"text_encoding": "auto", "session_id": "thread_alpha"},
        files={"file": ("alpha.txt", b"Alpha private launch date", "text/plain")},
    )

    assert response.status_code == 200
    assert engines["thread_alpha"].current_document_name == "alpha.txt"
    assert client.get(
        "/api/status",
        headers={"x-ai-loop-session-id": "thread_alpha"},
    ).json()["active_document"] == "alpha.txt"
    assert client.get(
        "/api/status",
        headers={"x-ai-loop-session-id": "thread_beta"},
    ).json()["active_document"] is None

    query_response = client.post(
        "/api/query",
        json={"message": "What is attached?", "session_id": "thread_beta"},
    )

    assert query_response.status_code == 200
    assert engines["thread_beta"] is not engines["thread_alpha"]
    assert engines["thread_beta"].current_document_name is None
    assert engines["thread_beta"].last_query_session_id == "thread_beta"
    assert query_response.json()["summary"]["document"] is None


def test_upload_document_rejects_missing_explicit_thread():
    engines = {}

    def engine_factory(session_id):
        engine = FakeQA()
        engines[session_id] = engine
        return engine

    client = TestClient(
        web_app.create_app(
            engine_factory=engine_factory,
            thread_store=ThreadStore.in_memory(),
        )
    )

    response = client.post(
        "/api/documents",
        data={"text_encoding": "auto", "session_id": "thread_missing"},
        files={"file": ("alpha.txt", b"Alpha private launch date", "text/plain")},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Thread not found."
    assert engines == {}


def test_upload_document_generation_change_does_not_drop_runtime():
    store = ThreadStore.in_memory()
    store.create_thread(thread_id="thread_alpha")
    engines = {}

    class ClearDuringReplacementQA(FakeQA):
        def process_document(self, document_path, text_encoding=None):
            status = super().process_document(document_path, text_encoding=text_encoding)
            if Path(document_path).name == "new.txt":
                assert store.clear_thread("thread_alpha") is not None
            return status

    def engine_factory(session_id):
        engine = ClearDuringReplacementQA()
        engines[session_id] = engine
        return engine

    client = TestClient(
        web_app.create_app(
            engine_factory=engine_factory,
            thread_store=store,
        )
    )

    old_response = client.post(
        "/api/documents",
        data={"text_encoding": "auto", "session_id": "thread_alpha"},
        files={"file": ("old.txt", b"Old active document", "text/plain")},
    )
    assert old_response.status_code == 200
    assert engines["thread_alpha"].current_document_name == "old.txt"

    replacement_response = client.post(
        "/api/documents",
        data={"text_encoding": "auto", "session_id": "thread_alpha"},
        files={"file": ("new.txt", b"Replacement document", "text/plain")},
    )

    assert replacement_response.status_code == 200
    assert replacement_response.json()["status"]["active_document"] == "new.txt"
    assert engines["thread_alpha"].current_document_name == "new.txt"
    status = client.get(
        "/api/status",
        headers={"x-ai-loop-session-id": "thread_alpha"},
    )
    assert status.status_code == 200
    assert status.json()["active_document"] == "new.txt"


def test_upload_document_rejects_deleted_thread_after_file_write(monkeypatch):
    store = ThreadStore.in_memory()
    store.create_thread(thread_id="thread_alpha")
    engines = {}

    def engine_factory(session_id):
        engine = FakeQA()
        engines[session_id] = engine
        return engine

    async def delete_thread_during_write(upload, upload_path):
        upload_path.write_bytes(await upload.read())
        assert store.delete_thread("thread_alpha") is True

    monkeypatch.setattr(web_app, "write_upload_file", delete_thread_during_write)
    client = TestClient(
        web_app.create_app(
            engine_factory=engine_factory,
            thread_store=store,
        )
    )

    response = client.post(
        "/api/documents",
        data={"text_encoding": "auto", "session_id": "thread_alpha"},
        files={"file": ("alpha.txt", b"Alpha private launch date", "text/plain")},
    )

    assert response.status_code == 409
    assert "Thread changed before document upload completed" in response.json()["detail"]
    assert "thread_alpha" not in engines
    assert client.get("/api/threads/thread_alpha").status_code == 404
    assert client.get(
        "/api/status",
        headers={"x-ai-loop-session-id": "thread_alpha"},
    ).status_code == 404


def test_upload_document_drops_runtime_when_thread_deleted_after_indexing():
    store = ThreadStore.in_memory()
    store.create_thread(thread_id="thread_alpha")
    engines = {}

    class DeletingAfterIndexQA(FakeQA):
        def process_document(self, document_path, text_encoding=None):
            status = super().process_document(document_path, text_encoding=text_encoding)
            assert store.delete_thread("thread_alpha") is True
            return status

    def engine_factory(session_id):
        engine = DeletingAfterIndexQA()
        engines[session_id] = engine
        return engine

    client = TestClient(
        web_app.create_app(
            engine_factory=engine_factory,
            thread_store=store,
        )
    )

    response = client.post(
        "/api/documents",
        data={"text_encoding": "auto", "session_id": "thread_alpha"},
        files={"file": ("alpha.txt", b"Alpha private launch date", "text/plain")},
    )

    assert response.status_code == 409
    assert "Thread changed before document upload completed" in response.json()["detail"]
    assert engines["thread_alpha"].current_document_name == "alpha.txt"
    assert client.get("/api/threads/thread_alpha").status_code == 404

    store.create_thread(thread_id="thread_alpha")
    status = client.get(
        "/api/status",
        headers={"x-ai-loop-session-id": "thread_alpha"},
    )

    assert status.status_code == 200
    assert status.json()["active_document"] is None
    assert engines["thread_alpha"].current_document_name is None


def test_upload_document_rejects_unknown_encoding():
    client = TestClient(web_app.create_app(FakeQA()))

    response = client.post(
        "/api/documents",
        data={"text_encoding": "not-real"},
        files={"file": ("demo.txt", b"Project Phoenix", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported text encoding."


def test_upload_document_rejects_oversized_body_before_processing(monkeypatch):
    fake_qa = FakeQA()
    client = TestClient(web_app.create_app(fake_qa))
    monkeypatch.setattr(web_app, "MAX_DOCUMENT_BYTES", 8)

    response = client.post(
        "/api/documents",
        data={"text_encoding": "auto"},
        files={"file": ("too-large.txt", b"more than eight", "text/plain")},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "Uploaded document exceeds the 25 MB limit."
    assert getattr(fake_qa, "status_calls", 0) == 0
    assert not hasattr(fake_qa, "document_path")


def test_upload_document_rejects_oversized_request_before_status(monkeypatch):
    fake_qa = FakeQA()
    client = TestClient(web_app.create_app(fake_qa))
    monkeypatch.setattr(web_app, "MAX_DOCUMENT_BYTES", 8)
    monkeypatch.setattr(web_app, "MAX_MULTIPART_OVERHEAD_BYTES", 0)

    response = client.post(
        "/api/documents",
        data={"text_encoding": "auto"},
        files={"file": ("too-large.txt", b"too-large", "text/plain")},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "Uploaded document exceeds the 25 MB limit."
    assert getattr(fake_qa, "status_calls", 0) == 0
    assert not hasattr(fake_qa, "document_path")


def test_upload_document_rejects_missing_content_length_before_parsing():
    fake_qa = FakeQA()
    api = web_app.create_app(fake_qa)
    sent_messages = []

    async def receive():
        return {
            "type": "http.request",
            "body": b"unparsed multipart body",
            "more_body": False,
        }

    async def send(message):
        sent_messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/documents",
        "raw_path": b"/api/documents",
        "query_string": b"",
        "headers": [],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "root_path": "",
    }

    asyncio.run(api(scope, receive, send))

    response_start = next(
        message for message in sent_messages if message["type"] == "http.response.start"
    )
    response_body = b"".join(
        message.get("body", b"")
        for message in sent_messages
        if message["type"] == "http.response.body"
    )
    assert response_start["status"] == 411
    assert json.loads(response_body)["detail"] == (
        "Content-Length is required for document uploads."
    )
    assert getattr(fake_qa, "status_calls", 0) == 0
    assert not hasattr(fake_qa, "document_path")


def test_upload_document_rejects_directory_like_filenames():
    fake_qa = FakeQA()
    client = TestClient(web_app.create_app(fake_qa))

    for filename in (".", ".."):
        response = client.post(
            "/api/documents",
            data={"text_encoding": "auto"},
            files={"file": (filename, b"content", "text/plain")},
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "Invalid upload filename."
    assert getattr(fake_qa, "status_calls", 0) == 0
    assert not hasattr(fake_qa, "document_path")


def test_safe_upload_name_rejects_control_and_overlong_names():
    for filename in (
        "bad\0.txt",
        "bad\n.txt",
        "bad\x7f.txt",
        "\nbad.txt",
        "bad.txt\n",
        "\tbad.txt",
        "a" * 181,
    ):
        with pytest.raises(web_app.HTTPException) as exc_info:
            web_app.safe_upload_name(filename)

        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "Invalid upload filename."


def test_upload_document_rejects_overlong_filename_before_filesystem():
    fake_qa = FakeQA()
    client = TestClient(web_app.create_app(fake_qa))

    response = client.post(
        "/api/documents",
        data={"text_encoding": "auto"},
        files={"file": ("a" * 181, b"content", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid upload filename."
    assert getattr(fake_qa, "status_calls", 0) == 0
    assert not hasattr(fake_qa, "document_path")


def test_visually_hidden_label_remains_accessible_to_assistive_tech():
    css = (REPO_ROOT / "src" / "web_static" / "styles.css").read_text(
        encoding="utf-8"
    )

    assert ".hidden {\n  display: none;\n}" in css
    assert ".visually-hidden {" in css
    assert "clip-path: inset(50%);" in css
    assert "position: absolute;" in css
    assert ".hidden,\n.visually-hidden" not in css


def test_upload_document_reports_processing_failure_without_losing_active_status():
    fake_qa = FakeQA()
    fake_qa.current_document_name = "good.txt"
    fake_qa.latest_processing_report = processed_report()

    def fail_process_document(document_path, text_encoding=None):
        fake_qa.text_encoding = text_encoding
        fake_qa.latest_processing_report = DocumentProcessingReport(
            attempted_document_name="bad.txt",
            active_document_name="good.txt",
            success=False,
            phase="load",
            file_extension=".txt",
            chunk_count=0,
            truncated=False,
            max_chunk_limit=2000,
            text_encoding_mode=text_encoding or "auto",
            backend="mock",
            model_label="MockLLM (explicit demo)",
            error_message="Could not decode text document",
        )
        raise DocumentProcessingError(
            "Error loading document: Could not decode text document",
            fake_qa.status(),
        )

    fake_qa.process_document = fail_process_document
    client = TestClient(web_app.create_app(fake_qa))

    response = client.post(
        "/api/documents",
        data={"text_encoding": "auto"},
        files={"file": ("bad.txt", b"bad", "text/plain")},
    )

    assert response.status_code == 400
    payload = response.json()
    assert "failed during load" in payload["message"]
    assert "Active file remains good.txt" in payload["message"]
    assert payload["status"]["active_document"] == "good.txt"
    assert payload["status"]["last_attempted_document"] == "bad.txt"
    assert payload["status"]["last_error"] == "Could not decode text document"


def test_upload_document_unexpected_error_uses_pre_upload_status():
    class UnexpectedFailureQA(FakeQA):
        def __init__(self):
            self.current_document_name = "good.txt"
            self.latest_processing_report = processed_report()
            self.status_calls = 0

        def process_document(self, document_path, text_encoding=None):
            self.current_document_name = "mutated.txt"
            self.latest_processing_report = processed_report(
                document_name="mutated.txt"
            )
            raise RuntimeError("unexpected boom")

    fake_qa = UnexpectedFailureQA()
    client = TestClient(web_app.create_app(fake_qa))

    response = client.post(
        "/api/documents",
        data={"text_encoding": "auto"},
        files={"file": ("bad.txt", b"bad", "text/plain")},
    )

    assert response.status_code == 500
    payload = response.json()
    assert fake_qa.status_calls == 1
    assert "failed during unexpected" in payload["message"]
    assert "Active file remains good.txt" in payload["message"]
    assert payload["status"]["active_document"] == "good.txt"
    assert payload["status"]["phase"] == "unexpected"
    assert payload["status"]["last_error"] == "unexpected boom"


def test_query_endpoint_returns_visible_loop_payload():
    fake_qa = FakeQA()
    fake_qa.current_document_name = "demo.txt"
    client = TestClient(web_app.create_app(fake_qa))

    response = client.post(
        "/api/query",
        json={"message": "What is Project Phoenix?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "Project Phoenix" in payload["answer"]
    assert payload["timeline"]["rows"][0]["step"] == "Input"
    assert payload["timeline"]["rows"][0]["signals"] == "-"
    assert payload["timeline"]["rows"][1]["phase"] == "Retrieve"
    assert payload["timeline"]["rows"][1]["signals"] == "-"
    assert payload["summary"]["document"] is None
    assert payload["summary"]["final_decision"] == "not_verified"
    assert payload["summary"]["conversation_context_count"] == 0
    assert payload["summary"]["semantic_memory_count"] == 0
    assert payload["summary"]["semantic_memory_status"] == "not_requested"
    assert payload["summary"]["recipe_id"] is None
    assert payload["summary"]["recipe_name"] is None
    assert payload["recipe"]["recipe_id"] == "recipe_general_loop"
    assert payload["trace"]["question"] == "What is Project Phoenix?"
    citation = payload["trace"]["citations"][0]
    assert citation["evidence_id"].startswith("evidence_")
    assert citation["source"] == citation["evidence_id"]
    assert citation["excerpt"] is None
    assert payload["trace"]["model_thinking"] == {
        "available": True,
        "redacted": False,
        "label": "Model Thinking (unverified)",
        "content": "I matched Project Phoenix against citation [1].",
        "note": (
            "Model-emitted thinking is useful for debugging the loop, but it is "
            "not verified evidence."
        ),
    }


def test_live_query_response_fails_closed_when_sibling_result_disagrees():
    projected_answer = "The canonical public answer."
    foreign_answer = "SECRET_FOREIGN_ANSWER"
    foreign_question = "SECRET_FOREIGN_QUESTION"
    foreign_thinking = "SECRET_FOREIGN_THINKING"
    foreign_excerpt = "SECRET_FOREIGN_CITATION"
    report_started_at = utc_now()
    evidence = (
        EvidenceReference.from_source(
            citation_id=1,
            provider="document",
            source_identity="canonical.txt",
            page=None,
            chunk_index=0,
            excerpt="Canonical evidence.",
        ),
    )
    report = LoopReport(
        run=LoopRun(
            run_id="run_bound_live_response",
            session_id="thread_bound_live_response",
            user_input="Canonical question",
            context_provider="document",
            backend="mock",
                model_label="MockLLM (explicit demo)",
                started_at=report_started_at,
                completed_at=report_started_at,
                steps=visible_not_verified_steps(
                answer=projected_answer,
                evidence=evidence,
                started_at=report_started_at,
                prefix="bound_live",
            ),
            evidence=evidence,
            final_decision=LoopDecision.NOT_VERIFIED,
            final_answer=projected_answer,
        )
    )
    result = QueryResult(
        answer=foreign_answer,
        trace=AnswerTrace(
            question=foreign_question,
            document_name="foreign.txt",
            backend="mock",
            model_label="MockLLM (explicit demo)",
            retrieved_chunk_count=1,
            citations=[
                AnswerCitation(
                    citation_id=99,
                    source_name="foreign.txt",
                    page=None,
                    chunk_index=0,
                    excerpt=foreign_excerpt,
                )
            ],
            self_check=AnswerSelfCheck(
                outcome="not_verified",
                reasons=[foreign_excerpt],
            ),
            model_thinking=foreign_thinking,
        ),
        loop_report=report,
    )

    payload = query_response_dict(result)
    serialized = json.dumps(payload)

    assert payload["answer"] == projected_answer
    assert payload["trace"]["answer"] == projected_answer
    assert payload["trace"]["question"] is None
    assert payload["trace"]["citations"][0]["id"] == 1
    assert payload["trace"]["citations"][0]["excerpt"] is None
    assert payload["trace"]["model_thinking"]["available"] is False
    assert payload["trace"]["model_thinking"]["content"] is None
    for secret in (
        foreign_answer,
        foreign_question,
        foreign_thinking,
        foreign_excerpt,
        "foreign.txt",
    ):
        assert secret not in serialized


def test_live_query_response_rejects_foreign_thinking_when_checked_fields_match():
    canonical_thinking = "Canonical thinking owned by run A."
    foreign_thinking = "SECRET_FOREIGN_THINKING_FROM_RUN_B"
    foreign_excerpt = "SECRET_FOREIGN_CITATION_FROM_RUN_B"
    foreign_self_check = "SECRET_FOREIGN_SELF_CHECK_FROM_RUN_B"
    report_started_at = utc_now()
    evidence = (
        EvidenceReference.from_source(
            citation_id=1,
            provider="document",
            source_identity="canonical.txt",
            page=None,
            chunk_index=0,
            excerpt="Canonical evidence.",
        ),
    )
    report = LoopReport(
        run=LoopRun(
            run_id="run_bound_thinking",
            session_id="thread_bound_thinking",
            user_input="Repeated question",
            context_provider="document",
            backend="mock",
                model_label="MockLLM (explicit demo)",
                started_at=report_started_at,
                completed_at=report_started_at,
                steps=visible_not_verified_steps(
                answer="Repeated answer",
                evidence=evidence,
                started_at=report_started_at,
                prefix="bound_thinking",
            ),
            evidence=evidence,
            final_decision=LoopDecision.NOT_VERIFIED,
            final_answer="Repeated answer",
            metadata={
                MODEL_THINKING_SHA256_METADATA_KEY: hashlib.sha256(
                    canonical_thinking.encode("utf-8")
                ).hexdigest(),
            },
        )
    )
    result = QueryResult(
        answer="Repeated answer",
        trace=AnswerTrace(
            question="Repeated question",
            document_name="foreign.txt",
            backend="mock",
            model_label="MockLLM (explicit demo)",
            retrieved_chunk_count=1,
            citations=[
                AnswerCitation(
                    citation_id=1,
                    source_name="foreign.txt",
                    page=99,
                    chunk_index=99,
                    excerpt=foreign_excerpt,
                )
            ],
            self_check=AnswerSelfCheck(
                outcome="supported",
                reasons=[foreign_self_check],
            ),
            model_thinking=foreign_thinking,
        ),
        loop_report=report,
    )

    payload = query_response_dict(result)
    serialized = json.dumps(payload)

    assert payload["answer"] == "Repeated answer"
    assert payload["trace"]["question"] == "Repeated question"
    assert payload["trace"]["citations"][0]["id"] == 1
    assert payload["trace"]["citations"][0]["excerpt"] is None
    assert payload["trace"]["self_check"] == {
        "outcome": "not_verified",
        "reasons": [],
        "retry_attempted": False,
    }
    assert payload["trace"]["model_thinking"]["available"] is False
    assert payload["trace"]["model_thinking"]["content"] is None
    for secret in (
        foreign_thinking,
        foreign_excerpt,
        foreign_self_check,
        "foreign.txt",
    ):
        assert secret not in serialized


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {MODEL_THINKING_SHA256_METADATA_KEY: None},
        {MODEL_THINKING_SHA256_METADATA_KEY: "A" * 64},
        {MODEL_THINKING_SHA256_METADATA_KEY: "0" * 63},
        {MODEL_THINKING_SHA256_METADATA_KEY: 7},
    ],
)
def test_live_query_response_requires_valid_lowercase_thinking_digest(metadata):
    thinking = "SECRET_UNBOUND_MODEL_THINKING"
    report_started_at = utc_now()
    report = LoopReport(
        run=LoopRun(
            run_id="run_unbound_thinking",
            session_id="thread_unbound_thinking",
            user_input="Question",
            context_provider="none",
            backend="mock",
                model_label="MockLLM (explicit demo)",
                started_at=report_started_at,
                completed_at=report_started_at,
                steps=visible_not_verified_steps(
                answer="Answer",
                evidence=(),
                started_at=report_started_at,
                prefix="unbound_thinking",
            ),
            final_decision=LoopDecision.NOT_VERIFIED,
            final_answer="Answer",
            metadata=metadata,
        )
    )
    result = QueryResult(
        answer="Answer",
        trace=AnswerTrace(
            question="Question",
            document_name=None,
            backend="mock",
            model_label="MockLLM (explicit demo)",
            retrieved_chunk_count=0,
            citations=[],
            model_thinking=thinking,
        ),
        loop_report=report,
    )

    payload = query_response_dict(result)

    assert payload["trace"]["model_thinking"]["available"] is False
    assert payload["trace"]["model_thinking"]["content"] is None
    assert thinking not in json.dumps(payload)


def test_query_response_without_loop_report_fails_closed():
    secrets = {
        "answer": "SECRET_MISSING_REPORT_ANSWER",
        "question": "SECRET_MISSING_REPORT_QUESTION",
        "citation": "SECRET_MISSING_REPORT_CITATION",
        "self_check": "SECRET_MISSING_REPORT_SELF_CHECK",
        "thinking": "SECRET_MISSING_REPORT_THINKING",
    }
    result = QueryResult(
        answer=secrets["answer"],
        trace=AnswerTrace(
            question=secrets["question"],
            document_name="secret.txt",
            backend="mock",
            model_label="MockLLM (explicit demo)",
            retrieved_chunk_count=1,
            citations=[
                AnswerCitation(
                    citation_id=1,
                    source_name="secret.txt",
                    page=None,
                    chunk_index=0,
                    excerpt=secrets["citation"],
                )
            ],
            self_check=AnswerSelfCheck(
                outcome="supported",
                reasons=[secrets["self_check"]],
            ),
            error_message="trace_unavailable",
            model_thinking=secrets["thinking"],
        ),
        loop_report=None,
    )

    payload = query_response_dict(result)
    serialized = json.dumps(payload)

    assert payload["answer"] is None
    assert payload["trace"]["answer"] is None
    assert payload["trace"]["question"] is None
    assert payload["trace"]["citations"] == []
    assert payload["trace"]["self_check"] is None
    assert payload["trace"]["model_thinking"]["available"] is False
    assert payload["trace"]["model_thinking"]["content"] is None
    for secret in secrets.values():
        assert secret not in serialized


def test_query_endpoint_passes_session_id_to_loop_runtime():
    fake_qa = FakeQA()
    client = TestClient(web_app.create_app(fake_qa))

    response = client.post(
        "/api/query",
        json={"message": "What is Project Phoenix?", "session_id": "thread_alpha"},
    )

    assert response.status_code == 200
    assert fake_qa.last_query_session_id == "thread_alpha"
    assert fake_qa.last_loop_recipe["recipe_id"] == "recipe_general_loop"
    assert response.json()["trace"]["loop_report"]["run"]["session_id"] == (
        "thread_alpha"
    )


def test_query_endpoint_passes_context_provider_to_loop_runtime():
    fake_qa = FakeQA()
    client = TestClient(web_app.create_app(fake_qa))

    response = client.post(
        "/api/query",
        json={
            "message": "What changed today?",
            "session_id": "thread_web",
            "context_provider": "web",
        },
    )

    assert response.status_code == 200
    assert fake_qa.last_query_session_id == "thread_web"
    assert fake_qa.last_context_provider == "web"


def test_query_endpoint_applies_selected_loop_recipe():
    fake_qa = FakeQA()
    store = ThreadStore.in_memory()
    recipe = store.create_recipe(
        recipe_id="recipe_custom",
        name="Custom recipe",
        goal="Answer with a custom tone.",
        instructions="Be direct.",
        success_criteria=("Uses the selected recipe.",),
    )
    client = TestClient(web_app.create_app(fake_qa, thread_store=store))

    response = client.post(
        "/api/query",
        json={
            "message": "What is Project Phoenix?",
            "session_id": "thread_alpha",
            "recipe_id": recipe.recipe_id,
        },
    )

    payload = response.json()
    thread = client.get("/api/threads/thread_alpha").json()
    assert response.status_code == 200
    assert fake_qa.last_loop_recipe["recipe_id"] == "recipe_custom"
    assert fake_qa.last_loop_recipe["success_criteria"] == [
        "Uses the selected recipe."
    ]
    assert payload["recipe"]["recipe_id"] == "recipe_custom"
    assert payload["summary"]["recipe_name"] is None
    assert "recipe_id" not in thread["loop_runs"][0]
    assert "recipe_name" not in thread["loop_runs"][0]


def test_query_endpoint_refreshes_stale_builtin_default_recipe():
    fake_qa = FakeQA()
    store = ThreadStore.in_memory()
    stale_default = LoopRecipe(
        recipe_id=DEFAULT_LOOP_RECIPE_ID,
        name="General assistant loop",
        description="Default evidence loop behavior.",
        goal="Answer using indexed context.",
        instructions="Use indexed context when present.",
        success_criteria=("Uses indexed context.",),
        context_provider="auto",
        metadata={"built_in": True},
    )
    with store._lock:
        store._insert_recipe(stale_default)
        store._conn.commit()
    client = TestClient(web_app.create_app(fake_qa, thread_store=store))

    response = client.post(
        "/api/query",
        json={"message": "Who is Jackie Chan?"},
    )

    assert response.status_code == 200
    assert fake_qa.last_loop_recipe["recipe_id"] == DEFAULT_LOOP_RECIPE_ID
    assert fake_qa.last_loop_recipe["context_provider"] == "smart"
    assert "web evidence" in fake_qa.last_loop_recipe["instructions"]
    assert "indexed context" not in fake_qa.last_loop_recipe["goal"].lower()


def test_thread_endpoints_create_list_get_rename_and_delete():
    fake_qa = FakeQA()
    store = ThreadStore.in_memory()
    client = TestClient(web_app.create_app(fake_qa, thread_store=store))

    initial = client.get("/api/threads")
    assert initial.status_code == 200
    assert len(initial.json()["threads"]) == 1

    created = client.post("/api/threads", json={"title": "Research thread"})
    assert created.status_code == 200
    thread_id = created.json()["id"]
    assert thread_id.startswith("thread_")
    assert created.json()["title"] == "Research thread"

    listed = client.get("/api/threads")
    assert listed.status_code == 200
    assert any(thread["id"] == thread_id for thread in listed.json()["threads"])

    fetched = client.get(f"/api/threads/{thread_id}")
    assert fetched.status_code == 200
    assert fetched.json()["messages"] == []

    renamed = client.patch(
        f"/api/threads/{thread_id}",
        json={"title": "Renamed thread"},
    )
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Renamed thread"

    deleted = client.delete(f"/api/threads/{thread_id}")
    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": True, "thread_id": thread_id}
    assert fake_qa.cleared_loop_session_id == thread_id
    assert client.get(f"/api/threads/{thread_id}").status_code == 404


def test_recipe_endpoints_manage_loop_recipes():
    client = TestClient(web_app.create_app(FakeQA(), thread_store=ThreadStore.in_memory()))

    listed = client.get("/api/recipes")
    assert listed.status_code == 200
    assert listed.json()["default_recipe_id"] == "recipe_general_loop"
    assert listed.json()["recipes"][0]["recipe_id"] == "recipe_general_loop"

    created = client.post(
        "/api/recipes",
        json={
            "recipe_id": "recipe_strict_reviewer",
            "name": "Strict reviewer",
            "description": "Review answers sharply.",
            "goal": "Find weak assumptions before final answer.",
            "instructions": "Call out uncertainty.",
            "success_criteria": ["Risks first.", "No vague praise."],
            "stop_condition": "Stop after a clear verdict.",
            "context_provider": "smart",
            "model_profile": "quality",
            "verifier": "human_review",
        },
    )
    assert created.status_code == 200
    recipe_id = created.json()["recipe_id"]
    assert recipe_id == "recipe_strict_reviewer"

    fetched = client.get(f"/api/recipes/{recipe_id}")
    assert fetched.status_code == 200
    assert fetched.json()["success_criteria"] == ["Risks first.", "No vague praise."]

    exported = client.get(f"/api/recipes/{recipe_id}/export")
    assert exported.status_code == 200
    assert exported.json()["recipe_id"] == recipe_id
    assert exported.json()["exported_from"] == "Loopwright"

    duplicate = client.post(
        "/api/recipes",
        json={
            "recipe_id": recipe_id,
            "name": "Duplicate",
            "goal": "Should fail.",
        },
    )
    assert duplicate.status_code == 400
    assert "already exists" in duplicate.json()["detail"]

    patched = client.patch(
        f"/api/recipes/{recipe_id}",
        json={"name": "Sharper reviewer", "success_criteria": ["No soft passes."]},
    )
    assert patched.status_code == 200
    assert patched.json()["name"] == "Sharper reviewer"
    assert patched.json()["success_criteria"] == ["No soft passes."]

    assert client.delete("/api/recipes/recipe_general_loop").status_code == 404
    deleted = client.delete(f"/api/recipes/{recipe_id}")
    assert deleted.status_code == 200
    assert client.get(f"/api/recipes/{recipe_id}").status_code == 404


def test_durable_public_payload_preserves_typed_memory_provenance_and_unknowns():
    query_result = FakeQA().query_with_trace(
        "What is Project Phoenix?",
        session_id="thread_memory_unknown",
        conversation_history=[{"role": "user", "content": "Earlier turn"}],
        semantic_memory=[
            {
                "message_id": 7,
                "role": "assistant",
                "content": "Recalled turn",
                "score": 0.9,
            }
        ],
        semantic_memory_status="retrieved",
    )

    payload = public_loop_payload_from_report(query_result.loop_report.to_public_dict())

    assert payload["summary"]["conversation_context_count"] == 1
    assert payload["summary"]["semantic_memory_count"] == 1
    assert payload["summary"]["semantic_memory_status"] == "retrieved"

    legacy_report = replace(
        query_result.loop_report,
        run=replace(query_result.loop_report.run, metadata={}),
    )
    legacy_payload = public_loop_payload_from_report(legacy_report.to_public_dict())

    assert legacy_payload["summary"]["conversation_context_count"] is None
    assert legacy_payload["summary"]["semantic_memory_count"] is None
    assert legacy_payload["summary"]["semantic_memory_status"] is None


def test_query_endpoint_derives_latest_from_canonical_run_not_message_payload():
    fake_qa = FakeQA()
    store = ThreadStore.in_memory()
    client = TestClient(web_app.create_app(fake_qa, thread_store=store))

    response = client.post(
        "/api/query",
        json={"message": "What is Project Phoenix?", "session_id": "thread_alpha"},
    )

    assert response.status_code == 200
    thread = client.get("/api/threads/thread_alpha").json()
    listed_thread = next(
        thread
        for thread in client.get("/api/threads").json()["threads"]
        if thread["id"] == "thread_alpha"
    )
    assert "latest" not in listed_thread
    assert thread["title"] == "What is Project Phoenix?"
    assert [message["role"] for message in thread["messages"]] == [
        "user",
        "assistant",
    ]
    assert thread["messages"][0]["content"] == "What is Project Phoenix?"
    assert "Project Phoenix" in thread["messages"][1]["content"]
    assert thread["messages"][1]["thinking"]["available"] is True
    assert thread["messages"][1]["loop_payload"] is None
    assert thread["latest"]["trace"]["loop_report"][
        "projection_schema_version"
    ] == "loop-public-report/v1"
    assert thread["latest"]["trace"]["loop_report"]["run"]["run_id"] == (
        "run_fake"
    )
    assert thread["latest"]["trace"]["question"] is None
    assert thread["latest"]["trace"]["citations"][0]["excerpt"] is None
    assert "demo.txt" not in json.dumps(thread["latest"])
    stored_assistant_payload = store._conn.execute(
        """
        SELECT loop_payload_json
        FROM messages
        WHERE thread_id = ? AND role = 'assistant'
        """,
        ("thread_alpha",),
    ).fetchone()
    assert stored_assistant_payload["loop_payload_json"] is None
    assert thread["memory_count"] == 2
    assert thread["loop_run_count"] == 1
    assert thread["loop_runs"][0]["run_id"] == "run_fake"
    assert thread["loop_runs"][0]["projection_status"] == "available"
    assert "recipe_id" not in thread["loop_runs"][0]
    response_payload = response.json()
    assert response_payload["run"]["run_id"] == "run_fake"
    assert response_payload["thread"]["memory_count"] == 2

    runs = client.get("/api/threads/thread_alpha/runs").json()
    run_detail = client.get("/api/threads/thread_alpha/runs/run_fake").json()
    assert runs["runs"][0]["run_id"] == "run_fake"
    assert run_detail["public"] is True
    assert run_detail["report"]["run"]["run_id"] == "run_fake"
    assert run_detail["report"]["run"]["session_id"] == "thread_alpha"
    assert "metadata" not in run_detail["report"]["run"]


@pytest.mark.parametrize("poisoned_raw_json", ('{"run":"legacy-shape"}', "not-json"))
def test_run_endpoints_quarantine_malformed_raw_without_serving_cached_public(
    poisoned_raw_json,
):
    store = ThreadStore.in_memory()
    client = TestClient(web_app.create_app(FakeQA(), thread_store=store))
    response = client.post(
        "/api/query",
        json={"message": "What happened?", "session_id": "thread_quarantine"},
    )
    assert response.status_code == 200
    run_id = response.json()["run"]["run_id"]
    with store._lock:
        store._conn.execute(
            """
            UPDATE loop_runs
            SET raw_report_json = ?, public_report_json = ?
            WHERE run_id = ?
            """,
            (
                poisoned_raw_json,
                json.dumps(
                    {
                        "run": {
                            "run_id": "run_foreign",
                            "session_id": "thread_foreign",
                            "final_answer": "CACHED_PUBLIC_SECRET",
                        }
                    }
                ),
                run_id,
            ),
        )
        store._conn.commit()

    listed = client.get("/api/threads/thread_quarantine/runs")
    thread = client.get("/api/threads/thread_quarantine")
    detail = client.get(f"/api/threads/thread_quarantine/runs/{run_id}")

    assert listed.status_code == 200
    assert thread.status_code == 200
    assert listed.json()["runs"] == [
        {
            "run_id": run_id,
            "thread_id": "thread_quarantine",
            "created_at": listed.json()["runs"][0]["created_at"],
            "projection_status": "quarantined",
            "quarantine_reason": "stored_loop_report_invalid",
        }
    ]
    assert detail.status_code == 409
    assert detail.json() == {
        "detail": "Stored loop run is quarantined and cannot be inspected."
    }
    assert "CACHED_PUBLIC_SECRET" not in listed.text
    assert "CACHED_PUBLIC_SECRET" not in thread.text
    assert "CACHED_PUBLIC_SECRET" not in detail.text


def test_run_detail_reprojects_valid_raw_and_ignores_poisoned_public_cache():
    store = ThreadStore.in_memory()
    client = TestClient(web_app.create_app(FakeQA(), thread_store=store))
    response = client.post(
        "/api/query",
        json={"message": "What happened?", "session_id": "thread_reproject"},
    )
    assert response.status_code == 200
    run_id = response.json()["run"]["run_id"]
    with store._lock:
        store._conn.execute(
            "UPDATE loop_runs SET public_report_json = ? WHERE run_id = ?",
            (
                json.dumps(
                    {
                        "projection_schema_version": "loop-public-report/v0",
                        "run": {
                            "run_id": "run_foreign",
                            "session_id": "thread_foreign",
                            "final_answer": "CACHED_PUBLIC_SECRET",
                        },
                    }
                ),
                run_id,
            ),
        )
        store._conn.commit()

    detail = client.get(f"/api/threads/thread_reproject/runs/{run_id}")

    assert detail.status_code == 200
    assert detail.json()["public"] is True
    assert detail.json()["run_id"] == run_id
    assert detail.json()["thread_id"] == "thread_reproject"
    assert detail.json()["report"]["projection_schema_version"] == (
        "loop-public-report/v1"
    )
    assert detail.json()["report"]["run"]["run_id"] == run_id
    assert detail.json()["report"]["run"]["session_id"] == "thread_reproject"
    assert "CACHED_PUBLIC_SECRET" not in detail.text


def test_query_endpoint_passes_recent_same_thread_history_to_runtime():
    fake_qa = FakeQA()
    store = ThreadStore.in_memory()
    store.create_thread(thread_id="thread_alpha")
    store.append_message(
        "thread_alpha",
        role="user",
        content="Do you know what dynamic programming is?",
    )
    store.append_message("thread_alpha", role="assistant", content="Yes.")
    store.create_thread(thread_id="thread_other")
    store.append_message(
        "thread_other",
        role="user",
        content="This should stay out of alpha.",
    )
    client = TestClient(web_app.create_app(fake_qa, thread_store=store))

    response = client.post(
        "/api/query",
        json={
            "message": "Please explain it in layman terms.",
            "session_id": "thread_alpha",
        },
    )

    assert response.status_code == 200
    assert fake_qa.last_query_session_id == "thread_alpha"
    assert fake_qa.last_conversation_history == [
        {
            "role": "user",
            "content": "Do you know what dynamic programming is?",
        },
        {"role": "assistant", "content": "Yes."},
    ]
    assert all(
        "stay out" not in entry["content"]
        for entry in fake_qa.last_conversation_history
    )
    assert response.json()["summary"]["conversation_context_count"] == 2


def test_query_endpoint_retrieves_older_semantic_thread_memory():
    fake_qa = FakeQA()
    store = ThreadStore.in_memory()
    store.create_thread(thread_id="thread_alpha")
    relevant = store.append_message(
        "thread_alpha",
        role="user",
        content="Dynamic programming means reusing answers to subproblems.",
    )
    unrelated = store.append_message(
        "thread_alpha",
        role="assistant",
        content="Banana bread uses ripe bananas.",
    )
    store.upsert_message_embedding(
        relevant,
        embedding_model="fake-memory",
        vector=[1.0, 0.0],
    )
    store.upsert_message_embedding(
        unrelated,
        embedding_model="fake-memory",
        vector=[0.0, 1.0],
    )
    for index in range(web_app.MAX_QUERY_HISTORY_MESSAGES):
        store.append_message(
            "thread_alpha",
            role="user" if index % 2 == 0 else "assistant",
            content=f"recent filler {index}",
        )
    other_thread_memory = store.append_message(
        "thread_other",
        role="user",
        content="Dynamic programming in another thread must not leak.",
    )
    store.upsert_message_embedding(
        other_thread_memory,
        embedding_model="fake-memory",
        vector=[1.0, 0.0],
    )
    client = TestClient(web_app.create_app(fake_qa, thread_store=store))

    response = client.post(
        "/api/query",
        json={
            "message": "Can you explain that algorithm simply?",
            "session_id": "thread_alpha",
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert fake_qa.last_semantic_memory_status == "retrieved"
    assert fake_qa.last_semantic_memory == [
        {
            "message_id": relevant.id,
            "role": "user",
            "content": "Dynamic programming means reusing answers to subproblems.",
            "score": 1.0,
        }
    ]
    assert all(
        "another thread" not in entry["content"]
        for entry in fake_qa.last_semantic_memory
    )
    assert all(
        "Dynamic programming means" not in entry["content"]
        for entry in fake_qa.last_conversation_history
    )
    memory_rows = [
        row for row in payload["timeline"]["rows"] if row["phase"] == "Context"
    ]
    assert memory_rows
    assert memory_rows[0]["signals"] == "-"
    assert payload["summary"]["semantic_memory_count"] == 1
    assert payload["summary"]["semantic_memory_status"] == "retrieved"
    assert payload["summary"]["conversation_context_count"] == (
        web_app.MAX_QUERY_HISTORY_MESSAGES
    )
    public_payload_text = json.dumps(payload)
    assert "reusing answers to subproblems" not in public_payload_text
    assert "another thread" not in public_payload_text
    indexed_memories = store.semantic_memories(
        "thread_alpha",
        embedding_model="fake-memory",
        query_vector=[1.0, 0.0],
        min_score=0.0,
    )
    assert any(
        memory.content == "Can you explain that algorithm simply?"
        for memory in indexed_memories
    )


@pytest.mark.parametrize(
    "failure_stage",
    ("runtime", "projection", "response_shaping", "persistence"),
)
def test_query_endpoint_rolls_back_pristine_auto_created_thread_on_failure(
    monkeypatch,
    failure_stage,
):
    fake_qa = FakeQA()
    store = ThreadStore.in_memory()

    if failure_stage == "runtime":
        def fail_runtime(*args, **kwargs):
            raise RuntimeError("backend unavailable")

        monkeypatch.setattr(fake_qa, "query_with_trace", fail_runtime)
    elif failure_stage == "projection":
        def fail_projection(self):
            raise RuntimeError("public projection unavailable")

        monkeypatch.setattr(LoopReport, "to_public_dict", fail_projection)
    elif failure_stage == "response_shaping":
        def fail_response_shaping(query_result):
            raise RuntimeError("response shaping unavailable")

        monkeypatch.setattr(
            web_app,
            "query_response_dict",
            fail_response_shaping,
        )
    else:
        def fail_persistence(*args, **kwargs):
            raise RuntimeError("persistence unavailable")

        monkeypatch.setattr(store, "append_turn", fail_persistence)

    client = TestClient(
        web_app.create_app(fake_qa, thread_store=store),
        raise_server_exceptions=False,
    )

    response = client.post(
        "/api/query",
        json={"message": "hello", "session_id": "thread_fail"},
    )

    assert response.status_code == 500
    assert client.get("/api/threads/thread_fail").status_code == 404


def test_query_failure_preserves_an_existing_empty_thread(monkeypatch):
    fake_qa = FakeQA()
    store = ThreadStore.in_memory()
    existing = store.create_thread(
        thread_id="thread_existing_failure",
        title="Keep this thread",
    )

    def fail_runtime(*args, **kwargs):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(fake_qa, "query_with_trace", fail_runtime)
    client = TestClient(
        web_app.create_app(fake_qa, thread_store=store),
        raise_server_exceptions=False,
    )

    response = client.post(
        "/api/query",
        json={"message": "hello", "session_id": existing.id},
    )

    assert response.status_code == 500
    thread = client.get(f"/api/threads/{existing.id}").json()
    assert thread["title"] == "Keep this thread"
    assert thread["messages"] == []
    assert thread["loop_run_count"] == 0


@pytest.mark.parametrize("mutation", ("rename", "append_message"))
def test_query_failure_preserves_concurrently_claimed_auto_created_thread(
    monkeypatch,
    mutation,
):
    fake_qa = FakeQA()
    store = ThreadStore.in_memory()

    def mutate_then_fail(*args, session_id="default", **kwargs):
        if mutation == "rename":
            store.rename_thread(session_id, "Claimed by the user")
        else:
            store.append_message(
                session_id,
                role="user",
                content="Concurrent durable message",
            )
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(fake_qa, "query_with_trace", mutate_then_fail)
    client = TestClient(
        web_app.create_app(fake_qa, thread_store=store),
        raise_server_exceptions=False,
    )

    response = client.post(
        "/api/query",
        json={"message": "hello", "session_id": "thread_claimed"},
    )

    assert response.status_code == 500
    thread = client.get("/api/threads/thread_claimed").json()
    if mutation == "rename":
        assert thread["title"] == "Claimed by the user"
        assert thread["messages"] == []
    else:
        assert [message["content"] for message in thread["messages"]] == [
            "Concurrent durable message"
        ]


def test_simultaneous_first_queries_serialize_creator_rollback_before_success():
    class FirstFailureQA(FakeQA):
        def __init__(self):
            self.call_lock = threading.Lock()
            self.call_count = 0
            self.first_started = threading.Event()
            self.release_first = threading.Event()
            self.clear_calls = []

        def query_with_trace(self, *args, **kwargs):
            with self.call_lock:
                self.call_count += 1
                call_number = self.call_count
            if call_number == 1:
                self.first_started.set()
                if not self.release_first.wait(timeout=5):
                    raise RuntimeError("timed out waiting to release first query")
                raise RuntimeError("first query failed")
            return super().query_with_trace(*args, **kwargs)

        def clear_loop_session(self, session_id="default"):
            self.clear_calls.append(session_id)

    fake_qa = FirstFailureQA()
    store = ThreadStore.in_memory()
    client = TestClient(
        web_app.create_app(fake_qa, thread_store=store),
        raise_server_exceptions=False,
    )
    responses = {}
    second_request_started = threading.Event()

    def run_first():
        responses["first"] = client.post(
            "/api/query",
            json={"message": "fail first", "session_id": "thread_race"},
        )

    def run_second():
        second_request_started.set()
        responses["second"] = client.post(
            "/api/query",
            json={"message": "then succeed", "session_id": "thread_race"},
        )

    first = threading.Thread(target=run_first)
    second = threading.Thread(target=run_second)
    first.start()
    assert fake_qa.first_started.wait(timeout=5)
    second.start()
    assert second_request_started.wait(timeout=5)
    fake_qa.release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert responses["first"].status_code == 500
    assert responses["second"].status_code == 200
    assert fake_qa.call_count == 2
    assert fake_qa.clear_calls == ["thread_race"]
    thread = client.get("/api/threads/thread_race").json()
    assert [message["content"] for message in thread["messages"]] == [
        "then succeed",
        "Project Phoenix is described in the indexed file.",
    ]
    assert thread["loop_run_count"] == 1


def test_existing_thread_persistence_failure_discards_only_runtime_run(monkeypatch):
    class DiscardTrackingQA(FakeQA):
        def __init__(self):
            self.discarded = []
            self.clear_calls = []

        def discard_loop_run(self, session_id, run_id):
            self.discarded.append((session_id, run_id))
            return True

        def clear_loop_session(self, session_id="default"):
            self.clear_calls.append(session_id)

    fake_qa = DiscardTrackingQA()
    store = ThreadStore.in_memory()
    store.create_thread(thread_id="thread_existing")

    def fail_persistence(*args, **kwargs):
        raise RuntimeError("persistence unavailable")

    monkeypatch.setattr(store, "append_turn", fail_persistence)
    client = TestClient(
        web_app.create_app(fake_qa, thread_store=store),
        raise_server_exceptions=False,
    )

    response = client.post(
        "/api/query",
        json={"message": "hello", "session_id": "thread_existing"},
    )

    assert response.status_code == 500
    assert fake_qa.discarded == [("thread_existing", "run_fake")]
    assert fake_qa.clear_calls == []
    thread = store.get_thread("thread_existing")
    assert thread is not None
    assert thread.messages == ()
    assert thread.loop_runs == ()


def test_retry_after_owned_rollback_builds_fresh_scoped_runtime(monkeypatch):
    store = ThreadStore.in_memory()
    runtimes = []

    def engine_factory(_session_id):
        current = FakeQA()
        runtimes.append(current)
        return current

    append_turn = store.append_turn
    call_count = 0

    def fail_first_append(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("persistence unavailable")
        return append_turn(*args, **kwargs)

    monkeypatch.setattr(store, "append_turn", fail_first_append)
    client = TestClient(
        web_app.create_app(
            thread_store=store,
            engine_factory=engine_factory,
        ),
        raise_server_exceptions=False,
    )

    failed = client.post(
        "/api/query",
        json={"message": "first", "session_id": "thread_retry"},
    )
    assert failed.status_code == 500
    assert store.get_thread("thread_retry") is None

    retried = client.post(
        "/api/query",
        json={"message": "second", "session_id": "thread_retry"},
    )

    assert retried.status_code == 200
    assert len(runtimes) == 2
    assert runtimes[0] is not runtimes[1]
    assert store.get_thread("thread_retry").message_count == 2


def test_query_endpoint_rejects_missing_loop_report_without_persisting_turn():
    class MissingReportQA(FakeQA):
        def query_with_trace(self, *args, **kwargs):
            result = super().query_with_trace(*args, **kwargs)
            return replace(
                result,
                answer="SECRET_MISSING_REPORT_ANSWER",
                trace=replace(
                    result.trace,
                    question="SECRET_MISSING_REPORT_QUESTION",
                    model_thinking="SECRET_MISSING_REPORT_THINKING",
                ),
                loop_report=None,
            )

    store = ThreadStore.in_memory()
    client = TestClient(
        web_app.create_app(
            thread_store=store,
            engine_factory=lambda _session_id: MissingReportQA(),
        ),
        raise_server_exceptions=False,
    )

    response = client.post(
        "/api/query",
        json={"message": "hello", "session_id": "thread_missing_report"},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Canonical loop report unavailable."}
    for secret in (
        "SECRET_MISSING_REPORT_ANSWER",
        "SECRET_MISSING_REPORT_QUESTION",
        "SECRET_MISSING_REPORT_THINKING",
    ):
        assert secret not in response.text
    assert client.get("/api/threads/thread_missing_report").status_code == 404

    store.create_thread(thread_id="thread_existing_missing_report")
    existing_response = client.post(
        "/api/query",
        json={
            "message": "hello again",
            "session_id": "thread_existing_missing_report",
        },
    )
    assert existing_response.status_code == 500
    thread = client.get("/api/threads/thread_existing_missing_report").json()
    assert thread["messages"] == []
    assert thread["message_count"] == 0
    assert thread["loop_run_count"] == 0
    assert thread["latest"] is None


def test_clear_chat_blocks_stale_in_flight_query_persistence():
    class BlockingQA(FakeQA):
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()

        def query_with_trace(
            self,
            message,
            session_id="default",
            conversation_history=None,
            semantic_memory=None,
            semantic_memory_status="not_requested",
            loop_recipe=None,
            context_provider=None,
        ):
            self.last_query_session_id = session_id
            self.last_conversation_history = list(conversation_history or [])
            self.last_semantic_memory = list(semantic_memory or [])
            self.last_semantic_memory_status = semantic_memory_status
            self.last_loop_recipe = dict(loop_recipe or {})
            self.last_context_provider = context_provider
            self.started.set()
            if not self.release.wait(timeout=5):
                raise RuntimeError("timed out waiting for test release")
            return super().query_with_trace(
                message,
                session_id=session_id,
                conversation_history=conversation_history,
                semantic_memory=semantic_memory,
                semantic_memory_status=semantic_memory_status,
                loop_recipe=loop_recipe,
                context_provider=context_provider,
            )

    fake_qa = BlockingQA()
    store = ThreadStore.in_memory()
    client = TestClient(web_app.create_app(fake_qa, thread_store=store))
    responses = {}

    def run_query():
        responses["query"] = client.post(
            "/api/query",
            json={"message": "stale me", "session_id": "thread_race"},
        )

    query_thread = threading.Thread(target=run_query)
    query_thread.start()
    assert fake_qa.started.wait(timeout=5)

    clear_response = client.post(
        "/api/chat/clear",
        json={"session_id": "thread_race"},
    )
    fake_qa.release.set()
    query_thread.join(timeout=5)

    assert not query_thread.is_alive()
    assert clear_response.status_code == 200
    assert responses["query"].status_code == 409
    assert responses["query"].json() == {
        "detail": (
            "Thread changed before query persistence completed. "
            "Please retry in the active thread."
        )
    }
    thread = client.get("/api/threads/thread_race").json()
    assert thread["messages"] == []
    assert thread["message_count"] == 0
    assert thread["loop_run_count"] == 0
    assert client.get("/api/threads/thread_race/runs").json()["runs"] == []
    assert thread["latest"] is None


def test_delete_thread_blocks_stale_in_flight_query_resurrection():
    class BlockingQA(FakeQA):
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()

        def query_with_trace(
            self,
            message,
            session_id="default",
            conversation_history=None,
            semantic_memory=None,
            semantic_memory_status="not_requested",
            loop_recipe=None,
            context_provider=None,
        ):
            self.last_query_session_id = session_id
            self.last_conversation_history = list(conversation_history or [])
            self.last_semantic_memory = list(semantic_memory or [])
            self.last_semantic_memory_status = semantic_memory_status
            self.last_loop_recipe = dict(loop_recipe or {})
            self.last_context_provider = context_provider
            self.started.set()
            if not self.release.wait(timeout=5):
                raise RuntimeError("timed out waiting for test release")
            return super().query_with_trace(
                message,
                session_id=session_id,
                conversation_history=conversation_history,
                semantic_memory=semantic_memory,
                semantic_memory_status=semantic_memory_status,
                loop_recipe=loop_recipe,
                context_provider=context_provider,
            )

    fake_qa = BlockingQA()
    store = ThreadStore.in_memory()
    client = TestClient(web_app.create_app(fake_qa, thread_store=store))
    responses = {}

    def run_query():
        responses["query"] = client.post(
            "/api/query",
            json={"message": "stale me", "session_id": "thread_race"},
        )

    query_thread = threading.Thread(target=run_query)
    query_thread.start()
    assert fake_qa.started.wait(timeout=5)

    delete_response = client.delete("/api/threads/thread_race")
    fake_qa.release.set()
    query_thread.join(timeout=5)

    assert not query_thread.is_alive()
    assert delete_response.status_code == 200
    assert responses["query"].status_code == 409
    assert set(responses["query"].json()) == {"detail"}
    assert client.get("/api/threads/thread_race").status_code == 404


def test_delete_recreate_blocks_stale_in_flight_query_resurrection():
    class BlockingQA(FakeQA):
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()

        def query_with_trace(
            self,
            message,
            session_id="default",
            conversation_history=None,
            semantic_memory=None,
            semantic_memory_status="not_requested",
            loop_recipe=None,
            context_provider=None,
        ):
            self.last_query_session_id = session_id
            self.last_conversation_history = list(conversation_history or [])
            self.last_semantic_memory = list(semantic_memory or [])
            self.last_semantic_memory_status = semantic_memory_status
            self.last_loop_recipe = dict(loop_recipe or {})
            self.last_context_provider = context_provider
            self.started.set()
            if not self.release.wait(timeout=5):
                raise RuntimeError("timed out waiting for test release")
            return super().query_with_trace(
                message,
                session_id=session_id,
                conversation_history=conversation_history,
                semantic_memory=semantic_memory,
                semantic_memory_status=semantic_memory_status,
                loop_recipe=loop_recipe,
                context_provider=context_provider,
            )

    fake_qa = BlockingQA()
    store = ThreadStore.in_memory()
    client = TestClient(web_app.create_app(fake_qa, thread_store=store))
    responses = {}

    def run_query():
        responses["query"] = client.post(
            "/api/query",
            json={"message": "stale me", "session_id": "thread_race"},
        )

    query_thread = threading.Thread(target=run_query)
    query_thread.start()
    assert fake_qa.started.wait(timeout=5)

    delete_response = client.delete("/api/threads/thread_race")
    recreated = store.create_thread(thread_id="thread_race")
    fake_qa.release.set()
    query_thread.join(timeout=5)

    assert not query_thread.is_alive()
    assert delete_response.status_code == 200
    assert responses["query"].status_code == 409
    assert set(responses["query"].json()) == {"detail"}
    thread = client.get("/api/threads/thread_race").json()
    assert thread["id"] == recreated.id
    assert thread["messages"] == []
    assert thread["message_count"] == 0
    assert thread["loop_run_count"] == 0
    assert thread["latest"] is None


def test_clear_lifecycle_gate_rejects_query_and_detects_runtime_cleanup_aba():
    class BlockingClearQA(FakeQA):
        def __init__(self):
            self.clear_started = threading.Event()
            self.release_clear = threading.Event()
            self.clear_calls = []

        def clear_loop_session(self, session_id="default"):
            self.clear_calls.append(session_id)
            self.clear_started.set()
            if not self.release_clear.wait(timeout=5):
                raise RuntimeError("timed out waiting to release runtime clear")

    session_id = "thread_clear_runtime_aba"
    store = ThreadStore.in_memory()
    original = store.create_thread(thread_id=session_id)
    store.append_message(session_id, role="user", content="obsolete message")
    runtimes = []

    def engine_factory(_session_id):
        current = BlockingClearQA() if not runtimes else FakeQA()
        runtimes.append(current)
        return current

    client = TestClient(
        web_app.create_app(thread_store=store, engine_factory=engine_factory),
        raise_server_exceptions=False,
    )
    assert client.get(
        "/api/status",
        headers={"x-ai-loop-session-id": session_id},
    ).status_code == 200
    assert len(runtimes) == 1

    responses = {}

    def run_clear():
        responses["clear"] = client.post(
            "/api/chat/clear",
            json={"session_id": session_id},
        )

    clear_thread = threading.Thread(target=run_clear)
    clear_thread.start()
    assert runtimes[0].clear_started.wait(timeout=5)
    replacement = None
    try:
        blocked_query = client.post(
            "/api/query",
            json={"message": "must be gated", "session_id": session_id},
        )
        assert blocked_query.status_code == 409
        assert len(runtimes) == 1

        assert store.delete_thread(session_id) is True
        replacement = store.create_thread(
            thread_id=session_id,
            title="Replacement thread",
        )
        store.append_message(
            session_id,
            role="user",
            content="replacement durable message",
        )
    finally:
        runtimes[0].release_clear.set()
    clear_thread.join(timeout=5)

    assert not clear_thread.is_alive()
    assert responses["clear"].status_code == 409
    assert replacement is not None
    assert replacement.instance_id != original.instance_id
    current = store.get_thread(session_id)
    assert current is not None
    assert current.instance_id == replacement.instance_id
    assert [message.content for message in current.messages] == [
        "replacement durable message"
    ]

    retried = client.post(
        "/api/query",
        json={"message": "use the replacement", "session_id": session_id},
    )

    assert retried.status_code == 200
    assert len(runtimes) == 2
    assert runtimes[1] is not runtimes[0]
    assert runtimes[1].last_query_session_id == session_id
    current = store.get_thread(session_id)
    assert current is not None
    assert current.instance_id == replacement.instance_id
    assert current.messages[0].content == "replacement durable message"


def test_delete_lifecycle_gate_uses_exact_identity_and_preserves_replacement(
    monkeypatch,
):
    class TrackingClearQA(FakeQA):
        def __init__(self):
            self.clear_calls = []

        def clear_loop_session(self, session_id="default"):
            self.clear_calls.append(session_id)

    session_id = "thread_delete_runtime_aba"
    store = ThreadStore.in_memory()
    original = store.create_thread(thread_id=session_id)
    store.append_message(session_id, role="user", content="obsolete message")
    runtimes = []

    def engine_factory(_session_id):
        current = TrackingClearQA()
        runtimes.append(current)
        return current

    client = TestClient(
        web_app.create_app(thread_store=store, engine_factory=engine_factory),
        raise_server_exceptions=False,
    )
    assert client.get(
        "/api/status",
        headers={"x-ai-loop-session-id": session_id},
    ).status_code == 200
    assert len(runtimes) == 1

    delete_started = threading.Event()
    release_delete = threading.Event()
    observed_expectations = []
    delete_thread = store.delete_thread

    def blocking_delete(
        thread_id,
        *,
        expected_instance_id=None,
        expected_generation=None,
    ):
        observed_expectations.append(
            (expected_instance_id, expected_generation)
        )
        delete_started.set()
        if not release_delete.wait(timeout=5):
            raise RuntimeError("timed out waiting to release durable delete")
        return delete_thread(
            thread_id,
            expected_instance_id=expected_instance_id,
            expected_generation=expected_generation,
        )

    monkeypatch.setattr(store, "delete_thread", blocking_delete)
    responses = {}

    def run_delete():
        responses["delete"] = client.delete(f"/api/threads/{session_id}")

    request_thread = threading.Thread(target=run_delete)
    request_thread.start()
    assert delete_started.wait(timeout=5)
    replacement = None
    try:
        blocked_query = client.post(
            "/api/query",
            json={"message": "must be gated", "session_id": session_id},
        )
        assert blocked_query.status_code == 409
        assert len(runtimes) == 1

        assert delete_thread(session_id) is True
        replacement = store.create_thread(
            thread_id=session_id,
            title="Replacement thread",
        )
        store.append_message(
            session_id,
            role="user",
            content="replacement durable message",
        )
    finally:
        release_delete.set()
    request_thread.join(timeout=5)

    assert not request_thread.is_alive()
    assert observed_expectations == [
        (original.instance_id, original.generation)
    ]
    assert responses["delete"].status_code == 409
    assert replacement is not None
    assert replacement.instance_id != original.instance_id
    current = store.get_thread(session_id)
    assert current is not None
    assert current.instance_id == replacement.instance_id
    assert [message.content for message in current.messages] == [
        "replacement durable message"
    ]

    retried = client.post(
        "/api/query",
        json={"message": "use the replacement", "session_id": session_id},
    )

    assert retried.status_code == 200
    assert len(runtimes) == 2
    assert runtimes[1] is not runtimes[0]
    assert runtimes[1].last_query_session_id == session_id
    current = store.get_thread(session_id)
    assert current is not None
    assert current.instance_id == replacement.instance_id
    assert current.messages[0].content == "replacement durable message"


def test_delete_missing_thread_cleans_and_drops_loaded_scoped_runtime():
    class TrackingClearQA(FakeQA):
        def __init__(self):
            self.clear_calls = []

        def clear_loop_session(self, session_id="default"):
            self.clear_calls.append(session_id)

    session_id = "thread_missing_with_stale_runtime"
    store = ThreadStore.in_memory()
    store.create_thread(thread_id=session_id)
    runtimes = []

    def engine_factory(_session_id):
        current = TrackingClearQA()
        runtimes.append(current)
        return current

    client = TestClient(
        web_app.create_app(thread_store=store, engine_factory=engine_factory),
        raise_server_exceptions=False,
    )
    assert client.get(
        "/api/status",
        headers={"x-ai-loop-session-id": session_id},
    ).status_code == 200
    assert store.delete_thread(session_id) is True

    missing = client.delete(f"/api/threads/{session_id}")

    assert missing.status_code == 404
    assert runtimes[0].clear_calls == [session_id]
    replacement = store.create_thread(thread_id=session_id)
    retried = client.post(
        "/api/query",
        json={"message": "use a fresh runtime", "session_id": session_id},
    )

    assert retried.status_code == 200
    assert len(runtimes) == 2
    assert runtimes[1] is not runtimes[0]
    assert runtimes[1].last_query_session_id == session_id
    current = store.get_thread(session_id)
    assert current is not None
    assert current.instance_id == replacement.instance_id


def test_delete_committed_durable_state_wins_when_runtime_cleanup_fails():
    class CleanupFailureQA(FakeQA):
        def __init__(self):
            self.clear_calls = []

        def clear_loop_session(self, session_id="default"):
            self.clear_calls.append(session_id)
            raise RuntimeError("runtime cleanup failed")

    session_id = "thread_delete_cleanup_failure"
    store = ThreadStore.in_memory()
    store.create_thread(thread_id=session_id)
    runtimes = []

    def engine_factory(_session_id):
        current = CleanupFailureQA() if not runtimes else FakeQA()
        runtimes.append(current)
        return current

    client = TestClient(
        web_app.create_app(thread_store=store, engine_factory=engine_factory),
        raise_server_exceptions=False,
    )
    assert client.get(
        "/api/status",
        headers={"x-ai-loop-session-id": session_id},
    ).status_code == 200

    deleted = client.delete(f"/api/threads/{session_id}")

    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": True, "thread_id": session_id}
    assert runtimes[0].clear_calls == [session_id]
    assert store.get_thread(session_id) is None

    replacement = store.create_thread(thread_id=session_id)
    retried = client.post(
        "/api/query",
        json={"message": "use a fresh runtime", "session_id": session_id},
    )

    assert retried.status_code == 200
    assert len(runtimes) == 2
    assert runtimes[1] is not runtimes[0]
    assert runtimes[1].last_query_session_id == session_id
    current = store.get_thread(session_id)
    assert current is not None
    assert current.instance_id == replacement.instance_id


def test_shared_engine_stale_query_cleans_history_appended_after_clear():
    class StaleAfterClearQA(FakeQA):
        def __init__(self):
            self.chat_history = []
            self.query_started = threading.Event()
            self.release_query = threading.Event()
            self.clear_calls = []

        def query_with_trace(self, *args, **kwargs):
            self.query_started.set()
            if not self.release_query.wait(timeout=5):
                raise RuntimeError("timed out waiting to release stale query")
            self.chat_history.append(
                {
                    "session_id": kwargs.get("session_id") or "default",
                    "question": "STALE_AFTER_CLEAR",
                }
            )
            return super().query_with_trace(*args, **kwargs)

        def clear_loop_session(self, session_id="default"):
            self.clear_calls.append(session_id)

    session_id = "thread_shared_clear_history_race"
    qa = StaleAfterClearQA()
    store = ThreadStore.in_memory()
    store.create_thread(thread_id=session_id)
    client = TestClient(
        web_app.create_app(qa, thread_store=store),
        raise_server_exceptions=False,
    )
    responses = {}

    def run_query():
        responses["query"] = client.post(
            "/api/query",
            json={"message": "become stale", "session_id": session_id},
        )

    query_thread = threading.Thread(target=run_query)
    query_thread.start()
    assert qa.query_started.wait(timeout=5)
    try:
        cleared = client.post(
            "/api/chat/clear",
            json={"session_id": session_id},
        )
        assert cleared.status_code == 200
        assert qa.chat_history == []
    finally:
        qa.release_query.set()
    query_thread.join(timeout=5)

    assert not query_thread.is_alive()
    assert responses["query"].status_code == 409
    assert qa.clear_calls == [session_id, session_id]
    assert qa.chat_history == []
    current = store.get_thread(session_id)
    assert current is not None
    assert current.generation == 1
    assert current.messages == ()


def test_stale_generation_query_preserves_advanced_runtime_and_document():
    class FirstQueryBlockingQA(FakeQA):
        def __init__(self):
            self.query_started = threading.Event()
            self.release_query = threading.Event()
            self.query_calls = 0
            self.clear_calls = []
            self.documents_seen_on_query = []

        def query_with_trace(self, *args, **kwargs):
            self.query_calls += 1
            self.documents_seen_on_query.append(self.current_document_name)
            if self.query_calls == 1:
                self.query_started.set()
                if not self.release_query.wait(timeout=5):
                    raise RuntimeError("timed out waiting to release stale query")
            return super().query_with_trace(*args, **kwargs)

        def clear_loop_session(self, session_id="default"):
            self.clear_calls.append(session_id)

    session_id = "thread_generation_runtime_race"
    store = ThreadStore.in_memory()
    original = store.create_thread(thread_id=session_id)
    runtimes = []

    def engine_factory(_session_id):
        current = FirstQueryBlockingQA()
        runtimes.append(current)
        return current

    client = TestClient(
        web_app.create_app(thread_store=store, engine_factory=engine_factory),
        raise_server_exceptions=False,
    )
    uploaded = client.post(
        "/api/documents",
        data={"text_encoding": "auto", "session_id": session_id},
        files={
            "file": (
                "attached.txt",
                b"ATTACHED_PRIVATE_DOCUMENT",
                "text/plain",
            )
        },
    )
    assert uploaded.status_code == 200
    assert uploaded.json()["status"]["active_document"] == "attached.txt"
    assert len(runtimes) == 1
    responses = {}

    def run_query():
        responses["query"] = client.post(
            "/api/query",
            json={"message": "become stale", "session_id": session_id},
        )

    query_thread = threading.Thread(target=run_query)
    query_thread.start()
    assert runtimes[0].query_started.wait(timeout=5)
    try:
        cleared = client.post(
            "/api/chat/clear",
            json={"session_id": session_id},
        )
        assert cleared.status_code == 200
    finally:
        runtimes[0].release_query.set()
    query_thread.join(timeout=5)

    assert not query_thread.is_alive()
    assert responses["query"].status_code == 409
    current = store.get_thread(session_id)
    assert current is not None
    assert current.instance_id == original.instance_id
    assert current.generation == 1
    assert current.messages == ()
    assert len(runtimes) == 1
    assert runtimes[0].clear_calls == [session_id, session_id]

    status = client.get(
        "/api/status",
        headers={"x-ai-loop-session-id": session_id},
    )
    assert status.status_code == 200
    assert status.json()["active_document"] == "attached.txt"
    assert len(runtimes) == 1

    retried = client.post(
        "/api/query",
        json={"message": "use attached context", "session_id": session_id},
    )

    assert retried.status_code == 200
    assert len(runtimes) == 1
    assert runtimes[0].query_calls == 2
    assert runtimes[0].documents_seen_on_query == [
        "attached.txt",
        "attached.txt",
    ]


def test_generation_refresh_gates_same_session_runtime_until_cleanup_finishes():
    class BlockingCleanupQA(FakeQA):
        def __init__(self):
            self.cleanup_started = threading.Event()
            self.release_cleanup = threading.Event()
            self.clear_calls = []

        def clear_loop_session(self, session_id="default"):
            self.clear_calls.append(session_id)
            self.cleanup_started.set()
            if not self.release_cleanup.wait(timeout=5):
                raise RuntimeError("timed out waiting to release runtime refresh")

    session_id = "thread_generation_refresh_gate"
    store = ThreadStore.in_memory()
    original = store.create_thread(thread_id=session_id)
    runtimes = []

    def engine_factory(_session_id):
        current = BlockingCleanupQA() if not runtimes else FakeQA()
        runtimes.append(current)
        return current

    client = TestClient(
        web_app.create_app(thread_store=store, engine_factory=engine_factory),
        raise_server_exceptions=False,
    )
    initial_status = client.get(
        "/api/status",
        headers={"x-ai-loop-session-id": session_id},
    )
    assert initial_status.status_code == 200
    assert len(runtimes) == 1
    assert runtimes[0].status_calls == 1

    advanced = store.clear_thread(
        session_id,
        expected_instance_id=original.instance_id,
        expected_generation=original.generation,
    )
    assert advanced is not None
    assert advanced.instance_id == original.instance_id
    assert advanced.generation == 1
    responses = {}

    def run_refresh():
        responses["refresh"] = client.get(
            "/api/status",
            headers={"x-ai-loop-session-id": session_id},
        )

    refresh_thread = threading.Thread(target=run_refresh)
    refresh_thread.start()
    assert runtimes[0].cleanup_started.wait(timeout=5)
    try:
        assert runtimes[0].status_calls == 1
        assert sum(
            getattr(current, "status_calls", 0) for current in runtimes[1:]
        ) == 0

        concurrent = client.get(
            "/api/status",
            headers={"x-ai-loop-session-id": session_id},
        )
        assert concurrent.status_code == 409
        assert runtimes[0].status_calls == 1
        assert sum(
            getattr(current, "status_calls", 0) for current in runtimes[1:]
        ) == 0
    finally:
        runtimes[0].release_cleanup.set()
    refresh_thread.join(timeout=5)

    assert not refresh_thread.is_alive()
    assert responses["refresh"].status_code == 200
    assert runtimes[0].clear_calls == [session_id]
    assert len(runtimes) == 1
    assert runtimes[0].status_calls == 2

    subsequent = client.get(
        "/api/status",
        headers={"x-ai-loop-session-id": session_id},
    )

    assert subsequent.status_code == 200
    assert len(runtimes) == 1
    assert runtimes[0].status_calls == 3


def test_stale_status_cannot_retire_newer_generation_runtime():
    class SecondStatusBlockingQA(FakeQA):
        def __init__(self):
            self.current_document_name = "PRESERVED"
            self.status_attempts = 0
            self.stale_status_started = threading.Event()
            self.release_stale_status = threading.Event()

        def status(self):
            self.status_attempts += 1
            if self.status_attempts == 2:
                self.stale_status_started.set()
                if not self.release_stale_status.wait(timeout=5):
                    raise RuntimeError("timed out waiting to release stale status")
            return super().status()

    session_id = "thread_stale_status_runtime_aba"
    store = ThreadStore.in_memory()
    original = store.create_thread(thread_id=session_id)
    runtimes = []

    def engine_factory(_session_id):
        current = SecondStatusBlockingQA() if not runtimes else FakeQA()
        runtimes.append(current)
        return current

    client = TestClient(
        web_app.create_app(thread_store=store, engine_factory=engine_factory),
        raise_server_exceptions=False,
    )
    initial = client.get(
        "/api/status",
        headers={"x-ai-loop-session-id": session_id},
    )
    assert initial.status_code == 200
    assert initial.json()["active_document"] == "PRESERVED"
    responses = {}

    def run_stale_status():
        responses["stale"] = client.get(
            "/api/status",
            headers={"x-ai-loop-session-id": session_id},
        )

    stale_thread = threading.Thread(target=run_stale_status)
    stale_thread.start()
    assert runtimes[0].stale_status_started.wait(timeout=5)
    try:
        advanced = store.clear_thread(
            session_id,
            expected_instance_id=original.instance_id,
            expected_generation=original.generation,
        )
        assert advanced is not None
        assert advanced.generation == 1
        refreshed = client.get(
            "/api/status",
            headers={"x-ai-loop-session-id": session_id},
        )
        assert refreshed.status_code == 200
        assert refreshed.json()["active_document"] == "PRESERVED"
    finally:
        runtimes[0].release_stale_status.set()
    stale_thread.join(timeout=5)

    assert not stale_thread.is_alive()
    assert responses["stale"].status_code == 409
    subsequent = client.get(
        "/api/status",
        headers={"x-ai-loop-session-id": session_id},
    )

    assert subsequent.status_code == 200
    assert subsequent.json()["active_document"] == "PRESERVED"
    assert len(runtimes) == 1


def test_stale_status_during_clear_cannot_retire_document_runtime():
    class BlockingStatusAndClearQA(FakeQA):
        def __init__(self):
            self.current_document_name = "PRESERVED"
            self.status_attempts = 0
            self.stale_status_started = threading.Event()
            self.release_stale_status = threading.Event()
            self.clear_started = threading.Event()
            self.release_clear = threading.Event()

        def status(self):
            self.status_attempts += 1
            if self.status_attempts == 2:
                self.stale_status_started.set()
                if not self.release_stale_status.wait(timeout=5):
                    raise RuntimeError("timed out waiting to release stale status")
            return super().status()

        def clear_loop_session(self, session_id="default"):
            self.clear_started.set()
            if not self.release_clear.wait(timeout=5):
                raise RuntimeError("timed out waiting to release clear")

    session_id = "thread_stale_status_during_clear"
    store = ThreadStore.in_memory()
    store.create_thread(thread_id=session_id)
    runtimes = []

    def engine_factory(_session_id):
        current = BlockingStatusAndClearQA() if not runtimes else FakeQA()
        runtimes.append(current)
        return current

    client = TestClient(
        web_app.create_app(thread_store=store, engine_factory=engine_factory),
        raise_server_exceptions=False,
    )
    assert client.get(
        "/api/status",
        headers={"x-ai-loop-session-id": session_id},
    ).json()["active_document"] == "PRESERVED"
    responses = {}

    def run_stale_status():
        responses["stale"] = client.get(
            "/api/status",
            headers={"x-ai-loop-session-id": session_id},
        )

    def run_clear():
        responses["clear"] = client.post(
            "/api/chat/clear",
            json={"session_id": session_id},
        )

    stale_thread = threading.Thread(target=run_stale_status)
    clear_thread = threading.Thread(target=run_clear)
    stale_thread.start()
    assert runtimes[0].stale_status_started.wait(timeout=5)
    clear_thread.start()
    assert runtimes[0].clear_started.wait(timeout=5)
    try:
        current = store.get_thread(session_id)
        assert current is not None
        assert current.generation == 1
        runtimes[0].release_stale_status.set()
        stale_thread.join(timeout=5)
        assert not stale_thread.is_alive()
        assert responses["stale"].status_code == 409
    finally:
        runtimes[0].release_clear.set()
    clear_thread.join(timeout=5)

    assert not clear_thread.is_alive()
    assert responses["clear"].status_code == 200
    subsequent = client.get(
        "/api/status",
        headers={"x-ai-loop-session-id": session_id},
    )
    assert subsequent.status_code == 200
    assert subsequent.json()["active_document"] == "PRESERVED"
    assert len(runtimes) == 1


def test_caller_injected_runtime_fails_closed_across_threads():
    store = ThreadStore.in_memory()
    store.create_thread(thread_id="thread_alpha")
    store.create_thread(thread_id="thread_beta")
    fake_qa = FakeQA()
    client = TestClient(
        web_app.create_app(fake_qa, thread_store=store),
        raise_server_exceptions=False,
    )

    uploaded = client.post(
        "/api/documents",
        data={"text_encoding": "auto", "session_id": "thread_alpha"},
        files={
            "file": (
                "alpha-private.txt",
                b"ALPHA_PRIVATE_DOCUMENT",
                "text/plain",
            )
        },
    )
    assert uploaded.status_code == 200
    assert uploaded.json()["status"]["active_document"] == "alpha-private.txt"

    beta_status = client.get(
        "/api/status",
        headers={"x-ai-loop-session-id": "thread_beta"},
    )
    beta_query = client.post(
        "/api/query",
        json={"message": "show private context", "session_id": "thread_beta"},
    )

    assert beta_status.status_code == 409
    assert beta_query.status_code == 409
    assert "alpha-private.txt" not in beta_status.text
    assert "ALPHA_PRIVATE_DOCUMENT" not in beta_status.text
    assert "alpha-private.txt" not in beta_query.text
    assert "ALPHA_PRIVATE_DOCUMENT" not in beta_query.text
    assert getattr(fake_qa, "last_query_session_id", None) != "thread_beta"


def test_failed_pristine_query_does_not_release_injected_runtime_owner(monkeypatch):
    store = ThreadStore.in_memory()
    fake_qa = FakeQA()
    fake_qa.current_document_name = "PRELOADED_PRIVATE_DOCUMENT.txt"
    original_append_turn = store.append_turn

    def fail_append_turn(*args, **kwargs):
        raise RuntimeError("persistence unavailable")

    monkeypatch.setattr(store, "append_turn", fail_append_turn)
    client = TestClient(
        web_app.create_app(fake_qa, thread_store=store),
        raise_server_exceptions=False,
    )
    failed = client.post(
        "/api/query",
        json={"message": "fail and roll back", "session_id": "thread_alpha"},
    )
    assert failed.status_code == 500
    assert store.get_thread("thread_alpha") is None

    monkeypatch.setattr(store, "append_turn", original_append_turn)
    store.create_thread(thread_id="thread_beta")
    beta_status = client.get(
        "/api/status",
        headers={"x-ai-loop-session-id": "thread_beta"},
    )

    assert beta_status.status_code == 409
    assert "PRELOADED_PRIVATE_DOCUMENT.txt" not in beta_status.text


def test_idle_delete_recreate_query_uses_runtime_bound_to_replacement_instance():
    class MarkerQA(FakeQA):
        def __init__(self, *, answer_marker, document_name):
            self.answer_marker = answer_marker
            self.current_document_name = document_name
            self.query_calls = 0

        def query_with_trace(self, *args, **kwargs):
            self.query_calls += 1
            result = super().query_with_trace(*args, **kwargs)
            answer_digest = answer_candidate_sha256(self.answer_marker)
            rebound_steps = []
            for step in result.loop_report.run.steps:
                metadata = dict(step.metadata)
                if ANSWER_CANDIDATE_SHA256_METADATA_KEY in metadata:
                    metadata[ANSWER_CANDIDATE_SHA256_METADATA_KEY] = answer_digest
                    step = replace(step, metadata=metadata)
                rebound_steps.append(step)
            rebound_report = replace(
                result.loop_report,
                run=replace(
                    result.loop_report.run,
                    steps=tuple(rebound_steps),
                    final_answer=self.answer_marker,
                ),
            )
            return replace(
                result,
                answer=self.answer_marker,
                loop_report=rebound_report,
            )

    session_id = "thread_idle_runtime_aba"
    old_answer = "OLD_PRIVATE_DOCUMENT_ANSWER"
    fresh_answer = "FRESH_RUNTIME_ANSWER"
    old_document = "OLD_PRIVATE_DOCUMENT.txt"
    fresh_document = "FRESH_RUNTIME_DOCUMENT.txt"
    store = ThreadStore.in_memory()
    original = store.create_thread(thread_id=session_id)
    runtimes = []

    def engine_factory(_session_id):
        if not runtimes:
            current = MarkerQA(
                answer_marker=old_answer,
                document_name=old_document,
            )
        else:
            current = MarkerQA(
                answer_marker=fresh_answer,
                document_name=fresh_document,
            )
        runtimes.append(current)
        return current

    client = TestClient(
        web_app.create_app(thread_store=store, engine_factory=engine_factory),
        raise_server_exceptions=False,
    )
    old_status = client.get(
        "/api/status",
        headers={"x-ai-loop-session-id": session_id},
    )
    assert old_status.status_code == 200
    assert old_status.json()["active_document"] == old_document
    assert len(runtimes) == 1

    assert store.delete_thread(session_id) is True
    replacement = store.create_thread(
        thread_id=session_id,
        title="Replacement thread",
    )
    store.append_message(
        session_id,
        role="user",
        content="replacement durable context",
    )

    queried = client.post(
        "/api/query",
        json={"message": "use only replacement state", "session_id": session_id},
    )

    assert queried.status_code == 200
    assert replacement.instance_id != original.instance_id
    assert len(runtimes) == 2
    assert runtimes[1] is not runtimes[0]
    assert runtimes[0].query_calls == 0
    assert runtimes[1].query_calls == 1
    assert queried.json()["answer"] == fresh_answer
    assert old_answer not in queried.text
    assert old_document not in queried.text
    fresh_status = client.get(
        "/api/status",
        headers={"x-ai-loop-session-id": session_id},
    )
    assert fresh_status.status_code == 200
    assert fresh_status.json()["active_document"] == fresh_document
    assert len(runtimes) == 2
    current = store.get_thread(session_id)
    assert current is not None
    assert current.instance_id == replacement.instance_id
    assert [message.content for message in current.messages] == [
        "replacement durable context",
        "use only replacement state",
        fresh_answer,
    ]


def test_query_does_not_send_aba_replacement_history_to_runtime(monkeypatch):
    class RecordingQA(FakeQA):
        def __init__(self):
            self.query_calls = []

        def query_with_trace(self, *args, **kwargs):
            self.query_calls.append(
                {
                    "conversation_history": list(
                        kwargs.get("conversation_history") or []
                    ),
                    "semantic_memory": list(kwargs.get("semantic_memory") or []),
                }
            )
            return super().query_with_trace(*args, **kwargs)

    session_id = "thread_history_read_aba"
    replacement_secret = "REPLACEMENT_PRIVATE_RECENT_HISTORY"
    store = ThreadStore.in_memory()
    original = store.create_thread(thread_id=session_id)
    store.append_message(session_id, role="user", content="original history")
    qa = RecordingQA()
    recent_read_started = threading.Event()
    release_recent_read = threading.Event()
    recent_messages = store.recent_messages

    def blocking_recent_messages(thread_id, limit=12, **kwargs):
        recent_read_started.set()
        if not release_recent_read.wait(timeout=5):
            raise RuntimeError("timed out waiting to release recent-message read")
        return recent_messages(thread_id, limit=limit, **kwargs)

    monkeypatch.setattr(store, "recent_messages", blocking_recent_messages)
    client = TestClient(
        web_app.create_app(qa, thread_store=store),
        raise_server_exceptions=False,
    )
    responses = {}

    def run_query():
        responses["query"] = client.post(
            "/api/query",
            json={"message": "read safely", "session_id": session_id},
        )

    query_thread = threading.Thread(target=run_query)
    query_thread.start()
    assert recent_read_started.wait(timeout=5)
    replacement = None
    try:
        assert store.delete_thread(session_id) is True
        replacement = store.create_thread(thread_id=session_id)
        store.append_message(
            session_id,
            role="user",
            content=replacement_secret,
        )
    finally:
        release_recent_read.set()
    query_thread.join(timeout=5)

    assert not query_thread.is_alive()
    assert responses["query"].status_code == 409
    assert replacement is not None
    assert replacement.instance_id != original.instance_id
    assert qa.query_calls == []
    current = store.get_thread(session_id)
    assert current is not None
    assert current.instance_id == replacement.instance_id
    assert [message.content for message in current.messages] == [
        replacement_secret
    ]


def test_query_does_not_send_aba_replacement_semantic_memory_to_runtime(
    monkeypatch,
):
    class RecordingQA(FakeQA):
        def __init__(self):
            self.query_calls = []

        def query_with_trace(self, *args, **kwargs):
            self.query_calls.append(
                {
                    "conversation_history": list(
                        kwargs.get("conversation_history") or []
                    ),
                    "semantic_memory": list(kwargs.get("semantic_memory") or []),
                }
            )
            return super().query_with_trace(*args, **kwargs)

    session_id = "thread_semantic_read_aba"
    replacement_secret = (
        "REPLACEMENT_PRIVATE_SEMANTIC_MEMORY about dynamic programming"
    )
    store = ThreadStore.in_memory()
    original = store.create_thread(thread_id=session_id)
    qa = RecordingQA()
    embedding_read_started = threading.Event()
    release_embedding_read = threading.Event()
    has_message_embeddings = store.has_message_embeddings

    def blocking_has_message_embeddings(thread_id, embedding_model, **kwargs):
        embedding_read_started.set()
        if not release_embedding_read.wait(timeout=5):
            raise RuntimeError("timed out waiting to release embedding read")
        return has_message_embeddings(thread_id, embedding_model, **kwargs)

    monkeypatch.setattr(
        store,
        "has_message_embeddings",
        blocking_has_message_embeddings,
    )
    client = TestClient(
        web_app.create_app(qa, thread_store=store),
        raise_server_exceptions=False,
    )
    responses = {}

    def run_query():
        responses["query"] = client.post(
            "/api/query",
            json={
                "message": "Explain that algorithm",
                "session_id": session_id,
            },
        )

    query_thread = threading.Thread(target=run_query)
    query_thread.start()
    assert embedding_read_started.wait(timeout=5)
    replacement = None
    try:
        assert store.delete_thread(session_id) is True
        replacement = store.create_thread(thread_id=session_id)
        private_memory = store.append_message(
            session_id,
            role="user",
            content=replacement_secret,
        )
        assert store.upsert_message_embedding(
            private_memory,
            embedding_model="fake-memory",
            vector=[1.0, 0.0],
        ) is True
    finally:
        release_embedding_read.set()
    query_thread.join(timeout=5)

    assert not query_thread.is_alive()
    assert responses["query"].status_code == 409
    assert replacement is not None
    assert replacement.instance_id != original.instance_id
    assert qa.query_calls == []
    current = store.get_thread(session_id)
    assert current is not None
    assert current.instance_id == replacement.instance_id
    assert [message.content for message in current.messages] == [
        replacement_secret
    ]


def test_query_endpoint_rejects_invalid_session_id():
    client = TestClient(web_app.create_app(FakeQA()))

    response = client.post(
        "/api/query",
        json={"message": "What is Project Phoenix?", "session_id": "../bad"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid session id."


def test_query_endpoint_allows_no_context_loop():
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    client = TestClient(web_app.create_app(qa))

    response = client.post(
        "/api/query",
        json={"message": "What can you do?", "context_provider": "none"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "mock response" in payload["answer"]
    assert payload["summary"]["context_provider"] == "none"
    assert payload["summary"]["document"] is None
    assert payload["summary"]["format_check"] == "passed"
    assert payload["summary"]["final_decision"] == "not_verified"
    assert payload["trace"]["retrieved_chunk_count"] == 0
    assert payload["trace"]["citations"] == []
    assert payload["trace"]["self_check"]["outcome"] == "not_verified"
    assert payload["timeline"]["rows"][0]["step"] == "Input"
    assert payload["timeline"]["rows"][1]["phase"] == "Context"
    assert payload["timeline"]["rows"][1]["signals"] == "-"
    assert payload["trace"]["model_thinking"]["available"] is False
    assert payload["trace"]["model_thinking"]["content"] is None


def test_query_endpoint_rejects_blank_message():
    client = TestClient(web_app.create_app(FakeQA()))

    response = client.post("/api/query", json={"message": "  "})

    assert response.status_code == 400
    assert response.json()["detail"] == "Message is required."


def test_clear_chat_resets_session_state():
    fake_qa = FakeQA()
    fake_qa.chat_history = [{"question": "old"}]
    client = TestClient(web_app.create_app(fake_qa))

    response = client.post("/api/chat/clear")

    assert response.status_code == 200
    assert fake_qa.chat_history == []
    assert fake_qa.cleared_loop_session_id == "default"
    assert response.json()["timeline"]["empty"] is True


def test_clear_chat_resets_requested_session_state():
    class StaleAppendDuringClearQA(FakeQA):
        def clear_loop_session(self, session_id="default"):
            super().clear_loop_session(session_id)
            self.chat_history.append(
                {"session_id": session_id, "question": "stale in-flight"}
            )

    fake_qa = StaleAppendDuringClearQA()
    fake_qa.chat_history = [
        {"session_id": "thread_beta", "question": "old"},
        {"session_id": "thread_other", "question": "keep"},
    ]
    client = TestClient(web_app.create_app(fake_qa))

    response = client.post("/api/chat/clear", json={"session_id": "thread_beta"})

    assert response.status_code == 200
    assert fake_qa.chat_history == [
        {"session_id": "thread_other", "question": "keep"}
    ]
    assert fake_qa.cleared_loop_session_id == "thread_beta"
    assert response.json()["timeline"]["empty"] is True


def test_clear_chat_resets_persisted_thread_messages_only():
    fake_qa = FakeQA()
    store = ThreadStore.in_memory()
    store.create_thread(thread_id="thread_beta")
    store.create_thread(thread_id="thread_other")
    store.append_message("thread_beta", role="user", content="old")
    store.append_message("thread_other", role="user", content="keep")
    client = TestClient(web_app.create_app(fake_qa, thread_store=store))

    response = client.post("/api/chat/clear", json={"session_id": "thread_beta"})

    assert response.status_code == 200
    assert client.get("/api/threads/thread_beta").json()["messages"] == []
    other_messages = client.get("/api/threads/thread_other").json()["messages"]
    assert [message["content"] for message in other_messages] == ["keep"]


def test_clear_chat_does_not_recreate_deleted_thread():
    fake_qa = FakeQA()
    store = ThreadStore.in_memory()
    store.create_thread(thread_id="thread_deleted")
    store.delete_thread("thread_deleted")
    client = TestClient(web_app.create_app(fake_qa, thread_store=store))

    response = client.post(
        "/api/chat/clear",
        json={"session_id": "thread_deleted"},
    )

    assert response.status_code == 200
    assert client.get("/api/threads/thread_deleted").status_code == 404


def test_clear_chat_rejects_invalid_session_id():
    client = TestClient(web_app.create_app(FakeQA()))

    response = client.post("/api/chat/clear", json={"session_id": "bad/session"})

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid session id."


def test_loop_contract_redacts_guardrail_blocked_draft_everywhere():
    blocked_draft = "Sensitive blocked draft should not be public."
    secret_question = "Generate unsafe content with SECRET_USER_QUESTION."
    blocked_answer = "Blocked answer with SECRET_PUBLIC_ANSWER."
    blocked_thinking = "Blocked model thinking with SECRET_MODEL_THINKING."
    secret_document = "SECRET_BOARD_DOCUMENT.txt"
    secret_excerpt = "SECRET_CITATION_EXCERPT"
    evidence = EvidenceReference.from_source(
        citation_id=1,
        provider="document",
        source_identity=secret_document,
        page=None,
        chunk_index=0,
        excerpt=secret_excerpt,
    )
    run_started_at = utc_now()
    loop_report = LoopReport(
        run=LoopRun(
                run_id="run_blocked",
                started_at=run_started_at,
                completed_at=run_started_at,
            user_input=secret_question,
            context_provider="document",
            backend="mock",
            model_label="MockLLM (explicit demo)",
            steps=(
                LoopStep(
                    phase=LoopPhase.VERIFY,
                    decision=LoopDecision.NOT_VERIFIED,
                    name="Answer verifier",
                    input_summary=secret_question,
                    output_summary=blocked_draft,
                    verification=VerificationResult(
                        outcome=VerificationOutcome.UNSUPPORTED,
                        reasons=(blocked_draft,),
                        verifier="mock",
                        raw_response=blocked_draft,
                        metadata={"debug": blocked_draft},
                    ),
                        metadata={"draft_preview": blocked_draft},
                        started_at=run_started_at,
                        ended_at=run_started_at,
                    ),
                LoopStep(
                    phase=LoopPhase.ERROR,
                    decision=LoopDecision.BLOCK,
                    name="Guardrail decision",
                    output_summary=blocked_draft,
                    error_message=blocked_draft,
                        metadata={
                        "guardrail_decision": "block",
                            "guardrail_reason": blocked_draft,
                        },
                        started_at=run_started_at,
                        ended_at=run_started_at,
                    ),
                ),
                evidence=(evidence,),
                final_decision=LoopDecision.BLOCK,
            final_answer=blocked_answer,
            error_message=blocked_draft,
            metadata={"guardrail_detail": blocked_draft},
        )
    )
    result = QueryResult(
        answer=blocked_answer,
        trace=AnswerTrace(
            question=secret_question,
            document_name=secret_document,
            backend="mock",
            model_label="MockLLM (explicit demo)",
            retrieved_chunk_count=1,
            citations=[
                AnswerCitation(
                    citation_id=1,
                    source_name=secret_document,
                    page=None,
                    chunk_index=0,
                    excerpt=secret_excerpt,
                )
            ],
            self_check=AnswerSelfCheck(
                outcome="needs_refusal",
                reasons=[secret_excerpt],
            ),
            error_message=blocked_draft,
            model_thinking=blocked_thinking,
        ),
        loop_report=loop_report,
    )

    public_payload = query_response_dict(result)
    durable_payload = public_loop_payload_from_report(loop_report.to_public_dict())
    public_json = json.dumps(public_payload)

    raw_report_json = json.dumps(loop_report.to_dict())
    assert blocked_draft in raw_report_json
    assert blocked_answer in raw_report_json
    assert secret_question in raw_report_json
    assert blocked_draft not in public_json
    assert blocked_answer not in public_json
    assert secret_question not in public_json
    assert blocked_thinking not in public_json
    assert secret_document not in public_json
    assert secret_excerpt not in public_json
    assert public_payload["trace"]["citations"] == []
    assert public_payload["trace"]["self_check"] == {
        "outcome": "unsupported",
        "reasons": [],
        "retry_attempted": False,
    }
    assert public_payload["summary"]["last_error"] == PUBLIC_REDACTION_REASON
    assert public_payload["timeline"]["rows"][0]["signals"] == (
        "verifier: unsupported"
    )
    assert public_payload["trace"]["question"] == PUBLIC_REDACTION_TEXT
    assert public_payload["answer"] == PUBLIC_REDACTION_TEXT
    assert public_payload["trace"]["answer"] == PUBLIC_REDACTION_TEXT
    assert public_payload["trace"]["model_thinking"] == {
        "available": False,
        "redacted": True,
        "label": "Model Thinking (unverified)",
        "content": "[redacted: terminal loop decision]",
        "note": (
            "Model-emitted thinking is useful for debugging the loop, but it is "
            "not verified evidence."
        ),
    }
    assert public_payload["trace"]["loop_report"]["public_redaction"]["applied"] is True
    assert (
        durable_payload["trace"]["model_thinking"]
        == public_payload["trace"]["model_thinking"]
    )


def test_answer_trace_redacts_model_thinking_for_refused_results_without_guardrail():
    secret_thinking = "SECRET_REFUSED_MODEL_THINKING"
    refusal_answer = "I could not find enough relevant information in the provided evidence."
    run_started_at = utc_now()
    result = QueryResult(
        answer=refusal_answer,
        trace=AnswerTrace(
            question="What is unsupported?",
            document_name="phoenix.txt",
            backend="ollama",
            model_label="Ollama (nemotron-3-nano:4b)",
            retrieved_chunk_count=1,
            citations=[],
            model_thinking=secret_thinking,
        ),
        loop_report=LoopReport(
            run=LoopRun(
                    run_id="run_refused",
                    started_at=run_started_at,
                    completed_at=run_started_at,
                user_input="What is unsupported?",
                context_provider="document",
                backend="ollama",
                model_label="Ollama (nemotron-3-nano:4b)",
                final_decision=LoopDecision.REFUSE,
                final_answer=refusal_answer,
            )
        ),
    )

    trace = answer_trace_dict(result)
    trace_without_thinking = answer_trace_dict(
        replace(result, trace=replace(result.trace, model_thinking=None))
    )
    payload = query_response_dict(result)

    assert payload["answer"] == PUBLIC_REDACTION_TEXT
    assert payload["trace"]["answer"] == PUBLIC_REDACTION_TEXT
    assert payload["trace"]["question"] == PUBLIC_REDACTION_TEXT
    assert payload["summary"]["last_error"] == PUBLIC_REDACTION_REASON
    assert payload["trace"]["loop_report"]["public_redaction"]["applied"] is True
    assert secret_thinking not in json.dumps(trace)
    assert trace["model_thinking"] == {
        "available": False,
        "redacted": True,
        "label": "Model Thinking (unverified)",
        "content": "[redacted: terminal loop decision]",
        "note": (
            "Model-emitted thinking is useful for debugging the loop, but it is "
            "not verified evidence."
        ),
    }
    assert trace_without_thinking["model_thinking"] == trace["model_thinking"]


def test_loop_timeline_fallback_error_rows_are_sequential():
    result = QueryResult(
        answer="A query error occurred.",
        trace=AnswerTrace(
            question="What failed?",
            document_name="demo.txt",
            backend="mock",
            model_label="MockLLM (explicit demo)",
            retrieved_chunk_count=0,
            citations=[],
            error_message="backend unavailable",
        ),
    )

    timeline = loop_timeline_dict(result)

    assert [row["index"] for row in timeline["rows"]] == [1]
    assert timeline["rows"][-1]["phase"] == "Error"


def test_runtime_status_dict_preserves_honest_readiness_scope():
    fake_qa = FakeQA()
    status = runtime_status_dict(fake_qa.status())

    assert status["ready_for_queries"] is False
    assert status["readiness_scope"] == "retrieval_pipeline"
    assert status["inference_validated"] is False
    assert status["max_output_tokens"] == 384
