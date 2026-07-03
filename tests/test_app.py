import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from src import app as web_app
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
    DEFAULT_LOOP_RECIPE_ID,
    LoopDecision,
    LoopPhase,
    LoopRecipe,
    LoopReport,
    LoopRun,
    LoopStep,
    PUBLIC_REDACTION_REASON,
    PUBLIC_REDACTION_TEXT,
    VerificationOutcome,
    VerificationResult,
)
from src.thread_store import ThreadStore
from src.web_contract import (
    answer_trace_dict,
    loop_summary_dict,
    loop_timeline_dict,
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
                ),
                LoopStep(
                    phase=LoopPhase.MECHANICAL_CHECK,
                    decision=LoopDecision.NOT_VERIFIED,
                    name="Mechanical checks",
                    output_summary="mechanical_checks_passed",
                ),
            ]
        )
        loop_report = LoopReport(
            run=LoopRun(
                run_id="run_fake",
                session_id=session_id,
                user_input=message,
                context_provider="document",
                backend=active_backend,
                model_label=active_model_label,
                steps=tuple(steps),
                final_decision=LoopDecision.NOT_VERIFIED,
                final_answer=answer,
                metadata={
                    "conversation_context_turns": len(self.last_conversation_history),
                    "semantic_memory_turns": len(self.last_semantic_memory),
                    "semantic_memory_status": semantic_memory_status,
                    "recipe_id": self.last_loop_recipe.get("recipe_id"),
                    "recipe_name": self.last_loop_recipe.get("name"),
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
                model_thinking="I matched Project Phoenix against citation [1].",
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


def test_static_frontend_is_served():
    client = TestClient(web_app.create_app(FakeQA()))

    response = client.get("/")
    script = client.get("/assets/app.js")
    styles = client.get("/assets/styles.css")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert "AI Loop Engine" in response.text
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
        check=True,
    )
    assert result.stderr == ""


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
        final_decision: "not_verified",
        context_provider: "none",
        backend: "mock",
        model: "MockLLM",
        recipe_id: "recipe_general_loop",
        recipe_name: "General assistant loop",
        step_count: 4,
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
await dom["clear-chat"].dispatch("click");
assert.ok(
  nodeText(dom["active-thread-memory"]).includes("0 memories indexed"),
  "clear should reset visible indexed memory count",
);
assert.ok(
  nodeText(dom["memory-status"]).includes("last run did not use thread memory"),
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
    return jsonResponse({ title: "AI Loop Engine" });
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
await staleDom["clear-chat"].dispatch("click");
assert.equal(staleDom["query-button"].disabled, false);
assert.equal(staleDom["query-input"].disabled, false);
assert.equal(staleDom["query-context"].disabled, false);
assert.equal(staleDom["final-decision"].textContent, "idle");
staleQuery.resolve();
await pendingSubmit;
const staleAssistant = findNode(staleDom.messages, (node) => node.className === "message assistant");
assert.equal(staleAssistant, null);
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
assert.equal(globalThis.localStorage.getItem("ai-loop-engine.active-thread.v1"), "thread_keep");
assert.deepEqual(detailFetches, ["thread_delete", "thread_keep"]);
assert.equal(dom["query-button"].disabled, false);

await dom["delete-thread"].dispatch("click");
await tick();
assert.deepEqual(deleted, ["thread_delete", "thread_keep"]);
assert.equal(createdCount, 1);
assert.equal(serverThreads.length, 1);
assert.equal(serverThreads[0].id, "thread_created_1");
assert.equal(dom["active-thread-title"].textContent, "New thread");
assert.equal(globalThis.localStorage.getItem("ai-loop-engine.active-thread.v1"), "thread_created_1");
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


def test_static_frontend_query_completion_during_thread_delete_stays_disabled():
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

const pendingDelete = dom["delete-thread"].dispatch("click");
await tick();
assert.equal(dom["query-button"].disabled, true);

queryGate.resolve();
await pendingQuery;
await tick();
assert.equal(
  dom["query-button"].disabled,
  true,
  "query completion must not re-enable controls while active thread deletion is pending",
);
const staleAssistant = findNode(
  dom.messages,
  (node) =>
    node.className === "message assistant" &&
    nodeText(node).includes("Stale answer should not render"),
);
assert.equal(staleAssistant, null);

deleteGate.resolve();
await pendingDelete;
await tick();
assert.equal(createdCount, 1);
assert.equal(dom["active-thread-title"].textContent, "New thread");
assert.equal(dom["query-button"].disabled, false);
'''
    )


def test_static_frontend_running_query_delete_does_not_enable_other_pending_delete():
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
  thread_a: deferred(),
  thread_b: deferred(),
};
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
    return jsonResponse({ answer: "Old answer", timeline: { rows: [], final_decision: "not_verified" }, summary: {}, trace: { model_thinking: null } });
  }
  if (url.startsWith("/api/threads/") && method === "DELETE") {
    const id = decodeURIComponent(url.slice("/api/threads/".length));
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
await clickThreadByTitle("Thread B");
const pendingDeleteB = dom["delete-thread"].dispatch("click");
await tick();
assert.equal(dom["active-thread-title"].textContent, "Thread B");
assert.equal(dom["query-button"].disabled, true);

gates.thread_a.resolve();
await pendingDeleteA;
await tick();
assert.equal(
  dom["query-button"].disabled,
  true,
  "finishing delete A must not enable query controls while active thread B is deleting",
);

queryGate.resolve();
await pendingQuery;
await tick();
assert.equal(dom["query-button"].disabled, true);

gates.thread_b.resolve();
await pendingDeleteB;
await tick();
assert.equal(createdCount, 1);
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
    return jsonResponse({ ...recipes.find((recipe) => recipe.recipe_id === "recipe_custom"), exported_from: "AI Loop Engine" });
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
assert.equal(exportedRecipe.exported_from, "AI Loop Engine");
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

    assert config["title"] == "AI Loop Engine"
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
    assert payload["timeline"]["rows"][0]["step"] == "Apply loop recipe"
    assert payload["timeline"]["rows"][0]["signals"] == "General assistant loop"
    assert payload["timeline"]["rows"][1]["phase"] == "Retrieve"
    assert payload["timeline"]["rows"][1]["signals"] == "1 prompt chunks; chunks: 1; citations: 1"
    assert payload["summary"]["document"] == "demo.txt"
    assert payload["summary"]["final_decision"] == "not_verified"
    assert payload["summary"]["conversation_context_count"] == 0
    assert payload["summary"]["semantic_memory_count"] == 0
    assert payload["summary"]["semantic_memory_status"] == "not_requested"
    assert payload["summary"]["recipe_id"] == "recipe_general_loop"
    assert payload["summary"]["recipe_name"] == "General assistant loop"
    assert payload["recipe"]["recipe_id"] == "recipe_general_loop"
    assert payload["trace"]["question"] == "What is Project Phoenix?"
    assert payload["trace"]["citations"][0]["source"] == "demo.txt"
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
    assert payload["summary"]["recipe_name"] == "Custom recipe"
    assert thread["loop_runs"][0]["recipe_id"] == "recipe_custom"


def test_query_endpoint_refreshes_stale_builtin_default_recipe():
    fake_qa = FakeQA()
    store = ThreadStore.in_memory()
    stale_default = LoopRecipe(
        recipe_id=DEFAULT_LOOP_RECIPE_ID,
        name="General assistant loop",
        description="Default local-first loop behavior.",
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
    assert exported.json()["exported_from"] == "AI Loop Engine"

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


def test_query_endpoint_persists_thread_messages_and_latest_payload():
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
    assert thread["latest"]["summary"]["final_decision"] == "not_verified"
    assert thread["memory_count"] == 2
    assert thread["loop_run_count"] == 1
    assert thread["loop_runs"][0]["run_id"] == "run_fake"
    assert thread["loop_runs"][0]["recipe_id"] == "recipe_general_loop"
    response_payload = response.json()
    assert response_payload["run"]["run_id"] == "run_fake"
    assert response_payload["thread"]["memory_count"] == 2
    assert thread["latest"]["trace"]["loop_report"]["run"]["session_id"] == (
        "thread_alpha"
    )

    runs = client.get("/api/threads/thread_alpha/runs").json()
    run_detail = client.get("/api/threads/thread_alpha/runs/run_fake").json()
    assert runs["runs"][0]["run_id"] == "run_fake"
    assert run_detail["public"] is True
    assert run_detail["report"]["run"]["run_id"] == "run_fake"
    assert run_detail["report"]["run"]["metadata"]["recipe_id"] == (
        "recipe_general_loop"
    )


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
        row
        for row in payload["timeline"]["rows"]
        if row["step"] == "Retrieve thread memory"
    ]
    assert memory_rows
    assert "memory: 1" in memory_rows[0]["signals"]
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


def test_query_endpoint_does_not_persist_partial_turn_on_runtime_failure():
    class FailingQA(FakeQA):
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
            raise RuntimeError("backend unavailable")

    fake_qa = FailingQA()
    store = ThreadStore.in_memory()
    client = TestClient(
        web_app.create_app(fake_qa, thread_store=store),
        raise_server_exceptions=False,
    )

    response = client.post(
        "/api/query",
        json={"message": "hello", "session_id": "thread_fail"},
    )

    assert response.status_code == 500
    thread = client.get("/api/threads/thread_fail").json()
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
    assert responses["query"].status_code == 200
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
    assert responses["query"].status_code == 200
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
    assert responses["query"].status_code == 200
    thread = client.get("/api/threads/thread_race").json()
    assert thread["id"] == recreated.id
    assert thread["messages"] == []
    assert thread["message_count"] == 0
    assert thread["loop_run_count"] == 0
    assert thread["latest"] is None


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
    assert payload["summary"]["format_check"] == "format_passed"
    assert payload["summary"]["final_decision"] == "not_verified"
    assert payload["trace"]["retrieved_chunk_count"] == 0
    assert payload["trace"]["citations"] == []
    assert payload["trace"]["self_check"]["outcome"] == "not_verified"
    assert payload["timeline"]["rows"][0]["step"] == "Apply loop recipe"
    assert payload["timeline"]["rows"][1]["phase"] == "Context"
    assert payload["timeline"]["rows"][1]["signals"] == "no_context_provider"
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
    loop_report = LoopReport(
        run=LoopRun(
            run_id="run_blocked",
            user_input=secret_question,
            context_provider="document",
            backend="mock",
            model_label="MockLLM (explicit demo)",
            steps=(
                LoopStep(
                    phase=LoopPhase.DRAFT,
                    decision=LoopDecision.CONTINUE,
                    name="Draft answer",
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
                ),
            ),
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
            document_name="phoenix.txt",
            backend="mock",
            model_label="MockLLM (explicit demo)",
            retrieved_chunk_count=0,
            citations=[],
            error_message=blocked_draft,
            model_thinking=blocked_thinking,
        ),
        loop_report=loop_report,
    )

    public_payload = query_response_dict(result)
    public_json = json.dumps(public_payload)

    raw_report_json = json.dumps(loop_report.to_dict())
    assert blocked_draft in raw_report_json
    assert blocked_answer in raw_report_json
    assert secret_question in raw_report_json
    assert blocked_draft not in public_json
    assert blocked_answer not in public_json
    assert secret_question not in public_json
    assert blocked_thinking not in public_json
    assert public_payload["summary"]["last_error"] == PUBLIC_REDACTION_REASON
    assert public_payload["timeline"]["rows"][0]["signals"] == (
        f"{PUBLIC_REDACTION_TEXT}; "
        "verifier: unsupported; "
        f"reasons: {PUBLIC_REDACTION_TEXT}"
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


def test_answer_trace_redacts_model_thinking_for_refused_results_without_guardrail():
    secret_thinking = "SECRET_REFUSED_MODEL_THINKING"
    refusal_answer = "I could not find enough relevant information in the provided evidence."
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
    payload = query_response_dict(result)

    assert payload["answer"] == refusal_answer
    assert payload["trace"]["answer"] == refusal_answer
    assert payload["trace"]["question"] == "What is unsupported?"
    assert payload["summary"]["last_error"] is None
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

    assert [row["index"] for row in timeline["rows"]] == [1, 2, 3]
    assert timeline["rows"][-1]["phase"] == "Error"


def test_runtime_status_dict_preserves_honest_readiness_scope():
    fake_qa = FakeQA()
    status = runtime_status_dict(fake_qa.status())

    assert status["ready_for_queries"] is False
    assert status["readiness_scope"] == "retrieval_pipeline"
    assert status["inference_validated"] is False
    assert status["max_output_tokens"] == 384
