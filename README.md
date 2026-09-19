# Loopwright
Loopwright is a local-first, evidence-backed flight recorder for AI answer
loops. Today it records and hardens context selection, retrieval, drafting,
format checks, citation checks, claim verification, retries, refusals,
middleware guardrails, and evals. The direction is a cross-runtime trust plane
for inspecting and comparing evidence from external AI loops without becoming
another orchestrator. The current context routes are Smart routing, DuckDuckGo
web snippets, optional uploaded files, thread memory, and direct model
knowledge. Only prompt-used web snippets and indexed-file chunks are treated as
retrieved evidence; memory and direct model knowledge remain `not_verified`
without such evidence. The product focus is trust: making behavior visible,
testable, and harder—but not impossible—to fake.

## Features
- **Loop Engineering Core:** Treats retrieval, drafting, format checks, self-checking, retry, refusal, middleware guardrails, and evals as the product surface rather than hidden plumbing
- **Observable Runtime Reports:** Emits structured loop evidence for context selection, prompt evidence, drafts, format checks, verifier decisions, retries, refusals, and local export
- **Durable Local Runs:** Stores public loop-run summaries and reports in the
  local thread database across process restarts, and lets the browser reopen a
  historical run's versioned public artifact projection. Historical reads
  ignore cached public JSON and re-project the canonical raw report; malformed
  or identity-inconsistent raw records are quarantined instead of displayed
  Compatibility boundary: pre-binding `loop-report/v1` rows with a visible
  terminal answer but no terminal answer/evidence-set digests, plus supported
  rows without exact draft/verifier answer-and-evidence provenance, are retained
  as raw SQLite data and intentionally quarantined from canonical/public
  serving. Loopwright cannot safely infer the emitted answer or which evidence
  a verifier saw, so this upgrade is destructive for public inspection rather
  than a false migration.
- **Visible Thread Memory:** Shows per-thread memory counts and last-run use of
  recent conversation or semantic memory without exposing raw recalled text
- **Loop Recipes:** Provides saved loop recipes for goal, instructions,
  success criteria, stop condition, context provider, model profile, and verifier
  metadata
- **Smart Evidence Routing:** Uses web evidence for lookup/current questions,
  indexed files when a file is active and relevant, or direct model knowledge
  for private/local tasks such as rewriting, coding, and reasoning
- **Private Runtime Path:** Recommended local runtime is Ollama; cloud or
  gateway deployment uses a generic OpenAI-compatible chat-completions backend
- **Vector Search:** Uses FAISS for efficient similarity search with
  provider-backed embedding models through Ollama or OpenAI-compatible gateways
- **FastAPI Web App:** Real backend API plus a static browser UI for local loop
  threads, persistent local chat history, context indexing, runtime status, a
  readable loop timeline, compact loop summaries, and answer traces
- **Model Thinking in Chat:** Shows Ollama model-emitted thinking inline under
  assistant messages and in the loop detail panel when the model supports it,
  clearly labeled as unverified debugging signal rather than evidence
- **External Model Runtime:** Uses Ollama or an OpenAI-compatible gateway for
  generation so Python document indexing stays lightweight and stable


![Loopwright flow](docs/loopwright-flow.svg)

The gated product direction is documented in
[`docs/loopwright-trust-plane-plan.md`](docs/loopwright-trust-plane-plan.md).

## Installation

### Prerequisites
- Python 3.11 or 3.12; Python 3.12 is what CI and Docker use
- `uv` 0.12.15 or newer for the complete local workflow, including audits
  ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
- Ollama installed for the recommended local runtime
- Optional: an OpenAI-compatible model gateway for cloud or remote deployment
  (`/v1/chat/completions` shape)
- Enough memory for the Ollama model you choose; small models are strongly
  recommended on memory-constrained Macs

### Local Setup

1. Clone the repository:
    
    ```bash
    git clone https://github.com/dtkmn/loopwright.git
    cd loopwright
    ``` 

2. Install dependencies with `uv`:

    ```bash
    uv sync --locked --dev
    ```

   uv creates the project environment automatically. `--locked` rejects an
   outdated lockfile rather than changing dependency versions during setup.

3. Run with Ollama:

    Terminal 1, unless the Ollama desktop app/service is already running:

    ```bash
    ollama serve
    ```

    Terminal 2:

    ```bash
    ollama pull nemotron-3-nano:4b
    ollama pull embeddinggemma
    cp .env.example .env
    ```

    Edit `.env` if you pulled a different chat or embedding model. The app
    loads `.env` and `.env.local` automatically when started with
    `uv run --locked loopwright`; shell exports still override
    file values for one-off runs. Both local files are ignored by git.

    `LLM_BACKEND` selects the provider runtime. `LLM_MODEL` chooses the chat
    model. `EMBEDDINGS_MODEL` chooses the retrieval embedding model.
    `MODEL_THINKING=true` shows Ollama model-emitted thinking for models that
    advertise the `thinking` capability; set it to `false` if you want final
    answers and loop evidence only. `OLLAMA_THINK_LEVEL` can be set to `low`,
    `medium`, `high`, or `max` for models that support levels. GPT-OSS accepts
    only `low`, `medium`, or `high`; when left on `auto`, Loopwright sends
    `medium` for GPT-OSS and `true` for other thinking-capable Ollama models.

4. (Optional) choose a different backend:

    Supported values:
    - `ollama` uses a local Ollama server. This is the recommended local path.
    - `auto` selects Ollama and fails closed if Ollama or the configured model
      is unavailable. It does not fall back to mock.
    - `openai-compatible` uses any server that implements OpenAI-style
      `/v1/chat/completions`.
    - `mock` disables real inference for demos/tests.

5. (Optional) set up a cloud or local OpenAI-compatible endpoint in `.env`:

    ```dotenv
    LLM_BACKEND=openai-compatible
    OPENAI_COMPAT_BASE_URL=http://localhost:8000/v1
    LLM_MODEL=gpt-oss:20b
    EMBEDDINGS_MODEL=text-embedding-local
    OPENAI_COMPAT_API_KEY=optional_token_here
    ```

    `OPENAI_COMPAT_API_KEY` is optional for local gateways such as vLLM,
    llama.cpp server, LM Studio, or a private proxy. Set it for hosted services
    that require bearer auth. Plain `http://` is accepted only for loopback
    local development; non-loopback endpoints must use `https://`.

6. (Optional) tune quality and answer length in `.env`:

    ```dotenv
    FAST_MODE=false
    MAX_OUTPUT_TOKENS=1024
    ```

7. (Optional) enable debug logs in `.env`:

    ```dotenv
    APP_DEBUG=true
    ```

    Thread messages are stored locally in SQLite at
    `~/.loopwright/threads.sqlite3` by default. If you already have the old
    `~/.ai-loop-engine/threads.sqlite3` database and no Loopwright database
    exists yet, Loopwright keeps using the old file so existing history does
    not disappear during the rename. Override this when you want project-local
    or container-mounted persistence:

    ```dotenv
    LOOPWRIGHT_THREAD_DB_PATH=.loopwright/threads.sqlite3
    ```

8. Run the application:

    ```bash
    uv run --locked loopwright
    ```

## 🐳 Docker Setup

The application is containerized for easy deployment.

### Build the Docker Image

   ```bash
   docker build -t loopwright .
   ```

### Run the Container

   ```bash
   docker run -p 7860:7860 \
     -e LLM_BACKEND=mock \
     loopwright
   ```

For a deployed model gateway:

   ```bash
   docker run -p 7860:7860 \
     -e LLM_BACKEND=openai-compatible \
     -e OPENAI_COMPAT_BASE_URL=https://your-gateway.example/v1 \
     -e LLM_MODEL=your-chat-model \
     -e EMBEDDINGS_MODEL=your-embedding-model \
     -e OPENAI_COMPAT_API_KEY=optional_token_here \
     loopwright
   ```

**Note:** `LLM_BACKEND=auto` is local-runtime and real-backend-only: it selects
Ollama and fails closed if Ollama is not reachable. Use explicit
`LLM_BACKEND=mock` only for deterministic demos/tests.


## Usage
1. Open your browser and go to `http://localhost:7860`
2. Start a new thread or use the default thread, then ask normally. Recent
   same-thread messages and retrieved
   semantic thread memories are supplied to the model as bounded conversation
   context, and threads/messages are restored after app restart from the local
   SQLite store.
3. Pick a Loop Recipe when you want a saved goal/instruction/checking profile.
   The default recipe is selected automatically.
4. Switch threads from the sidebar when you want separate local conversations,
   memory counts, durable run history, and loop traces.
5. Ask normally. The default Smart Evidence loop automatically decides whether
   to use DuckDuckGo snippets for lookup/current questions, indexed files when
   a file is active and relevant, or direct model knowledge for private/local
   tasks. If automatically selected web evidence fails or cannot verify an
   answer, Smart Evidence falls back to direct model knowledge and marks the
   result `not_verified`. Explicit `context_provider` overrides remain
   available through the API and recipes for tests or power-user workflows.
6. Optionally upload a file (PDF, DOCX, TXT, or MD; max 25 MB) when you want
   local file-grounded retrieval, citations, and verifier-backed support checks.
7. Click "Index File" to make the uploaded file available to the loop.
8. Inspect the Loop Timeline to see recipe selection, context selection, retrieve, draft, format,
   check, verify, retry, refusal, and final-decision steps in order
9. Inspect Durable Runs to see persisted run evidence for the active thread
10. Inspect the loop summary for memory usage, provider, draft count, checks,
   verifier outcome and typed model provenance, retry/refusal state, final
   decision, and whether an error occurred
11. Select a Durable Run to reopen its public evidence, or open the current
    answer trace for the detailed `loop-public-report/v1` projection

## Technical Details

### Loop Contract
- **Context mode:** Direct no-context chat is allowed, but it is reported as
  `not_verified` with no citations. Direct answers should match the depth the
  user asks for, but model knowledge and thread memory are not treated as
  verified evidence. Smart Evidence, explicit web search, and indexed files are
  context providers that can upgrade the loop into grounded retrieval plus
  citation/verifier checks.
- **Current context providers:** Smart Evidence, web search, indexed files, and
  no external evidence. `context_provider=smart` is the default; legacy
  `context_provider=auto` is accepted as an alias. Smart Evidence uses web
  snippets for lookup/current questions, uses active indexed files for file-
  relevant questions, and stays in no-external-evidence mode for private/local
  tasks such as rewriting, coding, and reasoning. Automatic web attempts may
  degrade to a direct `not_verified` answer when snippets or verifier checks
  are insufficient; explicit `context_provider=web` remains evidence-strict.
- **Current evidence loop shape:** select evidence -> retrieve -> draft answer -> run format checks -> run mechanical checks -> verify cited claims -> retry once or fail closed -> return trace/status
- **Context provider boundary:** `DocumentContextProvider` is the legacy class
  name for local indexed-file retrieval; per-query web search uses the same
  retrieve/draft/check/verify loop
  without becoming durable uploaded context.
- **Typed loop primitives:** `src/loop_engine.py` defines provider-neutral `LoopRecipe`, `LoopRun`, `LoopStep`, `LoopDecision`, `LoopReport`, `LoopSession`, `LoopPolicy`, `EvidenceReference`, `GuardrailDecision`, `LoopMiddleware`, `VerificationResult`, and `HumanReviewRequest`
- **Runtime reports:** `AILoopEngine.query_with_trace()` returns a `QueryResult` with both the legacy answer trace and a first-class `LoopReport`
- **Thread state:** browser threads are backed by a local SQLite store for
  thread metadata, messages, durable public loop-run records, and the latest
  public loop payload. Recent same-thread messages are passed into the runtime
  as bounded conversation context. Older same-thread messages may also be
  retrieved by local embedding similarity as semantic thread memory; browser
  storage is only used to remember the selected thread and recipe. Thread
  messages remain raw local data, and the thread APIs do not provide
  authentication or access control. Public UI surfaces show memory counts and
  last-run use, not raw recalled memory text, but the public artifact projection
  does not make the surrounding thread store confidential.
- **Loop recipes:** saved local recipes provide reusable goal, instruction,
  success-criteria, stop-condition, context-provider, profile, and verifier
  metadata. They guide the run and are recorded in loop metadata, but they do
  not grant tool permissions or scheduling by themselves.
- **Runtime session state:** completed loop reports are also retained in bounded
  in-memory `LoopSession` objects keyed by `session_id` for local artifact export
  during the running process.
- **Session artifacts:** local JSONL export writes one raw `LoopReport` per line,
  suitable as future inspect/diff input
- **Public trace surface:** the FastAPI/static web app shows a readable Loop
  Timeline, compact loop summary, and the versioned
  `loop-public-report/v1` artifact. One allowlist-only projector is used by the
  runtime report API, durable history, adapters, and export CLI. It keeps typed
  operational provenance but omits prompts, arbitrary step names and
  summaries, raw errors and metadata, verifier reasons and raw payloads, recipe
  text, and human-review bodies. A valid non-guardrail completion may expose its
  final answer. Refuse, block, review, or another guardrail-like terminal signal
  suppresses the answer, model identity, and evidence; contradictory terminal
  contracts fail closed instead of being projected.
- **Answer and evidence binding:** every publicly visible non-guardrail terminal
  answer requires a terminal final step carrying exact answer and evidence-set
  correlation digests. A public `supported` result additionally requires an
  ordered real-backend draft candidate and supported verifier result carrying
  those same exact digests. Inline citation numbers must resolve to projected
  evidence. These are internal consistency checks, not cryptographic signatures
  or authenticity guarantees.
- **Public evidence:** projected evidence contains a stable SHA-256-derived
  identity, provider, citation number, and page/chunk locator only. It never
  contains a filename, title, URL, or excerpt. The digest identifies the exact
  reference/content inputs used to construct it; it does not establish semantic
  equivalence, authenticity, or secrecy.
- **Projection boundary:** the public artifact is a data-minimization surface,
  not authentication, access control, or a general PII/secret scrub. A
  non-guardrail terminal answer and typed provenance can still be sensitive, so
  every exported artifact must be handled accordingly.
- **Middleware boundary:** loop middleware can observe runs/steps, block unsafe progress, request retry/refusal, or mark a human-review pending state without introducing autonomous tool use
  Once a run has refused, blocked, or requested human review, its terminal
  decision is fixed. `after_step` can observe a recorded terminal step but cannot
  replace it. Final-step hooks are skipped for that outcome; `after_run` can
  observe it, but cannot change it. Earlier checks and nonterminal completions
  still permit middleware enforcement.
- **Framework posture:** OpenAI trace-shaped JSON and LangGraph manifest JSON
  are export targets today; their adapter modules do not import or execute the
  framework SDKs. Microsoft Agent Framework remains a future export target. The
  adapter strategy lives in
  [`docs/framework-adapter-strategy.md`](docs/framework-adapter-strategy.md).

### Model
- **LLM backend:** Configurable via `LLM_BACKEND`
  - `ollama`: local Ollama server via `OLLAMA_BASE_URL`; recommended for local use
  - `auto` (default): local real-backend path; selects Ollama and fails closed if unavailable
  - `openai-compatible`: OpenAI-style `/v1/chat/completions` endpoint for cloud,
    private gateway, vLLM, llama.cpp server, LM Studio, or similar runtimes
  - `mock`: explicit deterministic demo/test backend; never used as fallback
- **Chat model:** configure with `LLM_MODEL`. Provider-specific aliases
  `OLLAMA_MODEL` and `OPENAI_COMPAT_MODEL` are still accepted for compatibility.
- **Embedding model:** configure with `EMBEDDINGS_MODEL`. Ollama defaults to
  `embeddinggemma`; OpenAI-compatible gateways require an explicit embedding
  model. Provider-specific aliases `OLLAMA_EMBED_MODEL` and
  `OPENAI_COMPAT_EMBED_MODEL` are accepted for compatibility.
- **Ollama:** optionally configurable with loopback-only `OLLAMA_BASE_URL`
  (default `http://localhost:11434`) and `OLLAMA_TIMEOUT`
- **OpenAI-compatible endpoint:** requires `OPENAI_COMPAT_BASE_URL` and
  `LLM_MODEL`; optionally set `OPENAI_COMPAT_API_KEY` and
  `OPENAI_COMPAT_TIMEOUT`
- **Mock embeddings:** `LLM_BACKEND=mock` uses deterministic local hashing
  embeddings (`local-hashing-384`) for demos/tests only.
- **Vector Store:** FAISS for efficient similarity search
- **Retrieval primitives:** lightweight LangChain Core interfaces plus
  `langchain-text-splitters`; Loopwright owns loop orchestration.

### Configuration
- **Response Length:** 1024 new tokens (quality) / 384 new tokens (fast) by
  default. Override with `MAX_OUTPUT_TOKENS` when you want longer or shorter
  local answers.
- **Generation mode:** temperature `0` / greedy-style settings reduce variance
  for context-grounded answers, but do not guarantee bit-for-bit replay across
  models, servers, hardware, or versions.
- **Chunk Size:** 1200/200 overlap (quality) / 900/120 overlap (fast)
- **Retrieval:** MMR retrieval with source/page grounding
  - **Quality:** `k=6`, `fetch_k=24`
  - **Fast:** `k=3`, `fetch_k=10`
- **Web search:** Smart Evidence lookup/current queries and explicit
  `context_provider=web` queries use fixed DuckDuckGo Instant Answer and result
  snippet endpoints for snippets only. The app does not fetch arbitrary result
  pages. Configure bounded provider behavior with `WEB_SEARCH_TIMEOUT` and
  `WEB_SEARCH_MAX_RESULTS`.
- **Safety limits:** Max upload size 25 MB, chunk cap 2,000 chunks per document
- **Native runtime defaults:** unless you override them, app entrypoints
  bootstrap `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `VECLIB_MAXIMUM_THREADS`, and tokenizer parallelism before FastAPI, NumPy,
  or FAISS load native libraries. This is intentional: upload stability beats
  native thread-pool surprises on local Macs.

## Runtime Direction
- Model inference and persistence can remain local by default and are portable
  across local or gateway runtimes. Smart Evidence may send lookup/current
  queries to DuckDuckGo; select a local-only context mode when that network
  disclosure is unacceptable. Ollama is the recommended path for Mac and
  workstation use because it keeps model setup outside the Python dependency
  graph and avoids requiring cloud credentials.
- Cloud/deployed inference should go through the generic OpenAI-compatible
  backend, not a provider-specific happy path.
- First-party model providers are intentionally limited to Ollama and generic
  OpenAI-compatible gateways. Do not add provider-specific token paths unless a
  new product decision makes that tradeoff explicit.
- New Loopwright features should work through the local Ollama path first
  and the OpenAI-compatible deployment path second.
- On Apple Silicon, keep generation and embeddings outside this Python process
  by using Ollama; mock mode keeps built-in hashing only for deterministic
  demos/tests.

## Loop Engineering Pattern
This repo is intentionally built around three loops:

- **Runtime answer loop:** evidence-backed route: retrieve -> cite -> draft ->
  format/mechanical checks -> cited-support decision -> bounded retry or
  terminal decision. Direct route: draft -> format check -> `not_verified`.
- **Guardrail loop:** middleware hooks can run before/after runs and steps, and
  can return typed decisions: continue, retry, refuse, block, or
  requires_review. A middleware retry request currently fails closed because a
  safe execution retry path is not implemented.
- **Engineering loop:** change one contract -> add focused regressions -> run
  golden loop evals -> run broad validation -> ask for review -> stage only
  intentional files.

Golden evals live in `tests/test_golden_document_eval.py` and the CLI lives in
`src/loop_eval.py`. The test suite is provider-free; the CLI can also write a
JSON artifact that includes the loop reports used to score each case:

```bash
uv run --locked python -m src.loop_eval --mode fake --artifact artifacts/loop-eval.json
uv run --locked pytest tests/test_golden_document_eval.py -q
uv run --locked pytest tests/test_loop_eval.py -q
uv run --locked pytest
```

Use these before adding planner loops, tools, multi-context memory, or more
agent-like behavior. Blunt rule: if the bounded answer loop is not
measurably honest, bigger agent features will only make the failure harder to see.

### Framework Adapter Strategy

Frameworks are interop surfaces, not the engine. The current plan is to export
Loopwright reports into framework-shaped artifacts before adding any live
framework runtime integration:

- OpenAI Agents SDK: trace-shaped export, dependency-free in
  `src.adapters.openai_trace`
- LangGraph: thread/checkpoint manifest export, dependency-free in
  `src.adapters.langgraph_manifest`
- Microsoft Agent Framework: workflow event-stream export first, not yet
  implemented

See [`docs/framework-adapter-strategy.md`](docs/framework-adapter-strategy.md)
for mappings, non-goals, and the dependency boundary.

Export a report or session locally when you need framework-shaped JSON for
inspection or downstream tooling:

```python
from src.adapters.openai_trace import export_report, export_session
from src.adapters.langgraph_manifest import export_session as export_langgraph_session

trace_payload = export_report(query_result.loop_report)
session_payload = export_session(qa_system.loop_session("default"))
langgraph_payload = export_langgraph_session(qa_system.loop_session("default"))
```

These helpers do not import the OpenAI Agents SDK, call OpenAI APIs, or mutate
the original loop reports. They also do not import or execute LangGraph.
Public export is the default and consumes the same versioned,
allowlist-only `loop-public-report/v1` projection used by the web and durable
run surfaces. It does not copy raw report fields and then try to redact known
secrets. Terminal guardrail-like outcomes suppress the answer, model identity,
and evidence. This projection is data minimization, not access control or a
general secret/PII scrub, so treat every exported artifact as potentially
sensitive.
Use `public=False` only for local diagnostics you are willing to treat as even
more sensitive.

Use the local export CLI when starting from a JSONL session artifact:

```bash
uv run --locked python -m src.loop_export \
  --adapter openai-trace \
  --input artifacts/loop-session-default.jsonl \
  --output artifacts/openai-trace.json

uv run --locked python -m src.loop_export \
  --adapter langgraph-manifest \
  --input artifacts/loop-session-default.jsonl \
  --output artifacts/langgraph-manifest.json
```

The CLI defaults to the same versioned public artifact projection. `--raw` is
intentionally explicit because raw loop reports can contain prompts, retrieved
excerpts, drafts, verifier payloads, and final answers. Public artifacts can
still contain a valid non-guardrail terminal answer and typed provenance; they are
not automatically safe to publish.

The CLI's `--raw` mode also accepts the original `loop-report/v1` JSONL shape
that predates evidence identities, terminal reasons, and verifier provenance.
Those missing fields remain empty or unknown. This compatibility path does not
invent evidence bindings or make legacy records eligible for public export.

### Local Session Artifacts

`AILoopEngine` keeps recent loop reports in memory per `session_id`. Export a
session locally when you need a raw diagnostic artifact or future inspect/diff
input:

```python
qa_system.export_loop_session_jsonl("artifacts/loop-session-default.jsonl")
```

Each JSONL line is a raw `loop-report/v1` object. Treat these files as local
developer diagnostics because they may include prompts, retrieved excerpts,
draft outputs, and final answers. Planned inspect/diff commands should look like:

```bash
uv run --locked python -m src.loop_replay inspect artifacts/loop-session-default.jsonl
uv run --locked python -m src.loop_replay diff before.jsonl after.jsonl
```

Those commands are intentionally not implemented yet. The report and public
projection shapes need to stay stable before inspect/diff becomes a real
product surface. Deterministic model re-execution is not implemented.

### Optional Live Ollama Model Eval

CI stays provider-free. When you want to compare a pulled local Ollama model,
run the unified loop eval command manually. Ollama mode exercises the configured
chat model and embedding model, so make sure both models are pulled first. Each
case performs document indexing, retrieval, answer, and verifier calls, so start
with one model and one case on memory-constrained Macs. The live eval command
only accepts loopback Ollama URLs such as
`http://localhost:11434` or `http://127.0.0.1:11434`; it is not an arbitrary
remote model benchmark tool.

```bash
uv run --locked python -m src.loop_eval \
  --mode ollama \
  --models nemotron-3-nano:4b \
  --case launch_date \
  --timeout 30 \
  --artifact artifacts/loop-eval-ollama-launch.json \
  --no-fail
```

Then run the full golden set for one model:

```bash
uv run --locked python -m src.loop_eval \
  --mode ollama \
  --models nemotron-3-nano:4b \
  --all-cases \
  --timeout 60 \
  --artifact artifacts/loop-eval-ollama-full.json \
  --no-fail
```

Score the artifacts by loop evidence: phases, citations, verifier decisions,
retry/refusal state, and final decision. Do not judge models by answer text
alone.

The command refuses multiple Ollama models by default so a comparison run does not
accidentally overload a local Mac. Prefer one model per command. Only use the
override when you have enough free unified memory and are comfortable watching
resource pressure:

```bash
uv run --locked python -m src.loop_eval \
  --mode ollama \
  --models nemotron-3-nano:4b qwen3:8b \
  --allow-multi-model \
  --all-cases \
  --timeout 60 \
  --artifact artifacts/loop-eval-ollama-compare.json \
  --no-fail
```

The command asks Ollama to unload each model after its run by default. If your
machine still feels memory pressure, stop the run and inspect resident models:

```bash
ollama ps
ollama stop nemotron-3-nano:4b
ollama stop qwen3:8b
```

## Security and Dependency Maintenance
- Dependencies are declared in `pyproject.toml` and locked in `uv.lock`.
  Local setup, CI, and Docker all install with `uv sync --locked`; requirements
  exports are no longer maintained. Docker installs runtime dependencies only
  and runs the installed app without resolving dependencies at startup.
- Add dependencies with `uv add`, or `uv add --dev` for development tools. Update
  an existing dependency with `uv lock --upgrade-package PACKAGE`, then run
  `uv sync --locked --dev` and the checks below. Commit `pyproject.toml` and
  `uv.lock` together when both change.
- Dependabot uses its uv ecosystem for weekly dependency updates.
- Audits use [`uv audit`](https://docs.astral.sh/uv/reference/cli/#uv-audit)
  directly against the lockfile. That command is currently experimental, so CI
  and Docker pin uv to 0.12.15. There is no project-wide uv version floor, so
  Dependabot can update the lockfile with its bundled uv. Use 0.12.15 or newer
  for local audits. Audit failures remain blocking; no intermediate requirements
  export is needed.
- The runtime uses FastAPI, Uvicorn, FAISS CPU, LangChain components, local
  text/document parsers, and the stdlib-HTTP Ollama/OpenAI-compatible model
  adapters. Treat `uv.lock` as the exact baseline; lower bounds remain in
  `pyproject.toml`.
- Direct and transitive dependency versions are resolved and locked (including
  `python-multipart`, `urllib3`, and `orjson`) so audits are reproducible without
  misrepresenting lower bounds as exact pins.
- Recommended recurring checks:

  ```bash
  uv sync --locked --dev
  uv lock --check
  uv run --locked pytest tests/test_loop_engine.py -q
  uv run --locked pytest tests/test_golden_document_eval.py -q
  uv run --locked pytest tests/test_loop_eval.py -q
  uv run --locked pytest tests/test_ollama_model_eval.py -q
  uv run --locked pytest
  uv audit --locked --no-dev
  uv pip check
  ```

## Agent-Assisted Development
- `AGENTS.md` contains repo-level instructions for coding agents: setup commands,
  validation expectations, backend honesty rules, encoding policy, and release
  guardrails.
- `.agents/skills/loop-engineering/SKILL.md` defines the focused loop-engineering
  skill for changes to loop contracts, evidence context, retrieval, model
  routing, UI status, evals, and CI publishing.
- Use the documented loop for non-trivial changes: explore, plan, act, observe,
  verify, review, and ship.

## Contributing
Feel free to submit issues and pull requests. Contributions are welcome!

## License
This project is open source and available under the MIT License.
