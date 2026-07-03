---
name: loop-engineering
description: Use when changing loop contracts, evidence context, retrieval, LLM backend routing, FastAPI/web upload/query behavior, evals, or CI release policy for AI Loop Engine.
---

# Loop Engineering

## Purpose

Use this skill to evolve the workbench without weakening its core promise: make
agent loops observable, testable, local-first, and honest. Smart Evidence,
web snippets, uploaded files, thread memory, and direct model knowledge are
evidence providers; none of them is the whole product boundary.

## Operating Loop

1. Perceive: identify which user-facing contract is affected: ingestion,
   encoding, retrieval, loop trace, LLM backend, UI status, dependency hygiene,
   or release automation.
2. Plan: write down the smallest behavior change that improves that contract.
3. Act: edit the owned files only. Keep unrelated refactors out of the patch.
4. Observe: run focused tests for the changed behavior and inspect failure modes,
   not just success paths.
5. Verify: run broader checks when shared behavior changes.
6. Review: when subagent tooling is available, ask a subagent for actionable
   findings on the actual diff. Otherwise, perform a direct review pass and
   report that no subagent tooling was available.
7. Ship: stage intentional files and report validation honestly.

## Runtime Contracts

- `auto` backend is local-first and real-backend-only: select Ollama and fail
  closed when Ollama or the configured model is unavailable. It must not fall
  back to mock.
- Ollama runtime URLs must be loopback local only. Remote/cloud model gateways
  must use `LLM_BACKEND=openai-compatible`, not `OLLAMA_BASE_URL`.
- Explicit real backends must fail closed: `ollama` and
  `openai-compatible`. Removed backend names such as `endpoint` and `local`
  must fail closed as invalid configuration.
- `openai-compatible` is the cloud/private-gateway deployment path. Use
  `OPENAI_COMPAT_BASE_URL`, `LLM_MODEL`, `EMBEDDINGS_MODEL`, optional
  `OPENAI_COMPAT_API_KEY`, and `OPENAI_COMPAT_TIMEOUT`. Plain HTTP is allowed
  only for loopback local development; remote gateways must use HTTPS.
- `LLM_BACKEND` is the single provider/runtime selector. Do not add a separate
  embedding backend variable. Configure the chat model with `LLM_MODEL` and the
  retrieval model with `EMBEDDINGS_MODEL`; provider-specific model env vars are
  compatibility aliases only.
- Native runtime defaults must be installed before FastAPI, NumPy, FAISS,
  or other native-heavy imports in app entrypoints. Use `src.native_runtime`
  instead of duplicating env setup in modules that may be imported too late.
- Local `.env` and `.env.local` files are loaded before native defaults in the
  app entrypoint. They must not override shell environment variables and must
  stay disabled under tests.
- Response length is explicit runtime configuration. `FAST_MODE` controls the
  speed/retrieval profile, while `MAX_OUTPUT_TOKENS` controls generation budget
  and must be reflected in runtime status.
- Runtime imports must not eagerly load FAISS or `src.retrieval`; keep retrieval
  lazy to indexing/search paths or explicit compatibility imports.
- Embedding runtime follows the selected provider. Ollama uses `/api/embed`;
  OpenAI-compatible gateways use `/embeddings`; mock mode uses built-in local
  hashing for deterministic demos/tests.
- `LLM_BACKEND=auto` must select Ollama only. It must not silently select mock,
  provider-specific hosted backends, or in-process model loading.
- Product identity is AI Loop Engine. Treat uploaded-file answering and web
  evidence as context provider capabilities, not the repo's strategic identity.
- File and web evidence are `ContextProvider`-shaped capabilities; keep
  provider identity in loop reports instead of hardcoding document-specific
  assumptions in UI code.
- Smart Evidence is the default query-time context mode. No-context answers may
  draft with the selected LLM backend, but they must report
  `context_provider="none"`, zero citations, and `not_verified`; do not run
  verifier support claims without prompt evidence. `context_provider=smart` may
  route lookup/current questions to fixed DuckDuckGo web snippets, file-relevant
  questions to indexed files, and private/local tasks such as rewriting, coding,
  or reasoning to direct model knowledge; legacy `context_provider=auto` is an
  alias for `smart`.
  `context_provider=web`, `document`, and `none` remain explicit user
  overrides.
- Typed loop records are the contract surface for future agent work. Add or
  update `LoopRecipe`, `LoopRun`, `LoopStep`, `LoopDecision`, `LoopReport`,
  `LoopPolicy`, `LoopSession`, `GuardrailDecision`, `LoopMiddleware`,
  `VerificationResult`, and `HumanReviewRequest` before adding planner,
  multi-agent, tool, replay, or framework-adapter behavior.
- `AILoopEngine.query_with_trace()` must expose a `LoopReport` that matches the
  actual query path: prompt evidence, draft, mechanical check, verifier outcome,
  retry/refusal state, and final answer.
- Completed query reports should be retained in bounded in-memory `LoopSession`
  state. Local JSONL export may write raw reports for developer replay/debug
  artifacts; public UI traces must continue using the redacted report surface.
- Web/API threads must pass an explicit validated `session_id` into the runtime.
  FastAPI owns local SQLite persistence for thread metadata, messages, durable
  public loop-run records, recipes, and latest public loop payloads. Recent
  same-thread messages should be passed as bounded conversation context so
  follow-up questions can resolve references without leaking across threads.
  Older same-thread messages may be retrieved by embedding similarity as
  semantic thread memory, but public loop traces should expose only memory
  counts/status, not raw memory text. Browser storage is only a
  selected-thread/recipe hint. Do not imply authenticated, cross-device, or
  account-backed history until that architecture exists.
- Loop recipes are reusable local skill definitions: goal, instructions,
  success criteria, stop condition, context provider, model profile, and
  verifier metadata. They may guide drafting and must be recorded in loop
  metadata, but they must not grant tool access, scheduling, or autonomous
  action by themselves.
- Loop middleware is a guardrail/telemetry boundary. It may block, refuse, retry,
  or request human review, but it must not introduce autonomous tools by itself.
- Upload status must say `indexed` for real backends, because endpoint readiness
  is not proven until the first inference call.
- Upload replacement must be transactional. Failed uploads must not replace the
  previous successful document, vector store, retrieval chain, or query behavior.
- UI upload/runtime status must come from `AILoopEngine.status()` and the latest
  `DocumentProcessingReport`, not ad hoc reads of internal attributes.
- Answer traces and citations must come from the same retrieved chunks used in
  the prompt. Keep `query()` string-compatible and expose richer evidence
  through structured APIs such as `query_with_trace()`.
- Answer-loop policy belongs in `src.answer_loop`; runtime should orchestrate
  loop state and middleware, not re-own mechanical citation validation,
  verifier parsing, retry instructions, or fail-closed self-check decisions.
- Answer self-checking should inspect the generated answer and trace. Cheap
  mechanical checks and deterministic refutation prefilters may reject bad
  answers, but only a real backend verifier may label an answer `supported`.
  Mock/demo mode must report mechanically valid answers as `not_verified`.
- Ollama model thinking is model-emitted debug signal. Show it only when the
  provider exposes it, label it as unverified, keep it out of final answer text,
  and drop/redact it for refused, blocked, or terminal-public-redacted results.
  GPT-OSS thinking must use `OLLAMA_THINK_LEVEL` with `low`, `medium`,
  or `high`; boolean `think` values are ignored by that model family.
- Golden evidence evals must exercise the full provider-free answer loop:
  upload, retrieval, cited answer, self-check, retry, and fail-closed refusal.
  Do not require a live Ollama backend for these CI checks.
- `src.loop_eval --mode fake` is the provider-free CLI surface for JSON loop
  eval artifacts. It should include scored `LoopReport` evidence so humans can
  inspect phases, citations, verifier decisions, retries, refusals, and final
  decisions without a live model.
- Live Ollama model comparison is optional and manual. Keep it unload-aware,
  keep multi-model runs behind an explicit override, and never make CI require
  resident local models. Keep the live eval base URL loopback-only and prefer
  one-model, one-case smoke runs before full-model sweeps.
- OpenAI Agents SDK, LangGraph, and Microsoft Agent Framework are future adapter
  targets, not core dependencies, until provider-neutral loop reports are real
  and test-covered.
- Framework adapter work must follow `docs/framework-adapter-strategy.md`.
  Export framework-shaped artifacts first; do not let any framework runtime own
  loop execution until the internal loop report contract remains stable under
  tests.
- OpenAI trace-shaped export, LangGraph manifest export, and `src.loop_export`
  must remain dependency-free. Public/redacted export is the default; raw
  diagnostics require explicit opt-in.
- Do not add autonomous tool use or multi-agent behavior until middleware,
  guardrail, telemetry, and human-review boundaries exist in the loop contract.
- Text encoding default is `Auto`. Ambiguous non-UTF legacy files must not be
  silently decoded as Western text.
- Explicit encoding selections are user intent. Preserve valid CP1250, CP1251,
  CP1252, CP1254, CP1257, Latin-1, UTF-8, UTF-16, and UTF-32 behavior when
  touching text ingestion.
- `pyproject.toml` is the local-development dependency contract. Keep
  `requirements.txt` and `requirements-dev.txt` synchronized as pip-compatible
  deployment exports.

## Files To Inspect First

- `src/ai_loop_engine.py`
- `src/app.py`
- `src/thread_store.py`
- `src/web_contract.py`
- `src/ai_loop_runtime.py`
- `src/context_providers.py`
- `src/retrieval.py`
- `src/retrieval_types.py`
- `src/document_config.py`
- `src/document_text.py`
- `src/document_ingestion.py`
- `src/env_file.py`
- `src/runtime_config.py`
- `src/model_adapters.py`
- `src/web_search.py`
- `src/answer_loop.py`
- `src/DocumentQA.py`
- `src/loop_engine.py`
- `src/golden_eval.py`
- `src/loop_eval.py`
- `src/ollama_model_eval.py`
- `src/adapters/`
- `src/loop_export.py`
- `docs/framework-adapter-strategy.md`
- `pyproject.toml`
- `requirements.txt`
- `requirements-dev.txt`
- `tests/test_document_qa.py`
- `tests/test_app.py`
- `tests/test_golden_document_eval.py`
- `tests/test_loop_engine.py`
- `tests/test_loop_eval.py`
- `tests/test_ollama_model_eval.py`
- `.github/workflows/tests.yml`
- `.github/workflows/docker-publish.yml`

## Validation Matrix

For ingestion or encoding changes:

- `python -m pytest tests/test_document_qa.py -q`
- Include replacement-failure tests that prove the previous document remains
  active and queryable.
- Add hostile tests for env pollution, mojibake, invalid bytes, or unsupported
  explicit encodings as appropriate.

For UI status changes:

- `python -m pytest tests/test_app.py -q`
- Confirm mock mode and real backend wording remain honest.

For backend routing changes:

- Exercise `LLM_BACKEND=auto`, `mock`, `ollama`, `openai-compatible`,
  and invalid removed backend names such as `endpoint` and `local` when the
  change touches backend selection.
- Clear or set `LLM_BACKEND`, `LLM_MODEL`, `EMBEDDINGS_MODEL`,
  `OLLAMA_MODEL`, `OLLAMA_EMBED_MODEL`, `OLLAMA_BASE_URL`,
  `OPENAI_COMPAT_BASE_URL`, `OPENAI_COMPAT_MODEL`,
  `OPENAI_COMPAT_EMBED_MODEL`, and `OPENAI_COMPAT_API_KEY` inside tests so
  shell state cannot poison CI.

For answer-loop or agent-pattern changes:

- `uv lock --check` when packaging metadata or dependency files changed.
- `python -m pytest tests/test_loop_engine.py -q`
- `python -m pytest tests/test_golden_document_eval.py -q`
- `python -m pytest tests/test_loop_eval.py -q`
- `python -m pytest tests/test_ollama_model_eval.py -q`
- `python -m pytest tests/test_openai_trace_adapter.py -q` when adapter export
  behavior changes.
- `python -m pytest tests/test_langgraph_manifest_adapter.py -q` when LangGraph
  manifest behavior changes.
- `python -m pytest tests/test_loop_export.py -q` when JSONL adapter export CLI
  behavior changes.
- Assert cited supported answers, unsupported-answer refusal, and retry behavior.
- Keep eval fixtures deterministic and provider-free.
- Keep live Ollama comparison manual. Prefer one-model, one-case smoke runs on
  memory-constrained Macs, and do not make multi-model live eval the default.

For release automation:

- Parse YAML if possible.
- `git diff --check`
- Ensure Docker publish remains limited to `push` on `refs/heads/main`.

## Stop Conditions

Stop and ask for direction if a change requires new model providers, persistent
storage, authentication, a database, or background job infrastructure. Those are
product architecture decisions, not cleanup.
