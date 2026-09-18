# Framework Adapter Strategy

Status: implemented for dependency-free OpenAI trace-shaped export and
LangGraph manifest export; Microsoft workflow event export remains planned.
Research baseline: 2026-06-24. Last reviewed: 2026-07-17.

## Purpose

Loopwright should interoperate with current agent frameworks without letting
any framework become the core product architecture. The internal contract remains
`LoopReport`, `LoopSession`, provider-neutral phases, explicit verifier
decisions, guardrail decisions, and local diagnostic artifacts that may become
future inspect/diff input.

This note covers optional adapter directions for:

- OpenAI Agents SDK trace/export surfaces
- LangGraph graph/checkpoint surfaces
- Microsoft Agent Framework workflow/event surfaces

No runtime dependency should be added for any of these until the internal report
schema is stable enough to make adapter behavior testable and boring.

## Source Baseline

Research references checked on 2026-06-24:

- OpenAI Agents SDK tracing: https://openai.github.io/openai-agents-python/tracing/
- LangGraph persistence: https://docs.langchain.com/oss/python/langgraph/persistence
- Microsoft Agent Framework workflows: https://learn.microsoft.com/en-us/agent-framework/workflows/
- Microsoft Agent Framework workflow events: https://learn.microsoft.com/en-us/agent-framework/workflows/events
- Microsoft Agent Framework Python workflow class: https://learn.microsoft.com/en-us/python/api/agent-framework-core/agent_framework.workflow

## Current Internal Surfaces

The useful internal adapter boundary is already mostly present:

- `LoopReport`: one completed run, including `run_id`, `session_id`, input,
  context provider, backend/model, policy, steps, final decision, final answer,
  error, and metadata.
- `LoopStep`: phase-level events for context selection, retrieval, drafting,
  format checks, mechanical checks, verifier decisions, retries, refusals, final answers, and
  errors.
- `LoopSession`: bounded in-memory run history keyed by `session_id`, with raw
  JSONL export for local diagnostics and future inspect/diff input. Loopwright
  does not currently replay or re-execute those records.
- `LoopPolicy`: explicit guardrail policy, including no autonomous tools by
  default.
- Public artifact projection: adapters default to a versioned, public data
  shape; raw local artifacts remain developer diagnostics. Projection is a
  data-shaping boundary, not access control or a general secret/PII scrub, so
  every artifact still requires review before sharing.

The adapter strategy should export these surfaces. It should not replace them.

## OpenAI Agents SDK Trace Adapter

### Observed Shape

The OpenAI Agents SDK traces an end-to-end workflow with spans for agent runs,
LLM generations, function tools, guardrails, handoffs, and custom events. It also
supports custom trace processors and can include sensitive generation/tool data
unless that capture is disabled.

### Mapping

| Loopwright | OpenAI Trace Concept | Notes |
| --- | --- | --- |
| `LoopSession.session_id` | trace `group_id` | Groups related runs in one conversation/session. |
| `LoopRun.run_id` | trace metadata and deterministic trace identity | Preserve the raw internal id in metadata. Adapter v2 hashes `session_id` plus `run_id` into an OpenAI-shaped trace id and rejects duplicate run ids within a session. |
| `LoopRun.context_provider` | trace metadata | Resolved provider, for example `document`, `web`, or `none`; requested provider metadata may be `smart` or legacy `auto`. |
| `LoopStep.phase=context_select/retrieve` | custom span | These are Loopwright-specific context phases. |
| `LoopStep.phase=draft` | custom span in adapter v2 | Current summaries are not typed OpenAI `GenerationSpanData`; do not claim a generation span yet. |
| `LoopStep.phase=format_check` | custom span | Presentation/readability gate, not evidence verification. |
| `LoopStep.phase=mechanical_check` | custom span | Deterministic check, not model generation. |
| `LoopStep.phase=verify` | custom span in adapter v2 | Model-verifier provenance is not proof that an OpenAI SDK guardrail ran. Preserve the projected verifier outcome without claiming a guardrail span. |
| `LoopStep.phase=retry/refuse/error/final` | custom span | These are loop-control decisions. |
| `LoopDecision` | span status/metadata | Keep original value. |

### Recommended Adapter

The offline exporter lives in `src.adapters.openai_trace`. It maps
`LoopReport` and `LoopSession` objects to OpenAI-trace-shaped JSON without
importing the OpenAI Agents SDK or sending data to a network exporter. Only later
add an optional live exporter that uses Agents SDK tracing processors.

Adapter v2 emits every current `LoopStep` as a custom span. It should specialize
to generation or guardrail spans only after the typed loop contract records the
fields required by those SDK span types, including explicit guardrail-trigger
provenance.

Each `span_data` object follows the OpenAI SDK custom-span export shape:
`type`, `name`, and a `data` property bag containing the Loopwright fields.

The first live adapter should be opt-in and should:

- live outside the core query path;
- require an explicit OpenAI API key;
- disable sensitive data capture by default where the SDK permits it;
- export the public artifact projection unless the caller explicitly requests
  raw local diagnostics, and treat both forms as potentially sensitive;
- flush only after a completed run, never during a partially built trace.

### Non-Goals

- Do not run the application through `Runner`.
- Do not model file or web context as OpenAI tools until autonomous tool boundaries
  exist.
- Do not send raw diagnostic artifacts to OpenAI by default.
- Do not make `openai-agents` a core dependency.

## LangGraph Adapter

### Observed Shape

LangGraph distinguishes thread-scoped checkpointers from cross-thread stores.
Checkpointers support short-term memory, human-in-the-loop workflows, time
travel, and fault tolerance through a `thread_id`; stores are for application
data across threads.

### Mapping

| Loopwright | LangGraph Concept | Notes |
| --- | --- | --- |
| `LoopSession.session_id` | `thread_id` | Direct conceptual match. |
| `LoopRun` | one graph invocation or checkpoint sequence | Keep `run_id` as metadata. |
| `LoopStep.phase` | node name or superstep label | Start as labels, not executable nodes. |
| `LoopStep.metadata` | checkpoint metadata | Must stay JSON-safe. |
| `LoopReport.final_decision` | terminal graph state | Preserve supported/not_verified/refuse/block/error. |
| `LoopSession` JSONL | diagnostic/future inspect-diff source | Not a replay engine or LangGraph checkpointer. |

### Recommended Adapter

The first dependency-free exporter lives in `src.adapters.langgraph_manifest`.
It maps `LoopReport` and `LoopSession` objects to a LangGraph-oriented
thread/checkpoint manifest without importing LangGraph or pretending to be a
checkpointer.

The manifest includes:

- `thread_id`: `session_id`
- `run_id`
- ordered phase list
- terminal decision
- checkpoint-like snapshots per step
- source JSONL line references only when a JSONL caller, such as
  `src.loop_export`, supplies that invocation-local metadata explicitly;
  in-memory session exports leave those fields unset, and a detached export does
  not identify its source artifact from a line number alone

Only after the manifest proves useful should we consider a real LangGraph
adapter. The first real adapter should likely be a read-only report and
checkpoint-shaped inspector, not a replay claim or a new execution engine.

### Non-Goals

- Do not implement a LangGraph `BaseCheckpointSaver` yet.
- Do not compile the current evidence-backed answer loop into a `StateGraph` yet.
- Do not introduce LangSmith/LangGraph deployment assumptions.
- Do not treat LangGraph stores as our memory model before Loopwright has a
  real cross-session memory product.

## Microsoft Agent Framework Adapter

### Observed Shape

Microsoft Agent Framework workflows are composed from executors and edges.
Workflow execution emits lifecycle, executor, superstep, output, error, warning,
and request-info events. Python workflows can run to completion or stream events;
checkpointing can capture executor state, messages in transit, and shared state
at superstep boundaries.

### Mapping

| Loopwright | Microsoft Agent Framework Concept | Notes |
| --- | --- | --- |
| `LoopRun` | workflow run result | One Loopwright run maps to one workflow execution record. |
| `LoopStep.phase` | executor event or custom event | Keep phase names as custom event discriminators first. |
| `LoopStep.started_at/ended_at` | event timing metadata | Microsoft events are streaming-oriented; our report is completed-run oriented. |
| `LoopDecision.REQUIRES_REVIEW` | `request_info` event | Natural match for future human-review/tool-approval paths. |
| `LoopDecision.ERROR/BLOCK/REFUSE` | failed/error/output events | Preserve exact decision in event data. |
| `LoopSession` | workflow instance/run collection | Do not rely on workflow instance state for our session model. |

### Recommended Adapter

Start with a JSON event-stream exporter:

- `started` event for `LoopRun`
- `executor_invoked` / `executor_completed`-like events for each `LoopStep`
- `request_info` event for human-review requests
- terminal `output`, `failed`, or `error` event based on `final_decision`

This gives Microsoft-oriented users a familiar event vocabulary without
importing the framework or committing to its workflow execution model.

### Non-Goals

- Do not wrap Loopwright as a Microsoft `Workflow` yet.
- Do not turn loop phases into real executors until tool and human-review
  boundaries are stronger.
- Do not use Microsoft checkpoint storage as the source of truth for session
  state.
- Do not introduce `agent-framework` as a core dependency.

## Shared Adapter Contract

Future adapters should implement this shape, regardless of target framework:

```python
class LoopReportAdapter:
    adapter_name: str
    adapter_schema_version: str

    def export_report(
        self,
        report: LoopReport,
        *,
        public: bool = True,
        session_id: str | None = None,
        source_jsonl_line: int | None = None,
    ) -> dict:
        ...

    def export_session(
        self,
        session: LoopSession,
        *,
        public: bool = True,
        source_jsonl_lines: Sequence[int | None] | None = None,
    ) -> dict:
        ...
```

Rules:

- Default to the versioned public artifact projection. Do not describe it as
  generic redaction: projection is not access control or a secret/PII scrub.
- Require an explicit `public=False` or `raw=True` flag for local diagnostics.
- Preserve stable `run_id`, `session_id`, phase order, final decision, and
  self-check outcome. Preserve content-bearing verifier or guardrail detail only
  when the selected projection permits it.
- Reject ambiguous adapter identities instead of generating colliding trace or
  manifest records. Only emit invocation-local source-line metadata when the
  caller provides it explicitly; a line number alone is not durable artifact
  identity.
- Never mutate `LoopReport` or `LoopSession`.
- Never call model providers, tool providers, or network exporters from the
  basic mapping function.
- Keep adapter schema versions separate from `loop-report/v1`.
- The explicit invocation-local line metadata and collision-resistant identifier
  contract is `openai-trace-export/v2` and
  `langgraph-manifest-export/v2`; do not backport those shape changes under the
  v1 labels.
- Provide CLI export from JSONL diagnostic artifacts through `src.loop_export`
  instead of making downstream tools import application internals.

## Current And Proposed File Layout

Current dependency-free adapter files:

```text
src/adapters/
  __init__.py
  base.py
  redaction.py
  langgraph_manifest.py
  openai_trace.py
src/loop_export.py
tests/test_langgraph_manifest_adapter.py
tests/test_loop_export.py
tests/test_openai_trace_adapter.py
```

Add later when the matching adapter is actually implemented:

```text
src/adapters/
  microsoft_workflow_events.py
tests/test_microsoft_workflow_events_adapter.py
```

Optional live integrations should be dependency-gated later:

```text
requirements-adapters.txt
```

or Python extras if packaging is introduced. Do not add these dependencies to
`requirements.txt`.

## Decision

The current implementation is a dependency-free adapter manifest/exporter, not
a framework runtime integration or replay engine.

Recommended order:

1. Add a dependency-free `LoopReport` adapter protocol and one JSON manifest
   exporter. Done for the OpenAI trace-shaped exporter.
2. Add tests proving versioned public-projection default export and raw explicit
   export. Done for OpenAI trace-shaped export, LangGraph manifest export, and
   the JSONL export CLI.
3. Add optional OpenAI trace-shaped export first, because its trace/span model is
   closest to current `LoopReport`. Done as an offline export, not a live SDK
   integration.
4. Add LangGraph manifest/checkpoint-shape export second. Done as an offline
   manifest, not a LangGraph runtime integration.
5. Add Microsoft event-stream export third.

The product should remain Loopwright. Frameworks are export targets and
interop layers, not the engine.
