# Loopwright Trust-Plane Plan

## Decision

Loopwright will prove a narrow product wedge before expanding its runtime:

> An evidence-backed flight recorder for AI answer loops today, evolving into a
> local-first trust plane that can inspect and compare external AI loops.

Loopwright is not another orchestrator. Model vendors and agent frameworks will
continue to commoditize loop execution. The defensible surface is the evidence
that shows what a loop saw, changed, checked, claimed, and what the available
evidence and checks support or fail to support.

This is a six-week proof plan, not a feature backlog. Every milestone has an
acceptance gate. Missing a gate means fixing or narrowing that milestone—not
quietly moving on to more agent features.

## Current Baseline

The current product runs a bounded answer loop with typed reports, durable
versioned public run records, and local raw JSONL export. Evidence-backed routes
retrieve, draft, mechanically check, verify, and may retry or refuse. Direct
routes draft without retrieved evidence and finish `not_verified`. A single
allowlist-only `loop-public-report/v1` projector supplies web responses,
durable history, adapters, the export CLI, and offline inspection/comparison.
`src.loop_replay inspect` reads raw session JSONL into readable summaries or
versioned JSON, using the public projection by default. `src.loop_replay diff`
compares two selected recorded runs, separating material and timing changes.
The product also exports OpenAI
trace-shaped JSON and LangGraph manifest JSON; those export modules do not
import or execute the target framework runtimes.

It does not yet provide:

- deterministic re-execution replay;
- inbound Codex or Claude trace normalization;
- Microsoft workflow-event export;
- cross-runtime comparison; or
- an autonomous scheduler, tool runner, or agent fleet.

The plan keeps those distinctions explicit.

## Delivery Status (2026-09-20)

The initial implementation tranche has completed the claims truth pass, the
runtime retry/termination portion of Week 1, the narrow historical public-run
inspector, and the Week 2 public-projection implementation. Middleware-requested
retry execution remains unavailable and fails closed; legacy or manually
constructed completed reports may retain an `unspecified` terminal reason.
The public boundary is now `loop-public-report/v1`: a versioned allowlist rather
than a raw-report copy with selective redaction. Hostile tests cover terminal
contradictions, identity mismatch, legacy re-projection, and malformed-record
quarantine. That baseline and the Week 3 inspection portion passed full validation
and were merged. Semantic comparison is now implemented locally, with explicit
run pairing, normalized operational fields, separate timing changes, and visible
unavailable provenance. This does not satisfy the operator usability proof:
recorded comparison demonstrations and unfamiliar-user feedback remain pending.
External trace ingestion is still a separate, unimplemented milestone.

## Success Metric

At the end of six weeks, Loopwright must ingest or load two runs of the same
bounded task, preserve their provenance, and make a material trust difference
obvious to an unfamiliar operator.

The product proof is successful when at least four of five unfamiliar users can
identify the less trustworthy run in under two minutes without reading raw
JSON. If that does not happen, the product surface or comparison contract is
not clear enough.

## Six-Week Delivery Plan

### Days 1–2: Make the Claims Honest

State the current product as an evidence-backed answer-loop flight recorder and
the cross-runtime trust plane as the direction.

Deliverables:

- Map every advertised capability to a working API, CLI, UI, test, or clearly
  marked future direction.
- Use **Loop Recipes**, not “skills”; recipes do not grant tools, scheduling, or
  autonomous action.
- Describe JSONL as a local session or diagnostic artifact with offline
  inspection and comparison. Keep re-execution explicitly planned.
- Keep Microsoft workflow-event export explicitly planned.
- Say that evidence makes unsupported behavior harder—but not impossible—to
  fake.

Acceptance gate: every product claim maps to a working surface or is visibly
labelled planned.

### Week 1: Make the Existing Contract Enforceable

Strengthen the typed loop contract before adding importers or broader runtime
abstractions.

Deliverables:

- Enforce one global retry budget across executable runtime retry causes.
  Middleware `retry` remains an explicit fail-closed/unavailable decision until
  a safe execution contract exists.
- Reject invalid retry limits instead of normalizing dangerous values silently.
- Record typed terminal reasons for runtime-produced success, refusal, block,
  human review, budget exhaustion, verifier failure, and runtime error.
- Make same-model drafter/verifier agreement explicit when identity is known;
  retain `unknown` rather than inferring independence from different clients.
- Preserve backward compatibility deliberately, with schema migration or
  explicit schema-version handling where required. The current explicit
  exception is pre-binding visible-answer `loop-report/v1` storage: rows without
  terminal answer/evidence-set provenance, or supported rows without exact
  draft/verifier answer-and-evidence provenance, retain their stored raw JSON
  value but are quarantined from canonical/public serving because the
  emitted/verified candidate cannot be inferred safely.
- Add deterministic tests for round-trip serialization, invalid limits,
  terminal decisions, and public redaction.

Acceptance gate: no executable runtime path can exceed its declared retry
budget, every runtime-produced completed run explains why it stopped, and
known same-model verification is visible. Legacy/manual reports may remain
`unspecified`; middleware retry execution is a later gated contract.

### Week 2: Make the Recorder Usable

Make durable public runs genuinely inspectable after process restart.

Deliverables:

- Allow a stored Durable Run to be selected and opened from the user surface.
- Show typed evidence identity and locator, evidence provider, mechanical and
  verifier outcomes, retries, non-guardrail model provenance, error-presence,
  terminal reason, and projection state. Do not expose verifier reasons, raw
  errors, or citation content as public evidence.
- Use one versioned, allowlist-only `loop-public-report/v1` projector for the
  runtime report API, durable history, framework-shaped adapters, and export
  CLI. It omits prompts, arbitrary step names and summaries, raw errors and
  metadata, verifier reasons and raw payloads, recipe text, and human-review
  bodies.
- Represent public evidence as a stable SHA-256-derived identity plus provider,
  citation number, and page/chunk locator. Never project a filename, title,
  URL, or excerpt. The identity binds the exact reference/content inputs used to
  construct it; it does not establish semantic equivalence, authenticity, or
  secrecy.
- Expose a final answer only for a structurally valid non-guardrail completion.
  A refuse, block, review, or other guardrail-like terminal signal suppresses
  the answer, model identity, and evidence. Reject contradictory terminal
  contracts rather than selecting the least restrictive interpretation.
- Ignore legacy cached public JSON on read and re-project from the canonical raw
  report. Quarantine malformed, unsupported, or identity-inconsistent raw rows
  instead of serving their cached public payload.
- Add restart and hostile-redaction fixtures.

Acceptance gate: every fixture run reopens with the same intentionally
projected public evidence and terminal decision after restart, while raw
content remains unavailable from the public report surface by default. This
gate is implemented and covered by focused hostile fixtures. The baseline
tranche passed full validation and was merged before Week 3 inspection began.

The projection is a data-minimization boundary, not authentication, access
control, or a general PII/secret scrub. Thread messages remain raw local data,
and the current local thread APIs have no authentication. Non-guardrail terminal
answers and typed provenance may still be sensitive.

### Week 3: Ship Deterministic Inspect and Diff

Implement local, dependency-free inspection over existing session JSONL. This
is artifact inspection and comparison, not deterministic model re-execution.

Deliverables:

- Implemented: `src.loop_replay inspect <session.jsonl>` for readable run
  summaries, `--format json`, and `--report-index` selection.
- Implemented: `src.loop_replay diff <before.jsonl> <after.jsonl>`, with explicit
  per-side line selectors when an artifact contains multiple runs.
- Compare phases, decisions, evidence identity/citation mapping, retry behavior,
  verification, model fields, typed policy, public memory provenance, completion
  state, error/review presence, terminal reason, and permitted final answers.
  Arbitrary configuration/recipe metadata and causal metadata references are
  outside the operational comparison contract; raw JSON can include complete
  source reports for manual diagnosis.
- Ignore generated identities and absolute timestamps as material differences;
  report run and matched-step durations separately. Align steps by phase and
  retry count in recorded order without claiming causal identity.
- Default to the versioned public artifact projection; require explicit opt-in
  for raw local diagnostics.
- Fail closed on malformed input and report exact JSONL line provenance.

Acceptance gate: semantically identical runs produce no material diff,
malformed artifacts identify the failing line, and the versioned public
artifact projection is the default. Inspection and comparison implement these
requirements, including validation of unselected records before output. Missing
or redacted fields remain unavailable, so no observed material changes cannot
establish full equivalence, shared task identity, or correctness. Provider-free
regression fixtures cover these boundaries; the operator usability proof remains
pending.

### Weeks 4–5: Add Inbound Observations

Normalize read-only external traces without letting provider runtimes own the
Loopwright contract.

Deliverables:

- Define a versioned, provider-neutral `loop-observation/v1` for observed
  external events. Keep it separate from the answer-specific `LoopReport`.
- Add a narrow importer protocol that preserves source, schema version,
  timestamps, event identity, and unmapped payload provenance.
- Start with documented Codex JSON event output, then add Claude Code stream
  JSON only after the first importer proves the contract.
- Preserve unknown events visibly; never infer that provider “completion” means
  verified success.
- Keep import local, read-only, dependency-light, and non-executing.
- Fail safely on malformed, truncated, oversized, or unsupported artifacts.

Acceptance gate: an external trace can become a Loopwright observation and
pass through the versioned public artifact projection for inspect and diff
without silent event loss or executing provider commands.

### Week 6: Run the Killer Comparison

Run the same bounded repository task against the same commit and acceptance
contract through Codex and Claude. Seed one plausible, easy-to-miss
requirement.

The comparison must show:

- what each runtime observed;
- what it changed;
- what checks it ran;
- what completion claims it made;
- what evidence supports those claims;
- retries, terminal reason, tokens or usage when available; and
- unknown or incomparable fields without inventing parity.

Recorded fixtures are acceptable for a deterministic demonstration. Live runs
remain optional and must not become a CI dependency.

Acceptance gate: at least four of five unfamiliar users identify the less
trustworthy run in under two minutes without reading raw JSON.

## Hostile Evaluation Track

Hostile, provider-free fixtures run through every applicable milestone:

- false completion with an unfulfilled requirement;
- infinite or competing retry causes exhausting the global budget;
- missing or stale evidence;
- same-model drafter/verifier agreement presented as independent proof;
- context loss across steps;
- prompt injection or attempted tool escalation in observed content;
- state mutation that disagrees with the reported outcome;
- malformed, truncated, oversized, or unknown external events;
- terminal refusal or block leaking hidden draft content; and
- public export or diff leaking raw diagnostic fields.

Each case needs a known-bad fixture that fails the relevant contract and a
regression test proving the failure remains visible.

## Ruthless Non-Goals

Until inspection, diffing, and inbound traces demonstrate demand:

- No orchestrator, scheduler, or autonomous tools.
- No multi-agent coordinator or agent fleet.
- No third outbound adapter or live adapter-runtime integration.
- No hosted accounts, cloud sync, team workspace, authentication, or remote
  persistence.
- No cost comparison until token and usage accounting are first-class and
  honestly comparable.
- No claim of deterministic re-execution replay; the initial inspect/diff scope
  is local artifact inspection and comparison.
- No new broad `LoopSpec` abstraction unless at least two real importers expose
  repeated contract gaps that existing `LoopRecipe`, `LoopPolicy`, and typed
  run records cannot represent cleanly.
- No UI polish ahead of proving the comparison contract. A clear CLI or local
  report is enough for the first product test.

## Sequencing Rules

1. Typed contracts before importers or new runtime behavior.
2. Keep every new inspect/diff surface on the versioned public artifact
   projection by default. Existing raw diagnostics remain explicit and local;
   the public projection is data minimization, not authentication, access
   control, or a generic secret/PII scrub.
3. One external source before two; Codex proves the importer boundary before
   Claude broadens it.
4. Inspect and diff before re-execution.
5. Deterministic provider-free fixtures before optional live comparisons.
6. Stop after a missed gate and fix the contract or narrow the claim.

The Days 1–2 truth pass, Week 1 runtime retry/termination work, narrow
historical-run inspector, and Week 2 public-projection implementation form the
current tranche. Middleware retry execution remains unavailable. Week 3 and all
later milestones remain gated until this tranche is fully validated and
merge-ready; they are not promises that justify speculative architecture.
