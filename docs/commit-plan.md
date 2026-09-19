# Commit plan for the pending Loopwright update

Historical record: these changes were merged before the uv-only migration.
The file groups and validation commands below describe that earlier update;
use the current README for dependency setup and maintenance.

This plan includes the existing staged work and the fixes from the September
2026 readiness review. Apply the commits in order. No commit or push is performed
by this document.

The runtime, public projection, persistence, browser behavior, adapters, and
their tests share a changed report contract. Keep them in one feature commit;
splitting those files mechanically would create inconsistent intermediate
versions.

## Prepare the index

The changes were already staged before the readiness review. To make the three
commits below, first clear the index while preserving every working-tree file:

```bash
git restore --staged -- .
```

Optionally create a feature branch before committing:

```bash
git switch -c feature/loop-report-hardening
```

For each group, stage only the listed files, inspect `git diff --cached`, and
commit with the suggested message. Do not use `git add .` between groups.

## Commit 1: dependencies and build validation

Suggested message:

```text
build: update PDF dependency and harden deployment packaging
```

Include these files:

```text
.dockerignore
.github/workflows/dependency-audit.yml
Dockerfile
pyproject.toml
requirements-dev.txt
requirements.txt
tests/test_packaging_metadata.py
tests/test_pdf_ingestion.py
uv.lock
```

This keeps the supported dependency floor, locked versions, pip exports,
dependency-audit workflow, Docker changes, and packaging checks together.

## Commit 2: loop reports, durable history, and their regression fixes

Suggested message:

```text
feat: harden loop reports and durable run inspection
```

Include these files:

```text
src/adapters/base.py
src/adapters/langgraph_manifest.py
src/adapters/openai_trace.py
src/adapters/redaction.py
src/ai_loop_runtime.py
src/answer_loop.py
src/app.py
src/golden_eval.py
src/loop_engine.py
src/loop_eval.py
src/loop_export.py
src/model_adapters.py
src/public_projection.py
src/thread_store.py
src/web_contract.py
src/web_static/app.js
src/web_static/styles.css
tests/fixtures/loop-report-v1-legacy.jsonl
tests/test_app.py
tests/test_document_qa.py
tests/test_golden_document_eval.py
tests/test_langgraph_manifest_adapter.py
tests/test_loop_engine.py
tests/test_loop_eval.py
tests/test_loop_export.py
tests/test_openai_trace_adapter.py
tests/test_thread_store.py
```

This commit contains the shared report/evidence contract and its
consumers, including the fixes for legacy raw export, terminal middleware
decisions, and browser query ownership.

## Commit 3: current capabilities and development guidance

Suggested message:

```text
docs: clarify Loopwright capabilities and development gates
```

Include these files:

```text
.agents/skills/loop-engineering/SKILL.md
AGENTS.md
README.md
docs/commit-plan.md
docs/framework-adapter-strategy.md
docs/loopwright-flow.svg
docs/loopwright-trust-plane-plan.md
```

## Before pushing

Review `git status --short` and `git log -3 --oneline`. Keep `.env`,
`.env.local`, local databases, diagnostic artifacts, and virtual environments
out of the commits.

If using the feature branch above, push it explicitly to the GitHub remote:

```bash
git push -u origin feature/loop-report-hardening
```

The repository also has a Hugging Face remote named `hf`; it is not the target
of this plan. Review the feature branch through a pull request before merging
to `main`.

## Readiness fixes completed

- Upgraded PyPDF from 6.14.2 to 6.19.0 and synchronized the dependency floor,
  lock, and pip export. Uploaded PDFs reach `PdfReader` and `extract_text` in
  `src/document_ingestion.py`; updating the parser fixes the dependency findings
  without changing the upload API or production parser limits. Added real PDF
  extraction and failed-replacement tests in `tests/test_pdf_ingestion.py`.
- Restored explicit raw export of the original `loop-report/v1` format in
  `src/loop_export.py`, with an actual historical-format fixture and regression
  tests. Public export still rejects records missing required evidence bindings.
- Preserved the first recorded refusal, block, or review decision across later
  middleware callbacks in `src/ai_loop_runtime.py`. Tests cover conflicting
  decisions and callback exceptions, including the refusal-step case found by
  independent review. Earlier checks still enforce middleware decisions.
- Prevented an upload in another thread from releasing controls or replacing
  an active query in `src/web_static/app.js`. A regression test in
  `tests/test_app.py` covers switching threads, uploading, attempting another
  query, and completing the original query.

## Validation — 18 September 2026

All 43 changed files are included exactly once above. The proposed commit
boundaries were also tested by overlaying each group onto a clean snapshot of
the existing `HEAD`, without moving the working branch or touching its index.

| Check | Result |
| --- | --- |
| Commit 1 snapshot, full pytest suite | 695 passed |
| Commits 1 + 2 snapshot, full pytest suite | 1,243 passed |
| Final working tree, full pytest suite | 1,243 passed |
| `src.loop_eval --mode fake` | 4/4 cases passed |
| `uv lock --check` | Passed |
| `python -m pip check` in the project environment | No broken requirements |
| Compile all source and test Python files | 45 files passed |
| `python -m pip install --dry-run -r requirements.txt` | Passed |
| Locked runtime dependency export + strict `pip_audit` | No known vulnerabilities found |
| Docker build, Linux ARM64 | Passed |
| Container smoke test, network disabled and explicit mock backend | App imported; document indexed; cited answer returned as `not_verified` |
| Independent PDF and behavior reviews | Confirmed in-scope findings resolved |
| `git diff --check` | Passed |

The PDF-specific checks cover successful multipage extraction and page metadata,
plus malformed, encrypted, empty, and parser-limit replacement failures. Each
failure preserves the previous document, answer, and citations. The scoped
parser-limit test also verifies recovery after the limit is removed. The
original dependency audit findings no longer appear with the new locked version;
this is not an exhaustive hostile-PDF corpus test.

The full suite reports one existing Starlette/httpx deprecation warning. The
browser race is verified through the existing Node/DOM test harness; no manual
browser run was performed for this fix. Container validation used explicit mock
mode; live Ollama was checked earlier in the readiness review, and its local
configuration was not changed.
