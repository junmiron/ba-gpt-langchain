# BA_langchain handoff

**Authoritative continuation note.** This handoff supersedes the old November 2 baseline notes. Continue in the LangChain repo; keep the sibling AutoGen app separate.

## Current status

- **Project:** `D:\Code\Ba_gpt\Ba_langchain` — the LangChain rewrite.
- **Branch / HEAD:** `main` / `387da7f` (`REDESIGN_PLAN: mark Phase 0 done with evidence; align Phase 1 and architecture with the no-LangGraph decision`).
- **Working tree:** clean at handoff creation. No code changes were made for this handoff; the untracked `uv.lock` generated during verification was removed.
- **Phase 0 is complete.** The plan records the offline suite growing from 58 to 167 tests.
- **Next planned work: Phase 1 — LangChain-core idioms.** Phase 2 deliverables and Phase 3 parity remain pending.
- `docs/REDESIGN_PLAN.md` is the implementation plan, but still contains stale/redundant references called out below. Reconcile those before treating every paragraph as current.

## Repository boundary — important

- This repo is `Ba_langchain`, at `D:\Code\Ba_gpt\Ba_langchain`.
- The sibling `D:\Code\Ba_gpt\TFM_BA_GPT` is the **AutoGen project**. The user wants it to remain exclusively AutoGen. Do not implement LangChain work, handoff material, or cross-project cleanups there.
- The LangChain repo may consult the sibling's implementation as a read-only reference when a port is explicitly planned (e.g. Typst PDF rendering); make all new implementation changes in `Ba_langchain` only.
- Keep credentials out of YAML, source, docs, and commits. Provider/model/limit settings are fine; credentials stay in environment variables or an untracked `.env`.

## Approved architecture and product decisions

1. **No LangGraph imports or runtime.** The user's decision is strict: pure `langchain-core` runnables; no `StateGraph`, checkpointers, LangGraph middleware/runtime, or `create_agent` (LangChain's `create_agent` compiles to LangGraph).
2. **The deterministic Python state machine owns control flow.** Models produce content only. Preserve typed Pydantic contracts, visible/bounded budgets, and resumability.
3. **Keep the hand-rolled LangChain-core tool loop.** Refactor toward reusable core `Runnable` composition only where it clarifies the code; do not hand orchestration to an agent factory that introduces LangGraph.
4. **PDF direction is decided:** Typst-native, hand-generated `.typ` (no pandoc in the new PDF path); ReportLab + svglib is the pure-Python fallback. Bake-off evidence is in `docs/REDESIGN_PLAN.md` §3.3.
5. **Transcript ingestion is approved for Phase 3.** Deliverables (Phase 2) precede feature parity work.
6. **Security scope:** localhost hardening shipped in Phase 0; token authentication is deferred unless non-local deployment is approved.
7. **Optional MCP integration is not Phase 1 scope.** Do not add beta MCP dependencies or LangGraph-based tooling without separate approval.
8. **Preserve provider abstraction.** LM Studio remains supported; Strata was added as a first-class profile and UI labels now follow the active provider.

## Completed work

### Baseline

- Business Analyst workflow: vanilla-JS web SPA + FastAPI/SSE service, SQLite persistence, deterministic async workflow, Jinja2 prompts, model gateway, local knowledge retrieval, Mermaid generation, and export.
- Original architecture and old AutoGen analysis: `docs/DESIGN.md`, `docs/ANALYSIS.md`.
- The old `HANDOFF.md` captured code maps, invariants, environment notes, real-model defect history, and system-specific execution quirks. Its baseline/test count and repository-boundary statements are stale; use this file as the authoritative current handoff.

### Phase 0 — Stabilize (complete, 2026-11-07)

- Fixed 12 analyzed roles-layer defects with regression tests (`9ad35e9`); details/rationale are in `docs/REDESIGN_PLAN.md` §1.2.
- Added Strata as a first-class provider profile (`c71db05`) and fixed UI labels to reflect the active provider (`35ff4ef`).
- Hardened server, persistence, and security: SSE replay/cursor behavior, CSRF/CORS/SSRF protections, upload guards, schema versioning, event retention and related tests (`b55afa8`).
- Added GitHub Actions CI: offline pytest on Python 3.11 and 3.13; no model, LM Studio, or external URLs are needed. See `.github/workflows/ci.yml`.
- Marked Phase 0 done and corrected the architectural decision in `docs/REDESIGN_PLAN.md` (`387da7f`).
- Recorded offline suite result after Phase 0: **167 passed**. Re-run to establish a fresh baseline before changing code.

### Real-model and deliverable research

- Earlier LangChain baseline work included real LM Studio runs and regression fixes for thinking-model empty output, prompt/consumer contract mismatch, and other defects; see `docs/DESIGN.md` §12–§13 and the historical notes in git.
- A four-engine PDF bake-off is documented in `docs/REDESIGN_PLAN.md` §3.3. Decision: hand-written Typst is the target; do not copy the sibling's pandoc-based route into the new architecture.
- No Phase 1, Phase 2, or Phase 3 implementation is recorded after Phase 0.

## Remaining work

### Phase 1 — LangChain-core idioms (next implementation phase)

Approved scope from `docs/REDESIGN_PLAN.md` §4 Phase 1:

- Keep the deterministic workflow and hand-rolled tool loop; any core runnable composition must be a thin, testable refactor, not an abstraction project.
- Reconcile structured-output selection with existing provider profiles and preserve the tested fallback ladder (schema mode → tool/function calling → prompt/repair as applicable).
- Expose local knowledge through a `langchain-core` Retriever interface while retaining the current hybrid retrieval implementation. Use `langchain-text-splitters` for heading-aware chunking only if needed and approved as a dependency change.
- Add typed prompt-context contracts and a test that detects undeclared Jinja variables.
- Preserve the scripted-gateway harness, deterministic behavior, artifact output, bounded loops, and visible degraded-mode warnings.
- **Never add `langgraph`, `langchain.agents.create_agent`, or orchestration that transfers workflow ownership to LangGraph.**

Acceptance from the plan: offline tests remain green; focused runnable/retriever/context-contract tests are added; workflow and artifact behavior remain unchanged. Ask before dependency or persisted-schema changes.

### Phase 2 — Deliverables

- Implement the approved Typst-native renderer in this repo, generating `.typ` directly from the spec/Markdown model without pandoc.
- Vendor the approved Typst executable and fonts as planned; retain ReportLab + svglib as fallback.
- Add deterministic Mermaid render/export handling and self-contained HTML with linked TOC/inline SVG.
- Add no-model-call PDF/HTML/Markdown generation and PDF QA: TOC/bookmarks, all FR/NFR rows, diagram-band cut safety, and no raw Markdown leakage. Full acceptance thresholds are in `docs/REDESIGN_PLAN.md` §4 Phase 2.

### Phase 3 — Feature parity

- Transcript upload/ingestion → triage/ledger → gap-only interview.
- Configuration UI for role/model lanes, budgets, and knowledge directories.
- Feedback/lessons store injected into future interview/writer context. Describe it accurately as feedback/retrieval, not reinforcement learning unless a reward/policy-learning mechanism is actually implemented.
- UI gaps: delete/rename sessions, export, ledger editing, busy state, pagination/live updates.
- Validate parity against the AutoGen sibling without changing that repository; add e2e transcript-mode coverage.

### Explicitly dropped / deferred

- **Phase 4 LangGraph engine is dropped**, despite stale text in the plan still describing it as optional. Do not implement it.
- MCP adapter/beta integration is deferred; it is not required for Phase 1 or 2.
- LAN exposure/authentication is deferred; keep the service localhost-only unless the user approves a deployment scope.
- Optional validations in the historical handoff: real-model `--request-changes-first` run, RAG with an indexed document, and post-delivery chat-tool smoke test. Prioritize if the user requests them and model time is available.

## Plan cleanup needed before/alongside Phase 1

`docs/REDESIGN_PLAN.md` was amended at `387da7f`, but still has proposal-era contradictions. Update it without changing the approved decisions:

- Remove or mark dropped the Phase 4 LangGraph engine section and its LangGraph-specific risks.
- Remove the stale “create_agent + middleware” recommendation and any text implying it can be adopted; it violates the no-LangGraph rule.
- Remove stale open-decision questions at the end: PDF, transcript ingestion, security scope, and work order have already been decided above.
- Correct old feature-parity wording that calls the feedback store “reinforcement learning”.
- Document the order as Phase 1 core idioms → Phase 2 deliverables → Phase 3 parity → release; do not imply LangGraph work.

Keep this reconciliation documentation-only and small; do not silently broaden it into implementation.

## Verification snapshot (2026-11-09)

- `uv sync --extra dev`: succeeded and restored the project venv with declared development extras. It generated an untracked `uv.lock`; that file was removed because it was absent/untracked before the check.
- `uv run pytest -q`: **167 passed in 25.03s**. Pytest-asyncio emitted a warning that the default async fixture loop scope is unset.
- `uv run ruff format --check src tests`: **failed** — 33 files would be reformatted. No files were changed to address this during handoff.
- `uv run ruff check src tests`: **failed with 105 findings** across source and tests under the currently available rules.
- `uv run mypy src`: **unavailable** (`mypy` executable is not installed in the project environment).
- Working tree was clean after removing the incidental `uv.lock`; `.venv/` is ignored.

Run gates individually and read their output. Passing pytest does not imply formatting, lint, or type gates pass.

## Commands and conventions

- From `D:\Code\Ba_gpt\Ba_langchain`, use `uv sync --extra dev` to create/update the environment, then `uv run pytest -q`.
- CI installs `pip install -e ".[dev]"` and runs `pytest -q` on Python 3.11 and 3.13.
- Run `uv run ruff format --check src tests`, `uv run ruff check src tests`, and `uv run mypy src` separately; baseline results are above.
- Read `docs/REDESIGN_PLAN.md` and `docs/DESIGN.md` before architectural edits. Inspect relevant code/tests first.
- Prompt/gateway/output-contract changes require real local-model verification in addition to offline tests. Tell the user the ETA before starting; real-model runs may take hours and share the local model backend/VRAM.
- Keep duplicated config defaults synchronized between `config.yaml` and `src/ba_agent/config.py`. Keep credentials out of tracked files.
- Make small, reviewable diffs; do not commit unless asked. Never touch `D:\Code\Ba_gpt\TFM_BA_GPT` for LangChain work.

## First next step

1. Read this handoff and reconcile the stale decisions in `docs/REDESIGN_PLAN.md` as listed above.
2. Re-run `uv run pytest -q` (expected baseline: 167 tests) and record the actual result.
3. Start Phase 1 with a scoped test/design pass around `src/ba_agent/agent_loop.py`, `models.py`, `knowledge.py`, and `prompts.py`; retain the workflow machine and artifact behavior. Add focused tests before refactoring and do not add dependencies without approval.
