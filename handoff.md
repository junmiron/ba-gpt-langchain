# BA_langchain handoff

**Authoritative continuation note.** This handoff supersedes the old November 2 baseline notes. Continue in the LangChain repo; keep the sibling AutoGen app separate.

## Current status

- **Project:** `D:\Code\Ba_gpt\Ba_langchain` — the LangChain rewrite.
- **Version:** 1.0.0. See `CHANGELOG.md` for what shipped.
- **Phases 0–3 are complete.** Phase 4 (LangGraph engine) is dropped. Only the Phase 5 release pass and the optional real-model validations below remain.
- **Suite:** **590 passed in ~86 s** offline. `ruff check`, `ruff format --check`, `mypy src` (39 source files) and `node --check src/ba_agent/web/app.js` are all clean.
- **Real-model runs:** two full end-to-end runs against Strata `q2_0` completed with `E2E RESULT: PASS`, covering `--request-changes-first`, `--rag` and the post-delivery chat-tool path. Both produced a 29–30 page Typst PDF that passed the QA gate with zero warnings.
- **Dead config:** 0 dead of 57 declared fields — every config key is read.

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
6. **Security scope:** localhost hardening shipped in Phase 0; optional bearer-token auth shipped in Phase 5, off by default. LAN exposure beyond the token gate still needs approval.
7. **Optional MCP integration is not Phase 1 scope.** Do not add beta MCP dependencies or LangGraph-based tooling without separate approval.
8. **Preserve provider abstraction.** LM Studio remains supported; Strata was added as a first-class profile and UI labels now follow the active provider.

## Completed work

### Phase 0 — Stabilize (complete, 2026-11-07)

- Fixed 12 analyzed roles-layer defects with regression tests (`9ad35e9`); details/rationale are in `docs/REDESIGN_PLAN.md` §1.2.
- Added Strata as a first-class provider profile (`c71db05`) and fixed UI labels to reflect the active provider (`35ff4ef`).
- Hardened server, persistence, and security: SSE replay/cursor behavior, CSRF/CORS/SSRF protections, upload guards, schema versioning, event retention and related tests (`b55afa8`).
- Added GitHub Actions CI: offline pytest on Python 3.11 and 3.13; no model, LM Studio, or external URLs are needed. See `.github/workflows/ci.yml`.
- Marked Phase 0 done and corrected the architectural decision in `docs/REDESIGN_PLAN.md` (`387da7f`).
- Suite grew from 58 to 167 tests.

### Phase 1 — LangChain-core idioms (complete, 2026-11-09)

- `agent_loop.py` is a `RunnableSequence`: context policy → model step → tool executor.
- Structured output walks a per-profile rung ladder: `json_schema` → `function_calling` → prompt + repair. LM Studio and Strata declare their own rung order.
- `retriever.py` exposes the SQLite knowledge base as a `langchain-core` `BaseRetriever`; the synchronous path is keyword-only so it never blocks on an embedding call.
- Heading-aware chunking via `langchain-text-splitters` when installed, with a pure-Python fallback that needs no extra package.
- `prompt_context.py` declares a typed context contract per template, and a test calls Jinja's `undeclared_variables()` on every template.

### Phase 2 — Deliverables (complete, 2026-11-09)

- Typst-native PDF from hand-generated `.typ` (no pandoc): `#outline()` TOC, native tables, banded figures. `scripts/fetch_typst.py` downloads the binary into gitignored `vendor/`; discovery order is `BA_TYPST_BIN` → PATH → `vendor/typst/**/typst(.exe)`.
- ReportLab fallback with bookmarks and a TOC, embedding diagrams as **vector paths** — `rlPyCairo` is absent in this environment, so nothing is rasterised.
- Self-contained HTML deliverable with a Contents block whose anchors are verified to resolve.
- `pdf_qa.json` records what the gate checked: pages, outline entries, clickable Contents links, missing requirement ids, Markdown leakage, band-cut weight.
- Band-cut safety is a geometry ink weight (1 per connector stroke, 3 per node box or label), not a dark-pixel count; warning threshold is weight > 6. Cuts snap to the quietest row in a half-band window.

### Phase 3 — Feature parity (complete, 2026-11-09)

- Transcript ingestion → triage → ledger → gap-only interview. Only subjects with enough transcript evidence (`interview.subject_coverage_threshold`, default 2) are skipped.
- Lessons store: frequency-ranked `(kind, text)` rows with `seen_count`, injected as pinned context into the next plan prompt. Only generalisable signals are recorded — never project-specific ids or section names.
- UI: rename, delete (removes the artefact folder), pagination, live refresh, busy indicator, export zip, ledger editing (a finished session reopens at DELIVER and rebuilds), settings page with model lanes, transcript attach.
- Schema version 3 with conditional `ALTER TABLE` migrations.
- Invariant: `status == WAITING_USER` iff `awaiting` is ANSWER/APPROVAL, enforced and tested (`tests/test_status_invariant.py` inspects every persisted snapshot).
- Diagram PNGs are written to disk and referenced by `png_path`; the schema rejects raw image bytes in state.

### Phase 5 — Release pass (in progress)

- Version bumped to 1.0.0; `CHANGELOG.md` written.
- Optional bearer-token auth shipped (`server.auth_token` / `BA_AGENT_TOKEN`), off by default. The static UI stays reachable so it can prompt for the token; SSE takes the token as a query parameter because `EventSource` cannot set headers.
- Raw artefact hashes are computed on bytes, not decoded text — a PNG hashed through a lossy decode is not a dedup key.
- Config synchronised with `config.yaml`; dead-config sweep reports 0 dead of 57.

### Follow-up usability pass (2026-11-10)

- Fixed suggestion-chip contrast, timeline card shrinking/scroll hijacking, and approval-vs-answer confusion; history restore now pages beyond 1,500 events.
- Added accessible phase/elapsed “Thinking” status and a Human lane that defaults to real input; model-backed stakeholder replies are labeled and marked in the generated FSD, while approval remains human.
- Structured-call trace records role, step, model, output method, elapsed time, attempts, and character counts without prompt contents. A covered-answer probe can now include the next subject question, skipping a separate model round trip with the existing question/budget guards and generator fallback.
- One live Strata covered-answer probe emitted two successful calls (triage + coverage/next question); local model variance is high, so do not treat a single turn as a controlled latency benchmark.

## Remaining work

- **Tag + GitHub release.** `git remote -v` is empty — there is no remote configured, so `v1.0.0` cannot be pushed or tagged from here. Add a remote, then tag.
- **Real LM Studio run.** Not yet done: `curl http://127.0.0.1:1234/v1/models` → exit 7 (LM Studio not running). Run `scripts/e2e_lmstudio.py --provider lmstudio` when a chat model is loaded, to confirm the provider-specific rung ladder on a second backend.
- **Vector retrieval is BM25-only under Strata** (no embedding endpoint). It works, but the UI should say so per provider rather than implying semantic search.
- **Real-model runs are slow** (~25–35 min for a full interview on a local quantised model). Budget `--deadline-minutes` accordingly; the driver now nudges stalled sessions via `/api/sessions/{id}/resume`.
- `docs/REDESIGN_PLAN.md` Part 2 still has proposal-era prose (rows describing `create_agent` and LangGraph as open decision points). Part 6 closes those decisions; the prose is historical and safe to trim, and nothing in it is implemented.

## Explicitly dropped / deferred

- **Phase 4 LangGraph engine is dropped.** Do not implement it.
- MCP adapter/beta integration is deferred; it is not required.
- LAN exposure beyond the token gate is deferred until a deployment scope is approved.

## Verification snapshot (2026-11-10)

Run each gate separately and read the output — passing pytest does not imply the lint, format or type gates pass.

| Gate | Command | Result |
|---|---|---|
| Tests | `uv run pytest -q` | 590 passed, 1 deselected |
| Lint | `uv run ruff check src tests scripts` | All checks passed |
| Format | `uv run ruff format --check src tests scripts` | clean |
| Types | `uv run mypy src` | no issues in 39 source files |
| UI syntax | `node --check src/ba_agent/web/app.js` | OK |
| Offline acceptance | `uv run python scripts/e2e_offline.py` | exit 0 |
| Live acceptance | `uv run python scripts/e2e_live.py --base http://127.0.0.1:8769` | 43 checks passed, 0 failed |
| Real model | `uv run python scripts/e2e_lmstudio.py --provider strata --request-changes-first --rag` | PASS |

`uv sync --extra dev` regenerates an untracked `uv.lock`; it is not committed.

## Commands and conventions

- From `D:\Code\Ba_gpt\Ba_langchain`, use `uv sync --extra dev` to create/update the environment, then `uv run pytest -q`.
- `pyproject.toml` sets `addopts = "-m 'not live'"`, so the default run never touches a model server. Run the live smoke test explicitly with `uv run pytest -m live` when a provider is up.
- CI installs `pip install -e ".[dev]"` and runs `pytest -q` on Python 3.11 and 3.13.
- Run `uv run ruff format --check src tests`, `uv run ruff check src tests`, and `uv run mypy src` separately; baseline results are above.
- Read `docs/REDESIGN_PLAN.md` and `docs/DESIGN.md` before architectural edits. Inspect relevant code/tests first.
- Prompt/gateway/output-contract changes require real local-model verification in addition to offline tests. Tell the user the ETA before starting; real-model runs may take hours and share the local model backend/VRAM.
- Keep duplicated config defaults synchronized between `config.yaml` and `src/ba_agent/config.py`. Keep credentials out of tracked files.
- Make small, reviewable diffs; do not commit unless asked. Never touch `D:\Code\Ba_gpt\TFM_BA_GPT` for LangChain work.

## First next step

1. Add a git remote, then tag `v1.0.0` and cut the release.
2. Run one real LM Studio pass when a chat model is loaded, to confirm the rung ladder on a second backend.
3. Everything else is polish, not pending work.
