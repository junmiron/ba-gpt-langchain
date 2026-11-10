# Ba_langchain — Full Analysis, Ecosystem Research, Target Architecture & Implementation Plan

Status: **approved and executed** — Phases 0–3 shipped on 2026-11-09; Phase 4 dropped; the release pass (Phase 5) is the remaining work.
Baseline at approval: `python -m pytest -q` → **58 passed in 16.7 s** (no model needed). **Current:** **590 passed in ~86 s**.

---

# Part 1 — Analysis of the current codebase

## 1.1 What the project is (as-built)

~9.2k lines: `src/ba_agent` (5,445 lines core), `tests` (~2,600), a build-less vanilla-JS SPA, Jinja2 prompts, SQLite persistence, FastAPI + SSE, LM Studio as the only model backend.

The pipeline:

```
intake → plan → interview ⇄ clarify → summary → approval
       → draft(section-by-section) → review ⇄ fix → diagram → deliver → done
```

The four pillars of the design are sound and are the reason this rewrite beats the AutoGen original:

1. **Control flow in code, language in the model** — `workflow.py` is a deterministic phase machine; the LLM only produces content. No magic strings (`READY`, `[ASSIGNMENT]`, `NEEDS_INFO`).
2. **Structured output as the inter-step contract** — every model step declares a Pydantic schema; `ModelGateway.structured_call` walks a ladder: `json_schema → function_calling → prompt + repair`.
3. **Durable state** — sessions, QA pairs, ledger, events, artifacts in SQLite; `events.id` is the SSE replay cursor; runs survive a server restart (`mark_interrupted` → resume).
4. **Bounded loops** — per-subject question caps, gap rounds, review rounds, summary rejections, `MAX_STEPS_PER_ADVANCE = 40`, wall-clock budgets.

## 1.2 Concrete defects found (verified, with file:line)

Roles layer:

| # | Defect | Evidence | Impact |
|---|---|---|---|
| R1 | `chat_agent.md.j2` has no `---USER---` marker → `render()` returns a generic system prompt and throws away the real instructions | `prompts.py:47-50`, `runtime.py:257` | The post-delivery chat agent ignores its persona and session context |
| R2 | `has_similar` dedup uses a 24-char prefix — *"The system shall provide"* is exactly 24 chars, so distinct requirements get merged | `schemas.py:500` | Silent loss of requirements |
| R3 | Mermaid label auto-quoting corrupts balanced-paren labels (`A[total (net)]` → `A["total (net)]"`) and the validator accepts the result | `mermaid_utils.py:20,135-141` | Diagrams render wrong |
| R4 | `verdict.issues = merged[:12]` can truncate **lint blockers** (authoritative) behind critic issues | `roles/review.py:63` | A review round can "pass" with blockers dropped |
| R5 | Strict `Literal` fields without lenient validators (`SubjectSpec.priority`, `DiagramDraft.diagram_type`) — one stray `"High"` fails the whole step | `schemas.py:128,330` | Whole plan/diagram set discarded |
| R6 | Critic only sees the first 14,000 chars of the spec | `roles/review.py:34` | Glossary/open questions/appendix never reviewed |
| R7 | Diagram repair compares against the original draft, not the auto-fixed baseline | `roles/diagram.py:49-51` | Repair can be rejected although it is better |
| R8 | `write_use_cases` dereferences `state.summary.scope_in` without a None guard | `roles/writing.py:74-75` | Possible crash |
| R9 | Fallback summary puts FR bodies into `scope_in`, so the scope gap never fires | `roles/analysis.py:139`, `spec.py:319` | Fake completeness in degraded runs |
| R10 | Threshold disagreement: FR floor hard-coded 5 in `deterministic_gaps` vs configured in `lint_spec`; NFR gap ≥2 vs lint ≥3; `"sla"` substring matches "island"/"slate" | `spec.py:229,286,314,326,327` | Inconsistent gating between code paths |
| R11 | Silent fallbacks without WARNING events (`probe` fails open, gap analysis, reviewer parse failure) | `interview.py:157`, `analysis.py:118`, `diagram.py:90` | Users never see degraded quality |
| R12 | Dead branches in `transcript_lines` (skipped/NA labels unreachable); broken package fallback path in `prompts.py:35-36`; unused `SectionDraft`, `available_prompts`, `ledger_lines` | `context.py:123-127` | Confusion, latent bugs |

Server / UI / persistence layer:

| # | Defect | Evidence | Impact |
|---|---|---|---|
| S1 | SSE replay is a single query with `limit=1500`; beyond that the middle of the timeline is never sent; `?after=` overrides the browser's `Last-Event-ID` | `server/app.py:211,216` | Silent missing output on reconnects |
| S2 | Fanout queue drops the oldest 500 events silently | `events.py:70-80` | Live events lost during fast phases |
| S3 | `runtime.py:154` calls `self._chat_step`, which is **not defined** | `runtime.py` | AttributeError if ever scheduled |
| S4 | No auth on any endpoint; CORS condition inverted (`"*"` *disables* the middleware); multipart `/upload` and form-style POSTs to `/cancel`,`/resume` are CSRF-able | `app.py:95-97` | Any local page can drive the agent |
| S5 | Uploads: 25 MB checked **after** the file is fully read; no count/extension limits; dedup keyed on path, not content hash | `app.py:249-259`, `knowledge.py:295-313` | Memory blowup, duplicate KB entries |
| S6 | `fetch_url` follows redirects to any host, no private-IP block, no size cap → SSRF | `knowledge.py:184-192` | Local network exposure |
| S7 | `deliverable.html` served same-origin, unsanitized model-generated SVG embedded, no CSP | `export.py:66-68`, `app.py:270-283` | Stored-XSS vector in the deliverable |
| S8 | `events` table unbounded (`trim_events` never called); full `state_json` rewritten every save; `mark_interrupted` only scans the 500 most recent sessions; `load_state` swallows validation errors returning `None` | `store.py:153-168,247`, `runtime.py:57` | Slow, leaky, resumable-fragile |
| S9 | No schema version/migrations; `jsonx.extract_json` is O(n²) and its apostrophe/True/None rewrites corrupt string contents (acknowledged in code) | `jsonx.py:35-66` | Fragile JSON recovery |
| S10 | No CI, no lint config, no lockfile; UI has no delete/rename, no export button, no ledger editing; `kb_all_chunks` loads all vectors per search | various | Ops debt |

## 1.3 What is genuinely good (keep)

- The **hybrid quality loop**: code linter as floor, LLM critic on top, auto-fix restricted to model-written sections, bounded rounds, open questions surfaced in §10 instead of looping forever.
- **Capability-aware degradation**: tool-less models get no-tool prompts; schema-less models walk down the ladder; degraded states are *documented*, not hidden.
- **Deterministic assembly of static sections** (title page, traceability, open questions) — never asks the model to render tables of ids.
- Sandbox with symlink containment; offline-friendly vendored UI; `faulthandler_timeout`; scripted-gateway tests that pin phase transitions.
- Load-aware timeouts, global VRAM semaphore, per-role model overrides threaded through all 11 call sites.

---

# Part 2 — LangChain ecosystem research (November 2026)

Versions researched: `langchain 1.4.6` (2026-10-14), `langchain-core 1.7.0`, `langgraph 1.2.20` (2026-11-05).
**As built, the project depends on `langchain-core`, `langchain-openai` and `langchain-text-splitters` only** —
there is no `langchain` package in `pyproject.toml`, so there is no LangGraph transitive dependency and no
version pin to track. (Verified: `uv run python -c "import langchain"` → `ModuleNotFoundError`, while
`langchain_core` 1.7.6 and `langchain_openai` 1.7.4 import cleanly.)

| Technology | Verdict | Why |
|---|---|---|
| `create_agent` + **middleware** | **Skip** (needs LangGraph runtime — see Part 6.1) | It is the standard agent loop, but it compiles to LangGraph, which the architecture rules forbid. Our own loop in `agent_loop.py` does the same job with explicit budgets, and prompt-contract validation catches missing placeholders before a model ever sees a prompt. |
| **Model profiles + structured-output strategies** (`ProviderStrategy` / `ToolStrategy`, auto-selected from capability data, `langchain>=1.1`) | **Not adopted — hand-rolled ladder instead** | Those strategies live in the `langchain` package, which this build does not depend on (only `langchain-core`, `langchain-openai`, `langchain-text-splitters`). The equivalent behaviour is implemented directly in `models.py`: per-profile rung order plus a per-model learned rung (`_learned_rung`) and `rung_counts`, so a model that fails `json_schema` stops being probed with it. LM Studio's OpenAI-compatible server *does* support `response_format: {"type":"json_schema"}` for capable models; the prompt+repair rung stays as the final fallback for weak local models. |
| **LangGraph** (checkpointers, `interrupt()` / `Command(resume)`, stores, durable execution) | **Skip — decided in Part 6.1** | It is the runtime `create_agent` already compiles to. Its killer features map exactly onto hand-rolled code: step-level durable resume (checkpointer), human question cards (`interrupt()`), cross-session memory (`store`). |
| `langchain[mcp]` (`MCPAdapter`, beta since 1.4) | **Deferred** | Would let the writer/critic call the sibling repo's MCP servers (Mermaid render, spec formatting), but both jobs are already done in-process: `artifacts/typst.py` renders the PDF and `mermaid_utils.py` validates and renders diagrams. Not Phase 1 scope, and a beta API adds no value here.
| `langchain-text-splitters` | **Adopted** | `knowledge.py` uses `MarkdownTextSplitter` when `heading_aware` is on, falling back to the built-in paragraph splitter so chunking still works without the extra package. BM25+RRF hybrid retrieval stays custom: it is torch-free and beats stock vector-only on a single-user local corpus. |
| **LangSmith** tracing | **Skip (privacy)** | Customer specs should not leave the machine. Keep the local typed event log + `metrics.json` + JSONL trace. Optional: self-hosted Langfuse later. |
| `deepagents` | **Skip** | Designed for open-ended long-horizon tasks; this is a fixed pipeline with budgets — the opposite. |
| ChromaDB | **Skip** | SQLite-backed hybrid store is already production-grade for single-user local scale and avoids `sentence-transformers`/torch. |

---

# Part 3 — Target architecture

## 3.1 Principles (unchanged, made explicit)

1. Code owns control flow; models own content.
2. Every step has a typed contract (Pydantic), validated, with a documented degradation path.
3. Durability is per-step, not per-phase.
4. Every loop is bounded; every fallback is visible in the UI and in the event log.
5. Local-first, zero cloud dependency for the core pipeline.

## 3.2 Module layout (as built)

The proposed `gateway/` / `agents/` / `workflow/` nesting was not adopted — a flat module set with two
subpackages (`roles/`, `artifacts/`) is enough at this size and keeps imports short.

```
src/ba_agent/
  config.py              # schema_version, env overrides, per-role model lanes, budgets
  schemas.py             # strict domain models — typed diagrams and artifacts, no raw dicts
  models.py              # ModelGateway: profiles, semaphore, load-aware timeouts,
                         # structured-output ladder json_schema → function_calling → prompt+repair
  agent_loop.py          # RunnableSequence: context policy → model step → tool executor
  prompt_context.py      # typed context contract per template + undeclared-variable test
  prompts.py             # Jinja render with contract validation
  retriever.py           # KnowledgeRetriever — BaseRetriever over the SQLite hybrid store
  knowledge.py           # hash-keyed dedup, size-capped ingest, heading-aware chunking, BM25+vector+RRF
  mermaid_utils.py       # pure-Python validation + optional local mmdc render (SVG, opt-in PNG)
  spec.py                # template + linter (single source of thresholds)
  export.py              # deterministic assembler: md, html, pdf, qa sidecar, zip
  workflow.py            # the phase machine and all budgets (no LangGraph)
  runtime.py             # per-session background task, locking, SSE fanout
  store.py               # schema_version 3 + migrations, event retention, lessons
  events.py              # typed Enum events, payload schemas, non-blocking fanout with loss reporting
  sanitize.py            # CSP + SVG sanitising for served deliverables
  server/app.py          # FastAPI: CSRF origin check, CORS, optional bearer token, SSE replay cursor
  roles/                 # analysis (triage, transcript, gaps, summary), interview, writing, review, diagram
  artifacts/             # typst.py, reportlab_pdf.py, html.py, qa.py
  web/                   # SPA: rename, delete, export, ledger editing, settings, transcripts, lessons
  tools/                 # sandbox.py, chat_tools.py
```

No `mcp.py`: MCP integration stayed deferred (Part 6). `langchain-text-splitters` is used inside
`knowledge.py` rather than a separate `chunking.py` module.

## 3.3 Deliverable pipeline — decided by a real bake-off (2026-11-07)

All four engines rendered the *same* real spec (`sess-20261106143438`): 4 mermaid diagrams (band-split to fit pages), 38 tables including the 57-row FR table, glossary, traceability appendix.

| Engine | Pages | Bookmarks | Clickable TOC links | Diagram fidelity | Size | Runs on Windows? |
|---|---|---|---|---|---|---|
| **Typst, hand-written `.typ` (no pandoc)** | 40 | 78 | 79 | **vector SVG, labels are real selectable text** | 1.07 MB | yes — vendored `typst.exe` (already in sibling repo) |
| Typst via pandoc (current sibling prod) | 48 | 79 | 80 | diagram labels baked into PNG raster bands (not selectable) | 1.17 MB | yes — vendored binaries |
| WeasyPrint | 41 | 79 | 77 | vector, real text labels | 0.32 MB | **no** — needs GTK/Pango (`choco gtk3` has no package; only works via WSL/Linux/CI) |
| ReportLab + svglib | 44 | 77 | 0 (TOC entries not hyperlinked) | SVG converted to vector paths; glyph outlines only (labels not selectable) | 0.25 MB | yes — pure Python, pip-only |

Verification details (all measured with PyMuPDF):
- Typst-native PDF: every `FR-001..FR-057` and `NFR-001..NFR-031` row present; diagram labels ("Stage 1", "Suggest next available window", "escalated to Owner") extracted as text on the diagram pages; TOC is `#outline()` — genuinely clickable.
- Band-cut ink check (quiet-row snapping ported from the sibling fix): cut rows carry 5–10 dark px out of 784 — all on whitespace/connector lines, no node boxes sliced.
- Typst tables split across pages natively with the header repeated — the pandoc `_split_markdown_tables` chunking hack and the `---` HR normalization are **not needed** if we generate `.typ` directly.

**Recommendation:** Typst-first, hand-generated `.typ` (drop pandoc from the path entirely); ReportLab+svglib as the pure-Python fallback when the vendored binary is absent. WeasyPrint only in CI/Linux.

## 3.4 The LangGraph decision — DECIDED

**User choice (2026-11-07): A — Idioms only: pure `langchain-core`, no LangGraph imports at all.**

Practical meaning:
- `workflow.py` stays the deterministic phase machine and the only owner of transitions.
- `create_agent` is **not** used anywhere: in LangChain 1.x it compiles to LangGraph, so it is out. Writer/critic/chat keep the hand-rolled `langchain-core` tool loop.
- No `StateGraph`, no `SqliteSaver`, no `interrupt()`/`Command(resume)`. Human gates stay hand-rolled (`awaiting` cursor + SSE), which is already tested and works.
- The hand-rolled loop's budget/termination rules move into middleware hooks where possible without LangGraph.

Phase 4 (LangGraph engine) is **dropped** from the plan.

## 3.5 Feature parity + additions vs the AutoGen original

Status is as-built (2026-11-09), not the original plan.

| Feature | Status here | Notes |
|---|---|---|
| Transcript → spec ingestion | **shipped** | Upload a transcript, the engine seeds the ledger from it, and only subjects with enough transcript evidence (`interview.subject_coverage_threshold`) skip the interview |
| Session continuation | **shipped** (phase-level) | Step-level checkpointing would need LangGraph, which is out — the resumable `SessionState` + event cursor is the deliberate equivalent |
| Professional PDF (TOC, tables, diagrams) | **shipped** | Typst-native with a ReportLab fallback; see Part 3.3 |
| Web deliverable w/ linked TOC | **shipped** | Self-contained HTML, TOC anchors verified by the QA gate |
| Config UI page | **shipped** | Settings are a delta over `config.yaml`; `POST /api/settings/reset` restores the values the server started with |
| Feedback / lessons store | **shipped** | Frequency-ranked `lessons` table (`seen_count`) injected as pinned context into the plan prompt. Retrieval, not RL |
| Fixed question box, download buttons | **shipped** | Pagination, live refresh, busy indicator, export zip, ledger editing |
| MCP integration | deferred | Not required; needs separate approval |
| CI | **shipped** | Offline pytest 3.11 + 3.13, ruff, mypy, `node --check`; `addopts = "-m 'not live'"` keeps model calls out of CI |
| Auth/CSRF/CSP | **shipped** | CSRF/CORS/SSRF/upload/CSP in Phase 0; optional bearer token in Phase 5, off by default |

---

# Part 4 — Step-by-step implementation plan

**Phase 0 — Stabilize (≈ 1–2 days).** No architecture change. **DONE 2026-11-07** — commits `c71db05`…`b55afa8`; suite went 58 → 167 tests, all green, offline.
1. Fix R1–R12 and S1–S3 (typed events; SSE cursor precedence + looped replay; delete `_chat_step` or implement it; truncation keeps lint blockers; lenient validators; unified thresholds in config).
2. Upload: cap stream-wise, extension allowlist, hash dedup; SSRF guard; CSP + sanitize SVG; CSRF origin check.
3. Remove dead code; add `schema_version` + retention for `events`.
4. GitHub Actions CI (pytest 3.11/3.13, ruff, UI static check).
*Acceptance:* existing 58 tests + new regression tests for every fix, CI green.

**Phase 1 — LangChain 1.x idioms, core-only (≈ 2–4 days).** **DONE 2026-11-09.**
1. Deps: bump `langchain-core>=1.7`, add `langchain-text-splitters`. No `langgraph`, no `langchain.agents.create_agent`
   (it compiles to LangGraph — see Part 6.1).
2. `AgentLoop` stays a hand-rolled loop but is refactored into `RunnableSequence[context policy | model step |
   tool executor]` composition so budget/termination rules are reusable runnables, not inline counters.
3. Structured output via model profiles: each profile declares its rung order (LM Studio: json_schema if the model
   advertises it, then function calling, then prompt+repair; Strata: json_schema — verified live on q2_0).
4. Knowledge access becomes a `Retriever` subclass so writer/critic/chat share one interface; chunking moves to
   `RecursiveCharacterTextSplitter` with heading-aware separators.
5. Prompt-context contract: per-template typed context model + a test extracting undeclared Jinja variables.
*Acceptance:* the 167-test suite stays green offline; new runnable/retriever tests; artifact output unchanged.

**Phase 2 — Deliverables (≈ 3–5 days).** *(updated by the bake-off: generate Typst directly, skip pandoc)* **DONE 2026-11-09.**
1. `typst.exe` is fetched on demand by `scripts/fetch_typst.py` into the gitignored `vendor/` dir; discovery order is `BA_TYPST_BIN` → PATH → `vendor/typst/**/typst.exe`.
2. `artifacts/typst.py`: Markdown → hand-generated `.typ` (sections, `#outline()` TOC, native tables, figures). Diagram banding with quiet-row snapping (ported logic, verified 5–10 px cut ink).
3. Vendored `mermaid-cli` render path (SVG + PNG; disabled gracefully if absent).
4. Pure-Python fallback: `ReportLab + svglib` when the Typst binary is missing.
5. HTML deliverable: linked TOC + inline SVG (self-contained).
6. QA tests on the PDF: clickable Contents annotations ≥ 70, bookmarks ≥ 75, all FR/NFR rows present, no raw
   Markdown leakage. Band-cut safety is measured as a **geometry ink weight** (1 per connector stroke, 3 per node
   box or label), not a raw dark-pixel count — a cut through a label is worse than a cut through a line. Warning
   threshold: weight > 6.
*Acceptance:* `functional_specification.pdf/html/md` generated deterministically without any model call; QA suite green.

**Phase 3 — Feature parity (≈ 4–7 days).** **DONE 2026-11-09.**
1. Transcript ingestion mode (upload transcript → triage → ledger → gap-only interview).
2. Config page UI (roles→model lanes, budgets, knowledge dirs).
3. Lessons/feedback store (session end: capture corrections; future sessions inject them as pinned context).
4. UI gaps: delete/rename sessions, export button, ledger editing, chat busy state.
5. Session list pagination + live updates.
*Acceptance:* parity checklist vs sibling repo; e2e runs covering transcript mode.

**Phase 4 — LangGraph engine: DROPPED.** It contradicts the user's Part 3.4 choice and the no-LangGraph
rule in Part 6.1. Durability stays what Phase 0 already built: a `state_json` snapshot after every step plus
SSE event replay. Nothing here is to be implemented.

**Phase 5 — Release.** **DONE 2026-11-09** except the tag: optional bearer-token auth, version bump to 1.0.0, `CHANGELOG.md`, README/docs reconciliation, dead-config sweep (0 dead of 55), and real-model validation against Strata. `git remote -v` is empty, so `v1.0.0` cannot be tagged or published from this machine.

---

# Part 5 — Risks

- `langchain[mcp]` is beta — deferred; nothing in the current plan needs it.
- No durable-execution framework is being adopted, so there is no idempotent-replay discipline to learn — the crash-resume path is the `state_json` snapshot, and it is already tested.
- Vendor binaries are **not** committed: `vendor/` is gitignored and `scripts/fetch_typst.py` fetches Typst on demand. If the binary is absent the ReportLab fallback renders the PDF instead.
- Phase 1 changes the agent loop that tests cover — mitigated by keeping the scripted gateway harness unchanged.

# Part 6 — Open decisions for approval

1. ~~LangGraph~~ — **decided: A (idioms only, no LangGraph imports)**. Consequence: `create_agent` is also
   out, because in LangChain 1.x it compiles to a LangGraph under the hood. Phase 1 keeps the deterministic
   engine and the hand-rolled tool loop (already pure `langchain-core`), and adopts core-only idioms:
   `RunnableSequence`, per-profile structured-output strategies, `langchain-text-splitters`, typed prompt contexts.
2. ~~PDF engine~~ — **decided: Typst-native** (hand-generated `.typ`, vendored `typst.exe`, ReportLab fallback).
3. ~~Transcript ingestion~~ — **decided: yes, Phase 3**.
4. ~~Security~~ — **decided: localhost hardening only** (done in Phase 0: CSRF origin check, CORS fix, SSRF
   guard, upload caps + dedup, CSP + SVG sanitising). Token auth deferred to Phase 5 if ever exposed.
5. ~~Order~~ — **decided: stabilize first** (Phase 0 shipped; Phase 2 before Phase 3-scale work).

## Part 6.1 — Two fixes found while verifying against real runs

- **Band cuts.** The picker originally minimised a single profile row while QA scored a ±2-row window, so the
  number the picker optimised and the number QA reported were different. Both now use the same window
  (`CUT_WINDOW_SPAN = 2`). `DiagramFigure` carries its own `source_svg` so a figure's cuts can never be scored
  against a different figure that happens to share its title.
- **Structured-output rungs.** On a quantised local model the `json_schema` rung times out the same way on every
  call, burning a full request timeout ~29 times per run. The gateway now remembers which rung last worked for
  each model and starts there, keeping the rest of the ladder as fallback. An explicitly declared
  `structured_output_rungs` still wins.

All five decisions are now closed. Remaining work is the release pass (Phase 5) and the optional
real-model validations listed in `handoff.md`.
