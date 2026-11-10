# Changelog

All dates are UTC. Version numbering: `0.x` is pre-release engineering, `1.0` is the first release
that a stakeholder can hand to a delivery team.

## 1.0.1 — 2026-11-09

### Providers
- **Strata is the default provider** (`provider: strata`, `http://127.0.0.1:47831/v1`, `q2_0` on the 5090), so an
  out-of-the-box run exercises the same model lane the AutoGen implementation is tested against. LM Studio stays
  fully selectable via `BA_PROVIDER`, `--provider`, `config.yaml`, or the Settings dialog.
- Switching providers at runtime re-binds the gateway: the `ChatOpenAI` client cache and the model catalogue are
  cleared, so a run cannot keep talking to the backend it started on. The choice is persisted and replayed by
  `apply_stored_model_choices`, which sets the provider *before* the model names (they are per-profile).
- Reset restores the provider the process started with, not the config file's value.
- Model lanes are stored **per provider** (`model_choices: {"strata": {...}, "lmstudio": {...}}`). A model id
  means nothing to the other backend, so switching providers no longer drags a stale `default_model` across,
  and switching back restores the lane you picked. Legacy flat keys (`default_model`, `embedding_model`,
  `model_overrides`) are migrated on read.
- Regression tests: `tests/test_provider_switch.py` (repoint, cache clear, rejected provider, restart parity,
  per-provider lanes, reset, UI control).

### Web UI
- `connectStream` registered no message handler, so the live SSE stream was opened and never read. Named frames
  (`event: question`, `event: token`, …) reach the browser only through a listener registered for that exact name,
  so every event emitted after the page loaded was dropped: an interview that had already asked a question still
  looked dead until the session was reopened (which replays the log over REST). `connectStream` now attaches one
  listener per type in `STREAM_EVENT_TYPES`, routed to `handleEvent`.
- `app.js` contained `es.onopen = () { … }` — a bare `() {}` after `=` is a syntax error, which stops the whole
  script from running: no sidebar, no cards, no question, and no error on the page.
- `node --check` is now a pytest gate (`tests/test_stream_event_types.py`), alongside a parity assertion that every
  `EventType` the server can publish has a client listener. Both were missing: CI ran `node --check`, but a broken
  list of event names was invisible.
- Suggestions explicitly use the light foreground token on their dark background and have a keyboard focus outline.
  Timeline cards no longer shrink into unreadable strips; new messages only auto-follow when the reader is already
  near the bottom, and restore loads event history in bounded pages instead of truncating at 1,500.
- Approval is a distinct composer state rebuilt from `awaiting=approval`; answer/skip controls are hidden and the
  user gets persistent **Approve** / **Request changes** controls. A phase-aware elapsed “Thinking” indicator is
  announced accessibly and stops when the run waits for input or finishes.
- Model lanes now include **Human**: **Real human** is the default; choosing a model simulates stakeholder answers.
  Simulated answers are labeled in the timeline and the generated FSD carries a simulation notice; summary approval
  remains a real-person decision.
- Structured calls emit role/step/model/rung timing and prompt/completion character counts to Trace without recording
  prompt text. On a covered interview answer, the coverage response can supply the next subject's question, avoiding a
  separate model round trip; deterministic duplicate and budget checks still gate it, with the existing generator as
  a fallback.

### Reliability
- A timed-out rung of the structured-output ladder gets one attempt and the ladder moves to the next method instead
  of retrying a hung request three times; the run reports `backend unresponsive` only after every rung has hung.
  Worst case for one role drops from 900 s to 360 s (`request_timeout_s: 300 → 120`).
- Regression tests: `tests/test_transport_errors.py` (hung rung retried once per rung; a timeout walks to the next
  rung instead of giving up).

## 1.0.0 — 2026-11-09

Full rewrite of the AutoGen `Ba_gpt` pipeline on `langchain-core`, with a deterministic phase machine
owning control flow instead of the LLM. **No LangGraph anywhere** — `create_agent` is out too, because in
LangChain 1.x it compiles to LangGraph underneath.

### Orchestration
- `WorkflowEngine` (`workflow.py`) drives `intake → plan → interview ⇄ clarify → summary → approval →
  draft → review ⇄ fix → diagram → deliver → done` in code. No magic tokens are regex-matched.
- Per-run budgets: questions per subject, gap rounds, review rounds, approval rejections, total
  questions, model timeouts, and one global concurrency slot. Exhausted budgets surface as visible
  warnings and open questions in the deliverable, never as a silent stall.
- State invariant: `status == WAITING_USER` iff `awaiting` is an answer or an approval. Previously a
  snapshot persisted mid-run could tell the UI "waiting for you" while the engine was working.
  Regression test: `tests/test_status_invariant.py` inspects every persisted snapshot.

### LangChain idioms (core only)
- `agent_loop.py` is a `RunnableSequence` of context policy → model step → tool executor.
- Structured output walks a per-profile rung ladder: `json_schema` → `function_calling` → prompt + repair.
  LM Studio and Strata declare their own rung order; a weak local model still completes.
- `KnowledgeRetriever` exposes the SQLite knowledge base as a `langchain-core` `BaseRetriever`.
- Heading-aware chunking via `langchain-text-splitters` when available, pure-Python fallback otherwise.
- Every Jinja template has a typed context contract (`prompt_context.py`), validated before a model is
  called; `undeclared_variables()` catches a template that grew a reference the caller never learned.

### Deliverables
- Typst-native PDF: hand-generated `.typ` (no pandoc), `#outline()` TOC, native tables, banded figures.
  `typst.exe` is fetched by `scripts/fetch_typst.py` into gitignored `vendor/`, or found on PATH, or
  pointed at with `BA_TYPST_BIN`.
- Pure-Python ReportLab fallback when Typst is absent: bookmarks, TOC, and diagrams embedded as vector
  paths (no `rlPyCairo`, so nothing is rasterised).
- Mermaid diagrams render to SVG locally via `mmdc`; tall diagrams are band-cut at rows the geometry
  profile says are quiet. Cut safety is a geometry ink weight (1 per connector stroke, 3 per node box or
  label), not a dark-pixel count. Warning threshold: weight > 6.
- Self-contained HTML deliverable with a Contents block whose anchors are verified to resolve.
- `pdf_qa.json` records what the gate actually checked: pages, outline entries, clickable Contents links,
  missing requirement ids, Markdown leakage, band-cut weight.
- Diagram PNGs are written to disk and referenced by path; the schema rejects raw image bytes in state.

### Features
- Transcript ingestion: a pasted or attached transcript seeds the ledger, and only the subjects it
  actually answers are skipped. `interview.subject_coverage_threshold` counts evidence units.
- Lessons store: each finished run records generalisable gaps, ranked by `seen_count`, and the next
  run's plan prompt pins them as context. Plain retrieval — not reinforcement learning.
- Session lifecycle: rename, delete (removes the artefact folder), pagination, live list refresh, busy
  indicator, resume, and a one-click export zip.
- Settings API/UI for budgets, retrieval, diagram rendering and per-role model lanes.
- Optional bearer-token auth (`server.auth_token` / `BA_AGENT_TOKEN`), off by default for localhost use.
  Already present from Phase 0: CSRF origin check, tight CORS, SSRF guard, upload caps + hash dedup,
  CSP and SVG sanitising on served deliverables.

### Release pass fixes
- **Settings are a delta, not a snapshot.** Saving one field used to persist every field in that section,
  permanently freezing untouched `config.yaml` values at their then-current values. Only edited keys are
  stored now; a later edit to `config.yaml` for a field you never touched still takes effect.
- **`POST /api/settings/reset`** ("Reset to defaults") clears the stored delta and restores the config the
  process started with. `get_config()` caches a singleton that *is* the manager's config object, so a naive
  reset copied a mutated object onto itself and silently did nothing.
- **Corrupt stored settings are ignored, not applied.** Restored values are validated through the section
  model; a bad row is logged and skipped instead of poisoning runtime config.
- **Embedding model selection is a real control.** The UI select was decorative; its value now round-trips
  through the API, the store, and `restore_model_choices()`, and an empty choice means BM25-only on purpose.
- **Knowledge-base status is honest.** `status()` returns `document_count` separately from the document
  list and a `vector_search_reason` that distinguishes "no embedding model configured" from "embedding
  model configured but the chunks were never vectorised".
- **Structured-output rungs are learned per model** (`models.py`), and `metadata.json` records which rung
  actually satisfied each call (`structured_output.rung_counts`), which PDF engine produced the file
  (`pdf_engine`), and how much model effort the run spent (`llm_usage`). A run that silently fell back to
  prompt-mode JSON for every step is now visible.
- **Architecture is enforced by tests**, not prose: `tests/test_architecture_rules.py` fails the build if
  `langgraph`, `deepagents`, `create_agent`, `StateGraph`, `interrupt` or `Command` appear in `src/`.
- **Live tests are opt-in.** `pyproject.toml` sets `addopts = "-m 'not live'"`, so the default suite never
  contacts a model server; `pytest -m live` runs the Strata smoke test when one is up.
- QA reports diagrams that failed to render (`diagram_render_failures`) instead of silently omitting them
  from the PDF, and band cuts are chosen by the same ±2-row window weight QA measures rather than a single
  row.
- **Reset restores the model lanes too**, not just the budget sections: `default_model`, `embedding_model`
  and `model_overrides` were left behind by the earlier reset.
- **CLI and browser agree.** `ba-agent index|doctor|sessions|export` used to read only `config.yaml`, so a
  setting saved in the UI silently changed nothing on the command line. Stored model choices and run
  settings are now applied before every CLI command; `tests/test_cli_settings.py` asserts the helper is
  wired into every service-building path via the AST, not by eyeballing the file.
- **A bad environment variable is reported, not ignored.** `BA_PORT=notanumber` used to fall back to
  `config.yaml` with no trace. It now logs which key, which value and why.
- **`scripts/config_audit.py` is a real gate.** The dead-config and config-drift probes existed only as
  throwaway scratch scripts. They now live in `ba_agent.config_audit`, run in CI, and fail the build on
  any declared-but-unread field or any `config.yaml` value that disagrees with the code default.
- **`logs/` is actually used.** Nothing configured the root logger, so the directory `ensure_dirs()`
  created was empty. `configure_logging` now attaches a console handler plus a rotating file handler.
- **`metadata.json` carries the QA verdict**, not just the PDF engine, so a deliverable that failed the
  gates is distinguishable from one that passed without opening a second file.
- **Run timing is attributed per phase.** `metrics.by_phase` records calls, model time and wall time for
  each phase; the artifacts panel and the timeline summary name the three slowest. "The run took 25
  minutes" is not actionable; "interview was 19 of them" is.
- **A down backend is diagnosed as unreachable, not as bad JSON.** Previously a refused connection entered
  the same `except` as a malformed reply, so the retry appended "Your previous reply was not valid for the
  required JSON schema" to a conversation where the model never answered, and the ladder walked every rung
  against a dead server. Transport errors now back off, retry the same rung, and give up with a clear
  reason (`tests/test_transport_errors.py`).
- **Banding no longer writes into the caller's working directory.** `render_spec_pdf` with no `working_dir`
  fell through to `Path.cwd()`, leaving stray `process_diagram_N_band_M.svg` files in the repo root.
- **Interview questions stop repeating.** The anti-repetition list was built by concatenating open and all
  questions and slicing the last 14, so a re-asked question appeared twice and the oldest open questions
  were the ones cut off. `previous_questions()` normalises, de-duplicates, and keeps every open question
  before trimming answered ones (`tests/test_question_dedup.py`).
- **Queued review blockers are actually asked.** `_step_review` and `_step_clarify` asked the first blocker
  and *queued* the rest with no drain: the extra questions were written to the Q&A store, counted against
  the budget, and never shown. `current_question` also returned the newest unresolved question, so the
  question on screen was the wrong one. Questions now carry an `emitted` flag and `_drain_queue()` surfaces
  them one at a time (`tests/test_question_queue.py`).
- **`max_total_questions` is a ceiling, not a suggestion.** Queueing bypassed it, so a critic with several
  blockers could record more questions than the configured limit. `_queue_question` refuses past the
  ceiling and the refused blockers are appended to the ledger as `open_question` entries so they still reach
  section 10 of the deliverable.
- **`read_artifact` cannot cross sessions.** The tool interpolated a model-supplied name straight into
  `f"{session_id}/{name}"`, so `../<other_session>/salary_data.md` escaped the session folder inside the
  sandbox's writable outputs root. The name is now reduced to its basename
  (`Path(name).name`) and invalid names return the available artefacts instead of reading foreign files.
- **Transient SSE frames no longer rewind the reconnect cursor.** Streamed tokens carried `id: 0`, which
  the browser stores as `Last-Event-ID: 0`; every reconnect after a token replayed the whole event log.
  `_sse` now omits the `id:` line for unpersisted events.

### Tooling
- CI: pytest on 3.11 and 3.13, `ruff check`, `ruff format --check`, `mypy src`, `node --check` on the UI,
  `python scripts/config_audit.py`, and `python scripts/e2e_offline.py` as a separate acceptance gate —
  unit tests can all pass while the export path is broken.
- Offline acceptance: `scripts/e2e_offline.py` runs the whole pipeline against a scripted gateway and
  asserts the deliverables; `scripts/e2e_live.py` does the same against a running server;
  `scripts/e2e_lmstudio.py` does it against a real model.
- 590 tests, ~86 s, no model, no VRAM, no network.

## 0.1.0 — 2026-11-07

Phase 0 stabilization of the initial rewrite: typed events, SSE replay cursor, CSRF/CORS/SSRF/upload
guards, schema versioning with migrations, event retention, and 12 role-layer defect fixes. Suite grew
from 58 to 167 tests.
