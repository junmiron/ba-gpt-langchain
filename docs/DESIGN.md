# Design — `Ba_langchain` (Business Analyst agent, LangChain reimplementation)

Scope: reimplement the pipeline reviewed in [ANALYSIS.md](ANALYSIS.md) on **LangChain only**
(`langchain-core` + `langchain-openai`, no LangGraph), with a web UI and LM Studio as the model
backend (model chosen from a dropdown at runtime).

---

## 1. Design principles

These are the lessons of the last year of agent engineering, applied deliberately:

1. **Control flow in code, language in the model.** The old system asked an LLM to route between
   agents and then regex-matched its prose. Here a deterministic phase machine owns every
   transition; models only produce *content* (a question, a section of spec, a critique).
   → No `READY` / `QUESTION:` / `[ASSIGNMENT]` / `NEEDS_INFO` magic strings anywhere.
2. **Structured output as the inter-step contract.** Every model step declares a Pydantic schema and
   is parsed/validated/repaired. If it fails, we retry with the validation error appended — never
   silently continue.
3. **Durable state per session.** Everything the flow needs to resume (phase, subject coverage,
   requirements ledger, artefacts, event log) lives in SQLite, not in a browser tab or in model
   context. Restarting the server mid-interview loses nothing.
4. **The transcript is not the memory.** Instead of replaying the whole conversation into every agent
   (old: shared ChromaDB "memory" on 6 agents), each step receives a *small, purpose-built* context:
   the structured ledger + only the retrieved knowledge it asked for. Cheaper, and works with 8k-context local models.
5. **Bounded everything.** Max questions per subject, max clarification rounds, max review rounds,
   max tool iterations, wall-clock budgets. A stuck loop terminates with a reason instead of running forever.
6. **Deterministic checks before LLM critique.** Spec completeness is *linted in code* (sections present,
   non-empty, requirement IDs unique, acceptance criteria exist). The LLM critic only adds judgement on top.
7. **Local-first and honest about it.** LM Studio may be off, may be loading a 30B model for two minutes,
   may not support JSON-schema or tool calling. Every one of those is a handled state with a clear UI message,
   not a stack trace.

---

## 2. Architecture

```
┌──────────────────────────── Browser (static SPA, no build step) ────────────────────────────┐
│ sessions │ phase stepper │ live timeline (agent cards, tool chips, reasoning)               │
│ model ⏷  │ question card (answer / skip / N-A) │ Spec · Diagram · Ledger · Artefacts · Trace│
└─────▲────────────────────────────▲──────────────────────────────▲───────────────────────────┘
      │ REST (commands)            │ SSE (typed events, resumable)│ files
┌─────┴────────────────────────────┴──────────────────────────────┴───────────────────────────┐
│ FastAPI  (ba_agent.server.app)                                                              │
│   /api/models  /api/sessions  /api/sessions/{id}/reply  /api/sessions/{id}/events          │
│   /api/knowledge …/index      /api/artifacts/{id}/{name}                                    │
└─────┬───────────────────────────────▲───────────────────────────────────────────────────────┘
      │ commands                      │ events (persisted → replayable)
┌─────▼───────────────────────────────┴───────────────────────────────────────────────────────┐
│ WorkflowEngine  (ba_agent/workflow.py) — deterministic phase state machine                  │
│   intake → plan → interview ⇄ clarify → summary → approval → draft → review ⇄ fix           │
│                                    → diagram → deliver → done                               │
│   · owns all transitions, budgets and escalation      · emits typed events                  │
└─────┬──────────────────────────┬──────────────────────────┬─────────────────────────────────┘
      │                          │                          │
┌─────▼───────────┐  ┌───────────▼───────────┐  ┌───────────▼───────────────┐  ┌─────────────┐
│ Roles           │  │ AgentLoop             │  │ Knowledge                 │  │ Store       │
│ interview,      │  │ pure LangChain:       │  │ pdf/docx/txt/url ingest   │  │ SQLite      │
│ triage, summary,│  │ bind_tools +          │  │ chunking → hybrid         │  │ sessions    │
│ writer(section),│  │ astream → tool msgs → │  │ retrieval (vector+BM25)   │  │ events      │
│ use_cases,      │  │ loop until final text │  │ embeddings via LM Studio  │  │ ledger      │
│ glossary,       │  │ + per-role model      │  └───────────┬───────────────┘  │ artefacts   │
│ critic,         │  └───────────┬───────────┘              │                  │ kb chunks   │
│ diagrammer,     │              │                          │                  └─────────────┘
│ mermaid_reviewer│              │                          │
└─────────────────┴──────────────▼──────────────────────────▼───────────────────────────────────┐
│ ModelGateway (ba_agent/models.py)                                                             │
│  · LM Studio catalog via /api/v0/models (state, tool_use capability, context length)          │
│  · per-role model override + global default, hot-swappable at runtime                         │
│  · structured_call(): json_schema → function_calling → prompt+repair fallback ladder           │
│  · concurrency semaphore, load-aware timeouts, retry/backoff, usage accounting                 │
└──────────────┬─────────────────────────────────────────────────────────────────────────────────┘
               │ OpenAI-compatible HTTP (httpx) :1234/v1
        LM Studio desktop (local models; embeddings model for RAG)
```

No LangGraph import anywhere: the loop is `llm.bind_tools(...).astream()` + tool execution, and the
orchestrator is a plain async state machine. (`langgraph` is not even a dependency.)

---

## 3. The pipeline (what changed phase by phase)

| Phase | Old (AutoGen) | New | Why better |
| --- | --- | --- | --- |
| Intake | free-text task to planner | brief → `DiscoveryPlan` (subjects, priorities, per-subject question budget, inferred facts) | plan is data; user can see/edit it before questions start |
| Interview | interviewer + user proxy + planner routing every turn, counters in prose | engine iterates subjects; one LLM call for *next question*, one for *covered?* probe. Coverage counters are code (`SubjectState.asked/status`) | ~3× fewer calls; no skipped/repeated subjects; skip / N-A / "answer later" supported |
| Answer capture | model must call `save_conversation_to_jsonl` with a 2-record JSON payload | engine persists the Q&A pair itself, then a **triage** LLM step extracts facts + requirements into the ledger | persistence can't be forgotten or malformed; extraction is validated |
| Clarify | checker's `NEEDS_INFO` → planner re-asks | deterministic gap analysis over the ledger (missing NFR categories, no acceptance criteria, unnamed roles…) + critic pass → targeted questions, bounded rounds | gaps found by rules are not dependent on a model remembering to complain |
| Summary | summarizer asks for literal `APPROVE SUMMARY` | structured summary card; UI buttons *Approve* / *Request changes*; free-text changes become new ledger entries and re-summary | no exact-string gate; auditable approval record |
| Draft | one giant generation, writer calls use_case/glossary via `AgentTool`, must call `write_md_file_tool` before READY | **section-by-section map** over the FSD template (use cases + glossary generated by their own roles *in code*), assembled deterministically, then written once | small local models handle 1 section well and a whole FSD badly; file write is guaranteed, not requested |
| Review | checker LLM reads file, emits `READY`/`NEEDS_INFO` | **linter** (code) + **critic** (LLM) → verdict `{complete, auto_fixable[], needs_user[]}`; auto-fix re-runs only failing sections; user-gaps go back to interview; hard cap on rounds | completeness is verifiable; fixes are surgical; loop terminates |
| Diagram | creator ↔ reviewer over `TextCanvasMemory`, MCP render | generator → **syntax validator** (code) → reviewer critique → 1 revision round → `.mmd` saved; rendering happens in the browser with vendored mermaid | removes xelatex/MCP/absolute-path fragility; instant live preview |
| Deliver | terminate_agent runs 9 MCP steps incl. `xelatex` | deterministic assembler: `functional_specification.md`, `process_diagram.mmd/.svg`, `interview.jsonl`, `summary.md`, self-contained `deliverable.html` (print-to-PDF), optional PDF if weasyprint/pandoc present, `metadata.json` | works on Windows with zero system deps; PDF is opt-in |

Mermaid syntax validation is pure Python (`mermaid_utils.validate_mermaid`) and rendering happens in the browser.
An external renderer (kroki) can be enabled via `diagram.validator_url`, but it ships **disabled** — sending a
customer specification to a third party should be an explicit choice, not a default.

Kept from v1: the 8-subject agenda and per-subject cap, use-case/glossary as dedicated specialists,
the "user-dependent vs internally-fixable gap" escalation rule, per-session artefact archive.

---

## 4. Data model (SQLite, `state/ba_agent.db`)

```
sessions(id, title, brief, phase, status, awaiting, created_at, updated_at, state_json, model_json)
qa_pairs(id, session_id, q_index, subject, question, answer, rationale, skipped, created_at)
ledger(id, session_id, kind[fact|requirement|risk|assumption|decision], ref_id, title, body,
       category, priority, acceptance, source_q, updated_at)
events(id AUTOINCREMENT, session_id, ts, type, role, step, payload_json)   ← SSE replay cursor
artifacts(id, session_id, name, kind, path, bytes, sha256, created_at)
kb_docs(id, source, uri, sha256, title, bytes, indexed_at)
kb_chunks(id, doc_id, chunk_index, text, meta_json, embedding BLOB, dim)
```

`events.id` is the SSE cursor: a reconnecting browser sends `Last-Event-ID` and gets everything it missed,
so closing a tab or restarting the UI never loses output. `state_json` holds the phase machine state
(coverage map, budgets spent, current question id, review rounds) — that is what makes runs resumable.

---

## 5. Model handling (LM Studio)

* **Catalog**: `GET /api/v0/models` → `{id, type, state, capabilities, max_context_length, quantization}`.
  The dropdown lists only loadable chat models (`type in {llm, vlm}`), badges *loaded / not loaded / tool-calling*,
  and shows context length. `/v1/models` is used as fallback if the admin API is unavailable.
* **Selection**: one global default + optional per-role override (interviewer on a fast model, writer on a big one).
  Changing it applies to the next step — no restart. Persisted per session and in `config.yaml`.
* **Capability-aware degradation**: if a model lacks `tool_use`, tool-calling steps fall back to
  "no-tool" prompts (the engine never *needs* tools — see §3, all side effects are code). If JSON-schema is
  rejected, `structured_call` walks the ladder: `json_schema → function_calling → prompt + repair`.
* **Load-aware timeouts**: first call after a model switch can take minutes; default timeout 600 s with a
  "model may be loading" status event. Retries with backoff on connection errors, and a global semaphore
  (`max_concurrency`, default 1) so two sessions never fight for VRAM.
* **Context budget**: gateway reports each model's `loaded_context_length`; the context builder trims to
  `min(model_ctx, config.max_context_tokens) * safety` by dropping retrieved chunks first, then older QA,
  never the schema or the ledger summary.

---

## 6. Knowledge / RAG

* Ingest: PDF (`pypdf`), DOCX (`python-docx`), TXT/MD, web pages (readability-style main-content extraction).
* Content-hash dedup → re-indexing a folder never duplicates chunks (fixes finding #13).
* Chunking: structure-aware recursive splitter (~900 chars, 15% overlap) implemented locally — no `torch`.
* Embeddings: LM Studio `/v1/embeddings` (e.g. `text-embedding-nomic-embed-text-v1.5`) → vectors in SQLite BLOBs.
  If the embedding model is unavailable, retrieval degrades to BM25-only and says so in the UI.
* Retrieval: **hybrid** — cosine over cached vectors + BM25, fused with Reciprocal Rank Fusion, k=6,
  min-score cut. Exposed both as an internal service (steps request context explicitly) and as a
  `search_knowledge` tool for the writer/critic agent loop.

---

## 7. Web UI

Single-page app, **no build step** (`src/ba_agent/web/`, vendored `marked` + `mermaid` so it works offline):

* **Left**: session list (resume any), new-interview form with brief + optional knowledge files/URLs.
* **Top**: phase stepper with per-phase status and budget counters; model dropdown (+ "advanced" per-role overrides); stop/resume.
* **Centre**: timeline of agent cards grouped by role, streaming tokens live, collapsible tool calls and reasoning,
  question card pinned while awaiting input (Answer / Skip / Not applicable), approval card with Approve/Request-changes.
* **Right tabs**: Spec (rendered markdown, updates as sections land) · Diagram (live mermaid render + raw source + SVG download) ·
  Ledger (requirements table with priority/acceptance status) · Artefacts (download links) · Trace (event log with timings).

Transport: REST for commands (`POST /reply`, `/approve`, `/cancel`, `/models`), SSE for the event stream.
The engine runs as a background asyncio task per session, so HTTP disconnects don't kill work.

---

## 8. Observability & evaluation

* Every step logs `{step, role, model, prompt_chars, completion_tokens_est, latency_ms, attempts, schema_ok}` to `events` → the Trace tab and `metrics.json` in the deliverable bundle.
* One JSONL trace per session under `logs/` for offline analysis.
* `tests/` covers the parts that must never break without a model: phase transitions & budgets, spec linter,
  chunking/dedup, RRF fusion, JSON repair, sandbox path safety, SSE replay cursor. Model-dependent behaviour is
  exercised by an end-to-end script (`scripts/e2e_lmstudio.py`) run manually against LM Studio.

---

## 9. Old → new mapping

| `Ba_gpt` | `Ba_langchain` |
| --- | --- |
| `SelectorGroupChat` + `selector_func` + planner `[ASSIGNMENT]` | `WorkflowEngine` phase machine (`workflow.py`) |
| `planning_agent` (router) | `plan` step → `DiscoveryPlan` data; routing deleted |
| `interviewer_agent` + `UserProxyAgent` + Chainlit `AskUserMessage` | `roles/interview.py` + question card over SSE |
| `save_conversation_to_jsonl` (+ `_pending_pair` hack, FileLock) | engine writes `qa_pairs` in SQLite; JSONL exported at delivery |
| `summarizer_agent` + literal `APPROVE SUMMARY` | `roles/analysis.summarise` + approval card (`awaiting=approval`) |
| `functional_spec_writer_agent` (+2 AgentTools) | `roles/writing.py`: section map, use-cases/glossary roles called in code |
| `functional_spec_checker_agent` (`NEEDS_INFO`) | `spec_linter` (code) + `critic` role → structured verdict |
| `diagram_creator` ↔ `mermaid_reviewer` over `TextCanvasMemory` | `roles/diagram.py`: generate → validate → review → revise, files as truth |
| `terminate_agent` + 3 MCP servers + xelatex | `export.py` deterministic assembler (HTML/SVG/MD/JSONL, PDF optional) |
| `ChromaDBVectorMemory` + sentence-transformers/torch | `knowledge.py`: SQLite hybrid store + LM Studio embeddings |
| `model_config_*.yaml` × 8 + private-API agent cloning | `ModelGateway` + runtime dropdown + per-role overrides |
| Chainlit (`app.py`, `app2.py`) | FastAPI + static SPA |

---

## 10. Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| Local model ignores JSON schema / emits prose | repair ladder + one retry with the validation error; on final failure the step degrades to a documented fallback (raw text kept, flagged in UI) instead of aborting |
| Model without tool-calling selected | engine never depends on tools; capability badge warns in dropdown |
| VRAM contention / model swap stalls | global semaphore, load-aware timeout + status event, `/api/models` shows loaded state |
| Section-by-section drafting loses coherence | each section call receives the ledger + previously written section headings/summaries; assembler enforces template order and cross-references |
| Interview fatigue (too many questions) | per-subject cap (default 2), "covered?" probe can end a subject early, skip/N-A always available, progress shown |
| Review loop oscillation | hard `max_review_rounds` (default 2); remaining issues are reported in the deliverable as *Open questions* rather than looping forever |

---

## 11. Non-goals for v1

Multi-user auth/tenancy, LangGraph-style checkpointing/interrupts, streaming ASR/TTS interviews
(the old realtime notebook), LaTeX-quality PDF typesetting, and editing the FSD in-browser (download + re-upload instead).

---

## 12. Implementation notes — what running the tests changed

The suite (`python -m pytest -q`, 55 tests, no model) was written before it was ever executed, and running it
exposed defects that reading had missed. Each one is now covered by a test:

| Defect found | Impact if shipped | Fix |
| --- | --- | --- |
| `sessions` INSERT had 9 columns / 8 placeholders | **Every** session creation failed at runtime | corrected placeholder count |
| Chunks inserted before their parent `kb_docs` row | Knowledge indexing always raised a FK error (100% failure) | upsert the document first |
| `sandbox.py` read `cfg.paths.knowledge.docs_dir` (nonexistent) | Server could not start with the default config | use `cfg.knowledge.docs_dir`; also create `data/` + `uploads/` on startup |
| `assemble_spec` never called `render_open_questions` / `render_traceability` | §10 and Appendix A — the two sections promised to be *code-rendered* — shipped as "_Section pending_" | added `STATIC_RENDERERS` + `resolved_section_body()` as one source of truth for assembler **and** linter |
| Linter demanded a model-written body for those static sections | Review burned every round on an unfixable blocker in **every real run** | linter now checks resolved bodies |
| Traceability check searched the whole document, appendix included | Vacuous: ids are always present once Appendix A renders | check narrative sections only |
| Lint diagnostics were shown to stakeholders as questions | e.g. "Only 1 functional requirements captured" asked verbatim; answers could not fix it → wasted rounds | `LintFinding.question_for_user` + `Issue.source`; only real stakeholder wording is asked (critic-authored issues may still fall back to their description) |
| Auto-fix could target static sections | Regenerated nothing, looped | auto-fix restricted to model-written kinds |
| Approval rejection had no ceiling | summary ⇄ approval ping-pong forever with a scripted client (v1's stall class of bug) | `summary_rejections` + `interview.max_summary_rejections`, then notes flow into drafting with a warning |
| No usable model → 9 ladder failures per step, then a hollow document | Unreadable error spam; deliverables full of placeholders | gateway returns one `no_model` result; engine halts early with an actionable, resumable message (`NO_MODEL_MESSAGE`) |
| Blank answers returned HTTP 200 from the API | UI believed the answer was recorded; nothing happened | validated in `SessionManager.answer()` before scheduling → 409 |
| `Sandbox.resolve` anchored relative reads on a **sorted** root list | **Artifact downloads 404'd** (`data/` sorted before `outputs/`) | preserve caller order + prefer an existing match; the artifact endpoint now resolves against the session's own output dir explicitly |
| Setting a new default model wiped per-role overrides | Silent loss of configuration | `overrides=None` means "leave unchanged" |
| Per-session model choice was stored but never read by any role | The dropdown in *New interview* did nothing | `session_model(state, role)` threaded into all 11 model calls; per-role override beats session default beats global default |
| Resume button hidden whenever status was `error` | After the (resumable) no-model halt the user had no way to continue | resume offered for `idle`, `cancelled` and `error` |
| Hard-coded 3-subject interview floor overriding config | `min_subjects` silently ignored | agenda padded to the configured floor instead |
| CLI printed `→` into a legacy Windows console (cp1252) | `ba-agent doctor` **crashed** with `UnicodeEncodeError`, and the `serve` banner would too — the first two commands in the README | `configure_console()` reconfigures stdout/stderr to UTF-8 (`errors="replace"`) at CLI entry; scripts use it too |
| Writer sections came back empty on thinking models (found by the **real** e2e run, §13) | On Qwen3-27B via LM Studio every narrative section (§1–§8) shipped as a placeholder: the model spent its whole `max_tokens` budget (2048/2560) on internal reasoning that is not exposed in the stream, so the completion ended with `finish_reason=length` and **zero visible content — no error raised** | writer/use_cases caps raised to 16384; the gateway now retries once **without a cap** when a capped call returns empty output (both `stream_text` and `structured`) |
| Section prompt asked for JSON but `write_section` expected raw Markdown (found by the second real run, §13) | Every narrative section was wrapped in `{"section_id": …, "markdown": …}` — the model followed its instructions faithfully; the code never parsed them. The scripted gateway returned plain Markdown, so no test could catch it | prompt now asks for Markdown only; `clean_markdown()` defensively unwraps a JSON envelope carrying a `markdown`/`body` field (whole-reply match only, so quoted JSON samples are untouched) |

Also added since the first draft of this document: `tests/test_api.py` (the HTTP + SSE surface driven by the
scripted gateway over a real uvicorn socket), `scripts/demo_offline.py` (browse the full UI with scripted answers,
no VRAM), `README.md`, and `faulthandler_timeout = 60` in pytest so a future hang prints stacks instead of
blocking forever.

## 13. Real-model end-to-end run (LM Studio, Qwen3-27B)

`scripts/e2e_lmstudio.py` drives the real HTTP API against LM Studio with canned stakeholder answers — no fakes,
no VRAM beyond what is already loaded (it defaults to the first *loaded* chat model, e.g. the one serving Bionic).
First full run: **PASS** — intake → interview (20 questions incl. probe re-asks and gap rounds) → summary →
approval → draft ⇄ review (the critic asked two stakeholder questions mid-review) → diagram → `done`, all six
artifacts downloadable through `/api/artifacts`. It also exercised Resume across a process death: the session was
reloaded from SQLite and the interrupted question re-presented correctly.

That run exposed the thinking-model defect above (empty writer sections), which is fixed in §12's last row. The
structured-output ladder itself needed no fallbacks at all — `json_schema` succeeded first try on every call, so
the lower rungs remain a safety net rather than a daily path.

A second full clean run (`ba-20261009-154450-6c7194`, ~3.5 h) then passed mechanically — all phases to `done`,
six artifacts downloadable, only two by-design warnings (gap-round budget; review budget with 8 unresolved
blockers surfaced in §10) — but exposed the JSON-envelope defect above: sections carried real content, wrapped
in raw JSON. The fix is in §12's second-to-last row; that session's stored section bodies were unwrapped and its
deliverables re-exported deterministically (no model calls), so `outputs/ba-20261009-154450-6c7194/` now holds a
clean 46 KB specification with zero placeholders.

Practical notes for thinking models: each narrative section costs ~5 min of invisible reasoning before any visible
text; a full run takes several hours. For faster runs, point the writer/use_cases roles at a non-thinking model via
per-role overrides (⋯ in the UI) — everything else can stay on the big one.
