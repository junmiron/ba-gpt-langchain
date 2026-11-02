# Handoff — BA Agent (LangChain rewrite)

**Read this first if you are a coding agent picking up work on this project.** It is written for
OpenCode / Hermes (or any other agent). Everything here was verified against the code and test
record as of the baseline commit `04f2b46` (2026-11-02).

---

## 1. What this project is

A **Business Analyst agent** that interviews a stakeholder about a software project, builds an
approved requirements summary, then drafts and reviews a full Functional Specification Document
(FSD) with use cases, glossary and Mermaid process diagrams — served as a web app backed by
**LM Studio** (local OpenAI-compatible API).

It is a rewrite of an older AutoGen-based agent at `D:\Code\Ba_gpt` (**read-only reference; do not
modify it**). The original request had 7 items, all completed:

1. Review/analyse the old code ✅
2. New improved design (see [docs/DESIGN.md](docs/DESIGN.md)) ✅
3. Web UI ✅ (vanilla-JS SPA, no build step)
4. LM Studio backend + runtime model dropdown ✅
5. **LangChain only — no LangGraph** (hard constraint, see §3) ✅
6. Implementation finished & signalled ✅
7. Real-model testing executed ✅ (record in DESIGN.md §12–§13)

The project originates from a Spanish TFM (final-year thesis); conversation with the user is in
English.

## 2. Current state — as of this handoff

- **Complete and tested.** Two full real-model e2e runs plus targeted re-verifications of every
  defect found; all defects fixed at root cause with regression tests.
- Test suite: **58 tests, green** (`pytest -q`, ~2 min). One test is flaky under system load —
  see §7.3.
- Git repo initialized; baseline commit `04f2b46`. Local identity set to `junmi <junmi@localhost>`
  (user may want to change it: `git config user.name / user.email`).
- A clean, real deliverable exists from the last full run:
  `outputs/ba-20261009-154450-6c7194/` — 46 KB specification (12 sections + Appendix A), use cases,
  glossary, Mermaid diagram, interview audit trail (`interview.jsonl`), self-contained HTML bundle.
  Note: `outputs/`, `state/`, `logs/` are git-ignored by design.

## 3. Hard constraints (do not violate)

| Constraint | Why |
|---|---|
| **No LangGraph.** Pure `langchain-core` + `langchain-openai`. | User requirement #5. Control flow is owned by a deterministic Python state machine (`workflow.py`), deliberately — LLMs do not own the pipeline. |
| LM Studio as the only model backend; runtime model dropdown in UI | User requirement #4. Endpoint: OpenAI-compatible `http://127.0.0.1:1234/v1`, admin API `/api/v0` for the dropdown (load state, context size, tool support). |
| Web UI required | User requirement #3. Vanilla JS SPA in `src/ba_agent/web/` — **no build step**, no node_modules; vendored libs in `web/vendor/`. |
| Budgets on every LLM-driven loop | No run may stall: max questions per subject, gap rounds, summary rejections (2), review rounds (2), diagram revisions (1), total question ceiling (40). All in `config.yaml` → `interview:` / `review:` / `diagram:`. |
| Fail-fast without a model, but **resumable** | If no LM Studio model is loaded the app shows `NO_MODEL_MESSAGE` and the session can be resumed later (`--resume SESSION_ID`, or resume from UI). Never crash mid-run on missing model. |

## 4. Architecture in one page

```
stakeholder ⇄ Web UI (SSE) ⇄ FastAPI (server/app.py) ⇄ Workflow state machine (workflow.py)
                                                        │  per-phase role functions (roles/*)
                                                        ▼
                                          ModelGateway (models.py) → LM Studio /v1
```

**Phases** (`Phase` enum in `workflow.py`, driven by `_step_*` methods):

```
INTAKE → INTERVIEW → CLARIFY (gap rounds, ≤2) → SUMMARY → APPROVAL (user gate; ≤2 rejections)
       → DRAFT (narrative sections + use cases + glossary) → REVIEW (critic ⇄ user loop, ≤2 rounds)
       → DIAGRAM (Mermaid + pure-Python validation) → DELIVER → DONE
```

- The state machine is **deterministic and owns all control flow**. LLM calls are leaf functions.
  `state.awaiting` (`ANSWER` / `APPROVAL`) pauses the loop for human input; everything else runs
  unattended once a session starts.
- **State**: one SQLite DB at `state/ba_agent.db` (sessions, events, QA rows, knowledge base).
  `SessionState` is a Pydantic model in `schemas.py`; it serializes whole to JSON per save — that
  is what makes resume trivial.
- **Events**: every step emits typed events (`events.py`) → SSE stream → UI timeline.
  `step_start`/`step_end` are balanced; warnings surface budget exhaustion instead of hiding it.

### File map (what lives where)

| Path | Responsibility |
|---|---|
| `src/ba_agent/workflow.py` | **The** phase state machine. Phase transitions, budgets, awaiting logic. Start here to understand flow. |
| `src/ba_agent/models.py` | `ModelGateway`: LM Studio client, structured-output ladder (§5), thinking-model retry guards, streaming (`stream_text`) and `structured()`. |
| `src/ba_agent/schemas.py` | Pydantic models: `SessionState`, role outputs (plan, questions, triage, summary, critic issues…), `clean_markdown()` helper. |
| `src/ba_agent/roles/interview.py` | Agenda planning, question generation, answer triage, gap detection. |
| `src/ba_agent/roles/writing.py` | `write_section`, `write_use_cases`, `write_glossary` — free-text (streamed) writers. |
| `src/ba_agent/roles/review.py` | Critic pass + code linter that runs **before** the LLM critic; review⇄user loop. |
| `src/ba_agent/roles/diagram.py` | Mermaid generation + reviewer role. |
| `src/ba_agent/roles/analysis.py` | Summary assembly / requirements ledger rendering helpers. |
| `src/ba_agent/spec.py` | `SPEC_TEMPLATE` (the 12-section FSD outline) and `assemble_spec()` — deterministic assembly; section numbering is added here, never by the model. |
| `src/ba_agent/export.py` | Deliverable writers: spec MD, requirements summary, interview JSONL, `.mmd`, self-contained HTML bundle, optional PDF (weasyprint if installed). **Note:** it writes `state.spec_markdown or assemble_spec(state)` — a cached assembly wins; clear the cache if you mutate sections after delivery. |
| `src/ba_agent/store.py` | SQLite persistence (sessions/events/QA/knowledge), sha256 helpers. |
| `src/ba_agent/services.py` | Composition root: builds gateway, store, knowledge, config into a `Services` object. |
| `src/ba_agent/config.py` | Config loading (`config.yaml` + env overrides) and `_default_roles()`. **Keep yaml and defaults in sync** — role caps live in both. |
| `src/ba_agent/jsonx.py` | Tolerant JSON extraction (fences, prose wrappers, trailing commas, Python literals). Leaf module. |
| `src/ba_agent/knowledge.py` | Optional RAG: chunking + embeddings via LM Studio embedding model + BM25 hybrid; docs dropped in `data/`. |
| `src/ba_agent/mermaid_utils.py` | Pure-Python Mermaid validation (kroki external validator off by default — privacy). |
| `src/ba_agent/server/app.py` | FastAPI app: REST + SSE endpoints, serves the SPA. |
| `src/ba_agent/tools/chat_tools.py`, `tools/sandbox.py` | Chat-agent tools (post-delivery Q&A about the spec) with a sandboxed execution helper. |
| `src/ba_agent/cli.py` | CLI entry (`ba-agent …`): `serve`, `doctor`, etc. Calls `configure_console()` for UTF-8 on Windows — keep it at every entry point, including scripts. |
| `prompts/*.md.j2` | **All** LLM prompts as Jinja2 templates (13 files). Edit here, not in code. |
| `src/ba_agent/web/` | SPA: `index.html`, `app.js`, `styles.css`, `vendor/`. No build step — edit directly. |
| `scripts/e2e_lmstudio.py` | Real-model e2e driver (the testing workhorse, §7). |
| `scripts/demo_offline.py` | Full UI walkthrough with a scripted gateway — no model needed. |
| `tests/` | `test_units.py`, `test_pipeline.py` (scripted-gateway pipeline), `test_api.py` (HTTP+SSE over real uvicorn socket), shared fakes in `conftest.py`. |

## 5. Invariants that must survive refactors

These exist because a **real model** broke them at some point. Each has regression tests:

1. **Structured-output ladder** (`models.py`): try `json_schema` → `function_calling` → prompt-based
   (with `jsonx.extract_json`). In every real run so far, `json_schema` succeeded first try; the
   lower rungs are a safety net for weaker models. Don't remove them.
2. **Thinking-model retry guard**: thinking models (e.g. Qwen3 via LM Studio) spend thousands of
   tokens on internal reasoning that is *not exposed* in the stream (`reasoning_content` empty;
   `/no_think` and `chat_template_kwargs.enable_thinking=false` are ignored by this model/server).
   A small cap → completion ends with zero visible output, **no error raised**. Guards: writer /
   use_cases `max_output_tokens: 16384` (in `config.yaml` **and** `_default_roles()` in config.py —
   keep both) plus a one-time retry **without a cap** when a capped call returns empty visible
   output, in both `stream_text` (via `_stream_once(token_cap)`) and `structured()`.
3. **Prompt/code response contracts**: every prompt that says "Return JSON …" must be consumed via
   `gateway.structured()` with a matching Pydantic schema; every free-text prompt must say
   *Markdown only* and be consumed as text. The section prompt previously asked for JSON while
   `write_section` expected Markdown — the model followed its instructions and shipped raw JSON in
   the spec (defect #3, §8). **When you change a prompt's output contract, check its consumer.**
4. **`clean_markdown()` defensive unwrap** (`schemas.py`): strips code fences / "Here is …"
   prefixes, and unwraps a whole-reply JSON envelope carrying a `markdown`/`body` string field —
   whole-reply match only, so prose that merely quotes a JSON sample is untouched. Used by all
   free-text writers (sections, use cases, glossary).
5. **Deterministic assembly**: the model never writes section numbers/titles; `assemble_spec()`
   does. A writer that ignores "no heading" gets its stray title stripped in `write_section`.
6. **Linter before critic** (`roles/review.py`): cheap deterministic checks (missing requirement
   ids, TBD density, structure) run before spending LLM tokens on the critic.
7. **Budgets are surfaced, not hidden**: when a budget exhausts, emit a WARNING event and ship what
   you have (open questions in §10 of the spec). Never silently loop or truncate without telling.

## 6. How to run everything

Environment: Windows, Python venv at `.venv`. Always invoke via `.venv/Scripts/python.exe`
(Git Bash paths work fine; see §7 for quirks).

```bash
# install (already done in .venv)
.venv/Scripts/python.exe -m pip install -e .

# unit + pipeline + api tests (~2 min)
.venv/Scripts/python.exe -m pytest -q

# run the app → http://127.0.0.1:8765  (port from config.yaml server.port)
.venv/Scripts/python.exe -m ba_agent serve        # or `ba-agent serve` if on PATH
.venv/Scripts/python.exe -m ba_agent doctor       # checks LM Studio reachability, model loaded

# full real-model e2e (~3.5 h on qwen3.8-27b — run in background, poll sparsely)
nohup .venv/Scripts/python.exe scripts/e2e_lmstudio.py --deadline-minutes 480 > logs/e2e_run.log 2>&1 &

# e2e flags: --model NAME (else first loaded chat model), --resume SESSION_ID,
#            --request-changes-first (exercise the approval-gate rejection path)

# offline demo — full UI with scripted answers, no model needed
.venv/Scripts/python.exe scripts/demo_offline.py
```

**Testing rule for this project:** any change to prompts, output handling, or the gateway must be
verified against a **real LM Studio model**, not just the scripted gateway. The scripted fakes have
already hidden two real defects (§8). Real-model runs are slow on this machine — that is expected,
not a bug (see §7.2 for pacing).

## 7. Environment quirks (this machine)

1. **Git Bash has no `/tmp`** (`bash.exe: warning: could not find /tmp`). Use the project's
   `logs/` or your scratchpad for temp files. Prefer writing a script file then running it over
   heredoc-piping into python (one heredoc produced empty output).
2. **LM Studio + Bionic cannot both hold models** — same server, VRAM contention. The user tests
   with the model *already loaded* in LM Studio while another agent runs; currently that is
   `qwen3.8-27b`. Do not attempt to switch/load models during a run unless asked. Pacing on this
   model: ~4–9 min per interview step, ~5–16 min per narrative section (invisible reasoning),
   **full e2e ≈ 3.5 h**. Agent conversation turns also queue on the same backend — keep polling
   sparse during runs. The embedding model for knowledge indexing is
   `text-embedding-nomic-embed-text-v1.5` (must be downloaded in LM Studio).
3. **Flaky test under load**: `tests/test_api.py::test_interview_over_http_reaches_done_with_downloadable_artifacts`
   occasionally fails with an httpx `ReadError` when the machine is busy (it passed on isolated
   rerun and for the whole file). If it's the only failure, rerun in isolation before investigating.
4. **Windows console encoding**: legacy cp1252 consoles crash on UTF-8 output (`→`, em-dashes).
   `configure_console()` in `cli.py` fixes stdout/stderr at entry — call it from any new script/entry point.
5. LM Studio's `api_key: "lm-studio"` in config.yaml is a placeholder the server ignores; keep it (client requires a value).

## 8. Defect history (all fixed — read before touching these areas)

| # | Defect | How found | Fix |
|---|---|---|---|
| 1 | CLI crashed with `UnicodeEncodeError` on Windows consoles (`ba-agent doctor`) | First manual run | `configure_console()` UTF-8 reconfig at every entry point |
| 2 | Writer sections came back **empty** on thinking models — model spent its whole token cap on invisible reasoning; completion ended silently with zero visible content | Real e2e run #1/2 (all narrative sections were placeholders) | writer/use_cases caps → 16384 + one-time uncapped retry guard in `stream_text` and `structured()` (§5.2). Verified: surgical re-run of all 8 failed calls, PASS 8/8 |
| 3 | Section prompt asked for JSON but `write_section` expected raw Markdown — model followed instructions; spec shipped with sections wrapped in `{"section_id": …, "markdown": …}` | Real e2e run #3 (content was good, format wrong) | Prompt now asks Markdown only; `clean_markdown()` defensively unwraps whole-reply JSON envelopes (§5.4). Run-3 deliverables repaired deterministically (unwrapped stored bodies + re-exported, no model calls) and re-verified with fresh model calls: PASS 8/8, zero envelopes |

**Lesson:** the scripted gateway returns exactly what tests expect; only a real model reveals
prompt/code contract drift. Full run record: DESIGN.md §13 (sessions `ba-20261007-110709-3fd951`
= runs 1/2, `ba-20261009-154450-6c7194` = run 3 with the clean deliverable).

## 9. Known gaps / candidate next work (none are blockers)

- `--request-changes-first` e2e path (approval-gate rejection → notes flow into drafting) is
  covered by scripted tests only — never exercised against a real model. Cheap to add: one more
  background run with that flag.
- External Mermaid validation (`diagram.validator_url`, e.g. kroki.io) is off by default for
  privacy; pure-Python validation is the active path.
- Real PDF export needs `weasyprint` (optional dep, not installed); HTML bundle prints to PDF fine.
- Knowledge/RAG path exists but real runs so far used no indexed documents — worth one e2e with a
  doc in `data/` if the user cares about that feature.
- Post-delivery chat agent (`tools/chat_tools.py`) is implemented and unit-tested; light manual
  exercise would be nice.

## 10. Working conventions for this codebase

- **Prompts live in `prompts/*.md.j2`** — never inline new prompt text in Python. Render via
  `prompts.render(name, **ctx)`.
- **Config precedence**: env vars > `config.yaml` > defaults in `config.py`. When you change a role
  default, update `_default_roles()` *and* `config.yaml` (they are duplicated on purpose so the yaml
  stays self-documenting).
- **New LLM call?** Give it: a role name (for per-role model/temperature/cap overrides), an emit
  step id (`"role:what"`), and a budget. Structured → `gateway.structured(schema=…)`; free text →
  `gateway.stream_text(…)`.
- **New phase or state field?** It must survive the JSON round-trip in `SessionState` (resume
  depends on it) and be reflected in `_session_summary()` if the UI should see it.
- **Tests**: unit-level in `test_units.py`; pipeline behavior with scripted gateway in
  `test_pipeline.py`; HTTP/SSE surface in `test_api.py`. Fakes live in `conftest.py` — extend them,
  don't duplicate. Remember §6's rule: prompt/gateway changes also need a real-model check.
- **Web UI**: plain JS/CSS, no framework, no build step. SSE for events; the model dropdown reads
  LM Studio's admin API through the backend (don't call :1234 from browser — CORS).
- Git: commit small and atomic; branch prefix `bionic/` if this agent creates branches (other
  agents may use their own convention — ask the user).

## 11. User preferences

- English conversation. Wants **clear PASS/FAIL status and final deliverables**, not process
  narration ("was the test completed?" is a recurring check-in).
- Long real-model runs are acceptable if signalled up front with an ETA; keep polling sparse while
  one runs (shared backend, §7.2).
- The old AutoGen project at `D:\Code\Ba_gpt` is reference-only — never modify it.

## 12. Quick orientation checklist for a new agent

1. Read this file, then [docs/DESIGN.md](docs/DESIGN.md) (architecture + §12 defect table + §13 real-run record).
2. Skim `workflow.py` (phases/budgets) and `models.py` (gateway/ladder/guards).
3. Run the suite: `.venv/Scripts/python.exe -m pytest -q` → expect 58 passed (~2 min).
4. If you touch prompts/output handling: plan a real-model verification (§6, §7.2) and tell the user the ETA before starting it.
5. Commit atomically; keep `config.yaml` ↔ `_default_roles()` in sync; keep prompt contracts matched to their consumers (§5.3).
