# BA Agent — Business Analyst interview agent (LangChain + LM Studio)

A local-first requirements elicitation agent. You describe a project; it interviews you subject by subject,
builds a traceable requirements ledger, gets your approval on a summary, then writes a **Functional
Specification Document**, use cases, a glossary and Mermaid process diagrams — with a web UI you can close and
reopen at any point.

This is the LangChain rewrite of a year-old AutoGen system (`Ba_gpt`). The review of that codebase is in
[docs/ANALYSIS.md](docs/ANALYSIS.md) and the reasoning behind this design is in [docs/DESIGN.md](docs/DESIGN.md).

**LangChain only — no LangGraph.** `langgraph` is not a dependency; orchestration is a plain async state
machine, and the interactive chat channel is a hand-written `bind_tools` loop.

---

## Quick start

Requirements: Python 3.11+, and [LM Studio](https://lmstudio.ai) running with the REST API enabled
(Settings → Developer → *Serve REST API*, default port `1234`). At least one chat model downloaded.

```bash
cd D:\Code\Ba_langchain

# 1. create the environment (uv is fastest; pip works too)
uv venv .venv --python 3.13
.venv\Scripts\activate
uv pip install -e ".[dev]"        # or: pip install -e ".[dev]"

# 2. check that LM Studio is reachable and see which models the UI will offer
python -m ba_agent doctor

# 3. start the app
python -m ba_agent serve          # → http://127.0.0.1:8765
```

Then press **New interview**, paste a project brief, and answer the questions. Pick a model any time from the
top-right dropdown; **⋯** opens per-role overrides (e.g. a fast model for the interview, a bigger one for
drafting).

Optional: drop reference documents (PDF/DOCX/MD/TXT) into `data/` and press **Index data/** in the sidebar, or
upload them — retrieval is hybrid (vector + BM25) when an embedding model is available in LM Studio.

---

## What it produces

Everything lands in `outputs/<session-id>/`:

| File | What it is |
|---|---|
| `functional_specification.md` | The FSD: 10 sections, assembled deterministically from the ledger |
| `process_diagram.mmd` (+ `.svg` if a renderer is enabled) | Mermaid process flow(s), syntax-validated |
| `interview.jsonl` | Every question/answer pair — the audit trail |
| `summary.md` | The approved requirements summary |
| `deliverable.html` | Self-contained, styled, print-to-PDF bundle (diagrams rendered inline) |
| `metadata.json` | Run stats: model calls, wall/model seconds, budgets consumed, open issues |

---

## How it works

```
Browser SPA ──REST──▶ FastAPI ──▶ WorkflowEngine (deterministic phase machine)
    ▲                    │              │
    └────SSE (replayable)┘        roles: plan · interview · triage · summary
                                  writer · use_cases · glossary · critic
                                  diagrammer · mermaid_reviewer
                                       │
                                  ModelGateway ──▶ LM Studio (:1234/v1)
```

The five ideas worth knowing, inherited from what went wrong in v1:

1. **Control flow lives in code.** Phases (`intake → interview ⇄ clarify → summary → approval → draft → review
   ⇄ fix → diagram → deliver`) are transitions in `workflow.py`. The model is never asked "who speaks next", and
   there are no magic tokens (`READY`, `QUESTION:`, `NEEDS_INFO`) to regex-match.
2. **Structured output everywhere.** Each step declares a Pydantic schema; the gateway walks a ladder —
   `json_schema` → `function_calling` → prompt + repair — so the same code runs on a 30B and on a 4B that only
   produces prose.
3. **Durable state.** Phase, coverage, ledger, artefacts and the event log are in SQLite (`state/ba_agent.db`).
   Restart the server or close the tab mid-interview: press **Resume** and continue exactly where you stopped.
4. **The transcript is not the memory.** Each step gets a small purpose-built context (ledger digest + only the
   knowledge it retrieved), not the whole conversation — cheap enough for 8k-context local models.
5. **Bounded everything.** Questions per subject, gap rounds, review rounds, approval rejections, total
   questions, model timeouts, and one global concurrency slot so two requests never fight for VRAM. A stuck run
   terminates with a reason instead of running forever.

Deterministic checks also *precede* the LLM: spec completeness is linted in code (sections present, requirement
counts, acceptance criteria, traceability), Mermaid syntax is validated by a Python parser before any reviewer
sees it, and the title page / open questions / traceability appendix are rendered from the ledger rather than
generated.

---

## Configuration

`config.yaml` holds everything; `.env` (see [.env.example](.env.example)) or real environment variables override
it. Model choice can also be changed at runtime in the UI and is persisted.

| Setting | Default | Notes |
|---|---|---|
| `lmstudio.base_url` | `http://127.0.0.1:1234/v1` | OpenAI-compatible endpoint (`LMSTUDIO_BASE_URL`) |
| `lmstudio.admin_url` | `…/api/v0` | Powers the dropdown: load state, tool support, context length |
| `lmstudio.default_model` | `null` → first **loaded** chat model | Set in the UI dropdown instead (`BA_MODEL`) |
| `lmstudio.embedding_model` | `text-embedding-nomic-embed-text-v1.5` | Empty ⇒ retrieval falls back to BM25 only |
| `lmstudio.max_concurrency` | `1` | Raise only if you have VRAM to spare |
| `lmstudio.structured_output_method` | `auto` | Force `json_schema` / `function_calling` / `prompt` for a stubborn model |
| `interview.max_questions_per_subject` | `2` | Initial question + one follow-up (`BA_MAX_QUESTIONS_PER_SUBJECT`) |
| `interview.max_gap_rounds` | `2` | Clarification rounds after the agenda (`BA_MAX_GAP_ROUNDS`) |
| `interview.max_summary_rejections` | `2` | "Request changes" at the approval gate, then notes go to drafting |
| `interview.max_total_questions` | `40` | Absolute ceiling so a run cannot stall |
| `review.min_functional_requirements` | `5` | Linter gate; below it the agent asks for more capabilities |
| `diagram.validator_url` | `null` | Optional kroki renderer. **Off by default** — it would send your spec to a third party |

Prompts are plain Jinja2 files in [`prompts/`](prompts) (`system` and `user` halves separated by a literal
`---USER---` line). Edit them and the next step picks the change up — no restart.

---

## CLI

```bash
python -m ba_agent serve    [--host 127.0.0.1] [--port 8765]   # web UI + API
python -m ba_agent doctor   [--json]                           # connectivity, models, embeddings, KB
python -m ba_agent index    [--dir data] [--url https://…]     # build the knowledge base
python -m ba_agent sessions [--limit 20]                       # stored sessions and their phase
python -m ba_agent export   <session-id>                       # re-assemble deliverables (no model calls)
```

## HTTP API

| | |
|---|---|
| `GET /api/health` · `GET/POST /api/models` | status, catalog, default + per-role models |
| `GET/POST /api/sessions` · `GET/DELETE /api/sessions/{id}` | session lifecycle |
| `POST /api/sessions/{id}/answer` · `/approve` · `/resume` · `/cancel` · `/chat` | commands (return immediately; progress arrives over SSE) |
| `GET /api/sessions/{id}/stream` | SSE, honours `Last-Event-ID` for lossless reconnects |
| `GET /api/sessions/{id}/events?after=N` | persisted event replay (the UI rebuilds the timeline from this) |
| `POST /api/knowledge/upload` · `/index` · `/search` · `GET /api/knowledge/status` | knowledge base |
| `GET /api/artifacts/{id}/{name}` | download a deliverable |

---

## Tests

```bash
python -m pytest -q          # 52 tests, ~6 s — no model, no VRAM, no network
```

The suite drives the real state machine and the real HTTP/SSE surface against a scripted gateway
(`tests/conftest.py`), so budgets, escalation paths, resumability and artefact writing are all covered without
loading anything. `docs/DESIGN.md` explains what each layer guarantees.

---

## Troubleshooting

**"No LM Studio chat model available."** — Nothing is loaded in LM Studio. Load a model (or pick one in the
dropdown) and press **Resume**; the run continues from the phase it stopped in, nothing is lost.

**Repeated "structured output failed … retrying"** — normal for small quantised models: the gateway is walking
down its ladder. If every rung fails the step degrades visibly (a warning card) and the run continues with a
placeholder or the deterministic fallback instead of dying. Pinning `structured_output_method: prompt` often
helps models that mangle tool-call schemas.

**Vector search off / "BM25 only"** — no embedding model in LM Studio, or `/v1/embeddings` unsupported. Keyword
retrieval still works; download an embedding model and re-index to enable vectors.

**A step seems slow** — switching models unloads/reloads weights, which can take minutes on a big GGUF. The UI
stays responsive (commands return immediately); `request_timeout_s` is deliberately 600 s. Only one request runs
at a time by design.

**Sections come back empty / "no visible output … retrying without a token cap"** — you are using a *thinking*
model (Qwen3 and similar): it spends thousands of tokens on internal reasoning before any visible text, which can
exhaust the role's `max_output_tokens` budget. The gateway retries once without a cap, so sections do get written —
but expect ~5 min per section. For faster runs, assign a non-thinking model to the writer/use_cases roles via ⋯ →
per-role overrides.

**Diagrams don't render as SVG files** — the browser renders Mermaid locally from vendored JS, and the `.mmd`
source ships in the bundle. Server-side SVG requires enabling `diagram.validator_url`.

---

## Privacy

Every model call goes to LM Studio on your machine. Nothing is sent anywhere else unless you explicitly set
`BA_KROKI_URL`, or add external URLs to the knowledge base (which fetches those pages).
