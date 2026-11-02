# Review — `Ba_gpt` (AutoGen Business Analyst agent)

Reviewed at `D:\Code\Ba_gpt`. Three generations live in that folder:

| Folder | What it is | State |
| --- | --- | --- |
| `TFM_BA_GPT/` | The original AutoGen 0.4 multi-agent system (Chainlit UI) — the one under review | `app.py` (config-driven) + `app2.py` (hand-wired duplicate) |
| `ba_gpt_next/` | A rewrite on Microsoft Agent Framework (MAF), CLI + AG-UI, Redis transcript archive | more mature prompts/spec logic than `TFM_BA_GPT` |
| `ba_lc/` | Empty scaffold (only a `.venv`) | abandoned first attempt at the LangChain port |

The product idea is strong: a BA that **interviews** a stakeholder across 8 subjects,
summarises, writes a Functional Specification Document (FSD), reviews it for completeness,
draws Mermaid process diagrams and exports a PDF deliverable. The findings below are about
the *engineering*, not the idea.

---

## 1. What actually runs today

```
Chainlit UI (app.py)
  └─ SelectorGroupChat  [planning, interviewer, summarizer, writer, checker,
                         diagram_creator, mermaid_reviewer, terminate] + user proxy
       ├─ selector_func()          ← hand-written Python router (lib/agents_tools_helpers.py:430)
       ├─ planning_agent           ← LLM asked to emit `[ASSIGNMENT] agent=... awaiting_user_reply=<bool>`
       ├─ magic tokens             ← QUESTION: / READY / NEEDS_INFO / TERMINATE / APPROVE SUMMARY
       ├─ ChromaDBVectorMemory     ← shared "memory" for 6 agents (rag_helpers.py)
       ├─ TextCanvasMemory         ← shared scratchpad for the diagram pair
       └─ MCP stdio servers        ← fs, mermaid render, pdf format (terminate_agent only)
```

Control flow is therefore **split across three places that can disagree**: a Python
`selector_func`, an LLM planner that must emit machine-readable markers, and regexes over
message text. That single decision is the root cause of most of the issues below.

---

## 2. Findings

### A. Control flow / orchestration (highest impact)

1. **The router is a no-op.** `selector_func` returns `PLANNING` in *every* branch except
   two (`[ASSIGNMENT]` marker honoured, and checker→writer for missing Use Cases/Glossary).
   See [agents_tools_helpers.py:430-542](../../Ba_gpt/TFM_BA_GPT/lib/agents_tools_helpers.py) — e.g.
   `return PLANNING if READY_TOKEN.match(last_txt) or True else PLANNING` (line 538) is
   literally always true. So the LLM planner is invoked **between every single agent turn**
   just to say "now run the interviewer". For a ~20-question interview that is roughly
   60–100 extra LLM calls of pure overhead, and it makes latency and cost dominated by
   bookkeeping rather than work.
2. **Control flow depends on exact strings from small local models.** `READY`, `QUESTION:`,
   `NEEDS_INFO`, `APPROVE SUMMARY`, `[ASSIGNMENT] agent=... awaiting_user_reply=true` are all
   regex-matched. The prompts spend ~40% of their tokens restating these protocols — a symptom,
   not a fix. Any paraphrase ("**ready**", "Ready ✅") silently desyncs the state machine.
3. **State lives in model output text.** `planning_agent` must keep an inline task list and an
   `awaiting_user_reply` flag; `interviewer_agent` must keep per-subject `asked=` counters and a
   `SESSION_ID`. Nothing is authoritative — the next turn re-reads its own prose. Counters drift,
   subjects get skipped or repeated, and there is no way to resume after a crash.
4. **Termination is effectively disabled**: `MaxMessageTermination(100000000)` (app.py:206). The only
   real stop is the model typing `TERMINATE`, so a stuck loop runs forever and burns VRAM/time.
5. **No per-phase budgets.** No cap on interview turns, review rounds or writer↔checker cycles;
   the checker→writer→checker loop has no iteration counter.

### B. Correctness & robustness bugs

6. `save_conversation_to_jsonl` buffers the interviewer question in a **function attribute**
   (`save_conversation_to_jsonl._pending_pair`, agents_tools_helpers.py:230) keyed by
   `(subject, q_index, session_id)`. Two sessions in parallel (or one retried turn) corrupt it, and
   answers whose key doesn't match return `"Error: No matching interviewer question stored"`.
7. `datetime.utcnow()` is deprecated (agents_tools_helpers.py:241/250) while the rest of the file uses
   `datetime.now(timezone.utc)` — mixed naive/aware timestamps in the same JSONL.
8. `_extract_last_question` and `_clone_agent_with_new_client` (app.py:110-190) reach into AutoGen
   **private** attributes (`_message_buffer`, `_system_messages`, `_tools`, `_model_client_stream`).
   Any library patch breaks them; the clone also silently drops `handoff_message`, `output_content_type`, etc.
9. `sys.modules["canvas"] = types.SimpleNamespace(...)` (app.py:57) is a hack so that the YAML string
   `canvas.update_file_tool` can be imported. It only works because of import order, and it breaks if two
   sessions start concurrently (global module overwritten).
10. Hard-coded Linux paths in a Windows project: `mcp_tools` kwargs point at
    `/run/media/jun/Jun_6T/Code/TFM_BA_GPT/` ([agents_config.yaml](../../Ba_gpt/TFM_BA_GPT/lib/agents_config.yaml), terminate_agent). The termination pipeline — i.e. the whole deliverable step — cannot run on this machine.
11. `model_loader.load_model_client("openai_npt")` is referenced by `app2.py:41` but **not present** in
    `configs` inside [src/model_loader.py](../../Ba_gpt/TFM_BA_GPT/src/model_loader.py) (it maps only 8 names, and `openai_npt` is missing while being listed in the dict lookup as a key of a file that does not exist). `app2.py` therefore raises at import time.
12. `FileLock.acquire` uses `O_CREAT|O_EXCL` with a 5 s timeout and no stale-lock recovery: if a process
    dies mid-write, every later write fails permanently until someone deletes `interview.jsonl.lock`.
13. RAG re-indexing is all-or-nothing: `rag_indexer_runner` records *file* paths in `index_state.json`, but on any new file it calls `load_and_index_documents(DOCS_DIR, ...)` which indexes **every** file again → duplicate chunks in Chroma (no content hashing, no doc IDs).
14. `lxml.html.clean.Cleaner(... allow_tags=[], host_whitelist=[])` — `host_whitelist` is not a Cleaner kwarg and `allow_tags=[]` with `remove_tags=[...]` is contradictory; the sanitiser config is mostly cargo-culted (rag_helpers.py:150-180).
15. Silent exception swallowing everywhere (`except Exception: pass`, tools returning `"Error: ..."` strings as *success* payloads). A tool that returns an error string is fed back to the model as if it were data, so the model often hallucinates success and still emits `READY`.

### C. Duplication & drift (maintenance)

16. **Two sources of truth for prompts**: `lib/agents_config.yaml` *and* ~700 lines of inline
    system messages in `app2.py`, already diverged (e.g. the checker's "INTERNAL FIX TRIGGER" exists only
    in YAML; `app2.py:530` starts with a typo `"YYou are the FUNCTIONAL_SPEC_CHECKER..."`).
17. **Two sources of truth for tools**: function tools in Python, MCP servers on disk, and agent-as-tool
    wiring done twice (YAML `agent_tools:` pass-2 rebuild + manual `AgentTool(...)` in `app2.py`).
18. `PLANNING/INTERVIEWER/USER` constants are defined **twice** in the same file (agents_tools_helpers.py:345 and :390).
19. The streaming/dedup logic is copy-pasted verbatim between `on_chat_start` and `on_message` (~70 lines each).
20. `.env` holds Azure/OpenAI keys, whisper/TTS config for an unrelated realtime experiment, and is committed to the repo folder (`.gitignore` does not list it — worth checking git history).

### D. Product / UX gaps

21. **Interactive replies are not wired.** `app.py:on_message` answers *"Thanks! I will enable interactive
    interview replies next."* whenever the workflow is already running; only `app2.py` handles them, and that file doesn't import (item 11). In practice the flagship feature — answering the interview — is broken in the runnable entry point.
22. No way to skip a question, answer "N/A", go back and edit an earlier answer, or see which subjects are
    covered. The user can only type free text into one box.
23. No live view of the artefacts being produced: the FSD is written to disk and only *linked* at the end;
    no spec preview, no diagram preview, no download while running.
24. Model choice is fixed per process (`load_model_client("azure")` + a hard-coded override list in app.py:170-190). Switching models means editing code and restarting.
25. No resume: closing the browser tab loses the interview (state only in `cl.user_session`).
26. No observability: no per-step timing, token or cost accounting; `mcp_debug.log` is a manual dump committed to the repo.

### E. What was genuinely good (keep it)

- The **domain pipeline** (interview → summary+approval → FSD → completeness check → diagram → review → export) is exactly right for BA work, and the 8-subject agenda with a per-subject question cap is a good elicitation design.
- Making `use_case_agent` / `glossary_agent` **sub-tools of the writer** (instead of peer speakers) was the correct instinct — it removes two agents from the routing problem.
- The checker distinguishing *user-dependent* gaps (`NEEDS_INFO`) from *internally fixable* ones (Use Cases/Glossary) is a smart escalation rule worth keeping as real code.
- Writing artefacts to disk with a lock, and archiving per-session into `experience/<SESSION_ID>/`, is the right shape for deliverables.
- Using MCP for the "heavy" side effects (fs ops, mermaid render, LaTeX export) keeps those out of the model loop — good boundary, just badly configured.

---

## 3. Scorecard

| Dimension | Now | Root cause |
| --- | --- | --- |
| Reliability of the flow | ✗ fragile | control flow delegated to LLM-emitted magic strings |
| Cost / latency | ✗ poor | planner call between every turn, no budgets |
| Resumability | ✗ none | state in prose + browser session |
| Concurrency (2+ sessions) | ✗ unsafe | globals: `sys.modules["canvas"]`, function-attribute buffers |
| Portability (Win/Linux) | ✗ broken | hard-coded absolute paths, xelatex/DejaVu assumptions |
| Maintainability | △ weak | duplicated prompts/tools/streaming, private-API coupling |
| UX | △ demo-grade | one text box, no artefact preview, replies unwired |
| Domain design (BA pipeline) | ✓ good | — |

**The rewrite should keep the BA pipeline and the escalation rules, and replace the orchestration
substrate entirely**: control flow in code, language in prompts, structured I/O between steps,
durable state per session. That is what [DESIGN.md](DESIGN.md) specifies.
