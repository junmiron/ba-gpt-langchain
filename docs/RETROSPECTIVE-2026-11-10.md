# Sprint retrospective — 2026-11-02 → 2026-11-10

Repo: `Ba_langchain` (`ba_agent`), published at `github.com/junmiron/ba-gpt-langchain`.
Scope: Phases 0–3 of `docs/REDESIGN_PLAN.md`, the Phase 5 release pass, and the 2026-11-10 usability pass triggered by user testing.

## Numbers

| Measure | Baseline `04f2b46` (Nov 2) | Now `04b2036` (Nov 10) |
|---|---|---|
| Test files | 4 | 59 |
| Test functions | 40 | 446 |
| Collected tests passing offline | 58 | 590 (582 + 8 environment-guarded skips in CI) |
| Branch coverage | — | 94% (6,058 statements, 366 missing) |
| Config keys read by code | — | 57 declared, 0 dead, 0 drift |
| Source files type-checked | 38 | 39 |
| Commits this sprint | — | 10 |
| Files changed this sprint | — | 120 (+20,169 / −1,570) |

Behaviour deltas measured on real Strata runs: full pipeline went from a 20-minute deadline **FAIL** to **PASS** in 8 min 16 s; the deliver phase dropped from 939.1 s to 112.2 s wall time; the worst-case structured-call ladder exposure went from 900 s (300 s × 3 attempts) to 360 s (120 s × 3 rungs, one attempt each). Evidence: `scratch_e2e/strata_smoke2.txt` ("FAIL: deadline hit before the run finished") versus `scratch_e2e/strata_smoke_patched.txt` ("RESULT: PASS", `run_summary` deliver 1m 52s).

## What worked

1. **Every bug class became a permanent gate.** The duplicate-question guard, the status invariant (`WAITING_USER` iff `awaiting ∈ {ANSWER, APPROVAL}`), the SSE listener parity test, the no-LangGraph architecture test, the settings-UI parity test, the config dead-key audit. Each one is cheap and stops the same defect recurring.
2. **Deterministic offline testing.** `FakeGateway` stubs in `tests/conftest.py` plus `addopts = "-m 'not live'"` make the whole suite run with no model, no GPU, no network — CI in 1 m 41 s on two Python versions.
3. **Reproducing the user's report before fixing it.** The "no question appears" bug looked like a backend failure; `curl` on the session snapshot showed a real `pending_question`, so the fault was client-side. Server first, browser last is the right order.
4. **Instrumenting instead of guessing on latency.** Per-call telemetry (role, step, model, method, attempts, elapsed, prompt/completion chars) replaced speculation about why the interviewer felt slow.
5. **Design approval before implementation.** The usability pass was analysed, designed, and approved by the user before any code was written; the scope shipped matched the approved scope exactly.
6. **Thematic commits at the close.** 116 working-tree entries were grouped into 10 subsystem commits (build, gateway, server, artifacts, interview, settings, web, docs) in dependency order, then verified with an empty `git status`.

## What did not work

1. **A day of uncommitted work.** 109 files / 6,100 insertions sat in the working tree across sessions with no remote and no CI. Nothing was reviewable, nothing was revertible, and the release pass could not even be started.
2. **First push failed CI.** `scripts/e2e_offline.py` asserted `functional_specification.typ` in the bundle unconditionally. CI has no Typst binary, so the ReportLab fallback legitimately produced no `.typ`. The gate encoded an environment assumption that was never tested under CI conditions; fixing it meant hiding `vendor/typst` locally to reproduce the runner.
3. **An unprovable performance claim.** The AutoGen-vs-LangChain interviewer comparison rested on two unmatched probes (different briefs, different answers, different call counts). I reported the delta as a finding, then had to walk it back in the docs. Local quantised model variance makes single-turn probes worthless as benchmarks.
4. **Doc drift ran ahead of the code.** Suite counts of 490/475/464 stayed in README/CHANGELOG/handoff long after the suite reached 590; `handoff.md` claimed `uv.lock` was untracked while it was committed; `REDESIGN_PLAN.md` said "0 dead of 55" while the audit reported 57.
5. **Tool friction cost real turns.** Repeated `patch` failures on CRLF files, `node --check` reading stdin inside a chained command and printing `stdin is not a tty` instead of a verdict, and a half-installed compiled mypy wheel. None are project bugs; all were avoidable by running checks standalone and inspecting line endings first.
6. **Side effects were left behind.** Diagnostic probes created six sessions in `state/ba_agent.db`, several background `serve` processes kept firing gateway notifications after their turn ended, and scratch scripts stayed on disk. Clean-up was not part of the loop until it became a task in itself.
7. **Known gaps still open.** No `pytest-timeout` in CI, so a hung test has no per-test guard. Coverage sits at 94% with the largest holes in `workflow.py`, `knowledge.py`, and `artifacts/qa.py`. Strata has no embeddings endpoint, so vector retrieval is BM25-only there and the UI should say so per provider. LM Studio has never been exercised end to end. Optional dependencies (`langchain-text-splitters`, `reportlab`, `svglib`, `weasyprint`) are still awaiting approval.

## Lessons learned

1. **A gate is only as good as the environment it runs in.** Before writing an acceptance assertion, enumerate what the runner does *not* have (no Typst, no mermaid-cli, no model server) and make the expectation conditional on the same probe the code uses (`find_typst()`).
2. **Never state a speedup without matched measurements.** Same input, same model, same run, per-call telemetry. Otherwise the honest sentence is "the architecture removed one round trip", not "it is faster".
3. **Commit by subsystem in dependency order, then verify with an empty `git status`.** Any leftover line is a group you forgot; an uncommitted file means the pushed repo is not the code you tested.
4. **Numeric claims in docs are claims about the code.** When a count changes, grep every doc for the old number in the same change that produced the new one.
5. **Order of investigation for "the UI is silent": API snapshot → raw stream → browser.** "The stream is open" is not evidence the client read it; `EventSource` needs a listener per named event.
6. **Run verification commands standalone.** Chained commands can feed stdin to a check and make its output meaningless. Read the exit code, not the echo.
7. **Clean up the fixtures you create.** Delete probe sessions, kill background servers, restore renamed directories. Untracked state in a live database is a user-visible defect, not housekeeping noise.
8. **Separate environment repair from project work.** A broken mypy wheel is not a code bug; say so, fix it, and keep the project gate honest.
9. **Instrumented degradation beats silent failure.** A timed-out rung that reports "the backend is unresponsive" is far more useful than three retries that burn 15 minutes on one role.

## Next sprint

1. Tag `v1.0.0` on the pushed `main` and cut the GitHub release.
2. Run one real LM Studio end-to-end pass to confirm the rung ladder on a second backend.
3. Make the retrieval panel state provider capability explicitly (BM25-only under Strata).
4. Close the coverage holes in `workflow.py`, `knowledge.py`, `artifacts/qa.py`.
5. Decide on `pytest-timeout` so CI has a per-test hang guard.
6. Re-run a controlled interview-latency comparison: identical brief, identical answers, 10 turns, both engines.
