# Interview experience and Human lane

Status: implemented; full parity benchmark remains non-comparable.

## Behavior

- Suggestion chips inherit the foreground token and expose a visible keyboard-focus outline.
- Conversation cards retain their natural height. The timeline is focusable and scrollable; new cards and streamed text follow the bottom only when the reader is already near it. Session restore reads history in bounded pages until exhausted.
- `awaiting=approval` renders a persistent approval panel and hides answer/skip controls. The panel is rebuilt from the session snapshot, so reopening a session does not depend on seeing the original approval event.
- While a step is running, the composer shows an accessible phase label and elapsed timer. It clears when the workflow waits for user input, errors, or completes.
- The Human model lane defaults to **Real human**. Selecting a model simulates only stakeholder answers; each answer is labeled as simulated, the generated specification carries a simulation notice, and summary approval still requires a real person. If a simulator response is unusable, the workflow falls back to waiting for a person.
- Structured-call Trace entries record role, step, model, output method, attempts, duration, and prompt/completion character counts; prompt text is never logged. When the coverage response also supplies a next-subject question, the workflow uses it instead of making a separate question-generation call. Duplicate and budget guards still apply; missing/invalid questions use the existing generator fallback.

## Files

- `src/ba_agent/web/styles.css`, `src/ba_agent/web/index.html`, `src/ba_agent/web/app.js`
- `src/ba_agent/config.py`, `config.yaml`, `src/ba_agent/schemas.py`, `src/ba_agent/workflow.py`, `src/ba_agent/roles/interview.py`, `src/ba_agent/prompt_context.py`, `prompts/probe.md.j2`, `prompts/human.md.j2`
- `src/ba_agent/models.py`, `src/ba_agent/spec.py`
- `tests/test_web_ui_contract.py`, `tests/test_human_lane.py`, `tests/test_interview_latency.py`, `tests/test_structured_ladder.py`

## Acceptance evidence

- UI regression tests cover chip styling, card sizing/follow behavior, approval mode, elapsed indicator, complete event paging, and Trace timing fields.
- Real-browser check: chip contrast measured 13.79:1; approval controls visible while answer form hidden; timeline scrollable/focusable; cards retain readable heights and appending does not move a reader at scroll position 0; Human selector offers `Real human` and `q2_0`; elapsed indicator ticked and stopped.
- Offline tests prove simulated answers are persisted/tagged and approval remains a real-person gate; generated specification has the simulation notice.
- A real Strata answer advancing from Product Overview to Objectives & KPIs produced two structured calls (triage plus coverage/next question). A separate pre-change probe produced three calls on a different, shorter answer. These runs are not a controlled latency comparison; use the new per-call Trace data for matched follow-up runs.
- Last full gate: `590 passed, 1 deselected`; Ruff, formatting, mypy (39 files), JS syntax, config audit (57 fields, 0 dead, 0 drift), architecture checks, and offline acceptance passed.
