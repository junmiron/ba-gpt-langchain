"""Failure paths of the engine: what happens when a step blows up or a phase has no handler.

A workflow that only handles the happy path loses sessions silently. These tests drive the error
branches directly and assert the state the UI and the Resume button will see afterwards.
"""

from __future__ import annotations

import asyncio

import pytest

from ba_agent.schemas import AwaitKind, Phase, RunStatus, SessionState
from ba_agent.workflow import WorkflowEngine, WorkflowHalt


def _run(engine: WorkflowEngine, state: SessionState) -> list[dict]:
    events: list[dict] = []

    async def emit(event_type, payload=None):
        events.append(
            {"type": str(event_type), "phase": state.phase.value, "message": (payload or {}).get("message", "")}
        )

    asyncio.run(engine.advance(state, emit=emit))
    return events


def test_a_phase_with_no_handler_fails_loudly_instead_of_stalling(services):
    engine = WorkflowEngine(services)
    state = SessionState(session_id="wf-no-handler", title="Claims Portal")
    #: `plan` is a legacy phase name still in the enum (and in old saved states) with no handler.
    #: A run that lands there must say so rather than sit in RUNNING forever.
    state.phase = Phase.PLAN
    state.awaiting = AwaitKind.NONE
    state.status = RunStatus.RUNNING

    events = _run(engine, state)
    assert state.status is RunStatus.ERROR, f"no handler for {state.phase.value}: {state.error}"
    assert "No handler" in (state.error or ""), state.error
    assert any(e["type"] == "error" for e in events), "the error must reach the event stream, not just the DB"
    assert state.phase is Phase.PLAN, "an unhandled phase must not be rewritten into a real one"


def test_a_crashing_step_is_reported_and_stops_the_loop(services):
    engine = WorkflowEngine(services)
    state = SessionState(session_id="wf-crash", title="Claims Portal")

    async def boom(session, emit):
        raise RuntimeError("plan generator exploded")

    engine._step_plan = boom  # inject a failure the scripted gateway never produces

    events = _run(engine, state)
    assert state.status is RunStatus.ERROR, f"a failed step must not leave the run looking alive: {state.status}"
    assert "RuntimeError" in (state.error or "") and "plan generator exploded" in (state.error or ""), state.error
    assert any(e["type"] == "error" and e["phase"] == "intake" for e in events), events
    assert state.phase is Phase.INTAKE, "a failed step must not advance the phase"


def test_draft_before_summary_is_a_halt(services):
    engine = WorkflowEngine(services)
    state = SessionState(session_id="wf-halt", title="Claims Portal")
    state.approved_summary = False
    state.summary = None

    async def emit(event_type, payload=None):
        del event_type, payload

    with pytest.raises(WorkflowHalt, match="before the requirements summary"):
        asyncio.run(engine._step_draft(state, emit))


def test_approving_when_there_is_nothing_to_approve_is_a_clear_error(services):
    engine = WorkflowEngine(services)
    state = SessionState(session_id="wf-early-approval", title="Claims Portal")
    state.phase = Phase.INTERVIEW

    async def emit(event_type, payload=None):
        del event_type, payload

    with pytest.raises(ValueError, match="Nothing to approve"):
        asyncio.run(engine.set_approval(state, True, emit=emit))


def test_answering_an_unknown_question_is_a_clear_error(services):
    engine = WorkflowEngine(services)
    state = SessionState(session_id="wf-bad-qid", title="Claims Portal")

    async def emit(event_type, payload=None):
        del event_type, payload

    with pytest.raises(ValueError, match="Unknown question id 99"):
        asyncio.run(engine.submit_answer(state, 99, answer="yes", emit=emit))
