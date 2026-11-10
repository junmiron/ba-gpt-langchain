"""The interview loop's exit paths.

The engine enters interview with a subject list, a global question budget and a per-subject budget.
Each can run out, and every exit has to close the loop in a way the rest of the pipeline can read.
Left untested, an exhausted budget strands the run: subjects stay open, `next_subject` keeps returning
one, and the engine re-enters interview forever while the UI shows nothing happening.
"""

from __future__ import annotations

import pytest

from ba_agent.knowledge import KnowledgeService
from ba_agent.models import StructuredResult
from ba_agent.schemas import AwaitKind, Phase, QuestionState, SubjectState
from ba_agent.services import Services
from ba_agent.workflow import WorkflowEngine
from conftest import FakeGateway

BRIEF = "We need a portal for insurance brokers to submit and track claims. Today everything is email."


class _NoQuestionGateway(FakeGateway):
    """The model fails to produce a question for the subject: empty catalogue, refused call."""

    async def structured(self, schema, *, system_prompt="", user_prompt="", role="", step="", **kwargs):
        if role == "interview" and step.startswith("question"):
            return StructuredResult(
                parsed=None, raw_text="", method="failed", attempts=2, degraded=True, error="no question produced"
            )
        return await super().structured(
            schema, system_prompt=system_prompt, user_prompt=user_prompt, role=role, step=step, **kwargs
        )


@pytest.mark.asyncio
async def test_exhausting_the_question_budget_closes_every_open_subject(services):
    cfg = services.config
    cfg.interview.max_total_questions = 2

    engine = WorkflowEngine(services)
    events: list[tuple[str, dict]] = []

    async def emit(event_type, payload=None):
        events.append((event_type, payload or {}))

    state = services.store.create_session(brief=BRIEF)
    state.phase = Phase.INTERVIEW
    state.subjects = [
        SubjectState(name="Product Overview", priority="high"),
        SubjectState(name="Scope In and Out", priority="medium"),
    ]
    state.questions = [
        QuestionState(id=1, subject="Product Overview", text="What is the portal for?", answer="A claims portal."),
        QuestionState(id=2, subject="Scope In and Out", text="What is in scope?", answer="Intake only."),
    ]

    await engine.advance(state, emit=emit)

    warnings = [payload for event_type, payload in events if event_type == "warning"]
    assert any("budget" in str(payload.get("message", "")).lower() for payload in warnings), warnings
    assert all(subject.closed for subject in state.subjects), [(s.name, s.status) for s in state.subjects]
    assert state.next_subject() is None, "an open subject survived the budget and would be asked again"
    assert len(state.questions) <= cfg.interview.max_total_questions
    assert state.phase is not Phase.INTERVIEW, f"interview never exited: {state.phase.value}"


@pytest.mark.asyncio
async def test_a_subject_with_no_generatable_question_is_closed_not_retried(services):
    gateway = _NoQuestionGateway()
    fresh = Services.create(services.config)
    fresh.gateway = gateway
    fresh.knowledge = KnowledgeService(fresh.store, gateway, services.config)

    engine = WorkflowEngine(fresh)
    state = fresh.store.create_session(brief=BRIEF)
    state.phase = Phase.INTERVIEW
    state.subjects = [SubjectState(name="Integration", why="legacy policy system", priority="high")]

    async def noop(event_type, payload=None):
        del event_type, payload

    await engine.advance(state, emit=noop)

    assert state.subjects[0].status == "done", state.subjects[0].status
    assert "no question" in (state.subjects[0].notes or "").lower()
    assert state.awaiting is not AwaitKind.ANSWER, "the engine waits on a question that does not exist"


@pytest.mark.asyncio
async def test_the_ceiling_is_enforced_when_a_question_is_recorded_not_only_before_it(services):
    """The budget is checked before generating a question and again when it is recorded.

    The gap between those two checks is what lets a queued review or gap question push the run past
    the ceiling the stakeholder was told the interview would not exceed.
    """
    cfg = services.config
    cfg.interview.max_total_questions = 2

    engine = WorkflowEngine(services)
    state = services.store.create_session(brief=BRIEF)
    state.subjects = [SubjectState(name="Scope In and Out", priority="high")]
    state.questions = [
        QuestionState(id=1, subject="Scope In and Out", text="What is in scope?", answer="Intake."),
        QuestionState(id=2, subject="Scope In and Out", text="What is out of scope?", answer="Payments."),
    ]

    asked = await engine._ask(state, subject="Scope In and Out", text="Who approves exceptions?", emit=lambda *_: None)
    assert asked is None, "the ceiling was checked before generating but not when recording"
    assert len(state.questions) == 2, f"{len(state.questions)} questions recorded against a ceiling of 2"
