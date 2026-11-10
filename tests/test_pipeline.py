"""End-to-end state-machine tests driven by the scripted FakeGateway — no model, no VRAM."""

from __future__ import annotations

import pytest

from ba_agent.schemas import AwaitKind, Phase, RunStatus
from ba_agent.workflow import WorkflowEngine
from conftest import FakeGateway

BRIEF = (
    "We need a portal for insurance brokers to submit and track claims. Today everything is email. "
    "About 300 brokers, peak in winter, must integrate with the existing policy system over REST."
)


async def drive(
    services,
    *,
    brief: str = BRIEF,
    answer: str = "Around 400 claims per month, mostly PDFs.",
    approve: bool = True,
    max_turns: int = 60,
):
    """Run the engine to completion, answering every pending question like a stakeholder would."""
    state = services.store.create_session(brief=brief)
    engine = WorkflowEngine(services)
    events: list[tuple[str, dict]] = []

    async def emit(event_type: str, payload: dict | None = None) -> None:
        events.append((event_type, payload or {}))

    await engine.advance(state, emit=emit)
    for _ in range(max_turns):
        if state.phase is Phase.DONE or state.status is RunStatus.ERROR:
            break
        if state.awaiting is AwaitKind.ANSWER:
            question = state.current_question
            assert question is not None, "engine said it was waiting but no question is pending"
            await engine.submit_answer(state, question.id, answer=answer, emit=emit)
        elif state.awaiting is AwaitKind.APPROVAL:
            await engine.set_approval(state, approve, "" if approve else "please add the approval step", emit=emit)
        else:
            pytest.fail(
                f"engine stalled in phase={state.phase.value} awaiting={state.awaiting.value} error={state.error}"
            )
    return state, events


def types_of(events) -> list[str]:
    return [event_type for event_type, _ in events]


@pytest.mark.asyncio
async def test_happy_path_reaches_done_and_writes_deliverables(services):
    state, events = await drive(services)

    assert state.phase is Phase.DONE, f"error={state.error}"
    assert state.status is RunStatus.DONE
    assert state.awaiting is AwaitKind.FINISHED

    # one agenda question per subject (the probe says "covered") + no clarification rounds
    agenda = [q for q in state.questions if q.purpose == "agenda"]
    assert len(agenda) == len(state.subjects) == 2
    assert all(q.resolved for q in state.questions)

    # the ledger is the source of truth: ids are stable and traceable to a question
    requirements = state.requirements()
    assert requirements and requirements[0].id.startswith("FR-")
    assert requirements[0].source_q == agenda[0].id

    spec = state.spec_markdown
    assert "Functional Specification" in spec
    assert requirements[0].id in spec, "spec must reference the ledger requirement id"
    assert "UC-01" in spec and "**Claim**" in spec
    assert "Appendix A" in spec  # deterministic traceability table

    names = {artifact.name for artifact in state.artifacts}
    assert {"functional_specification.md", "interview.jsonl", "deliverable.html", "metadata.json"} <= names
    out_dir = services.config.out_dir(state.session_id)
    assert (out_dir / "functional_specification.md").read_text(encoding="utf-8").strip()
    assert len((out_dir / "interview.jsonl").read_text(encoding="utf-8").splitlines()) == 2

    # diagrams validated and shipped
    assert state.mermaid_code.startswith("flowchart")
    assert (out_dir / "process_diagram.mmd").exists()

    assert "question" in types_of(events) and "artifact" in types_of(events)


@pytest.mark.asyncio
async def test_review_needs_user_escalates_back_to_stakeholder(config, fake_gateway_cls):
    from ba_agent.services import Services

    gateway = fake_gateway_cls(critic_script=["needs_user", "complete"])
    services = Services.create(config)
    services.gateway = gateway
    from ba_agent.knowledge import KnowledgeService

    services.knowledge = KnowledgeService(services.store, gateway, config)

    state, _events = await drive(services)

    review_questions = [q for q in state.questions if q.purpose == "review"]
    assert len(review_questions) == 1
    assert review_questions[0].answer == "Around 400 claims per month, mostly PDFs."
    assert state.review_rounds == 1
    assert state.phase is Phase.DONE


@pytest.mark.asyncio
async def test_review_auto_fix_regenerates_only_the_flagged_section(config, fake_gateway_cls):
    from ba_agent.knowledge import KnowledgeService
    from ba_agent.services import Services

    gateway = fake_gateway_cls(critic_script=["auto_fixable", "complete"])
    services = Services.create(config)
    services.gateway = gateway
    services.knowledge = KnowledgeService(services.store, gateway, config)

    state, _ = await drive(services)

    assert state.phase is Phase.DONE
    assert state.review_rounds == 1
    # the critic asked for a new glossary → exactly two glossary generations (initial + fix)
    glossary_calls = [call for call in gateway.calls if call[0] == "glossary"]
    assert len(glossary_calls) == 2


@pytest.mark.asyncio
async def test_probe_that_never_covers_is_bounded_by_subject_budget(config, fake_gateway_cls):
    from ba_agent.knowledge import KnowledgeService
    from ba_agent.services import Services

    gateway = fake_gateway_cls(always_probe_followup=True)
    services = Services.create(config)
    services.gateway = gateway
    services.knowledge = KnowledgeService(services.store, gateway, config)

    state, _ = await drive(services, max_turns=80)

    agenda = [q for q in state.questions if q.purpose == "agenda"]
    # The probe keeps asking "and who approves it?" for every subject; the dedup guard stops the
    # engine from re-asking the identical question, so the budget is a ceiling, not a quota.
    assert len(agenda) <= len(state.subjects) * config.interview.max_questions_per_subject
    per_subject = {q.subject for q in agenda}
    assert all(
        sum(1 for x in agenda if x.subject == subject) <= config.interview.max_questions_per_subject
        for subject in per_subject
    )
    assert len(state.questions) <= config.interview.max_total_questions
    assert state.phase is Phase.DONE


@pytest.mark.asyncio
async def test_triage_failure_keeps_the_answer_verbatim(config, fake_gateway_cls):
    from ba_agent.knowledge import KnowledgeService
    from ba_agent.services import Services

    gateway = fake_gateway_cls(fail_triage=True)
    services = Services.create(config)
    services.gateway = gateway
    services.knowledge = KnowledgeService(services.store, gateway, config)

    state, events = await drive(services)

    assert state.phase is Phase.DONE
    facts = [i for i in state.ledger if i.kind == "fact" and "400 claims" in i.body]
    assert facts, "an unparseable answer must still be stored verbatim"
    assert any(t == "warning" for t in types_of(events))


@pytest.mark.asyncio
async def test_skipped_answers_close_subjects_instead_of_looping(services):
    state = services.store.create_session(brief=BRIEF)
    engine = WorkflowEngine(services)

    async def emit(event_type: str, payload: dict | None = None) -> None:
        return None

    await engine.advance(state, emit=emit)
    for _ in range(30):
        if state.phase is Phase.DONE or state.status is RunStatus.ERROR:
            break
        if state.awaiting is AwaitKind.ANSWER:
            question = state.current_question
            await engine.submit_answer(state, question.id, skipped=True, emit=emit)
        elif state.awaiting is AwaitKind.APPROVAL:
            await engine.set_approval(state, True, "", emit=emit)
        else:
            pytest.fail("stalled")

    assert state.phase is Phase.DONE
    assert all(subject.closed for subject in state.subjects)
    assert len([q for q in state.questions if q.purpose == "agenda"]) == len(state.subjects)


@pytest.mark.asyncio
async def test_run_is_resumable_from_sqlite(services):
    """Simulate a restart: drop the in-memory state and reload it from the database."""
    state = services.store.create_session(brief=BRIEF)
    engine = WorkflowEngine(services)

    async def emit(event_type: str, payload: dict | None = None) -> None:
        return None

    await engine.advance(state, emit=emit)
    assert state.awaiting is AwaitKind.ANSWER
    pending_id = state.current_question.id

    reloaded = services.store.load_state(state.session_id)
    assert reloaded is not None
    assert reloaded.phase is Phase.INTERVIEW
    assert reloaded.current_question is not None and reloaded.current_question.id == pending_id

    await engine.submit_answer(reloaded, pending_id, answer="400 claims per month", emit=emit)
    assert reloaded.requirements(), "answer recorded after a restart must still reach the ledger"


@pytest.mark.asyncio
async def test_run_stops_with_an_actionable_error_when_no_model_is_available(services):
    """An offline backend must produce one clear, resumable error — not a hollow document."""
    from ba_agent.models import no_model_message

    class NoModelGateway(FakeGateway):
        async def resolve_model(self, requested: str | None) -> str | None:
            return requested  # nothing loaded, no default configured → nothing callable

    gateway = NoModelGateway()
    services.gateway = gateway
    from ba_agent.workflow import WorkflowEngine

    state = services.store.create_session(brief=BRIEF)
    engine = WorkflowEngine(services)
    seen: list[tuple[str, dict]] = []

    async def emit(event_type: str, payload: dict | None = None) -> None:
        seen.append((event_type, payload or {}))

    await engine.advance(state, emit=emit)

    assert state.status is RunStatus.ERROR
    assert no_model_message(services.config.provider) in (state.error or ""), state.error
    assert [event_type for event_type, _ in seen] == ["error"]  # one legible failure, no retry storm
    assert gateway.calls == []  # nothing was attempted
    assert state.phase is Phase.INTAKE  # resumable from where it stopped


@pytest.mark.asyncio
async def test_summary_rejection_feeds_back_into_the_ledger(services):
    state = services.store.create_session(brief=BRIEF)
    engine = WorkflowEngine(services)

    async def emit(event_type: str, payload: dict | None = None) -> None:
        return None

    await engine.advance(state, emit=emit)

    rejected = False
    for _ in range(60):  # bounded so a regression fails the test instead of hanging it
        if state.phase is Phase.DONE or state.status is RunStatus.ERROR:
            break
        if state.awaiting is AwaitKind.ANSWER:
            await engine.submit_answer(state, state.current_question.id, answer="fine", emit=emit)
        elif state.awaiting is AwaitKind.APPROVAL:
            if not rejected:
                rejected = True  # ask for changes once, then accept the revised summary
                await engine.set_approval(state, False, "Mobile access is out of scope.", emit=emit)
            else:
                await engine.set_approval(state, True, "", emit=emit)
        else:
            pytest.fail(f"engine stalled in phase={state.phase.value} awaiting={state.awaiting.value}")

    assert rejected, "the run never reached the approval gate"
    assert state.summary_rejections == 1
    assert any("out of scope" in item.body.lower() for item in state.ledger if item.kind == "decision")
    assert state.phase is Phase.DONE, f"error={state.error}"
