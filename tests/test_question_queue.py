"""Review/gap questions that are queued must actually reach the stakeholder."""

from __future__ import annotations

import pytest

from ba_agent.config import AppConfig
from ba_agent.knowledge import KnowledgeService
from ba_agent.models import StructuredResult
from ba_agent.schemas import (
    AwaitKind,
    CriticVerdict,
    GapAnalysis,
    Phase,
    PlanResult,
    ProbeDecision,
    QuestionDraft,
    RequirementDraft,
    RunStatus,
    SummaryDoc,
    TriageResult,
)
from ba_agent.services import Services
from ba_agent.spec import assemble_spec
from ba_agent.tools.sandbox import Sandbox
from ba_agent.workflow import WorkflowEngine
from conftest import FakeGateway

BLOCKERS = [
    {
        "section_id": "non_functional_requirements",
        "kind": "user",
        "severity": "blocker",
        "description": "No claim volume captured",
        "question_for_user": "How many claims arrive per month?",
    },
    {
        "section_id": "non_functional_requirements",
        "kind": "user",
        "severity": "blocker",
        "description": "No latency target captured",
        "question_for_user": "What upload latency is acceptable?",
    },
    {
        "section_id": "scope",
        "kind": "user",
        "severity": "blocker",
        "description": "No retention period captured",
        "question_for_user": "How long must claim attachments be retained?",
    },
]


class MultiBlockerGateway(FakeGateway):
    """The critic reports three distinct stakeholder blockers on the first review round."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.critic_rounds = 0

    async def structured(self, schema, *, role: str = "", step: str = "", **kwargs):
        self.calls.append((role, step))
        self.routed.append((role, step, kwargs.get("model_override")))
        self.usage["calls"] += 1

        if role == "critic":
            self.critic_rounds += 1
            if self.critic_rounds == 1:
                parsed = CriticVerdict(
                    status="needs_user",
                    summary="three volumes unknown",
                    issues=[
                        {
                            "section_id": "non_functional_requirements",
                            "kind": "user",
                            "severity": "blocker",
                            "description": "No claim volume captured",
                            "question_for_user": "How many claims arrive per month?",
                        },
                        {
                            "section_id": "non_functional_requirements",
                            "kind": "user",
                            "severity": "blocker",
                            "description": "No latency target captured",
                            "question_for_user": "What upload latency is acceptable?",
                        },
                        {
                            "section_id": "scope",
                            "kind": "user",
                            "severity": "blocker",
                            "description": "No retention period captured",
                            "question_for_user": "How long must claim attachments be retained?",
                        },
                    ],
                )
            else:
                parsed = CriticVerdict(status="complete", summary="good enough", issues=[])
        elif role == "plan":
            parsed = PlanResult(
                project_name="Claims Portal",
                objective="Let brokers submit claims online.",
                summary="A portal replacing email intake.",
                subjects=[{"name": "Product Overview", "why": "baseline", "priority": "high"}],
                known_facts=["Claims arrive by email today"],
                assumptions=["Existing policy system stays"],
            )
        elif role == "interview" and step.startswith("question"):
            parsed = QuestionDraft(
                question="What is the team's daily routine?", rationale="scope", kind="open", options=[]
            )
        elif role == "interview" and step.startswith("probe"):
            parsed = ProbeDecision(covered=True, reason="enough detail")
        elif role == "interview" and step == "gap_analysis":
            parsed = GapAnalysis(gaps=[])
        elif role == "triage":
            parsed = TriageResult(
                facts=["Claims are validated manually today"],
                requirements=[
                    RequirementDraft(
                        title="Submit claim online",
                        description="The system shall let a broker submit a claim with attachments.",
                        acceptance_criteria=["A claim with 5 PDFs uploads in under 10 seconds"],
                    )
                ],
                risks=[],
                assumptions=[],
                decisions=[],
                coverage_note="",
            )
        elif role == "summary":
            parsed = SummaryDoc(
                project_name="Claims Portal",
                objective="Online claim submission.",
                background="Replaces email intake.",
                objectives=["Reduce handling time"],
                stakeholders=["Brokers"],
                scope_in=["Claim intake"],
                scope_out=["Payments"],
                key_requirements=["FR-001 Submit claim online"],
                non_functional=["NFR-001 upload under 10 seconds"],
                assumptions=["Policy system stays"],
                open_questions=[],
            )
        else:
            parsed = schema.model_validate({})

        return StructuredResult(parsed=parsed, raw_text="", method="native", attempts=1, degraded=False)


@pytest.fixture
def blocker_config(tmp_path) -> AppConfig:
    cfg = AppConfig(root=tmp_path)
    cfg.lmstudio.embedding_model = None
    cfg.interview.max_questions_per_subject = 2
    cfg.interview.min_subjects = 2
    cfg.interview.max_subjects = 4
    cfg.interview.max_gap_rounds = 1
    cfg.review.max_review_rounds = 5
    cfg.review.min_functional_requirements = 1
    cfg.ensure_dirs()
    (tmp_path / "data").mkdir(exist_ok=True)
    return cfg


def build_services(cfg: AppConfig) -> Services:
    services = Services.create(cfg)
    services.gateway = MultiBlockerGateway()  # type: ignore[assignment]
    services.knowledge = KnowledgeService(services.store, services.gateway, cfg)  # type: ignore[arg-type]
    services.sandbox = Sandbox(writable_roots=[cfg.root / "outputs", cfg.root / "data"])
    return services


@pytest.mark.asyncio
async def test_every_queued_review_question_reaches_the_stakeholder(blocker_config):
    services = build_services(blocker_config)
    state = services.store.create_session(brief="Portal for brokers to submit claims; today it is all email.")
    engine = WorkflowEngine(services)
    events: list[tuple[str, dict]] = []

    async def emit(event_type: str, payload: dict | None = None) -> None:
        events.append((event_type, payload or {}))

    await engine.advance(state, emit=emit)

    asked: list[int] = []
    for _ in range(60):
        if state.phase is Phase.DONE or state.status is RunStatus.ERROR:
            break
        if state.awaiting is AwaitKind.ANSWER:
            question = state.current_question
            assert question is not None
            asked.append(question.id)
            await engine.submit_answer(state, question.id, answer="about 400 per month", emit=emit)
        elif state.awaiting is AwaitKind.APPROVAL:
            await engine.set_approval(state, True, "", emit=emit)
        else:
            pytest.fail(f"engine stalled phase={state.phase.value} awaiting={state.awaiting.value}")

    blocker_questions = {issue["question_for_user"] for issue in BLOCKERS}
    emitted_questions = {payload["question"] for event_type, payload in events if event_type == "question"}
    missing = blocker_questions - emitted_questions
    assert not missing, (
        f"the critic queued {len(missing)} blocker question(s) that were never emitted to the stakeholder: "
        f"{sorted(missing)}"
    )
    answered = {q["question_id"] for event_type, q in events if event_type == "answer"}
    assert len(answered & set(asked)) >= 3, f"only {len(answered & set(asked))} of 3 blockers were answered"


@pytest.mark.asyncio
async def test_queued_questions_do_not_exceed_the_total_budget(blocker_config):
    """`max_total_questions` is a hard ceiling, including questions created by queueing."""
    cfg = blocker_config
    cfg.interview.max_total_questions = 3
    services = build_services(cfg)
    state = services.store.create_session(brief="Portal for brokers to submit claims; today it is all email.")
    engine = WorkflowEngine(services)

    async def emit(event_type: str, payload: dict | None = None) -> None:
        pass

    await engine.advance(state, emit=emit)
    for _ in range(40):
        if state.phase is Phase.DONE or state.status is RunStatus.ERROR:
            break
        if state.awaiting is AwaitKind.ANSWER:
            question = state.current_question
            if question is None:
                break
            await engine.submit_answer(state, question.id, answer="about 400 per month", emit=emit)
        elif state.awaiting is AwaitKind.APPROVAL:
            await engine.set_approval(state, True, "", emit=emit)
        else:
            break

    assert len(state.questions) <= cfg.interview.max_total_questions, (
        f"{len(state.questions)} questions recorded, ceiling is {cfg.interview.max_total_questions}"
    )


@pytest.mark.asyncio
async def test_budget_dropped_blockers_survive_in_the_deliverable(blocker_config):
    """A blocker the budget cannot ask must still show up in section 10, not disappear."""
    cfg = blocker_config
    cfg.interview.max_total_questions = 2
    cfg.interview.max_questions_per_subject = 1
    cfg.interview.min_subjects = 1
    cfg.interview.max_subjects = 1
    services = build_services(cfg)
    state = services.store.create_session(brief="Portal for brokers to submit claims; today it is all email.")
    engine = WorkflowEngine(services)

    async def emit(event_type: str, payload: dict | None = None) -> None:
        pass

    await engine.advance(state, emit=emit)
    for _ in range(40):
        if state.phase is Phase.DONE or state.status is RunStatus.ERROR:
            break
        if state.awaiting is AwaitKind.ANSWER:
            question = state.current_question
            if question is None:
                break
            await engine.submit_answer(state, question.id, answer="about 400 per month", emit=emit)
        elif state.awaiting is AwaitKind.APPROVAL:
            await engine.set_approval(state, True, "", emit=emit)
        else:
            break

    assert len(state.questions) <= cfg.interview.max_total_questions
    spec = assemble_spec(state)
    asked = {q.text for q in state.questions}
    blocker_questions = {issue["question_for_user"] for issue in BLOCKERS}
    unasked = blocker_questions - asked
    if unasked:
        section = state.sections.get("open_questions", "") or spec
        for question in unasked:
            assert question in section, f"unasked blocker {question!r} is missing from the open-questions section"
