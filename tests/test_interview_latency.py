import pytest

from ba_agent.models import StructuredResult
from ba_agent.schemas import AwaitKind, Phase, ProbeDecision
from ba_agent.workflow import WorkflowEngine
from conftest import FakeGateway


class ProbeWithNextQuestion(ProbeDecision):
    next_question: str
    next_rationale: str = ""


class FastTransitionGateway(FakeGateway):
    async def structured(self, schema, **kwargs):
        if kwargs.get("step", "").startswith("probe:"):
            step = kwargs["step"]
            self.calls.append((kwargs.get("role", ""), step))
            self.usage["calls"] += 1
            self.routed.append((kwargs.get("role", ""), step, kwargs.get("model_override")))
            return StructuredResult(
                parsed=ProbeWithNextQuestion(
                    covered=True,
                    reason="enough detail",
                    next_question="Which staff roles need access in the first release?",
                    next_rationale="Define the access roles for the next subject.",
                ),
                method="fake",
                attempts=1,
            )
        return await super().structured(schema, **kwargs)


@pytest.mark.asyncio
async def test_covered_answer_uses_probe_generated_next_question_without_another_call(config, services):
    config.interview.max_questions_per_subject = 2
    gateway = FastTransitionGateway()
    services.gateway = gateway
    services.knowledge.gateway = gateway
    state = services.store.create_session(brief="A claims portal for insurance brokers.")
    engine = WorkflowEngine(services)
    await engine.advance(state)
    assert state.awaiting is AwaitKind.ANSWER
    prior_question = state.current_question
    calls_before = len(gateway.calls)

    await engine.submit_answer(state, prior_question.id, answer="Brokers submit claims and check their status.")

    assert len(gateway.calls) - calls_before == 2  # triage + coverage/next-question, not three calls
    assert state.phase is Phase.INTERVIEW
    assert state.awaiting is AwaitKind.ANSWER
    assert state.current_question is not None
    assert state.current_question.subject != prior_question.subject
    assert state.current_question.text == "Which staff roles need access in the first release?"
