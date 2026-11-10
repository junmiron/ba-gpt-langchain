import asyncio
from pathlib import Path

import pytest

from ba_agent.config import RoleParams
from ba_agent.models import StructuredResult
from ba_agent.runtime import SessionManager
from ba_agent.schemas import AwaitKind
from ba_agent.spec import assemble_spec
from ba_agent.workflow import WorkflowEngine
from conftest import FakeGateway

ROOT = Path(__file__).resolve().parents[1]


class StakeholderGateway(FakeGateway):
    def __init__(self):
        super().__init__()
        self.human_calls = 0

    async def structured(self, schema, **kwargs):
        if kwargs.get("role") == "human":
            self.human_calls += 1
            return StructuredResult(
                parsed=schema(answer="Staff use the app to track jobs; payments are out of scope."),
                method="test",
                attempts=1,
                model="fake-human",
            )
        return await super().structured(schema, **kwargs)


def test_human_lane_is_available_and_defaults_to_a_real_person(config, services) -> None:
    assert config.roles["human"].model is None
    manager = SessionManager(services)
    assert "human" in manager.model_settings()["roles"]
    assert RoleParams().model is None
    app = (ROOT / "src" / "ba_agent" / "web" / "app.js").read_text(encoding="utf-8")
    assert 'human: "Human"' in app
    assert 'role === "human" ? "Real human"' in app


def test_human_lane_model_selection_can_be_cleared_back_to_real_human(services) -> None:
    manager = SessionManager(services)
    asyncio.run(manager.set_models(overrides={"human": "simulator-model"}))
    assert services.config.roles["human"].model == "simulator-model"
    asyncio.run(manager.set_models(overrides={"human": ""}))
    assert services.config.roles["human"].model is None
    assert "human" not in manager.model_settings()["overrides"]


@pytest.mark.asyncio
async def test_selected_human_model_answers_questions_but_keeps_approval_human(config, services) -> None:
    config.roles["human"].model = "fake-human"
    config.interview.max_questions_per_subject = 1
    gateway = StakeholderGateway()
    services.gateway = gateway
    services.knowledge.gateway = gateway
    state = services.store.create_session(brief="A small claims portal for brokers.")
    events = []

    async def emit(event_type, payload=None):
        events.append((str(event_type), payload or {}))

    await WorkflowEngine(services).advance(state, emit=emit)

    assert gateway.human_calls >= 1
    assert state.awaiting is AwaitKind.APPROVAL
    answered = [question for question in state.questions if question.answer]
    assert answered and all(question.answer_source == "simulated" for question in answered)
    answer_events = [payload for kind, payload in events if kind == "answer"]
    assert answer_events and all(event["source"] == "simulated" for event in answer_events)
    assert "simulated stakeholder" in assemble_spec(state).lower()
