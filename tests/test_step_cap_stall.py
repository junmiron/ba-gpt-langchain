"""The per-advance step cap must not silently strand a run.

`advance` loops at most `MAX_STEPS_PER_ADVANCE` times. When that ceiling is reached the function simply
returns, and the tail of the loop sets `status = RUNNING` because the run is neither `DONE` nor waiting
on the user. The runtime's task then ends, so `busy` is False; the UI shows a Resume button only for
`idle`/`cancelled`/`error`. A run that exhausts the cap is therefore stuck: reported as "running", with
nothing scheduled and no way to continue.

The cap is a guard against engine bugs, not a budget the pipeline is allowed to hit legitimately. It has
to scale with the configured budgets, and hitting it must be visible.
"""

from __future__ import annotations

import asyncio
import time

from ba_agent.schemas import AwaitKind, Phase, RunStatus
from ba_agent.workflow import WorkflowEngine
from conftest import FakeGateway

BRIEF = "Portal for brokers to submit and track claims; email today."
ANSWER = "About 400 claims per month, mostly PDFs."


def _services(services, **review_overrides):
    cfg = services.config
    cfg.interview.max_questions_per_subject = 1
    cfg.interview.min_subjects = 2
    cfg.interview.max_subjects = 2
    cfg.interview.max_gap_rounds = 0
    for key, value in review_overrides.items():
        setattr(cfg.review, key, value)
    services.gateway = FakeGateway(critic_script=["auto_fixable"])
    return services


async def _drive(engine, state, events: list[tuple[str, dict]]) -> None:
    """Answer/approve as a stakeholder would, advancing until done, error, or the budget runs out."""

    async def emit(event_type: str, payload: dict | None = None) -> None:
        events.append((event_type, payload or {}))

    for _ in range(12):
        if state.phase is Phase.DONE or state.status is RunStatus.ERROR:
            break
        if state.awaiting is AwaitKind.ANSWER:
            question = state.current_question
            if question is None:
                break
            await engine.submit_answer(state, question.id, answer=ANSWER, emit=emit)
        elif state.awaiting is AwaitKind.APPROVAL:
            await engine.set_approval(state, True, "", emit=emit)
        else:
            await engine.advance(state, emit=emit)


def test_a_review_budget_larger_than_the_cap_still_finishes(services):
    """25 review rounds need ~50 steps; a single advance must cover them, not truncate at 40."""
    services = _services(services, max_review_rounds=25, min_functional_requirements=1)
    engine = WorkflowEngine(services)
    state = services.store.create_session(brief=BRIEF)
    events: list[tuple[str, dict]] = []
    asyncio.run(_drive(engine, state, events))

    assert state.phase is Phase.DONE, (
        f"run never finished: phase={state.phase.value} status={state.status.value} "
        "— the step cap silently truncated a legitimate run"
    )
    assert state.review_rounds >= 10, f"expected the review budget to be exercised, got {state.review_rounds}"
    assert not any("stalled" in payload.get("message", "").lower() for _, payload in events), (
        "a run that legitimately needs more steps than the old flat cap must not be reported as stalled"
    )


def test_the_budget_scales_with_the_configured_rounds(services):
    engine = WorkflowEngine(services)
    state = services.store.create_session(brief=BRIEF)
    small = engine._step_budget(state)

    services.config.review.max_review_rounds = 50
    big = engine._step_budget(state)
    assert big > small, f"budget did not grow with the review rounds: {small} -> {big}"
    assert big >= 2 * services.config.review.max_review_rounds, (
        f"budget {big} cannot cover {services.config.review.max_review_rounds} draft+review rounds"
    )


def test_an_exhausted_cap_leaves_the_run_resumable(services):
    """If the engine ever does hit the ceiling, the session must be recoverable, not frozen."""
    services = _services(services, max_review_rounds=500, min_functional_requirements=1)
    engine = WorkflowEngine(services)
    # Force the guard to trip: a cap far below what the config asks for.
    engine._step_budget = lambda state: 6

    state = services.store.create_session(brief=BRIEF)
    events: list[tuple[str, dict]] = []
    asyncio.run(_drive(engine, state, events))

    if state.phase is not Phase.DONE:
        assert state.status in (RunStatus.IDLE, RunStatus.ERROR), (
            f"stalled run is reported as {state.status.value}; the UI only offers Resume for "
            f"idle/cancelled/error, so the user cannot continue"
        )
        assert any(
            payload.get("message", "").lower().startswith("stalled") or "stalled" in payload.get("message", "").lower()
            for event_type, payload in events
            if event_type == "warning"
        ), "the cap being hit must be reported, not swallowed"


def test_a_stalled_session_can_be_resumed_until_it_finishes(services):
    """The UI's Resume button is the only escape from a stalled run, so it has to make progress.

    The defect this guards against is the state where a run reports RUNNING (so the UI shows no Resume
    button) while nothing is actually scheduled. A stalled run must instead stay resumable and advance
    every time the user presses Resume.
    """
    from ba_agent.runtime import SessionManager

    services = _services(services, max_review_rounds=500, min_functional_requirements=1)
    engine = WorkflowEngine(services)
    engine._step_budget = lambda state: 6  # force the guard to trip repeatedly

    state = services.store.create_session(brief=BRIEF)
    events: list[tuple[str, dict]] = []
    asyncio.run(_drive(engine, state, events))
    if state.phase is Phase.DONE:
        return

    manager = SessionManager(services)
    resumes = 0
    for _ in range(40):
        current = services.store.load_state(state.session_id)
        if current is None or current.phase is Phase.DONE:
            break
        runtime = manager._runtime(state.session_id)
        assert not (current.status is RunStatus.RUNNING and not runtime.busy), (
            f"session reports status={current.status.value} with no scheduled task: the UI shows neither "
            "Stop nor Resume, so the user has no way forward"
        )
        if current.status in (RunStatus.IDLE, RunStatus.ERROR, RunStatus.CANCELLED):
            asyncio.run(manager.resume(state.session_id))
            resumes += 1
        time.sleep(0.05)

    final = services.store.load_state(state.session_id)
    assert final is not None
    assert final.phase is Phase.DONE, (
        f"{resumes} resumes did not finish the run; stuck at phase={final.phase.value} status={final.status.value}"
    )
    assert resumes >= 2, "a stalled run should need more than one resume to reach the end"
