"""Regression: a persisted snapshot never claims WAITING_USER without an awaiting action.

`awaiting` clears the moment an answer is submitted, but `status` stayed WAITING_USER until the end of the
advance loop. Every `save_state` in between therefore told the UI "waiting for you" while the engine was
mid-step — exactly the `waiting_user awaiting=none` state a real-model run showed.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ba_agent.runtime import SessionManager
from ba_agent.schemas import AwaitKind, RunStatus
from conftest import FakeGateway


class StoreSpy:
    """Wraps the real store and records every state the engine persists."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.snapshots: list[Any] = []

    def save_state(self, state) -> None:
        self.snapshots.append(state.model_copy(deep=True))
        return self.inner.save_state(state)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


async def drive(manager: SessionManager, sid: str, answer: str) -> None:
    for _ in range(150):
        state = manager.get_state(sid)
        if state is None or state.phase.value == "done" or state.status is RunStatus.ERROR:
            break
        if state.awaiting is AwaitKind.ANSWER and state.current_question:
            await manager.answer(sid, state.current_question.id, answer=answer)
        elif state.awaiting is AwaitKind.APPROVAL:
            await manager.approve(sid, True)
        else:
            await manager.schedule(sid, "advance")
        # `schedule` spawns a background task; wait for it so the snapshot list is complete.
        runtime = manager.runtimes.get(sid)
        if runtime and runtime.task is not None:
            await runtime.task


def assert_invariant(spy: StoreSpy) -> None:
    assert spy.snapshots, "the engine should persist intermediate snapshots"
    for snapshot in spy.snapshots:
        waiting = snapshot.status is RunStatus.WAITING_USER
        awaits = snapshot.awaiting in (AwaitKind.ANSWER, AwaitKind.APPROVAL)
        if waiting != awaits:
            raise AssertionError(
                f"phase={snapshot.phase.value}: status={snapshot.status.value} but awaiting={snapshot.awaiting.value}"
            )
        if waiting and snapshot.awaiting is AwaitKind.ANSWER and snapshot.current_question is None:
            raise AssertionError(f"phase={snapshot.phase.value}: waiting for an answer with no question")


def test_status_and_awaiting_never_disagree(services) -> None:
    spy = StoreSpy(services.store)
    services.store = spy  # type: ignore[assignment]

    async def run() -> str:
        manager = SessionManager(services)
        state = await manager.create_session(brief="Claims portal for brokers, 400 claims a month.")
        await drive(manager, state.session_id, "Adjusters triage claims within 24 hours.")
        return state.session_id

    asyncio.run(run())

    assert_invariant(spy)
    final = spy.snapshots[-1]
    assert final.phase.value == "done"
    assert final.status is RunStatus.DONE and final.awaiting is AwaitKind.FINISHED


def test_a_multi_step_run_persists_a_running_snapshot(services) -> None:
    spy = StoreSpy(services.store)
    services.store = spy  # type: ignore[assignment]
    services.gateway = FakeGateway(critic_script=["fix", "fix", "complete"])  # type: ignore[assignment]

    async def run() -> None:
        manager = SessionManager(services)
        state = await manager.create_session(brief="Claims portal for brokers, 400 claims a month.")
        await drive(manager, state.session_id, "Adjusters triage claims within 24 hours.")

    asyncio.run(run())

    assert_invariant(spy)
    running = [s for s in spy.snapshots if s.status is RunStatus.RUNNING]
    assert running, "a multi-step run must persist at least one RUNNING snapshot"
