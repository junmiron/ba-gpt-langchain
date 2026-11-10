"""Runtime command paths: cancel, error surfaces, and settings validation.

The engine's error handling is what the UI depends on. If a cancelled run stays `RUNNING` or a failed
step never reaches the event stream, the user sees a spinner with no way out.
"""

from __future__ import annotations

import asyncio

from ba_agent.runtime import SessionManager
from ba_agent.schemas import AwaitKind, Phase, RunStatus


def _manager(services, *, start: bool = True) -> SessionManager:
    manager = SessionManager(services)
    asyncio.run(manager.create_session("Claims Portal", start=start))
    return manager


def test_cancel_stops_a_running_session_and_leaves_it_resumable(services):
    manager = _manager(services, start=False)
    session_id = manager.list_sessions()[0]["id"]

    async def slow_step(state, emit):
        del state, emit
        await asyncio.sleep(5)  # a step that is genuinely in flight when the user presses Stop

    manager.engine._step_plan = slow_step

    async def drive():
        await manager.schedule(session_id, "advance")
        await asyncio.sleep(0.05)
        cancelled = await manager.cancel(session_id)
        await asyncio.sleep(0.2)  # the task's CancelledError handler runs on the next tick
        return cancelled

    assert asyncio.run(drive()) is True, "cancel must report that a task was actually stopped"
    state = manager.get_state(session_id)
    assert state is not None
    assert state.status is RunStatus.CANCELLED, f"a cancelled run must not still look alive: {state.status}"
    assert state.phase is not Phase.DONE, "a cancelled run is unfinished by definition"
    assert state.awaiting not in (AwaitKind.ANSWER, AwaitKind.APPROVAL), "a cancelled run is not waiting on the user"
    assert not manager._runtime(session_id).busy, "the runtime must release the session after cancellation"

    # Resume is the only way forward from CANCELLED, so it has to work.
    assert asyncio.run(manager.resume(session_id))["queued"] is True


def test_cancel_on_an_idle_session_reports_false(services):
    manager = _manager(services)
    session_id = manager.list_sessions()[0]["id"]
    assert asyncio.run(manager.cancel(session_id)) is False, "nothing was running, so nothing was cancelled"


def test_a_step_that_raises_surfaces_an_error_event(services):
    manager = _manager(services, start=False)
    session_id = manager.list_sessions()[0]["id"]

    async def boom(state, emit):
        del state, emit
        raise RuntimeError("intake exploded")

    manager.engine._step_plan = boom

    async def drive():
        await manager.schedule(session_id, "advance")
        await asyncio.sleep(0.6)

    asyncio.run(drive())

    state = manager.get_state(session_id)
    assert state is not None and state.status is RunStatus.ERROR, state
    assert "intake exploded" in (state.error or ""), state.error
    events = manager.events_after(session_id)
    assert any(e["type"] == "error" for e in events), "the failure must reach the stream the UI reads"


def test_an_unknown_session_id_raises_keyerror(services):
    manager = SessionManager(services)
    try:
        manager._runtime("no-such-session")
    except KeyError as exc:
        assert "Unknown session" in str(exc)
    else:
        raise AssertionError("_runtime must raise for an unknown session")


def test_invalid_setting_values_are_rejected_with_a_readable_message(services):
    manager = SessionManager(services)

    for patch, expected in [
        ({"interview": {"max_total_questions": "not a number"}}, "number"),
        ({"interview": "flat string"}, "object"),
        ({"review": {"max_review_rounds": -3}}, "between"),
        ({"bogus": {"x": 1}}, "not editable"),
    ]:
        try:
            manager.update_run_settings(patch)
        except ValueError as exc:
            assert expected in str(exc).lower(), f"{patch} -> {exc}"
        else:
            raise AssertionError(f"{patch} should not be accepted")


def test_a_docs_dir_escaping_the_project_root_is_refused(services):
    manager = SessionManager(services)
    for bad in ("/etc", "../outside", ""):
        try:
            manager.update_run_settings({"knowledge": {"docs_dir": bad}})
        except ValueError as exc:
            assert "docs_dir" in str(exc), f"{bad!r} accepted: {exc}"
        else:
            raise AssertionError(f"docs_dir {bad!r} must be refused")
