"""Session runtime: background execution, per-session locking and the event pipeline to the UI.

Design notes
* HTTP handlers never wait on the model. ``answer``/``approve`` schedule a task and return; the browser watches
  progress over SSE. That keeps requests short even when a 30B model is loading.
* One asyncio lock per session serialises engine steps; chat mutations take it briefly, reads do not.
* Events are persisted (except token/reasoning streams) *and* fanned out live, so a reconnecting client can replay
  from its last event id — closing a tab never loses output.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from .agent_loop import run_agent
from .config import AppConfig
from .events import EventFanout, EventType, now_iso
from .models import ModelGateway
from .prompts import render as render_prompt
from .schemas import AwaitKind, Phase, RunStatus, SessionState
from .services import Services
from .tools.chat_tools import ChatContext, build_chat_tools
from .workflow import WorkflowEngine

#: Live-only events: streamed text would bloat the event table and is useless after the fact.
TRANSIENT_EVENTS = {EventType.TOKEN, EventType.REASONING}


@dataclass
class SessionRuntime:
    session_id: str
    state: SessionState
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    task: asyncio.Task | None = None

    @property
    def busy(self) -> bool:
        return self.task is not None and not self.task.done()


class SessionManager:
    def __init__(self, services: Services | None = None) -> None:
        self.services = services or Services.create()
        self.config: AppConfig = self.services.config
        self.engine = WorkflowEngine(self.services)
        self.gateway: ModelGateway = self.services.gateway
        self.fanout = EventFanout()
        self.runtimes: dict[str, SessionRuntime] = {}

    # ------------------------------------------------------------------ lifecycle

    def mark_interrupted(self) -> int:
        """Called on startup: a run interrupted by a restart is resumable, not silently 'running'."""
        count = 0
        for row in self.services.store.list_sessions(limit=500):
            if row["status"] == RunStatus.RUNNING.value:
                state = self.services.store.load_state(row["id"])
                if state is None:
                    continue
                state.status = RunStatus.IDLE
                self.services.store.save_state(state)
                self.log_event(state.session_id, EventType.WARNING, {
                    "message": "run was interrupted by a server restart — press Resume to continue"})
                count += 1
        return count

    async def create_session(self, brief: str, title: str = "", model_overrides: dict[str, str] | None = None,
                             start: bool = True) -> SessionState:
        state = self.services.store.create_session(brief=brief, title=title, model_overrides=model_overrides or {})
        runtime = SessionRuntime(session_id=state.session_id, state=state)
        self.runtimes[state.session_id] = runtime
        self.log_event(state.session_id, EventType.SESSION, {"event": "created", "title": state.title,
                                                             "brief": state.brief[:500]})
        if start:
            await self.schedule(state.session_id, "start")
        return state

    def get_state(self, session_id: str) -> SessionState | None:
        runtime = self.runtimes.get(session_id)
        if runtime is not None:
            return runtime.state
        state = self.services.store.load_state(session_id)
        if state is not None:
            self.runtimes[session_id] = SessionRuntime(session_id=session_id, state=state)
        return state

    def list_sessions(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.services.store.list_sessions(limit=limit)
        for row in rows:
            runtime = self.runtimes.get(row["id"])
            row["busy"] = bool(runtime and runtime.busy)
        return rows

    async def delete_session(self, session_id: str) -> None:
        await self.cancel(session_id)
        self.runtimes.pop(session_id, None)
        self.services.store.delete_session(session_id)

    # ------------------------------------------------------------------ events

    def log_event(self, session_id: str, event_type: str, payload: dict[str, Any] | None = None, *,
                  role: str = "", step: str = "") -> dict[str, Any]:
        payload = payload or {}
        if event_type in TRANSIENT_EVENTS:
            event = {"id": 0, "session_id": session_id, "ts": now_iso(), "type": event_type, "role": role,
                     "step": step, "payload": payload}
            self.fanout.publish(session_id, event)
            return event

        event_id = self.services.store.log_event(session_id, event_type, payload, role=role, step=step)
        event = {"id": event_id, "session_id": session_id, "ts": now_iso(), "type": event_type, "role": role,
                 "step": step, "payload": payload}
        self.fanout.publish(session_id, event)
        return event

    def emitter(self, session_id: str):
        async def emit(event_type: str, payload: dict[str, Any] | None = None) -> None:
            data = dict(payload or {})
            role = str(data.get("role") or "")
            step = str(data.get("step") or "")
            self.log_event(session_id, event_type, data, role=role, step=step)

        return emit

    def events_after(self, session_id: str, after_id: int = 0, limit: int = 2000) -> list[dict[str, Any]]:
        return self.services.store.get_events(session_id, after_id=after_id, limit=limit)

    # ------------------------------------------------------------------ commands

    async def schedule(self, session_id: str, reason: str = "advance", **kwargs: Any) -> dict[str, Any]:
        """Start a background task for this session (no-op if one is already running)."""
        runtime = self._runtime(session_id)
        if runtime.busy:
            return {"queued": False, "busy": True, "reason": reason}

        runtime.task = asyncio.create_task(self._run(runtime, reason, **kwargs),
                                           name=f"ba-{session_id}-{reason}")
        return {"queued": True, "busy": True, "reason": reason}

    def _runtime(self, session_id: str) -> SessionRuntime:
        state = self.get_state(session_id)
        if state is None:
            raise KeyError(f"Unknown session '{session_id}'")
        return self.runtimes[session_id]

    async def _run(self, runtime: SessionRuntime, reason: str, **kwargs: Any) -> None:
        emit = self.emitter(runtime.session_id)
        async with runtime.lock:
            state = runtime.state
            try:
                if reason == "chat":
                    await self._chat_step(state, str(kwargs.get("message", "")), emit)
                    return
                if reason in ("answer",):
                    await self.engine.submit_answer(
                        state, int(kwargs["question_id"]), answer=kwargs.get("answer"),
                        skipped=bool(kwargs.get("skipped")), not_applicable=bool(kwargs.get("not_applicable")),
                        emit=emit)
                    return
                if reason == "approve":
                    await self.engine.set_approval(state, bool(kwargs.get("approved")),
                                                   str(kwargs.get("notes", "")), emit=emit)
                    return
                await self.engine.advance(state, emit=emit)
            except asyncio.CancelledError:
                state.status = RunStatus.CANCELLED
                state.awaiting = AwaitKind.NONE if state.phase is not Phase.DONE else state.awaiting
                self.services.store.save_state(state)
                await emit(EventType.SESSION, {"event": "cancelled", "phase": state.phase.value})
                raise
            except Exception as exc:  # noqa: BLE001 — surfaced through the event stream
                state.error = f"{type(exc).__name__}: {exc}"
                state.status = RunStatus.ERROR
                self.services.store.save_state(state)
                await emit(EventType.ERROR, {"message": state.error, "reason": reason})

    async def cancel(self, session_id: str) -> bool:
        runtime = self.runtimes.get(session_id)
        if runtime and runtime.busy and runtime.task:
            runtime.task.cancel()
            try:
                await runtime.task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            return True
        return False

    async def answer(self, session_id: str, question_id: int, *, answer: str | None = None, skipped: bool = False,
                     not_applicable: bool = False) -> dict[str, Any]:
        state = self.get_state(session_id)
        if state is None:
            raise KeyError(f"Unknown session '{session_id}'")
        question = next((q for q in state.questions if q.id == question_id), None)
        if question is None:
            raise ValueError(f"Unknown question id {question_id}")
        if question.resolved:
            return {"ok": False, "error": "already answered"}
        # Validate here, not only inside the background task: otherwise an empty reply returns HTTP 200 and the
        # UI believes it was recorded while nothing happens.
        if not (skipped or not_applicable) and not (answer or "").strip():
            raise ValueError("Provide an answer, or skip the question.")

        # The engine records the answer inside the lock so a double submit cannot slip through.
        return await self._exclusive(session_id, "answer", question_id=question_id, answer=answer,
                                     skipped=skipped, not_applicable=not_applicable)

    async def approve(self, session_id: str, approved: bool, notes: str = "") -> dict[str, Any]:
        return await self._exclusive(session_id, "approve", approved=approved, notes=notes)

    async def resume(self, session_id: str) -> dict[str, Any]:
        state = self.get_state(session_id)
        if state is None:
            raise KeyError(f"Unknown session '{session_id}'")
        if state.phase is Phase.DONE:
            return {"ok": False, "error": "session already finished"}
        state.status = RunStatus.RUNNING
        state.error = None
        self.services.store.save_state(state)
        return await self.schedule(session_id, "advance")

    async def _exclusive(self, session_id: str, reason: str, **kwargs: Any) -> dict[str, Any]:
        """Schedule a command; if the session is busy, report it instead of silently dropping the input."""
        runtime = self._runtime(session_id)
        if runtime.busy:
            return {"ok": False, "busy": True, "error": "the agent is working on something else — try again in a moment"}
        await self.schedule(session_id, reason, **kwargs)
        return {"ok": True, "queued": True}

    # ------------------------------------------------------------------ chat channel

    async def mutate(self, session_id: str, apply_state):  # pragma: no cover - used via ChatContext
        runtime = self._runtime(session_id)
        async with runtime.lock:
            message = apply_state(runtime.state)
            self.services.store.save_state(runtime.state)
        return message

    async def chat(self, session_id: str, message: str) -> dict[str, Any]:
        """Interactive Q&A with the BA while a run is in flight (tool-calling agent loop)."""
        runtime = self._runtime(session_id)
        emit = self.emitter(session_id)

        async def mutate(apply_state):
            async with runtime.lock:
                result = apply_state(runtime.state)
                self.services.store.save_state(runtime.state)
            return result

        state = runtime.state
        ctx = ChatContext(session_id=session_id, get_state=lambda: runtime.state, mutate=mutate,
                          knowledge=self.services.knowledge, sandbox=self.services.sandbox,
                          store=self.services.store)
        tools = build_chat_tools(ctx)

        system_prompt, _ = render_prompt(
            "chat_agent",
            project_name=(state.plan.project_name if state.plan else state.title),
            phase=state.phase.value, status=state.status.value, awaiting=state.awaiting.value,
            coverage_summary=", ".join(f"{s.name}: {s.status}" for s in state.subjects) or "(none)",
            requirement_count=len(state.requirements()),
        )

        self.log_event(session_id, EventType.ANSWER, {"channel": "chat", "answer": message})
        result = await run_agent(system_prompt=system_prompt, user_message=message, tools=tools,
                                 gateway=self.gateway, model=state.model_overrides.get("chat"),
                                 emit=emit, step="chat")

        self.log_event(session_id, EventType.SESSION, {"event": "chat_reply", "text": result.text,
                                                       "iterations": result.iterations,
                                                       "tools": [t.name for t in result.tool_calls],
                                                       "error": result.error})
        return {"ok": not result.error, "reply": result.text, "tool_calls": len(result.tool_calls),
                "model": result.model, "error": result.error}

    # ------------------------------------------------------------------ model settings

    async def set_models(self, default_model: str | None = None,
                        overrides: dict[str, str] | None = None) -> dict[str, Any]:
        """Update the global default and/or per-role overrides.

        ``overrides=None`` means *leave them alone* — otherwise picking a new default in the top bar would
        silently erase every per-role override the user had saved.
        """
        if default_model is not None:
            self.config.active_profile.default_model = default_model or None
            self.services.store.set_setting("default_model", default_model)
        if overrides is not None:
            clean = {k: v for k, v in overrides.items() if v}
            for role in sorted(set(list(self.config.roles) + list(clean))):
                params = self.config.roles.get(role) or self.config.role(role)
                params.model = clean.get(role) or None
                self.config.roles[role] = params
            self.services.store.set_setting("model_overrides", clean)
        return {"default_model": self.config.active_profile.default_model,
                "overrides": {r: p.model for r, p in self.config.roles.items() if p.model}}

    def model_settings(self) -> dict[str, Any]:
        overrides = {role: params.model for role, params in self.config.roles.items() if params.model}
        stored_overrides = self.services.store.get_setting("model_overrides", {}) or {}
        return {"default_model": self.config.active_profile.default_model,
                "provider": self.config.provider,
                "overrides": {**stored_overrides, **overrides},
                "embedding_model": self.config.active_profile.embedding_model,
                "roles": sorted(self.config.roles.keys())}

    async def aclose(self) -> None:
        for runtime in list(self.runtimes.values()):
            if runtime.busy and runtime.task:
                runtime.task.cancel()
        await self.gateway.aclose()
        self.services.store.close()


_manager: SessionManager | None = None


def get_manager(services: Services | None = None) -> SessionManager:
    global _manager
    if _manager is None:
        _manager = SessionManager(services)
    return _manager
