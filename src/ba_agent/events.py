"""Typed event vocabulary shared by the engine, the store and the web UI."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable


class EventType:
    SESSION = "session"              # lifecycle: created/status/phase changed
    STEP_START = "step_start"        # engine entered a step
    STEP_END = "step_end"            # step finished (with metrics)
    TOKEN = "token"                  # streamed model output
    REASONING = "reasoning"          # streamed reasoning content (gpt-oss style models)
    STATUS = "status"                # transient notice ("model may be loading…")
    QUESTION = "question"            # a question is on screen, awaiting the user
    ANSWER = "answer"                # user answer recorded
    LEDGER = "ledger"                # ledger items added/updated
    SPEC = "spec"                    # spec section / document updated
    DIAGRAM = "diagram"              # mermaid produced/updated
    REVIEW = "review"                # lint/critic verdicts
    TOOL_CALL = "tool_call"          # agent loop invoked a tool
    ARTIFACT = "artifact"            # file written and downloadable
    APPROVAL = "approval"            # approval requested/granted
    WARNING = "warning"              # recoverable problem (schema repair, degraded mode)
    ERROR = "error"                  # step/session failure
    LOG = "log"                      # free-form debug line


EventPayload = dict[str, Any]
Emitter = Callable[[str, EventPayload], Awaitable[None]]


class NullEmitter:
    async def __call__(self, event_type: str, payload: EventPayload | None = None) -> None:  # noqa: D401
        return None


null_emitter = NullEmitter()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class EventFanout:
    """Per-session in-process pub/sub used by the SSE endpoint.

    Events are also persisted (see :mod:`ba_agent.store`), so a reconnecting client can replay
    from its last id; this class only handles live delivery.
    """

    def __init__(self, maxlen: int = 500) -> None:
        self._subscribers: dict[str, set[asyncio.Queue]] = {}
        self._maxlen = maxlen

    def subscribe(self, session_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._maxlen)
        self._subscribers.setdefault(session_id, set()).add(queue)
        return queue

    def unsubscribe(self, session_id: str, queue: asyncio.Queue) -> None:
        subs = self._subscribers.get(session_id)
        if subs:
            subs.discard(queue)
            if not subs:
                self._subscribers.pop(session_id, None)

    def publish(self, session_id: str, event: dict[str, Any]) -> None:
        for queue in list(self._subscribers.get(session_id, ())):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Slow consumer: drop the oldest frame, never block the engine.
                try:
                    queue.get_nowait()
                    queue.put_nowait(event)
                except Exception:
                    pass
