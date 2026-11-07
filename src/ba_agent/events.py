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


#: Default live buffer per subscriber. Large enough that a whole review round never overflows a healthy tab.
DEFAULT_FANOUT_CAPACITY = 5000


class EventFanout:
    """Per-session in-process pub/sub used by the SSE endpoint.

    Events are also persisted (see :mod:`ba_agent.store`), so a reconnecting client can replay
    from its last id; this class only handles live delivery.

    When a subscriber falls behind and its buffer overflows, the oldest frames are dropped (the engine never
    blocks). The drop is *not* silent: the subscriber receives a ``warning`` event with the number of frames lost,
    so the UI can tell the user to reload and replay from the event log.
    """

    def __init__(self, maxlen: int = DEFAULT_FANOUT_CAPACITY) -> None:
        self._subscribers: dict[str, set[asyncio.Queue]] = {}
        self._maxlen = max(4, int(maxlen))

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
                self._make_room(session_id, queue, event)

    def _make_room(self, session_id: str, queue: asyncio.Queue, event: dict[str, Any]) -> None:
        """Slow consumer: drop the oldest frames (never block the engine) and report the loss in-band."""
        dropped = 0
        for _ in range(2):  # one slot for the warning, one for the event itself
            try:
                queue.get_nowait()
                dropped += 1
            except asyncio.QueueEmpty:  # pragma: no cover - defensive
                break
        warning = {
            "id": 0, "session_id": session_id, "ts": now_iso(), "type": EventType.WARNING, "role": "", "step": "",
            "payload": {"message": f"live stream fell behind and dropped {dropped} event(s); reload the page to "
                                   "replay the full history", "dropped": dropped, "source": "fanout"},
        }
        for item in (warning, event):
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:  # pragma: no cover - two slots were just freed
                pass
