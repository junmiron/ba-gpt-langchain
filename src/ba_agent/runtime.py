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
import contextlib
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .agent_loop import run_agent
from .config import AppConfig
from .events import EventFanout, EventType, now_iso
from .models import ModelGateway
from .prompts import render as render_prompt
from .schemas import AwaitKind, LedgerItem, Phase, RunStatus, SessionState
from .services import Services
from .tools.chat_tools import ChatContext, build_chat_tools
from .workflow import WorkflowEngine

log = logging.getLogger(__name__)

#: Live-only events: streamed text would bloat the event table and is useless after the fact.
TRANSIENT_EVENTS = {EventType.TOKEN, EventType.REASONING}


def _model_choices(store) -> dict[str, dict[str, Any]]:
    """Saved model lanes, bucketed by provider.

    A model id belongs to one backend: `q2_0` means nothing to LM Studio. Keeping the choices in a
    per-provider bucket means switching providers and back does not drag a stale model name along.
    """
    saved = store.get_setting("model_choices", None)
    if isinstance(saved, dict):
        return {provider: dict(bucket) for provider, bucket in saved.items() if isinstance(bucket, dict)}
    # Migration path for stores written before provider switching existed: flat keys belong to
    # whichever provider they were set on, which is the one recorded in `provider`.
    provider = store.get_setting("provider", None) or "lmstudio"
    bucket: dict[str, Any] = {}
    rename = {"model_overrides": "overrides"}
    for key in ("default_model", "embedding_model", "model_overrides"):
        value = store.get_setting(key, None)
        # "" is a saved decision ("no embeddings"), not an absent one, so only None is skipped.
        if value is not None:
            bucket[rename.get(key, key)] = value
    return {provider: bucket} if bucket else {}


def apply_stored_model_choices(config: AppConfig, store) -> None:
    """Re-apply the model/embedding choices the UI saved, for any entry point that builds services by hand.

    The server does this in its lifespan; the CLI and the scripts must do the same or `ba-agent index`
    and `ba-agent export` quietly disagree with what the user configured in the browser.
    """
    stored_provider = store.get_setting("provider", None)
    if stored_provider and stored_provider in ("lmstudio", "strata"):
        # Provider first: the model ids below are provider-scoped, so restoring them before the
        # provider would put them on the wrong lane.
        config.provider = stored_provider
    bucket = _model_choices(store).get(config.provider, {})
    if bucket.get("default_model"):
        config.active_profile.default_model = bucket["default_model"]
    if bucket.get("embedding_model") is not None:
        # "" is a real choice: the user said this provider has no embedding endpoint.
        config.active_profile.embedding_model = bucket["embedding_model"] or None
    for role, model in (bucket.get("overrides") or {}).items():
        params = config.roles.get(role) or config.role(role)
        params.model = model
        config.roles[role] = params


def apply_stored_run_settings(config: AppConfig, store, settable: dict[str, tuple[str, ...]]) -> None:
    """Overlay the stored settings delta onto `config`, validating each section before applying it.

    Validation goes through the section's own model rather than a bare `setattr`: Pydantic v2 does not
    validate on assignment, so a stale or hand-edited row would land in the config with the wrong type and
    blow up the budget arithmetic that reads it. A bad row is dropped and logged, never applied.
    """
    for section, values in (store.get_setting("run_settings", {}) or {}).items():
        if section not in settable or not isinstance(values, dict):
            continue
        target = getattr(config, section)
        merged = target.model_dump()
        merged.update({key: value for key, value in values.items() if key in settable[section]})
        try:
            setattr(config, section, type(target).model_validate(merged))
        except ValidationError as exc:
            log.warning("ignoring stored %s settings: %s", section, exc.errors()[0]["msg"])


@dataclass
class SessionRuntime:
    """A live run's in-memory handle: its state plus the asyncio task driving it."""

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
        # Snapshot the section values this process started with. "Reset" means "undo every override
        # this manager made", so it must go back to *this* config - re-reading config.yaml would also
        # wipe a programmatic AppConfig (CLI --provider, a test harness) that never came from a file.
        self._baseline: dict[str, dict[str, Any]] = {
            section: getattr(self.config, section).model_dump() for section in self.SETTABLE
        }
        # Model choices are stored under their own keys, not in `run_settings`, so a reset that only
        # cleared `run_settings` would leave the model/embedding overrides in place and re-apply them on
        # the next restart. Snapshot them here so "Reset" means reset everything.
        self._model_baseline: dict[str, Any] = {
            "provider": self.config.provider,
            "default_model": self.config.active_profile.default_model,
            "embedding_model": self.config.active_profile.embedding_model,
            "overrides": {role: params.model for role, params in self.config.roles.items() if params.model},
        }
        self.engine = WorkflowEngine(self.services)
        self.gateway: ModelGateway = self.services.gateway
        self.fanout = EventFanout()
        self.runtimes: dict[str, SessionRuntime] = {}

    # ------------------------------------------------------------------ lifecycle

    def mark_interrupted(self) -> int:
        """Called on startup: a run interrupted by a restart is resumable, not silently 'running'.

        Scans every session in the database, not a page of the most recent ones, so an old interrupted run is
        still marked resumable.
        """
        count = 0
        for session_id in self.services.store.session_ids_with_status(RunStatus.RUNNING.value):
            state = self.services.store.load_state(session_id)
            if state is None:
                continue
            state.status = RunStatus.IDLE
            self.services.store.save_state(state)
            self.log_event(
                state.session_id,
                EventType.WARNING,
                {"message": "run was interrupted by a server restart — press Resume to continue"},
            )
            count += 1
        return count

    async def create_session(
        self,
        brief: str,
        title: str = "",
        model_overrides: dict[str, str] | None = None,
        transcript: str = "",
        transcript_source: str = "",
        start: bool = True,
    ) -> SessionState:
        state = self.services.store.create_session(
            brief=brief,
            title=title,
            model_overrides=model_overrides or {},
            transcript=transcript,
            transcript_source=transcript_source,
        )
        runtime = SessionRuntime(session_id=state.session_id, state=state)
        self.runtimes[state.session_id] = runtime
        self.log_event(
            state.session_id, EventType.SESSION, {"event": "created", "title": state.title, "brief": state.brief[:500]}
        )
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

    def list_sessions(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """A page of sessions. The total count rides in the response header so the UI can paginate
        without breaking the existing bare-list contract."""
        rows = self.services.store.list_sessions(limit=limit, offset=offset)
        for row in rows:
            runtime = self.runtimes.get(row["id"])
            row["busy"] = bool(runtime and runtime.busy)
        return rows

    def session_count(self) -> int:
        return self.services.store.count_sessions()

    def edit_ledger(self, session_id: str, item_id: str, patch_data: dict[str, Any]) -> dict[str, Any] | None:
        """Edit one ledger entry in place (title, body, category, priority, acceptance).

        The merged item is re-validated through :class:`LedgerItem` rather than setattr'd field by field, so an
        invalid priority or a malformed acceptance list is rejected instead of silently corrupting the enum.
        The spec is regenerated lazily: the change is persisted and `spec_markdown` is cleared so the writer
        rebuilds the document from the corrected ledger on the next pass.

        If the run is already finished, the session is reopened at DELIVER. Assembly is deterministic (the
        LLM-written sections are cached in `state.sections`), so a post-delivery correction costs no model
        calls and the deliverables are rewritten from the corrected ledger.
        """
        state = self.get_state(session_id)
        if state is None:
            return None
        position = next((i for i, item in enumerate(state.ledger) if item.id == item_id), None)
        if position is None:
            return None
        merged = {**state.ledger[position].model_dump(), **{k: v for k, v in patch_data.items() if v is not None}}
        try:
            updated = LedgerItem.model_validate(merged)
        except ValidationError as exc:
            raise ValueError(f"invalid ledger edit: {exc.errors()[0]['msg']}") from exc
        was_finished = state.phase is Phase.DONE
        state.ledger[position] = updated
        state.spec_markdown = ""
        if was_finished:
            state.phase = Phase.DELIVER
            state.status = RunStatus.RUNNING
            state.awaiting = AwaitKind.NONE
        self.services.store.save_state(state)
        self.log_event(
            session_id,
            EventType.LEDGER,
            {"event": "edited", "id": updated.id, "title": updated.title, "reopen_deliverables": was_finished},
        )
        return updated.model_dump()

    def rename_session(self, session_id: str, title: str) -> bool:
        title = (title or "").strip()
        if not title:
            return False
        if self.get_state(session_id) is None:
            return False
        self.services.store.rename_session(session_id, title)
        runtime = self.runtimes.get(session_id)
        if runtime:
            runtime.state.title = title
        return True

    async def delete_session(self, session_id: str) -> bool:
        await self.cancel(session_id)
        self.runtimes.pop(session_id, None)
        removed = self.services.store.delete_session(session_id)
        if removed:
            # The database row is gone; the artefact folder must go with it, otherwise the outputs directory
            # accumulates orphans that no API can list or restore. Contained to outputs_root by construction.
            directory = self.services.config.session_dir(session_id)
            if directory.exists():
                shutil.rmtree(directory)
        return removed

    # ------------------------------------------------------------------ events

    def log_event(
        self, session_id: str, event_type: str, payload: dict[str, Any] | None = None, *, role: str = "", step: str = ""
    ) -> dict[str, Any]:
        payload = payload or {}
        if event_type in TRANSIENT_EVENTS:
            event = {
                "id": 0,
                "session_id": session_id,
                "ts": now_iso(),
                "type": event_type,
                "role": role,
                "step": step,
                "payload": payload,
            }
            self.fanout.publish(session_id, event)
            return event

        event_id = self.services.store.log_event(session_id, event_type, payload, role=role, step=step)
        event = {
            "id": event_id,
            "session_id": session_id,
            "ts": now_iso(),
            "type": event_type,
            "role": role,
            "step": step,
            "payload": payload,
        }
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

        runtime.task = asyncio.create_task(self._run(runtime, reason, **kwargs), name=f"ba-{session_id}-{reason}")
        return {"queued": True, "busy": True, "reason": reason}

    def _runtime(self, session_id: str) -> SessionRuntime:
        state = self.get_state(session_id)
        if state is None:
            raise KeyError(f"Unknown session '{session_id}'")
        return self.runtimes[session_id]

    async def _run(self, runtime: SessionRuntime, reason: str, **kwargs: Any) -> None:
        emit = self.emitter(runtime.session_id)
        if reason == "chat":
            # Deliberately outside runtime.lock: the chat tools call mutate(), which takes that same lock.
            await self._run_chat(runtime, str(kwargs.get("message", "")), emit)
            return
        async with runtime.lock:
            state = runtime.state
            try:
                if reason in ("answer",):
                    await self.engine.submit_answer(
                        state,
                        int(kwargs["question_id"]),
                        answer=kwargs.get("answer"),
                        skipped=bool(kwargs.get("skipped")),
                        not_applicable=bool(kwargs.get("not_applicable")),
                        emit=emit,
                    )
                    return
                if reason == "approve":
                    await self.engine.set_approval(
                        state, bool(kwargs.get("approved")), str(kwargs.get("notes", "")), emit=emit
                    )
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

    async def _run_chat(self, runtime: SessionRuntime, message: str, emit) -> None:
        """Background chat turn; a failure is reported in the event stream and never fails the session run."""
        try:
            await self._chat_step(runtime.state, message, emit)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            await emit(EventType.ERROR, {"message": f"chat failed: {type(exc).__name__}: {exc}", "reason": "chat"})

    async def cancel(self, session_id: str) -> bool:
        runtime = self.runtimes.get(session_id)
        if runtime and runtime.busy and runtime.task:
            runtime.task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await runtime.task
            return True
        return False

    async def answer(
        self,
        session_id: str,
        question_id: int,
        *,
        answer: str | None = None,
        skipped: bool = False,
        not_applicable: bool = False,
    ) -> dict[str, Any]:
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
        return await self._exclusive(
            session_id, "answer", question_id=question_id, answer=answer, skipped=skipped, not_applicable=not_applicable
        )

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
            return {
                "ok": False,
                "busy": True,
                "error": "the agent is working on something else — try again in a moment",
            }
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
        """Interactive Q&A with the BA while a run is in flight (tool-calling agent loop). Returns the reply."""
        runtime = self._runtime(session_id)
        return await self._chat_step(runtime.state, message, self.emitter(session_id))

    async def _chat_step(self, state: SessionState, message: str, emit) -> dict[str, Any]:
        """One chat turn: build the tool-using agent for this session, run it, and persist the outcome.

        Tools mutate the session through :meth:`mutate`, which takes ``runtime.lock``; callers must therefore not
        hold that lock when invoking this method.
        """
        session_id = state.session_id
        runtime = self._runtime(session_id)

        async def mutate(apply_state):
            async with runtime.lock:
                result = apply_state(runtime.state)
                self.services.store.save_state(runtime.state)
            return result

        ctx = ChatContext(
            session_id=session_id,
            get_state=lambda: runtime.state,
            mutate=mutate,
            knowledge=self.services.knowledge,
            retriever=self.services.retriever,
            sandbox=self.services.sandbox,
            store=self.services.store,
        )
        tools = build_chat_tools(ctx)

        system_prompt, _ = render_prompt(
            "chat_agent",
            project_name=(state.plan.project_name if state.plan else state.title),
            phase=state.phase.value,
            status=state.status.value,
            awaiting=state.awaiting.value,
            coverage_summary=", ".join(f"{s.name}: {s.status}" for s in state.subjects) or "(none)",
            requirement_count=len(state.requirements()),
        )

        self.log_event(session_id, EventType.ANSWER, {"channel": "chat", "answer": message})
        result = await run_agent(
            system_prompt=system_prompt,
            user_message=message,
            tools=tools,
            gateway=self.gateway,
            model=state.model_overrides.get("chat"),
            emit=emit,
            step="chat",
        )

        self.log_event(
            session_id,
            EventType.SESSION,
            {
                "event": "chat_reply",
                "text": result.text,
                "iterations": result.iterations,
                "tools": [t.name for t in result.tool_calls],
                "error": result.error,
            },
        )
        return {
            "ok": not result.error,
            "reply": result.text,
            "tool_calls": len(result.tool_calls),
            "model": result.model,
            "error": result.error,
        }

    # ------------------------------------------------------------------ model settings

    async def set_models(
        self,
        default_model: str | None = None,
        overrides: dict[str, str] | None = None,
        embedding_model: str | None = None,
        provider: str | None = None,
    ) -> dict[str, Any]:
        """Update the backend, the global default, per-role overrides, and/or the embedding model.

        ``overrides=None`` means *leave them alone* — otherwise picking a new default in the top bar would
        silently erase every per-role override the user had saved. ``embedding_model=""`` clears it, which is
        the honest way to say "this provider serves no embeddings, retrieval is BM25 only".

        Switching ``provider`` re-binds the gateway in place: a run already in progress keeps its session
        state, and the next step talks to the new backend without restarting the server.
        """
        if provider:
            name = provider.strip().lower()
            if name not in ("lmstudio", "strata"):
                raise ValueError(f"provider must be 'lmstudio' or 'strata', got {provider!r}")
            if name != self.config.provider:
                self.config.provider = name
                rebind = getattr(self.gateway, "rebind", None)
                if callable(rebind):
                    rebind()
                self.services.store.set_setting("provider", name)
        choices = _model_choices(self.services.store)
        bucket = choices.setdefault(self.config.provider, {})
        if default_model is not None:
            self.config.active_profile.default_model = default_model or None
            bucket["default_model"] = default_model
        if embedding_model is not None:
            self.config.active_profile.embedding_model = embedding_model or None
            bucket["embedding_model"] = embedding_model or ""
        if overrides is not None:
            clean = {k: v for k, v in overrides.items() if v}
            for role in sorted(set(list(self.config.roles) + list(clean))):
                params = self.config.roles.get(role) or self.config.role(role)
                params.model = clean.get(role) or None
                self.config.roles[role] = params
            bucket["overrides"] = clean
        self.services.store.set_setting("model_choices", choices)
        return {
            "default_model": self.config.active_profile.default_model,
            "embedding_model": self.config.active_profile.embedding_model,
            "overrides": {r: p.model for r, p in self.config.roles.items() if p.model},
        }

    def model_settings(self) -> dict[str, Any]:
        overrides = {role: params.model for role, params in self.config.roles.items() if params.model}
        stored_overrides = _model_choices(self.services.store).get(self.config.provider, {}).get("overrides", {})
        return {
            "default_model": self.config.active_profile.default_model,
            "provider": self.config.provider,
            "providers": ["lmstudio", "strata"],
            "base_url": self.config.active_profile.base_url,
            "overrides": {**stored_overrides, **overrides},
            "embedding_model": self.config.active_profile.embedding_model,
            "roles": sorted(self.config.roles),
        }

    # ------------------------------------------------------------------ run settings

    #: The knobs the UI may change, with the range each one accepts. Anything not listed is refused rather
    # than silently accepted, so a stray POST cannot turn a budget into a runaway (or empty) loop.
    SETTABLE: dict[str, tuple[str, ...]] = {
        "interview": (
            "max_questions_per_subject",
            "min_subjects",
            "max_subjects",
            "max_gap_rounds",
            "max_gap_questions_per_round",
            "max_summary_rejections",
            "max_total_questions",
            "max_transcript_chunks",
            "transcript_chunk_chars",
            "subject_coverage_threshold",
        ),
        "review": ("max_review_rounds", "min_functional_requirements", "require_acceptance_criteria"),
        "diagram": ("max_revision_rounds", "max_diagrams", "local_renderer", "render_timeout_seconds", "emit_png"),
        "knowledge": ("docs_dir", "chunk_size", "chunk_overlap", "heading_aware", "top_k", "min_score"),
    }
    #: Count-like knobs must stay in a sane range; booleans and scores are checked by their own types.
    #: Inclusive bounds for every numeric knob above. A budget of 0 max_subjects or a 10000-question
    #: ceiling is technically valid to the model but produces a useless run, so the UI cannot set it.
    BOUNDS: dict[str, tuple[float, float]] = {
        "max_questions_per_subject": (1, 5),
        "min_subjects": (1, 10),
        "max_subjects": (2, 25),
        "max_gap_rounds": (0, 5),
        "max_gap_questions_per_round": (1, 10),
        "max_summary_rejections": (0, 5),
        "max_total_questions": (5, 100),
        "max_transcript_chunks": (1, 100),
        "transcript_chunk_chars": (200, 5000),
        "subject_coverage_threshold": (1, 10),
        "max_review_rounds": (0, 5),
        "min_functional_requirements": (0, 50),
        "max_revision_rounds": (0, 5),
        "max_diagrams": (0, 10),
        "render_timeout_seconds": (10, 600),
        "chunk_size": (200, 4000),
        "chunk_overlap": (0, 1000),
        "top_k": (1, 20),
        "min_score": (0.0, 1.0),
    }

    def _validate_docs_dir(self, value: Any) -> None:
        """The knowledge directory must stay inside the project root — no escaping via `..` or absolute paths."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("'docs_dir' must be a non-empty path")
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            raise ValueError("'docs_dir' must be relative to the project root")
        root = Path(self.config.root).resolve()
        resolved = (root / candidate).resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError("'docs_dir' must be inside the project root")

    def _validate_setting(self, key: str, value: Any) -> None:
        bounds = self.BOUNDS.get(key)
        if bounds is None:
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"'{key}' must be a number")
        low, high = bounds
        if not low <= value <= high:
            raise ValueError(f"'{key}' must be between {low} and {high}")

    def run_settings(self) -> dict[str, dict[str, Any]]:
        """The current, effective budgets for every settable section."""
        return {
            section: {key: getattr(getattr(self.config, section), key) for key in keys}
            for section, keys in self.SETTABLE.items()
        }

    def update_run_settings(self, patch: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Apply a partial settings patch to the live config and persist it for the next restart.

        Precedence is config.yaml first, then this store: only the keys the user actually touched are stored,
        so untouched fields keep following config.yaml. Validation is per-key against the model that owns it;
        assigning ``max_subjects = 0`` would let a plan produce an empty agenda, so the whole patch is rejected
        before any field is written. The return value is the full section, which is what the UI renders.
        """
        applied: dict[str, dict[str, Any]] = {}
        persisted: dict[str, dict[str, Any]] = {}
        for section, values in (patch or {}).items():
            if section not in self.SETTABLE:
                raise ValueError(f"settings section '{section}' is not editable")
            if not isinstance(values, dict):
                raise ValueError(f"settings section '{section}' must be an object")
            unknown = [key for key in values if key not in self.SETTABLE[section]]
            if unknown:
                raise ValueError(f"settings keys not editable in '{section}': {', '.join(sorted(unknown))}")
            for key, value in values.items():
                self._validate_setting(key, value)
                if key == "docs_dir":
                    self._validate_docs_dir(value)
            target = getattr(self.config, section)
            merged = target.model_dump()
            merged.update(values)
            try:
                validated = type(target).model_validate(merged)
            except ValidationError as exc:
                raise ValueError(f"invalid {section} settings: {exc.errors()[0]['msg']}") from exc
            setattr(self.config, section, validated)
            applied[section] = {key: getattr(validated, key) for key in self.SETTABLE[section]}
            # Persist only what the user actually changed. Storing the whole section snapshots every other
            # field at its value *at this moment*, so a later config.yaml edit to an untouched field would
            # be permanently shadowed by this snapshot on the next restart.
            persisted[section] = values

        stored = self.services.store.get_setting("run_settings", {}) or {}
        for section, values in persisted.items():
            stored.setdefault(section, {}).update(values)
        self.services.store.set_setting("run_settings", stored)
        return applied

    def reset_run_settings(self) -> dict[str, dict[str, Any]]:
        """Clear every stored override and return the config to the values this process started with.

        This covers the model lanes as well as the budget sections: `default_model`, `embedding_model` and
        `model_overrides` live under their own store keys, so clearing only `run_settings` would leave them
        behind and `restore_model_choices()` would re-apply them at the next restart.
        """
        self.services.store.set_setting("run_settings", {})
        for section, values in self._baseline.items():
            setattr(self.config, section, type(getattr(self.config, section)).model_validate(values))

        if self._model_baseline.get("provider"):
            self.config.provider = self._model_baseline["provider"]
            rebind = getattr(self.gateway, "rebind", None)
            if callable(rebind):
                rebind()
        # Clear the whole per-provider map, not just the current bucket: a stale LM Studio lane must
        # not reappear when the user switches back to it after a Reset.
        self.services.store.set_setting("model_choices", {})
        self.config.active_profile.default_model = self._model_baseline["default_model"]
        self.config.active_profile.embedding_model = self._model_baseline["embedding_model"]
        for role, params in list(self.config.roles.items()):
            params.model = self._model_baseline["overrides"].get(role)
            self.config.roles[role] = params
        return self.run_settings()

    def restore_model_choices(self) -> None:
        """Re-apply saved model/embedding choices at startup, for every entry point, not just the server.

        Putting this in the lifespan alone means the CLI silently ignores what the UI saved.
        """
        apply_stored_model_choices(self.config, self.services.store)

    def restore_run_settings(self) -> None:
        """Re-apply persisted run settings at startup (config.yaml is the default, the store is the override).

        Validation is re-run through the section's own model rather than a bare `setattr`: Pydantic v2 does
        not validate on assignment, so a stale or hand-edited row in the settings table would otherwise land
        in the config as the wrong type and blow up the budget arithmetic that reads it.
        """
        self.restore_model_choices()
        apply_stored_run_settings(self.config, self.services.store, self.SETTABLE)

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
