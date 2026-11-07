"""HTTP layer — REST commands + a resumable SSE event stream + the static single-page UI.

The browser never waits on the model: commands are queued by :class:`~ba_agent.runtime.SessionManager` and progress
arrives over ``GET /api/sessions/{id}/stream`` (with ``Last-Event-ID`` replay from SQLite, so refreshes and network
hiccups lose nothing).
"""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__
from ..config import AppConfig, get_config
from ..events import EventFanout
from ..knowledge import KnowledgeService
from ..runtime import SessionManager
from ..services import Services
from ..spec import spec_stats
from ..tools.sandbox import SandboxError

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

#: One inline ``<svg>…</svg>`` block inside a served deliverable (non-greedy, case-insensitive, spans lines).
_SVG_BLOCK_RE = re.compile(r"<svg\b.*?</svg\s*>", re.IGNORECASE | re.DOTALL)

#: Events per keyset page when replaying a backlog over SSE/REST.
REPLAY_PAGE_SIZE = 1000
#: Hard cap on one upload request body and on one file within it.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
#: Largest number of files accepted in one upload request.
MAX_UPLOAD_FILES = 20
#: Content-Security-Policy for the served deliverable: same-origin resources only, no script from anywhere but the
#: page itself (its own inline script is needed for the Markdown view), no plugins, no framing, no form posts.
DELIVERABLE_CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
                   "img-src 'self' data:; object-src 'none'; base-uri 'none'; form-action 'none'; "
                   "frame-ancestors 'none'")


def _parse_cursor(value: Any) -> int:
    """A non-negative integer event cursor; anything else (missing, garbage, negative) means 'from the start'."""
    try:
        return max(0, int(str(value).strip())) if value not in (None, "") else 0
    except (TypeError, ValueError):
        return 0


# --------------------------------------------------------------------------- request models


class CreateSessionRequest(BaseModel):
    brief: str = Field(min_length=4, description="What the project is about, in the stakeholder's own words.")
    title: str = ""
    urls: list[str] = Field(default_factory=list, description="Optional reference pages to index before starting.")
    model_overrides: dict[str, str] = Field(default_factory=dict)


class AnswerRequest(BaseModel):
    question_id: int
    answer: str | None = None
    skipped: bool = False
    not_applicable: bool = False


class ApproveRequest(BaseModel):
    approved: bool
    notes: str = ""


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)


class ModelsRequest(BaseModel):
    default_model: str | None = None
    overrides: dict[str, str] | None = None


class IndexRequest(BaseModel):
    directory: str | None = None
    urls: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- app factory


def create_app(config: AppConfig | None = None) -> FastAPI:
    cfg = config or get_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        services = Services.create(cfg)
        manager = SessionManager(services)
        interrupted = manager.mark_interrupted()
        stored_overrides = services.store.get_setting("model_overrides", {}) or {}
        if stored_overrides:
            for role, model in stored_overrides.items():
                cfg.roles.setdefault(role, cfg.role(role)).model = model
        app.state.services = services
        app.state.manager = manager
        if interrupted:
            print(f"[ba-agent] {interrupted} interrupted run(s) marked resumable")
        try:
            yield
        finally:
            await manager.aclose()

    app = FastAPI(title="BA Agent (LangChain)", version=__version__, lifespan=lifespan)
    # CORS: allow only the explicitly configured origins. A wildcard is *not* honoured as "allow all" here — the
    # app is local-first and a credentialed wildcard would let any web page read the API. Cross-origin
    # state-changing requests are rejected separately by _require_same_origin (CSRF).
    cors_origins = [o for o in cfg.server.cors_origins if o != "*"]
    if cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_credentials=True,
                           allow_methods=["*"], allow_headers=["*"])

    def mgr(request: Request) -> SessionManager:
        return request.app.state.manager  # type: ignore[return-value]

    def _require_same_origin(request: Request) -> None:
        """Refuse state-changing requests whose ``Origin`` is not this server (CSRF defence, localhost-only).

        Browsers send ``Origin`` on cross-site form posts and fetches. Requests without an Origin (curl, the
        test client, same-origin GETs) are not browser CSRF, so they pass.
        """
        origin = request.headers.get("origin")
        if origin is None:
            return
        if origin not in _allowed_origins(request):
            raise HTTPException(status_code=403, detail="cross-origin request refused")

    def _allowed_origins(request: Request) -> set[str]:
        host = request.headers.get("host", "")
        allowed = {f"http://{host}", f"https://{host}"} if host else set()
        allowed.update(cors_origins)
        return allowed

    # -- meta ---------------------------------------------------------------

    @app.get("/api/health")
    async def health(request: Request) -> dict[str, Any]:
        manager = mgr(request)
        gateway_health = await manager.gateway.health()
        return {"status": "ok" if gateway_health["reachable"] else "backend_unreachable",
                "version": __version__, "gateway": gateway_health,
                "knowledge": manager.services.knowledge.status(),
                "config": {"max_questions_per_subject": cfg.interview.max_questions_per_subject,
                           "max_gap_rounds": cfg.interview.max_gap_rounds,
                           "max_review_rounds": cfg.review.max_review_rounds,
                           "max_concurrency": cfg.active_profile.max_concurrency}}

    @app.get("/api/models")
    async def models(request: Request) -> dict[str, Any]:
        manager = mgr(request)
        catalog = await manager.gateway.list_models(refresh=True)
        settings = manager.model_settings()
        return {"provider": cfg.provider, "base_url": cfg.active_profile.base_url, "reachable": bool(catalog),
                "chat_models": [m.model_dump() for m in catalog if m.is_chat],
                "embedding_models": [m.model_dump() for m in catalog if not m.is_chat],
                "default_model": settings["default_model"], "overrides": settings["overrides"],
                "roles": settings["roles"], "embedding_model": settings["embedding_model"]}

    @app.post("/api/models")
    async def set_models(request: Request, payload: ModelsRequest = Body(...)) -> dict[str, Any]:
        _require_same_origin(request)
        return await mgr(request).set_models(payload.default_model, payload.overrides)

    # -- sessions -----------------------------------------------------------

    @app.get("/api/sessions")
    async def list_sessions(request: Request, limit: int = 100) -> list[dict[str, Any]]:
        return mgr(request).list_sessions(limit=limit)

    @app.post("/api/sessions")
    async def create_session(request: Request, payload: CreateSessionRequest) -> dict[str, Any]:
        _require_same_origin(request)
        manager = mgr(request)
        indexed = None
        if payload.urls:
            indexed = await manager.services.knowledge.index_urls(payload.urls)
        state = await manager.create_session(brief=payload.brief, title=payload.title,
                                             model_overrides=payload.model_overrides)
        return {"session": _session_summary(manager, state.session_id), "indexed": indexed}

    @app.get("/api/sessions/{session_id}")
    async def get_session(request: Request, session_id: str) -> dict[str, Any]:
        manager = mgr(request)
        state = manager.get_state(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail="unknown session")
        return _session_summary(manager, session_id, detailed=True)

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(request: Request, session_id: str) -> dict[str, Any]:
        _require_same_origin(request)
        await mgr(request).delete_session(session_id)
        return {"ok": True}

    @app.post("/api/sessions/{session_id}/answer")
    async def answer(request: Request, session_id: str, payload: AnswerRequest) -> dict[str, Any]:
        _require_same_origin(request)
        try:
            return await mgr(request).answer(session_id, payload.question_id, answer=payload.answer,
                                             skipped=payload.skipped, not_applicable=payload.not_applicable)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/sessions/{session_id}/approve")
    async def approve(request: Request, session_id: str, payload: ApproveRequest) -> dict[str, Any]:
        _require_same_origin(request)
        try:
            return await mgr(request).approve(session_id, payload.approved, payload.notes)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/sessions/{session_id}/resume")
    async def resume(request: Request, session_id: str) -> dict[str, Any]:
        _require_same_origin(request)
        try:
            return await mgr(request).resume(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/sessions/{session_id}/cancel")
    async def cancel(request: Request, session_id: str) -> dict[str, Any]:
        _require_same_origin(request)
        stopped = await mgr(request).cancel(session_id)
        return {"ok": True, "stopped": stopped}

    @app.post("/api/sessions/{session_id}/chat")
    async def chat(request: Request, session_id: str, payload: ChatRequest) -> dict[str, Any]:
        _require_same_origin(request)
        try:
            return await mgr(request).chat(session_id, payload.message)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    # -- events -------------------------------------------------------------

    @app.get("/api/sessions/{session_id}/events")
    async def events(request: Request, session_id: str, after: int = 0, limit: int = 1000) -> list[dict[str, Any]]:
        return mgr(request).events_after(session_id, after_id=after, limit=limit)

    @app.get("/api/sessions/{session_id}/stream")
    async def stream(request: Request, session_id: str, after: int | None = Query(default=None)) -> StreamingResponse:
        manager = mgr(request)
        if manager.get_state(session_id) is None:
            raise HTTPException(status_code=404, detail="unknown session")

        # Resume point. The browser's EventSource sends Last-Event-ID automatically on reconnect; the UI also
        # puts a ``?after=`` on the URL, which can be stale. Honour whichever cursor is further ahead so neither
        # a stale query string nor a missing header causes events to be re-sent or skipped.
        start_after = max(_parse_cursor(after), _parse_cursor(request.headers.get("last-event-id")))
        queue = manager.fanout.subscribe(session_id)

        async def generator() -> AsyncIterator[str]:
            try:
                cursor = start_after
                while True:
                    # Keyset pages: a backlog of any size is replayed in order without silent truncation.
                    page = manager.events_after(session_id, after_id=cursor, limit=REPLAY_PAGE_SIZE)
                    for event in page:
                        yield _sse(event)
                    if len(page) < REPLAY_PAGE_SIZE:
                        break
                    cursor = int(page[-1]["id"])
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=20.0)
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    yield _sse(event)
            finally:
                manager.fanout.unsubscribe(session_id, queue)

        return StreamingResponse(generator(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no", "Connection": "keep-alive"})

    # -- knowledge ----------------------------------------------------------

    @app.get("/api/knowledge/status")
    async def knowledge_status(request: Request) -> dict[str, Any]:
        return mgr(request).services.knowledge.status()

    @app.post("/api/knowledge/index")
    async def knowledge_index(request: Request, payload: IndexRequest) -> dict[str, Any]:
        _require_same_origin(request)
        service = mgr(request).services.knowledge
        result: dict[str, Any] = {}
        if payload.directory or not payload.urls:
            result["directory"] = await service.index_directory(payload.directory)
        if payload.urls:
            result["urls"] = await service.index_urls(payload.urls)
        return result

    @app.post("/api/knowledge/upload")
    async def knowledge_upload(request: Request) -> dict[str, Any]:
        """Multipart upload. The body is read in chunks and rejected with 413 as soon as it exceeds the cap, so an
        oversized request never has to fit in memory. Each file is then validated by the knowledge service."""
        _require_same_origin(request)
        service = mgr(request).services.knowledge
        content_type = request.headers.get("content-type", "")
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES + 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"upload larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        if "multipart/form-data" not in content_type:
            raise HTTPException(status_code=415, detail="expected multipart/form-data")

        files = await _read_capped_uploads(request)
        outcomes = []
        for name, data in files:
            outcomes.append({"name": name, **(await service.index_upload(name or "upload.txt", data))})
        return {"results": outcomes, "status": service.status()}

    @app.post("/api/knowledge/search")
    async def knowledge_search(request: Request, payload: dict = Body(...)) -> list[dict[str, Any]]:
        _require_same_origin(request)
        query = str(payload.get("query", ""))
        results = await mgr(request).services.knowledge.search(query, k=int(payload.get("k", 5)))
        return [{"source": r.source, "title": r.title, "score": round(r.score, 4), "text": r.text[:600]}
                for r in results]

    # -- artifacts ----------------------------------------------------------

    @app.get("/api/artifacts/{session_id}/{name}")
    async def artifact(request: Request, session_id: str, name: str) -> Response:
        manager = mgr(request)
        if ".." in session_id or "\\" in session_id or "/" in session_id:
            raise HTTPException(status_code=400, detail="invalid session id")
        # Serve from this session's own output folder; the sandbox still has to approve the resolved path.
        target = manager.services.config.session_dir(session_id) / Path(name).name
        try:
            path = manager.services.sandbox.resolve(target, write=False)
        except (SandboxError, OSError) as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail="artifact not found")
        response: Response = FileResponse(path, filename=path.name)
        if path.suffix.lower() in (".html", ".htm"):
            # The deliverable runs model-influenced markup on the app's own origin: sanitise the embedded SVG and
            # lock the document down with a CSP, so even a missed payload cannot execute.
            body = path.read_text(encoding="utf-8", errors="replace")
            body = _sanitize_inline_svg(body)
            response = HTMLResponse(content=body)
            response.headers["Content-Security-Policy"] = DELIVERABLE_CSP
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Referrer-Policy"] = "no-referrer"
        return response

    # -- UI -----------------------------------------------------------------

    if WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    else:  # pragma: no cover - packaging safety net
        @app.get("/")
        async def index_missing() -> JSONResponse:
            return JSONResponse({"detail": f"web UI not found at {WEB_DIR}"}, status_code=404)

    return app


async def _read_capped_uploads(request: Request) -> list[tuple[str, bytes]]:
    """Parse a multipart body, stopping with 413 as soon as the running total exceeds the cap.

    Starlette's ``UploadFile`` would spool the whole request before any check runs; this consumes the raw body
    chunk by chunk and aborts early, so a hostile client can never make the server hold more than
    ``MAX_UPLOAD_BYTES`` (plus one chunk) of upload data. Returns ``(filename, bytes)`` per file part.
    """
    from starlette.datastructures import UploadFile as StarletteUploadFile
    from starlette.formparsers import MultiPartParser

    received = 0

    async def capped_stream():
        nonlocal received
        async for chunk in request.stream():
            received += len(chunk)
            if received > MAX_UPLOAD_BYTES + 1024 * 1024:  # +1 MB multipart overhead allowance
                raise HTTPException(status_code=413, detail=f"upload larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
            yield chunk

    parser = MultiPartParser(request.headers, capped_stream())
    try:
        form = await parser.parse()
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — malformed multipart is a client error, not a crash
        raise HTTPException(status_code=400, detail=f"could not parse upload: {exc}") from exc

    files: list[tuple[str, bytes]] = []
    for _, value in form.multi_items():
        if not isinstance(value, StarletteUploadFile) or value.filename is None:
            continue
        if len(files) >= MAX_UPLOAD_FILES:
            raise HTTPException(status_code=413, detail=f"at most {MAX_UPLOAD_FILES} files per upload")
        data = await value.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"file larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        files.append((value.filename, data))
    if not files:
        raise HTTPException(status_code=400, detail="no files in upload")
    return files


def _sanitize_inline_svg(html_text: str) -> str:
    """Replace each ``<svg>...</svg>`` block in a served deliverable with its sanitised form (or remove it)."""
    from ..sanitize import sanitize_svg

    def _swap(match: re.Match[str]) -> str:
        return sanitize_svg(match.group(0))

    return _SVG_BLOCK_RE.sub(_swap, html_text)


def _sse(event: dict[str, Any]) -> str:
    payload = {"id": event.get("id", 0), "ts": event.get("ts"), "type": event.get("type"),
               "role": event.get("role", ""), "step": event.get("step", ""), "payload": event.get("payload", {})}
    data = json.dumps(payload, ensure_ascii=False)
    return f"id: {event.get('id', 0)}\nevent: {event.get('type')}\ndata: {data}\n\n"


def _session_summary(manager: SessionManager, session_id: str, detailed: bool = False) -> dict[str, Any]:
    state = manager.get_state(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="unknown session")

    runtime = manager.runtimes.get(session_id)
    summary: dict[str, Any] = {
        "id": state.session_id,
        "title": state.title,
        "brief": state.brief,
        "phase": state.phase.value,
        "status": state.status.value,
        "awaiting": state.awaiting.value,
        "error": state.error,
        "created_at": state.created_at,
        "updated_at": state.updated_at,
        "busy": bool(runtime and runtime.busy),
        "project_name": state.plan.project_name if state.plan else state.title,
        "objective": state.plan.objective if state.plan else "",
        "subjects": [s.model_dump() for s in state.subjects],
        "pending_question": (state.current_question.model_dump() if state.current_question else None),
        "questions_total": len(state.questions),
        "requirements_total": len(state.requirements()),
        "metrics": state.metrics,
    }

    if detailed:
        summary.update({
            "plan": state.plan.model_dump() if state.plan else None,
            "questions": [q.model_dump() for q in state.questions],
            "ledger": [i.model_dump() for i in state.ledger],
            "summary_markdown": _summary_markdown(state),
            "spec_markdown": state.spec_markdown,
            "sections": state.sections,
            "diagrams": state.diagrams,
            "mermaid_code": state.mermaid_code,
            "open_issues": [i.model_dump() for i in state.open_issues],
            "lint_report": state.lint_report,
            "artifacts": manager.services.store.list_artifacts(session_id),
            "stats": spec_stats(state),
            "approved_summary": state.approved_summary,
            "review_rounds": state.review_rounds,
            "gap_rounds": state.gap_round,
        })
    return summary


def _summary_markdown(state) -> str:
    from ..context import render_summary_markdown

    return render_summary_markdown(state.summary)


# --------------------------------------------------------------------------- entry point


def run(config: AppConfig | None = None, *, host: str | None = None, port: int | None = None) -> None:
    import uvicorn

    cfg = config or get_config()
    uvicorn.run(create_app(cfg), host=host or cfg.server.host, port=int(port or cfg.server.port), log_level="info")
