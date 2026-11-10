"""HTTP + SSE wiring end to end, driven by the scripted gateway — no model, no VRAM.

These cover what the unit tests cannot: that commands are accepted, that progress reaches the browser as typed
events with a replayable cursor, and that artefacts are downloadable through the same API the UI uses.

The app runs on a real uvicorn socket (not an in-process transport) because the UI talks SSE over HTTP and
``TestClient``'s portal does not exercise streaming the way a browser does.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import uvicorn

from ba_agent import server as server_package
from ba_agent.knowledge import KnowledgeService
from ba_agent.server.app import create_app
from ba_agent.services import Services
from conftest import FakeGateway

BRIEF = (
    "We need a portal for insurance brokers to submit and track claims. Today everything is email. "
    "About 300 brokers, peak in winter, must integrate with the existing policy system over REST."
)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture()
def api(config, monkeypatch):
    """A live server whose gateway is the scripted one (the real one would need LM Studio + VRAM)."""
    real_create = Services.create
    gateway = FakeGateway()

    def create_with_fake(cfg=None):
        services = real_create(cfg or config)
        services.gateway = gateway  # type: ignore[assignment]
        services.knowledge = KnowledgeService(services.store, gateway, config)  # type: ignore[arg-type]
        return services

    monkeypatch.setattr(server_package.app, "Services", SimpleNamespace(create=create_with_fake))

    app = create_app(config)
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", lifespan="on", access_log=False)
    )
    thread = threading.Thread(target=server.run, name="ba-test-server", daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    with httpx.Client(base_url=base_url, timeout=20.0) as client:
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                if client.get("/api/health").status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.05)
        else:  # pragma: no cover - defensive
            raise AssertionError("test server did not come up")

        client.ba_gateway = gateway  # type: ignore[attr-defined]
        client.ba_manager = app.state.manager  # type: ignore[attr-defined]
        yield client

    server.should_exit = True
    thread.join(timeout=10)


def _get(client: httpx.Client, url: str) -> httpx.Response:
    """GET with a short retry. The app is a real uvicorn socket, and on Windows a busy socket can
    reset (WinError 10054) mid-suite; a transient reset is not a product failure."""
    for attempt in range(5):
        try:
            return client.get(url)
        except httpx.HTTPError:
            if attempt == 4:
                raise
            time.sleep(0.1)
    raise AssertionError(f"unreachable: {url}")  # pragma: no cover


def wait_until(
    client: httpx.Client, session_id: str, predicate, *, timeout: float = 25.0, what: str = "condition"
) -> dict:
    """Poll the session endpoint until *predicate* holds (the run happens in a background task)."""
    deadline = time.time() + timeout
    last: dict = {}
    while time.time() < deadline:
        response = _get(client, f"/api/sessions/{session_id}")
        assert response.status_code == 200, response.text
        last = response.json()
        if predicate(last):
            return last
        time.sleep(0.05)
    raise AssertionError(
        f"timed out waiting for {what}; last state: {last.get('phase')}/{last.get('awaiting')} "
        f"error={last.get('error')}"
    )


def create_session(client: httpx.Client, brief: str = BRIEF) -> str:
    response = client.post("/api/sessions", json={"brief": brief})
    assert response.status_code == 200, response.text
    return response.json()["session"]["id"]


# --------------------------------------------------------------------------- meta endpoints


def test_health_reports_gateway_and_config(api):
    payload = api.get("/api/health").json()

    assert payload["status"] == "ok"
    assert payload["config"]["max_questions_per_subject"] == 2
    assert "knowledge" in payload


def test_models_endpoint_separates_chat_and_embedding_models(config, monkeypatch):
    """The dropdown must work while LM Studio is idle, so it cannot depend on a loaded model."""
    from ba_agent.models import ModelInfo

    class CatalogOnlyGateway(FakeGateway):
        async def list_models(self, refresh: bool = False):
            return [
                ModelInfo(id="loaded-one", state="loaded", capabilities=["tool_use"]),
                ModelInfo(id="idle-one", state="not-loaded"),
                ModelInfo(id="text-embedding-x", type="embeddings"),
            ]

    real_create = Services.create

    def create_with_fake(cfg=None):
        services = real_create(cfg or config)
        services.gateway = CatalogOnlyGateway()  # type: ignore[assignment]
        return services

    monkeypatch.setattr(server_package.app, "Services", SimpleNamespace(create=create_with_fake))
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(config), host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                payload = httpx.get(f"http://127.0.0.1:{port}/api/models", timeout=5).json()
                if payload.get("chat_models"):
                    break
            except (httpx.HTTPError, KeyError):
                time.sleep(0.05)
        else:  # pragma: no cover - defensive
            raise AssertionError("catalog endpoint never became ready")
    finally:
        server.should_exit = True
        thread.join(timeout=10)

    assert [m["id"] for m in payload["chat_models"]] == ["loaded-one", "idle-one"]
    assert [m["id"] for m in payload["embedding_models"]] == ["text-embedding-x"]
    assert payload["reachable"] is True


def test_model_selection_is_persisted(api):
    response = api.post("/api/models", json={"default_model": "some-model", "overrides": {"writer": "other"}})

    assert response.status_code == 200
    assert response.json()["default_model"] == "some-model"
    assert api.get("/api/models").json()["overrides"]["writer"] == "other"


def test_changing_only_the_default_keeps_role_overrides(api):
    """Picking a new default in the top bar must not wipe per-role overrides."""
    api.post("/api/models", json={"default_model": "first", "overrides": {"writer": "big-writer"}})

    api.post("/api/models", json={"default_model": "second"})

    assert api.get("/api/models").json()["overrides"]["writer"] == "big-writer"


def test_session_model_choice_reaches_every_role(api):
    """The model chosen when creating an interview has to be the one actually called."""
    response = api.post("/api/sessions", json={"brief": BRIEF, "model_overrides": {"default": "chosen-model"}})
    session_id = response.json()["session"]["id"]

    wait_until(api, session_id, lambda s: s["pending_question"] is not None)

    routed = api.get(f"/api/sessions/{session_id}")  # ensure the run settled before reading the gateway
    assert routed.status_code == 200
    calls = api.ba_gateway.routed  # type: ignore[attr-defined]
    assert calls, "no model call was made"
    assert {override for _, _, override in calls} == {"chosen-model"}


def test_per_role_override_beats_the_session_default(api):
    response = api.post(
        "/api/sessions", json={"brief": BRIEF, "model_overrides": {"default": "session-model", "plan": "planner-model"}}
    )
    session_id = response.json()["session"]["id"]

    wait_until(api, session_id, lambda s: s["pending_question"] is not None)

    routed = api.ba_gateway.routed  # type: ignore[attr-defined]
    plan_calls = [override for role, _, override in routed if role == "plan"]
    question_calls = [
        override for role, step, override in routed if role == "interview" and step.startswith("question")
    ]

    assert plan_calls == ["planner-model"]
    assert question_calls == ["session-model"]


# --------------------------------------------------------------------------- session lifecycle


def test_unknown_session_is_404(api):
    assert api.get("/api/sessions/nope").status_code == 404
    assert api.get("/api/sessions/nope/stream").status_code == 404
    assert api.post("/api/sessions/nope/answer", json={"question_id": 1, "answer": "x"}).status_code == 404


def test_interview_over_http_reaches_done_with_downloadable_artifacts(api):
    session_id = create_session(api)

    waiting = wait_until(api, session_id, lambda s: s["pending_question"] is not None, what="first question")
    assert waiting["phase"] == "interview"
    assert waiting["pending_question"]["subject"]

    # Answer every question the engine asks, then approve the summary — like the UI would.
    answered = 0
    for _ in range(40):
        state = wait_until(api, session_id, lambda s: not s["busy"])
        if state["phase"] == "done":
            break
        if state["awaiting"] == "answer" and state["pending_question"]:
            response = api.post(
                f"/api/sessions/{session_id}/answer",
                json={"question_id": state["pending_question"]["id"], "answer": "Around 400 claims per month."},
            )
            assert response.status_code == 200, response.text
            answered += 1
        elif state["awaiting"] == "approval":
            assert api.post(f"/api/sessions/{session_id}/approve", json={"approved": True}).status_code == 200
        else:
            raise AssertionError(f"stalled in {state['phase']} / {state['awaiting']} error={state['error']}")

    done = wait_until(api, session_id, lambda s: s["phase"] == "done", what="completion")
    assert answered >= 1
    assert done["requirements_total"] >= 1

    detail = api.get(f"/api/sessions/{session_id}").json()
    names = {artifact["name"] for artifact in detail["artifacts"]}
    assert {"functional_specification.md", "interview.jsonl", "deliverable.html", "metadata.json"} <= names
    assert detail["spec_markdown"].strip()
    assert detail["mermaid_code"].startswith("flowchart")

    download = api.get(f"/api/artifacts/{session_id}/functional_specification.md")
    assert download.status_code == 200
    assert "Functional Specification" in download.text


def test_answering_a_question_twice_is_not_accepted(api):
    session_id = create_session(api)
    state = wait_until(api, session_id, lambda s: s["pending_question"] is not None)
    question_id = state["pending_question"]["id"]

    assert (
        api.post(f"/api/sessions/{session_id}/answer", json={"question_id": question_id, "answer": "first"}).status_code
        == 200
    )
    wait_until(api, session_id, lambda s: not s["busy"])

    again = api.post(f"/api/sessions/{session_id}/answer", json={"question_id": question_id, "answer": "again"})
    assert again.status_code in (200, 409)
    if again.status_code == 200:
        assert again.json().get("ok") is not True or again.json().get("error")


def test_blank_answer_is_rejected(api):
    session_id = create_session(api)
    state = wait_until(api, session_id, lambda s: s["pending_question"] is not None)

    response = api.post(
        f"/api/sessions/{session_id}/answer", json={"question_id": state["pending_question"]["id"], "answer": "   "}
    )

    assert response.status_code == 409


def test_session_can_be_cancelled_and_listed(api):
    session_id = create_session(api)
    wait_until(api, session_id, lambda s: s["pending_question"] is not None)

    assert api.post(f"/api/sessions/{session_id}/cancel").status_code == 200
    listed = api.get("/api/sessions").json()

    assert [row["id"] for row in listed] == [session_id]
    assert "brief" in listed[0]


# --------------------------------------------------------------------------- events / SSE


def test_event_replay_is_ordered_and_cursorable_over_http(api):
    session_id = create_session(api)
    wait_until(api, session_id, lambda s: s["pending_question"] is not None)

    events = api.get(f"/api/sessions/{session_id}/events").json()
    ids = [event["id"] for event in events]

    assert ids == sorted(ids)
    assert any(event["type"] == "question" for event in events)

    cursor = ids[len(ids) // 2]
    tail = api.get(f"/api/sessions/{session_id}/events", params={"after": cursor}).json()
    assert all(event["id"] > cursor for event in tail)


def test_sse_stream_replays_events_with_ids(api):
    session_id = create_session(api)
    wait_until(api, session_id, lambda s: s["pending_question"] is not None)

    lines = _collect_stream(api, session_id, want="question")

    assert any(line.startswith("id: ") for line in lines)
    assert any(line == "event: question" for line in lines)


def test_sse_resumes_from_last_event_id_header(api):
    session_id = create_session(api)
    wait_until(api, session_id, lambda s: s["pending_question"] is not None)
    all_events = api.get(f"/api/sessions/{session_id}/events").json()
    first_id = all_events[0]["id"]

    with api.stream(
        "GET", f"/api/sessions/{session_id}/stream", headers={"Last-Event-ID": str(first_id)}, timeout=20
    ) as stream:
        assert stream.status_code == 200
        first_data = ""
        for line in stream.iter_lines():
            if line.startswith("data: "):
                first_data = line
                break

    assert first_data, "expected the stream to replay what came after Last-Event-ID"
    assert json.loads(first_data[len("data: ") :])["id"] > first_id


def test_sse_replay_does_not_double_deliver_events_published_during_replay(api):
    """The stream subscribes to the live fanout *before* reading the backlog.

    Anything emitted between those two steps is in the DB page and in the queue, so a naive
    generator sends it twice. A client reconnecting mid-run is exactly this situation.
    """
    session_id = create_session(api)
    wait_until(api, session_id, lambda s: s["pending_question"] is not None)
    manager = api.ba_manager

    real_after = manager.events_after
    injected: dict = {}

    def events_after_during_replay(sid: str, after_id: int = 0, limit: int = 2000):
        # Published before the DB read: it is in the page *and* already sitting in the queue.
        injected["id"] = manager.log_event(sid, "log", {"message": "emitted during replay"})["id"]
        return real_after(sid, after_id, limit)

    manager.events_after = events_after_during_replay  # type: ignore[assignment]
    try:
        seen_injected = 0
        with api.stream("GET", f"/api/sessions/{session_id}/stream?after=0", timeout=30) as stream:
            for line in stream.iter_lines():
                # A keep-alive comment means the live queue has been drained: every event the
                # generator could deliver has been delivered by this point.
                if line.startswith(": keep-alive"):
                    break
                if line.startswith("data: ") and json.loads(line[len("data: ") :])["id"] == injected["id"]:
                    seen_injected += 1
                if seen_injected > 1:
                    break
    finally:
        manager.events_after = real_after  # type: ignore[assignment]

    assert seen_injected == 1, f"event {injected['id']} delivered {seen_injected} times"


def test_sse_transient_frames_do_not_clobber_the_browser_cursor():
    """A streamed token must not rewrite the client's resume cursor.

    Per the SSE spec, any frame carrying an `id:` line sets `Last-Event-ID`. Transient events
    (TOKEN/REASONING) are never persisted, so they carry `id: 0` — meaning a browser that reconnects
    after receiving one would resume from 0 and replay the whole log. Transient frames therefore omit
    the `id:` line entirely, leaving the cursor at the last persisted event.
    """
    from ba_agent.server.app import _sse

    persisted = _sse({"id": 42, "type": "question", "ts": "2026-01-01T00:00:00Z", "payload": {}})
    transient = _sse({"id": 0, "type": "token", "ts": "2026-01-01T00:00:01Z", "payload": {"text": "hi"}})

    assert persisted.splitlines()[0] == "id: 42"
    assert not transient.splitlines()[0].startswith("id:"), (
        f"transient frame emits an id line: {transient.splitlines()[0]!r}"
    )
    # The payload still carries the (zero) id so the UI can tell tokens apart from real events.
    assert '"id": 0' in transient


def _collect_stream(client: httpx.Client, session_id: str, want: str, limit: int = 400) -> list[str]:
    """Read the SSE stream until an event of type *want* appears (bounded: never hangs the suite)."""
    lines: list[str] = []
    with client.stream("GET", f"/api/sessions/{session_id}/stream?after=0", timeout=20) as stream:
        assert stream.status_code == 200
        assert stream.headers["content-type"].startswith("text/event-stream")
        for line in stream.iter_lines():
            if not line:
                continue
            lines.append(line)
            if line == f"event: {want}":
                break
            if len(lines) > limit:  # pragma: no cover - defensive
                pytest.fail(f"no '{want}' event arrived; saw: {lines[:20]}")
    return lines


# --------------------------------------------------------------------------- knowledge + UI


def test_knowledge_upload_and_search_over_http(api):
    content = (
        "# Interview guide\n\nAsk about GDPR, latency budgets and availability targets for the claims "
        "portal. Repeat per stakeholder group.\n\n" + "Unrelated filler. " * 60
    )

    upload = api.post("/api/knowledge/upload", files={"files": ("guide.md", content.encode("utf-8"), "text/markdown")})
    assert upload.status_code == 200, upload.text
    assert upload.json()["results"][0]["ok"] is True

    hits = api.post("/api/knowledge/search", json={"query": "GDPR latency availability", "k": 3}).json()
    assert hits and "gdpr" in hits[0]["text"].lower()
    # Every hit must name the signal that matched, so a run that silently degraded to keyword-only is
    # distinguishable from a hybrid one.
    assert all(h["mode"] in {"bm25", "vector", "bm25+vector"} for h in hits), [h.get("mode") for h in hits]

    status = api.get("/api/knowledge/status").json()
    assert status["chunks"] > 0
    assert isinstance(status["document_count"], int) and status["document_count"] >= 1
    assert status["vector_search_reason"], "the status must explain why vector search is off"


def test_health_reports_the_pdf_engine_the_box_can_use(api):
    """The UI must not promise a Typst PDF on a machine with no Typst binary."""
    pdf = api.get("/api/health").json()["pdf"]
    assert pdf["engine"] in {"typst", "reportlab", "none"}
    if pdf["engine"] == "typst":
        assert Path(pdf["typst_binary"]).is_file()
    else:
        assert pdf["typst_binary"] is None


def test_ui_is_served_with_its_assets(api):
    page = api.get("/")
    assert page.status_code == 200 and "model" in page.text.lower()

    for asset in ("/styles.css", "/app.js", "/vendor/marked.min.js", "/vendor/mermaid.min.js"):
        response = api.get(asset)
        assert response.status_code == 200, f"{asset} -> {response.status_code}"
        assert len(response.content) > 100, f"{asset} looks empty"
