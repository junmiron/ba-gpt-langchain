"""Regression tests for the server / persistence / security defects S1-S9 (see docs/REDESIGN_PLAN.md §1.2).

Everything here is offline: a scripted gateway, a loopback server, and synthetic events. No model, no network.
"""

from __future__ import annotations

import asyncio
import json
import socket
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import uvicorn

from ba_agent import server as server_package
from ba_agent.events import DEFAULT_FANOUT_CAPACITY, EventFanout, EventType
from ba_agent.jsonx import extract_json
from ba_agent.knowledge import MAX_FETCH_BYTES, KnowledgeService, UnsafeURLError, check_fetch_url, fetch_url
from ba_agent.sanitize import sanitize_svg
from ba_agent.server.app import DELIVERABLE_CSP, REPLAY_PAGE_SIZE, create_app
from ba_agent.services import Services
from ba_agent.store import SCHEMA_VERSION, Store
from conftest import FakeGateway

# --------------------------------------------------------------------------- helpers


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture()
def live(config, monkeypatch):
    """A running server on loopback with the scripted gateway. Yields ``(client, manager)``."""
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
    thread = threading.Thread(target=server.run, name="ba-fix-server", daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    with httpx.Client(base_url=base, timeout=20.0) as client:
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                if client.get("/api/health").status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.05)
        else:  # pragma: no cover - defensive
            raise AssertionError("test server did not come up")
        yield client, app.state.manager
    server.should_exit = True
    thread.join(timeout=10)


def _seed_events(store: Store, session_id: str, count: int) -> None:
    for index in range(count):
        store.log_event(session_id, EventType.LOG, {"n": index})


def _event_ids(sse_text: str) -> list[int]:
    ids: list[int] = []
    for line in sse_text.splitlines():
        if line.startswith("id: "):
            ids.append(int(line[4:]))
    return ids


# --------------------------------------------------------------------------- S1: SSE replay


def test_s1_replay_sends_every_event_of_a_2000_event_backlog(live):
    """The old single query stopped at 1500; a 2000-event session must replay completely, in order."""
    client, manager = live
    state = manager.services.store.create_session(brief="backlog replay")
    _seed_events(manager.services.store, state.session_id, 2000)

    with client.stream("GET", f"/api/sessions/{state.session_id}/stream?after=0") as response:
        assert response.status_code == 200
        ids: list[int] = []
        for line in response.iter_lines():
            if line.startswith("id: "):
                ids.append(int(line[4:]))
            if len(ids) >= 2000:
                break

    assert len(ids) == 2000
    assert ids == sorted(ids)
    assert len(set(ids)) == 2000
    assert REPLAY_PAGE_SIZE < 2000  # the replay really had to page


def test_s1_last_event_id_header_is_honoured_when_no_query_cursor(live):
    client, manager = live
    state = manager.services.store.create_session(brief="header cursor")
    _seed_events(manager.services.store, state.session_id, 10)
    all_ids = [e["id"] for e in manager.events_after(state.session_id, 0, 100)]
    cursor = all_ids[4]

    with client.stream(
        "GET", f"/api/sessions/{state.session_id}/stream", headers={"Last-Event-ID": str(cursor)}
    ) as response:
        seen: list[int] = []
        for line in response.iter_lines():
            if line.startswith("id: "):
                seen.append(int(line[4:]))
            if len(seen) >= 5:
                break
    assert seen == all_ids[5:10]


def test_s1_stale_query_cursor_does_not_override_a_newer_last_event_id(live):
    """The UI always sends ``?after=`` (possibly stale). A newer ``Last-Event-ID`` must win, not be ignored."""
    client, manager = live
    state = manager.services.store.create_session(brief="stale query")
    _seed_events(manager.services.store, state.session_id, 12)
    all_ids = [e["id"] for e in manager.events_after(state.session_id, 0, 100)]

    with client.stream(
        "GET", f"/api/sessions/{state.session_id}/stream?after=0", headers={"Last-Event-ID": str(all_ids[7])}
    ) as response:
        seen: list[int] = []
        for line in response.iter_lines():
            if line.startswith("id: "):
                seen.append(int(line[4:]))
            if len(seen) >= 4:
                break
    assert seen == all_ids[8:12]


def test_s1_malformed_cursor_is_treated_as_from_start(live):
    client, manager = live
    state = manager.services.store.create_session(brief="garbage cursor")
    _seed_events(manager.services.store, state.session_id, 3)
    with client.stream(
        "GET", f"/api/sessions/{state.session_id}/stream", headers={"Last-Event-ID": "not-a-number"}
    ) as response:
        assert response.status_code == 200
        seen = []
        for line in response.iter_lines():
            if line.startswith("id: "):
                seen.append(line)
            if len(seen) >= 3:
                break
    assert len(seen) == 3


# --------------------------------------------------------------------------- S2: fanout overflow


def test_s2_overflow_drops_oldest_and_reports_it_in_band():
    async def scenario() -> list[dict]:
        fanout = EventFanout(maxlen=8)
        queue = fanout.subscribe("s")
        for index in range(20):
            fanout.publish("s", {"id": index + 1, "type": "log", "payload": {}})
        drained: list[dict] = []
        while not queue.empty():
            drained.append(queue.get_nowait())
        return drained

    drained = asyncio.run(scenario())
    warnings = [e for e in drained if e["type"] == EventType.WARNING]
    assert warnings, "a silent drop is a defect: the subscriber must be told"
    assert warnings[0]["payload"]["dropped"] > 0
    assert warnings[0]["payload"]["source"] == "fanout"
    # The most recent event always survives an overflow.
    assert drained[-1]["id"] == 20


def test_s2_default_capacity_absorbs_a_full_review_burst_without_loss():
    assert DEFAULT_FANOUT_CAPACITY >= 2000

    async def scenario() -> int:
        fanout = EventFanout()
        queue = fanout.subscribe("s")
        for index in range(1500):
            fanout.publish("s", {"id": index + 1, "type": "log", "payload": {}})
        return queue.qsize()

    assert asyncio.run(scenario()) == 1500


# --------------------------------------------------------------------------- S3: chat step


def test_s3_chat_step_is_implemented_and_runs_through_the_background_path(services, config):
    from ba_agent.runtime import SessionManager

    manager = SessionManager(services)
    state = services.store.create_session(brief="chat step", title="chat")

    async def scenario():
        from ba_agent.runtime import SessionRuntime

        manager.runtimes[state.session_id] = SessionRuntime(session_id=state.session_id, state=state)
        emitted: list[str] = []

        async def emit(event_type, payload=None):
            emitted.append(event_type)

        # The background entry point used by _run("chat") must exist and must not deadlock on runtime.lock.
        await asyncio.wait_for(manager._run_chat(manager.runtimes[state.session_id], "hello there", emit), timeout=20)
        return emitted

    asyncio.run(scenario())
    assert hasattr(manager, "_chat_step")
    kinds = [e["type"] for e in manager.services.store.get_events(state.session_id, 0, 100)]
    assert EventType.ANSWER in kinds and EventType.SESSION in kinds


def test_s3_scheduled_chat_does_not_raise_attribute_error(services):
    from ba_agent.runtime import SessionManager

    manager = SessionManager(services)
    state = services.store.create_session(brief="scheduled chat")
    manager.get_state(state.session_id)

    async def scenario():
        await manager.schedule(state.session_id, "chat", message="status?")
        await asyncio.sleep(0.5)
        task = manager.runtimes[state.session_id].task
        if task is not None:
            await asyncio.wait_for(task, timeout=20)

    asyncio.run(scenario())
    events = services.store.get_events(state.session_id, 0, 100)
    assert not any(e["type"] == EventType.ERROR and "_chat_step" in json.dumps(e["payload"]) for e in events)


# --------------------------------------------------------------------------- S4: CORS + CSRF


def test_s4_cancel_from_a_foreign_origin_is_refused(live):
    client, manager = live
    state = manager.services.store.create_session(brief="csrf probe")
    response = client.post(f"/api/sessions/{state.session_id}/cancel", headers={"Origin": "https://evil.example"})
    assert response.status_code == 403


# Each body is *valid* for its route, so the origin guard (not body validation) is what decides the outcome.
@pytest.mark.parametrize(
    "path,body",
    [
        ("/api/sessions/x/resume", None),
        ("/api/sessions/x/answer", {"question_id": 1, "answer": "yes"}),
        ("/api/sessions/x/approve", {"approved": True}),
        ("/api/sessions/x/chat", {"message": "hello"}),
        ("/api/knowledge/index", {"urls": []}),
        ("/api/models", {"default_model": "m"}),
        ("/api/sessions", {"brief": "a valid brief"}),
    ],
)
def test_s4_every_state_changing_route_refuses_a_foreign_origin(live, path, body):
    client, _ = live
    response = client.post(path, json=body, headers={"Origin": "https://evil.example"})
    assert response.status_code == 403, f"{path} accepted a cross-origin request ({response.status_code})"


def test_s4_upload_from_a_foreign_origin_is_refused(live):
    client, _ = live
    response = client.post(
        "/api/knowledge/upload",
        files={"files": ("a.md", b"x", "text/markdown")},
        headers={"Origin": "https://evil.example"},
    )
    assert response.status_code == 403


def test_s4_same_origin_and_originless_requests_still_work(live):
    client, _ = live
    own = client.post(
        "/api/sessions", json={"brief": "same origin ok"}, headers={"Origin": str(client.base_url).rstrip("/")}
    )
    assert own.status_code == 200, own.text
    plain = client.post("/api/sessions", json={"brief": "no origin header"})
    assert plain.status_code == 200, plain.text


def test_s4_cors_wildcard_is_not_honoured_as_allow_all(config):
    """The original condition inverted the check: a wildcard *disabled* CORS. Now a wildcard never enables it."""
    config.server.cors_origins = ["*"]
    app = create_app(config)
    assert not any("CORS" in type(m.cls).__name__ for m in app.user_middleware)


def test_s4_explicit_origin_gets_cors_headers(config):
    """Behavioural check: a preflight from an allowed origin is answered; a foreign one is not."""
    config.server.cors_origins = ["http://127.0.0.1:8765"]
    app = create_app(config)

    async def preflight(origin: str) -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8765") as client:
            return await client.options(
                "/api/health", headers={"Origin": origin, "Access-Control-Request-Method": "GET"}
            )

    allowed = asyncio.run(preflight("http://127.0.0.1:8765"))
    foreign = asyncio.run(preflight("https://evil.example"))
    assert allowed.headers.get("access-control-allow-origin") == "http://127.0.0.1:8765"
    assert "access-control-allow-origin" not in foreign.headers


# --------------------------------------------------------------------------- S5: uploads


def test_s5_upload_over_the_cap_is_rejected_with_413(live):
    client, _ = live
    big = b"a" * (26 * 1024 * 1024)
    response = client.post("/api/knowledge/upload", files={"files": ("big.md", big, "text/markdown")})
    assert response.status_code == 413


def test_s5_unallowed_extension_is_refused_without_touching_disk(config, services):
    import asyncio as _a

    target = config.root / config.knowledge.docs_dir
    before = set(target.glob("*")) if target.exists() else set()
    outcome = _a.run(services.knowledge.index_upload("payload.exe", b"MZ\x00\x00"))
    assert outcome["ok"] is False and "not allowed" in outcome["error"]
    after = set(target.glob("*")) if target.exists() else set()
    assert after == before


def test_s5_same_content_under_another_name_is_deduplicated(config, services):
    import asyncio as _a

    first = _a.run(services.knowledge.index_upload("guide.md", b"# Guide\n\nSame content for both uploads."))
    second = _a.run(services.knowledge.index_upload("renamed.md", b"# Guide\n\nSame content for both uploads."))
    assert first["ok"] is True and not first.get("duplicate")
    assert second["ok"] is True and second.get("duplicate") is True
    docs = services.store.kb_docs()
    assert len(docs) == 1


def test_s5_distinct_content_is_stored_separately(config, services):
    import asyncio as _a

    _a.run(services.knowledge.index_upload("a.md", b"# A\n\nFirst document body."))
    _a.run(services.knowledge.index_upload("b.md", b"# B\n\nSecond document body."))
    assert len(services.store.kb_docs()) == 2


def test_s5_multipart_upload_over_http_is_stored_and_listed(live):
    client, _manager = live
    response = client.post("/api/knowledge/upload", files={"files": ("notes.txt", b"alpha beta gamma", "text/plain")})
    assert response.status_code == 200, response.text
    assert response.json()["results"][0]["ok"] is True


# --------------------------------------------------------------------------- S6: SSRF guard


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",
        "http://127.0.0.1:8765/api/health",
        "http://localhost/secret",
        "http://10.0.0.5/internal",
        "http://192.168.1.10/router",
        "http://172.16.4.2/x",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/x",
        "http://0.0.0.0/x",
        "file:///etc/passwd",
        "ftp://example.com/x",
        "gopher://example.com/x",
    ],
)
def test_s6_private_loopback_and_non_http_urls_are_rejected(url):
    with pytest.raises(UnsafeURLError):
        check_fetch_url(url, resolver=lambda host: ["93.184.216.34"])


def test_s6_hostname_resolving_to_a_private_address_is_rejected():
    with pytest.raises(UnsafeURLError):
        check_fetch_url("https://intranet.example/page", resolver=lambda host: ["10.1.2.3"])


def test_s6_hostname_with_mixed_public_and_private_answers_is_rejected():
    with pytest.raises(UnsafeURLError):
        check_fetch_url("https://mixed.example/", resolver=lambda host: ["93.184.216.34", "127.0.0.1"])


def test_s6_public_host_is_accepted():
    check_fetch_url("https://example.com/docs", resolver=lambda host: ["93.184.216.34"])


def test_s6_redirect_into_a_private_address_is_blocked(monkeypatch):
    """A public URL that redirects to 127.0.0.1 must not be followed. Uses a real local redirect server."""
    import ba_agent.knowledge as knowledge

    class FakeTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            if request.url.host == "public.example":
                return httpx.Response(302, headers={"location": "http://127.0.0.1:9/secret"}, request=request)
            return httpx.Response(200, text="<html><body>secret</body></html>", request=request)

    real_client = httpx.AsyncClient

    def client_with_fake(**kwargs):
        kwargs["transport"] = FakeTransport()
        return real_client(**kwargs)

    monkeypatch.setattr(knowledge.httpx, "AsyncClient", client_with_fake)

    async def scenario():
        with pytest.raises(UnsafeURLError):
            await fetch_url("https://public.example/start", resolver=lambda host: ["93.184.216.34"])

    asyncio.run(scenario())


def test_s6_response_larger_than_the_cap_is_refused(monkeypatch):
    import ba_agent.knowledge as knowledge

    payload = b"x" * (MAX_FETCH_BYTES + 10)

    class FakeTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(200, content=payload, headers={"content-type": "text/plain"}, request=request)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        knowledge.httpx, "AsyncClient", lambda **kw: real_client(**{**kw, "transport": FakeTransport()})
    )

    async def scenario():
        with pytest.raises(UnsafeURLError, match="larger than"):
            await fetch_url("https://example.com/big.txt", resolver=lambda host: ["93.184.216.34"])

    asyncio.run(scenario())


def test_s6_ordinary_public_page_is_still_fetched(monkeypatch):
    import ba_agent.knowledge as knowledge

    class FakeTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            body = "<html><head><title>Guide</title></head><body><main>" + ("Text. " * 80) + "</main></body></html>"
            return httpx.Response(200, text=body, headers={"content-type": "text/html"}, request=request)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        knowledge.httpx, "AsyncClient", lambda **kw: real_client(**{**kw, "transport": FakeTransport()})
    )

    async def scenario():
        return await fetch_url("https://example.com/guide", resolver=lambda host: ["93.184.216.34"])

    title, text = asyncio.run(scenario())
    assert title == "Guide" and "Text." in text


# --------------------------------------------------------------------------- S7: SVG sanitiser + CSP


MALICIOUS_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" onload="alert(1)">'
    '<script>alert("x")</script>'
    '<style>@import "https://evil.example/x.css";</style>'
    '<rect width="5" height="5" onclick="steal()"/>'
    '<a href="javascript:alert(2)"><text>click</text></a>'
    '<foreignObject><iframe src="https://evil.example"/></foreignObject>'
    '<image href="https://evil.example/t.png"/>'
    '<use href="data:image/svg+xml;base64,AAAA#x"/>'
    "</svg>"
)


def test_s7_sanitize_svg_removes_script_style_and_event_attributes():
    cleaned = sanitize_svg(MALICIOUS_SVG).lower()
    assert "<script" not in cleaned
    assert "<style" not in cleaned
    assert "foreignobject" not in cleaned
    assert "<iframe" not in cleaned
    assert "onload" not in cleaned and "onclick" not in cleaned
    assert "javascript:" not in cleaned
    assert "evil.example" not in cleaned
    assert "<rect" in cleaned  # legitimate shapes survive


def test_s7_sanitize_svg_keeps_a_benign_diagram():
    benign = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
        '<g id="a"><rect x="1" y="1" width="4" height="4" fill="#fff"/><text x="2" y="4">Stage</text></g>'
        '<a href="#a"><circle cx="5" cy="5" r="1"/></a></svg>'
    )
    cleaned = sanitize_svg(benign)
    assert "<rect" in cleaned and "Stage" in cleaned and 'href="#a"' in cleaned


@pytest.mark.parametrize(
    "bad",
    [
        "",
        None,
        "not svg at all",
        "<html><script>x</script></html>",
        "<!DOCTYPE svg [<!ENTITY x SYSTEM 'file:///etc/passwd'>]><svg>&x;</svg>",
        "<svg><unclosed>",
    ],
)
def test_s7_unparseable_or_hostile_input_yields_nothing(bad):
    assert sanitize_svg(bad) == ""


def test_s7_served_deliverable_has_csp_and_sanitised_svg(live):
    client, manager = live
    state = manager.services.store.create_session(brief="deliverable csp")
    directory = manager.services.config.session_dir(state.session_id)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "deliverable.html").write_text(
        "<html><body><p>spec</p>" + MALICIOUS_SVG + "</body></html>", encoding="utf-8"
    )

    response = client.get(f"/api/artifacts/{state.session_id}/deliverable.html")
    assert response.status_code == 200
    assert response.headers["content-security-policy"] == DELIVERABLE_CSP
    assert "default-src 'self'" in response.headers["content-security-policy"]
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert "<script>alert" not in response.text
    assert "onload=" not in response.text and "onclick=" not in response.text


def test_s7_csp_is_not_added_to_non_html_artifacts(live):
    client, manager = live
    state = manager.services.store.create_session(brief="md csp")
    directory = manager.services.config.session_dir(state.session_id)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "functional_specification.md").write_text("# Spec", encoding="utf-8")
    response = client.get(f"/api/artifacts/{state.session_id}/functional_specification.md")
    assert response.status_code == 200
    assert "content-security-policy" not in response.headers


# --------------------------------------------------------------------------- S8: persistence


def test_s8_trim_events_keeps_only_the_newest(tmp_path):
    store = Store(tmp_path / "t.sqlite", event_retention=100)
    state = store.create_session(brief="trim me")
    _seed_events(store, state.session_id, 250)
    removed = store.trim_events(state.session_id, keep=100)
    assert removed == 150
    remaining = store.get_events(state.session_id, 0, 1000)
    assert len(remaining) == 100
    assert remaining[0]["payload"]["n"] == 150  # the 100 newest survive
    store.close()


def test_s8_trim_is_a_noop_under_the_cap(tmp_path):
    store = Store(tmp_path / "t.sqlite")
    state = store.create_session(brief="small")
    _seed_events(store, state.session_id, 20)
    assert store.trim_events(state.session_id, keep=5000) == 0
    assert len(store.get_events(state.session_id, 0, 1000)) == 20
    store.close()


def test_s8_save_state_applies_retention(tmp_path):
    store = Store(tmp_path / "t.sqlite", event_retention=100)
    state = store.create_session(brief="retention on save")
    _seed_events(store, state.session_id, 300)
    store.save_state(state)
    assert len(store.get_events(state.session_id, 0, 1000)) == 100


def test_s8_schema_version_is_recorded_and_stable(tmp_path):
    store = Store(tmp_path / "v.sqlite")
    assert store.schema_version == SCHEMA_VERSION
    store.close()
    reopened = Store(tmp_path / "v.sqlite")
    assert reopened.schema_version == SCHEMA_VERSION
    reopened.close()


def test_s8_legacy_database_without_version_table_is_migrated(tmp_path):
    path = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(str(path))
    conn.executescript(
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, title TEXT, brief TEXT, phase TEXT, status TEXT, "
        "awaiting TEXT, created_at TEXT, updated_at TEXT, state_json TEXT);"
    )
    conn.commit()
    conn.close()
    store = Store(path)
    assert store.schema_version == SCHEMA_VERSION
    store.close()


def test_s8_database_from_a_newer_build_is_refused(tmp_path):
    path = tmp_path / "future.sqlite"
    store = Store(path)
    store._conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION + 5,))
    store._conn.commit()
    store.close()
    with pytest.raises(RuntimeError, match="newer than this build"):
        Store(path)


def test_s8_mark_interrupted_finds_a_run_outside_the_500_most_recent(services):
    from ba_agent.runtime import SessionManager
    from ba_agent.schemas import RunStatus

    store = services.store
    old = store.create_session(brief="old interrupted run")
    old.status = RunStatus.RUNNING
    store.save_state(old)
    for index in range(520):
        store.create_session(brief=f"filler {index}")

    manager = SessionManager(services)
    assert manager.mark_interrupted() == 1
    assert store.load_state(old.session_id).status == RunStatus.IDLE


def test_s8_load_state_logs_a_warning_instead_of_silently_returning_none(services):
    state = services.store.create_session(brief="corrupt me")
    services.store._conn.execute(
        "UPDATE sessions SET state_json=? WHERE id=?", ('{"session_id": "x", "phase": "not-a-phase"}', state.session_id)
    )
    services.store._conn.commit()

    assert services.store.load_state(state.session_id) is None
    warnings = [e for e in services.store.get_events(state.session_id, 0, 100) if e["type"] == EventType.WARNING]
    assert warnings, "a validation failure must leave a WARNING event behind"
    assert "failed validation" in warnings[0]["payload"]["message"]


# --------------------------------------------------------------------------- S9: JSON extraction


def test_s9_string_contents_survive_repair_untouched():
    text = '{"quote": "it\'s True and None, really", "n": True,}'
    value = extract_json(text)
    assert value == {"quote": "it's True and None, really", "n": True}


def test_s9_apostrophe_inside_double_quoted_value_is_not_corrupted():
    text = 'Here you go: {"text": "the client\'s portal isn\'t live"}'
    assert extract_json(text) == {"text": "the client's portal isn't live"}


def test_s9_braces_inside_strings_do_not_break_block_detection():
    text = 'prefix {"template": "use {name} and [items]", "ok": 1} suffix'
    assert extract_json(text) == {"template": "use {name} and [items]", "ok": 1}


def test_s9_block_scan_is_linear_on_adversarial_input():
    """The old code restarted a scan from every opener (O(n^2)). 60k openers must finish quickly."""
    started = time.perf_counter()
    result = extract_json("{" * 60_000)
    elapsed = time.perf_counter() - started
    assert result is None
    assert elapsed < 5.0, f"extract_json took {elapsed:.1f}s on adversarial input"


def test_s9_deeply_nested_valid_json_is_parsed():
    text = "x " + ("[" * 200) + ("]" * 200)
    assert extract_json(text) is not None


def test_s9_python_literal_with_single_quoted_string_is_recovered():
    assert extract_json("{'title': 'Plan', 'done': False}") == {"title": "Plan", "done": False}


def test_s9_raw_newline_inside_string_is_recovered():
    assert extract_json('{"text": "line one\nline two"}') == {"text": "line one\nline two"}


def test_s9_prose_without_json_returns_none():
    assert extract_json("There is no structured data in this sentence.") is None


def test_s9_the_first_valid_block_wins_over_later_noise():
    assert extract_json('noise {"a": 1} more {"b": 2}') == {"a": 1}


# --------------------------------------------------------------------------- S10: CI workflow


def test_s10_ci_workflow_matrix_and_offline_steps_are_declared():
    import yaml

    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
    assert workflow.exists()
    spec = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    job = next(iter(spec["jobs"].values()))
    versions = job["strategy"]["matrix"]["python-version"]
    assert {"3.11", "3.13"} <= {str(v) for v in versions}
    run_text = "\n".join(step.get("run", "") for step in job["steps"])
    assert 'pip install -e ".[dev]"' in run_text or "pip install -e ." in run_text
    assert "pytest -q" in run_text


def test_s5_body_is_cut_off_before_being_read_in_full():
    """The guard must stop pulling the request stream near the cap, not read everything and reject afterwards.

    Measured on the server side: the counter increments only when the guard itself pulls a chunk from the
    request stream. (A client-side counter would measure how far the test client's generator ran ahead.)
    """
    from fastapi import HTTPException
    from starlette.datastructures import Headers

    from ba_agent.server.app import MAX_UPLOAD_BYTES, _read_capped_uploads

    boundary = "BOUNDARY123"
    head = (
        b"--" + boundary.encode() + b"\r\n"
        b'Content-Disposition: form-data; name="files"; filename="huge.txt"\r\n'
        b"Content-Type: text/plain\r\n\r\n"
    )
    tail = b"\r\n--" + boundary.encode() + b"--\r\n"
    chunk = b"x" * (1024 * 1024)
    total_chunks = 60  # 60 MB offered; the cap is 25 MB
    pulled = {"bytes": 0}

    async def server_side_stream():
        yield head
        for _ in range(total_chunks):
            pulled["bytes"] += len(chunk)
            yield chunk
        yield tail

    class FakeRequest:
        headers = Headers({"content-type": f"multipart/form-data; boundary={boundary}"})

        def stream(self):
            return server_side_stream()

    async def scenario():
        with pytest.raises(HTTPException) as caught:
            await _read_capped_uploads(FakeRequest())
        return caught.value.status_code

    assert asyncio.run(scenario()) == 413
    assert pulled["bytes"] < MAX_UPLOAD_BYTES + 4 * 1024 * 1024, (
        f"pulled {pulled['bytes']} bytes before rejecting; expected to stop near the cap"
    )
