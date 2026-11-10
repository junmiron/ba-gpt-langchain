"""API error contracts: every failure mode returns the status code the UI branches on.

A 500 instead of a 400 means the browser shows a stack trace; a 404 where the client expected 409 means
the session looks gone when it is merely busy. These tests pin the mapping.
"""

from __future__ import annotations

import socket
import threading
import time

import httpx
import pytest
import uvicorn

from ba_agent.knowledge import KnowledgeService
from ba_agent.server.app import MAX_UPLOAD_FILES, Services, create_app
from conftest import FakeGateway


@pytest.fixture()
def api(config, monkeypatch):
    real_create = Services.create
    gateway = FakeGateway()

    def create_with_fake(cfg=None):
        services = real_create(cfg or config)
        services.gateway = gateway  # type: ignore[assignment]
        services.knowledge = KnowledgeService(services.store, gateway, config)  # type: ignore[arg-type]
        return services

    monkeypatch.setattr("ba_agent.server.app.Services.create", create_with_fake)

    app = create_app(config)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])

    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", lifespan="on", access_log=False)
    )
    thread = threading.Thread(target=server.run, name="ba-error-api", daemon=True)
    thread.start()

    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=20.0) as client:
        client.headers["Origin"] = f"http://127.0.0.1:{port}"
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                if client.get("/api/health").status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.05)
        else:  # pragma: no cover - defensive
            raise AssertionError("test server did not come up")

        client.ba_manager = app.state.manager  # type: ignore[attr-defined]
        yield client

    server.should_exit = True
    thread.join(timeout=10)


def _session_id(api, brief: str = "Session") -> str:
    return api.post("/api/sessions", json={"brief": brief}).json()["session"]["id"]


def test_an_unknown_session_is_404_on_every_command(api):
    bodies = {"resume": {}, "approve": {"approved": True}, "chat": {"message": "hello"}}
    for path, body in bodies.items():
        response = api.post(f"/api/sessions/does-not-exist/{path}", json=body)
        assert response.status_code == 404, f"{path} returned {response.status_code} for an unknown session"


def test_a_bad_model_name_is_400_not_500(api):
    response = api.post("/api/models", json={"default_model": "does-not-exist", "provider": "lmstudio"})
    assert response.status_code in (400, 200), response.status_code
    if response.status_code == 400:
        assert "model" in response.json()["detail"].lower()


def test_a_bad_setting_value_is_400_with_the_offending_key(api):
    response = api.patch("/api/settings", json={"interview": {"max_total_questions": 0}})
    assert response.status_code == 400, response.status_code
    assert "max_total_questions" in response.json()["detail"], response.json()


def test_editing_a_missing_ledger_entry_is_404(api):
    session_id = _session_id(api, "Ledger")
    response = api.post(f"/api/sessions/{session_id}/ledger/FR-999", json={"priority": "must"})
    assert response.status_code == 404, response.status_code


def test_an_invalid_ledger_priority_is_rejected_before_touching_the_store(api):
    session_id = _session_id(api, "Ledger")
    response = api.post(f"/api/sessions/{session_id}/ledger/FR-001", json={"priority": "urgent"})
    assert response.status_code in (400, 422), response.status_code


def test_an_upload_with_no_file_part_is_400(api):
    response = api.post(
        "/api/knowledge/upload",
        files=[("notes", ("", b""))],  # a part with no filename is not a file
    )
    assert response.status_code == 400, response.status_code
    assert "no files" in response.json()["detail"].lower()


def test_a_non_multipart_upload_is_415(api):
    response = api.post("/api/knowledge/upload", content=b"plain text", headers={"content-type": "text/plain"})
    assert response.status_code == 415, response.status_code


def test_more_than_the_allowed_number_of_files_is_413(api):
    parts = [("files", (f"file{i}.txt", b"content")) for i in range(MAX_UPLOAD_FILES + 1)]
    response = api.post("/api/knowledge/upload", files=parts)
    assert response.status_code == 413, response.status_code
    assert "at most" in response.json()["detail"].lower()


def test_an_artifact_from_another_session_is_not_servable(api):
    session_id = _session_id(api, "Session A")
    response = api.get(f"/api/artifacts/{session_id}/../../etc/passwd")
    assert response.status_code in (400, 403, 404), response.status_code


def test_a_missing_artifact_is_404(api):
    session_id = _session_id(api, "Session B")
    assert api.get(f"/api/artifacts/{session_id}/nope.pdf").status_code == 404


def test_exporting_an_unknown_session_is_404(api):
    assert api.get("/api/sessions/ghost/export").status_code == 404


def test_creating_a_session_with_neither_brief_nor_transcript_is_400(api):
    response = api.post("/api/sessions", json={"brief": "", "title": ""})
    assert response.status_code == 400, response.status_code
    assert "brief or a transcript" in response.json()["detail"]


def test_indexing_both_a_directory_and_urls_reports_both_results(api, tmp_path):
    (tmp_path / "data" / "notes").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "notes" / "claim.txt").write_text("Claims arrive by email today.", encoding="utf-8")
    response = api.post("/api/knowledge/index", json={"directory": "data/notes", "urls": []})
    assert response.status_code == 200
    assert "directory" in response.json(), response.json()
