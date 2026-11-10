"""Phase 5: optional bearer-token auth on the API."""

from __future__ import annotations

import socket
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
import uvicorn

from ba_agent.config import AppConfig
from ba_agent.knowledge import KnowledgeService
from ba_agent.server.app import create_app
from ba_agent.services import Services
from conftest import FakeGateway

TOKEN = "a-long-enough-local-token"


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture()
def auth_client(monkeypatch):
    config = AppConfig()
    config.server.auth_token = TOKEN
    gateway = FakeGateway()

    def create_with_fake(cfg=None):
        services = Services.create(cfg or config)
        services.gateway = gateway  # type: ignore[assignment]
        services.knowledge = KnowledgeService(services.store, gateway, config)  # type: ignore[arg-type]
        return services

    monkeypatch.setattr("ba_agent.server.app.Services", SimpleNamespace(create=create_with_fake))

    app = create_app(config)
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", lifespan="on", access_log=False)
    )
    thread = threading.Thread(target=server.run, name="ba-auth-server", daemon=True)
    thread.start()

    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=20.0) as client:
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                if client.get("/api/health", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.2)
        yield client


def test_api_is_closed_without_the_token(auth_client) -> None:
    response = auth_client.get("/api/sessions")
    assert response.status_code == 401
    assert "token" in response.json()["detail"].lower()


def test_api_is_open_with_the_token(auth_client) -> None:
    response = auth_client.get("/api/sessions", headers={"Authorization": f"Bearer {TOKEN}"})
    assert response.status_code == 200


def test_x_api_key_header_also_works(auth_client) -> None:
    response = auth_client.get("/api/health", headers={"X-API-Key": TOKEN})
    assert response.status_code == 200


def test_a_wrong_token_is_refused(auth_client) -> None:
    response = auth_client.get("/api/sessions", headers={"Authorization": "Bearer short-wrong-token-value"})
    assert response.status_code == 401


def test_the_static_page_is_still_served_without_a_token(auth_client) -> None:
    """The UI has to load so it can ask for the token."""
    response = auth_client.get("/")
    assert response.status_code == 200
    assert "interview" in response.text.lower()


def test_sse_stream_accepts_the_token_as_a_query_param(auth_client) -> None:
    created = auth_client.post(
        "/api/sessions",
        json={"brief": "Claims portal for brokers, 400 claims a month."},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert created.status_code == 200
    session_id = created.json()["session"]["id"]

    # EventSource cannot send headers, so the stream path must accept ?token=.
    with auth_client.stream(
        "GET", f"/api/sessions/{session_id}/stream?after=0", params={"token": TOKEN}, timeout=8.0
    ) as stream:
        first = b""
        for chunk in stream.iter_bytes():
            first += chunk
            if len(first) > 20:
                break
    assert b"event:" in first or b"data:" in first


def test_short_tokens_are_rejected_at_config_time() -> None:
    with pytest.raises(Exception) as exc:
        AppConfig.model_validate({"server": {"auth_token": "abc"}})
    assert "16 characters" in str(exc.value)


def test_auth_is_off_by_default() -> None:
    assert AppConfig().server.auth_token == ""
