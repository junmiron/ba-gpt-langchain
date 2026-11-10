"""`POST /api/settings/reset` clears the stored delta so config.yaml's defaults apply again."""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import httpx
import uvicorn

from ba_agent.config import AppConfig
from ba_agent.knowledge import KnowledgeService
from ba_agent.runtime import SessionManager
from ba_agent.server.app import create_app
from ba_agent.services import Services
from ba_agent.store import Store
from conftest import FakeGateway


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _manager(tmp_path: Path) -> SessionManager:
    config = AppConfig()
    config.paths.state_dir = str(tmp_path)
    return SessionManager(Services.create(config))


def test_reset_restores_config_defaults(tmp_path: Path):
    config = AppConfig()
    config.paths.state_dir = str(tmp_path)
    config.server.port = _free_port()

    # Keep the test off the network: the scripted gateway answers health checks instantly, so
    # readiness polling does not wait on a real model server's connect timeout.
    gateway = FakeGateway()
    # Capture the real factory *before* patching: calling `Services.create` from inside the
    # replacement would recurse into itself and blow the stack during lifespan startup.
    real_create = Services.create

    def create_with_fake(cfg=None):
        services = real_create(cfg or config)
        services.store = Store(config.db_path)
        services.gateway = gateway
        services.knowledge = KnowledgeService(services.store, gateway, config)
        return services

    Services.create = create_with_fake  # type: ignore[assignment]
    server = None
    thread = None
    try:
        app = create_app(config)
        server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=config.server.port, log_level="error", lifespan="on")
        )
        thread = threading.Thread(target=server.run, name="ba-reset-test-server", daemon=True)
        thread.start()

        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                if httpx.get(f"http://127.0.0.1:{config.server.port}/api/health", timeout=5.0).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.05)
        else:
            raise RuntimeError("test server never became healthy")

        base = f"http://127.0.0.1:{config.server.port}"
        shipped_default = config.interview.max_total_questions
        with httpx.Client(base_url=base, timeout=20.0) as client:
            client.patch("/api/settings", json={"interview": {"max_total_questions": 12}})
            assert client.get("/api/settings").json()["settings"]["interview"]["max_total_questions"] == 12

            response = client.post("/api/settings/reset")
            assert response.status_code == 200
            assert response.json()["settings"]["interview"]["max_total_questions"] == shipped_default, (
                "reset did not return the budget to the config default"
            )

        manager = _manager(tmp_path)
        assert manager.services.store.get_setting("run_settings", {}) == {}, "the delta was not cleared"
        manager.restore_run_settings()
        assert manager.config.interview.max_total_questions == shipped_default, "config.yaml defaults did not survive"
    finally:
        Services.create = real_create
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(timeout=10)
            assert not thread.is_alive(), "the test server did not shut down"
