"""`POST /api/settings/reset` clears the stored delta and restores the config the process started with.

Settings are a delta, so "reset" is the only way a user gets their defaults back. The trap is that
`get_config()` caches one singleton that *is* the manager's config object: a reset that re-reads that object
copies a mutated config onto itself and silently does nothing. The manager snapshots its section values at
construction and restores from that snapshot, which also protects a config that never came from a file
(CLI `--provider`, a test harness).
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from ba_agent import config as config_mod
from ba_agent.config import AppConfig
from ba_agent.runtime import SessionManager
from ba_agent.server.app import create_app
from ba_agent.services import Services
from ba_agent.store import Store

_original_create = Services.create
_original_get_config = config_mod.get_config


def _client(tmp_path: Path) -> TestClient:
    config = AppConfig()
    config.paths.state_dir = str(tmp_path)

    def create_with_store(cfg=None):
        services = _original_create(cfg or config)
        services.store = Store(config.db_path)
        return services

    Services.create = create_with_store  # type: ignore[assignment]
    # Entering the context runs the lifespan, which builds the SessionManager the routes need.
    client = TestClient(create_app(config))
    client.__enter__()
    return client


def _restore() -> None:
    Services.create = _original_create
    config_mod.get_config = _original_get_config


def test_reset_clears_the_stored_delta(tmp_path: Path):
    client = _client(tmp_path)
    try:
        client.patch("/api/settings", json={"interview": {"max_total_questions": 12}})
        assert client.get("/api/settings").json()["settings"]["interview"]["max_total_questions"] == 12

        response = client.post("/api/settings/reset")
        assert response.status_code == 200
        assert response.json()["settings"]["interview"]["max_total_questions"] == 40

        # Untouched fields were never re-frozen either.
        assert client.get("/api/settings").json()["settings"]["interview"]["max_subjects"] == 10
    finally:
        client.__exit__(None, None, None)
        _restore()


def test_reset_requires_the_same_origin(tmp_path: Path):
    client = _client(tmp_path)
    try:
        response = client.post("/api/settings/reset", headers={"Origin": "http://evil.example"})
        assert response.status_code == 403
    finally:
        client.__exit__(None, None, None)
        _restore()


def test_reset_restores_the_snapshot_not_the_mutated_singleton(tmp_path: Path, monkeypatch):
    """In production `create_app` builds the manager from `get_config()`, which caches one singleton — so
    the manager's config *is* what a naive reset would "restore". Reset must re-read the file instead."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("interview:\n  max_total_questions: 40\n  max_subjects: 10\n", encoding="utf-8")
    monkeypatch.setenv("BA_CONFIG", str(config_file))
    config_mod._config = None  # start from a clean cache, like a fresh process

    config = config_mod.get_config()  # the singleton the app would use
    manager = SessionManager(Services.create(config))

    assert manager.config is config_mod.get_config(), "the manager holds the cached singleton"

    manager.update_run_settings({"interview": {"max_total_questions": 12}})
    assert manager.config.interview.max_total_questions == 12

    manager.reset_run_settings()
    assert manager.config.interview.max_total_questions == 40, "reset did not go back to the startup config"
    assert manager.services.store.get_setting("run_settings", {}) == {}

    # Reset must not fork the config: the gateway and knowledge service hold the same object, and a second
    # object would mean the model ladder and the budgets disagree mid-run.
    assert manager.services.gateway.config is manager.config
    assert manager.services.knowledge.config is manager.config

    config_mod._config = None


def test_reset_also_clears_the_model_lanes(tmp_path: Path):
    """`default_model`, `embedding_model` and `model_overrides` are separate store keys.

    Resetting only `run_settings` leaves them behind, and `restore_model_choices()` re-applies them on the
    next restart — so a user who pressed Reset and restarted would still be on the model they discarded.
    """
    import asyncio

    config = AppConfig()
    config.root = tmp_path
    manager = SessionManager(Services.create(config))
    manager.restore_run_settings()

    start_default = manager.config.active_profile.default_model
    start_embedding = manager.config.active_profile.embedding_model
    asyncio.run(manager.set_models(default_model="model-x", embedding_model="", overrides={"writer": "model-y"}))
    assert manager.config.active_profile.default_model == "model-x"
    assert manager.config.active_profile.embedding_model is None

    manager.reset_run_settings()
    # Reset goes back to what this process started with — for the Strata default that is `q2_0`,
    # not "unset". What must never survive is the lane the user explicitly discarded.
    assert manager.config.active_profile.default_model == start_default
    assert manager.config.active_profile.embedding_model == start_embedding
    assert (manager.config.roles.get("writer") or manager.config.role("writer")).model is None

    # The cleared choices must survive a restart, not come back from the store.
    restarted = SessionManager(Services.create(config))
    restarted.restore_run_settings()
    assert restarted.config.active_profile.default_model != "model-x"
    assert restarted.config.active_profile.embedding_model == start_embedding
