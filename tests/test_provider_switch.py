"""Switching the backend from the UI must actually re-point the gateway.

The AutoGen reference app runs its agents on Strata (q2_0 on the 5090). To test this app against the
same lane, `provider` has to be changeable at runtime — not only by editing config.yaml and restarting.
A runtime switch that left the cached `ChatOpenAI` clients pointing at the old base_url would look like
it worked (the health pill changes) while every call still went to the previous backend.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ba_agent.config import AppConfig
from ba_agent.models import ModelGateway
from ba_agent.runtime import SessionManager, apply_stored_model_choices

INDEX_HTML = Path("src/ba_agent/web/index.html")
APP_JS = Path("src/ba_agent/web/app.js")


def _manager() -> SessionManager:
    config = AppConfig(provider="lmstudio")
    services = Services_stub(config)
    return SessionManager(services)


class Services_stub:
    """Services with an in-memory settings store and a fake gateway — no DB, no model."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.gateway = ModelGateway(config)
        self.store = SimpleStore()
        self.knowledge = None
        self.sandbox = None

    def close(self) -> None:
        self.store.close()


class SimpleStore:
    def __init__(self) -> None:
        self._settings: dict[str, object] = {}

    def get_setting(self, key: str, default=None):
        return self._settings.get(key, default)

    def set_setting(self, key: str, value: object) -> None:
        self._settings[key] = value

    def close(self) -> None:
        pass


@pytest.fixture
def manager() -> SessionManager:
    return _manager()


async def test_switching_provider_repoints_the_gateway(manager: SessionManager):
    before = manager.gateway._client_kwargs["base_url"]
    await manager.set_models(provider="strata")
    after = manager.gateway._client_kwargs["base_url"]
    assert before != after
    assert after == manager.config.strata.base_url
    assert manager.config.provider == "strata"
    assert manager.config.active_profile is manager.config.strata


async def test_a_provider_switch_clears_the_stale_client_cache(manager: SessionManager):
    manager.gateway.chat("model-a")
    assert manager.gateway._cache, "the client cache should be warm before the switch"
    await manager.set_models(provider="strata")
    assert not manager.gateway._cache, "clients built for the previous backend must not be reused"


async def test_an_unknown_provider_is_rejected_not_silently_accepted(manager: SessionManager):
    with pytest.raises(ValueError, match="provider must be"):
        await manager.set_models(provider="ollama")
    assert manager.config.provider == "lmstudio", "a rejected switch must leave the config untouched"


async def test_the_provider_choice_survives_a_restart(manager: SessionManager):
    """The CLI builds its own config; the provider the browser chose has to be applied there too."""
    await manager.set_models(provider="strata")
    fresh = AppConfig()
    apply_stored_model_choices(fresh, manager.services.store)
    assert fresh.provider == "strata"
    assert fresh.active_profile is fresh.strata


async def test_reset_returns_to_the_provider_the_process_started_with(manager: SessionManager):
    await manager.set_models(provider="strata")
    manager.reset_run_settings()
    assert manager.config.provider == "lmstudio", "reset means undo what this UI changed, not change defaults"
    assert manager.gateway._client_kwargs["base_url"] == manager.config.lmstudio.base_url


def test_the_settings_api_exposes_the_provider_options():
    models = _manager().model_settings()
    assert models["provider"] in models["providers"]
    assert models["base_url"], "the UI shows where the provider points so a bad switch is visible"


def test_the_settings_dialog_has_a_provider_control():
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert 'id="provider-select"' in html, "no provider selector in the settings dialog"
    js = APP_JS.read_text(encoding="utf-8")
    assert "renderProviderSelect" in js and "saveProvider" in js, "the provider control is never wired up"


async def test_model_lanes_are_kept_per_provider(manager: SessionManager):
    """`q2_0` is a Strata model id. Applying it to LM Studio after a switch is nonsense."""
    await manager.set_models(provider="strata", default_model="q2_0")
    await manager.set_models(provider="lmstudio")
    assert manager.config.active_profile.default_model is None, "the Strata lane leaked onto LM Studio"

    # Switching back must bring the lane home rather than losing it.
    await manager.set_models(provider="strata")
    assert manager.config.active_profile.default_model == "q2_0"


async def test_a_model_choice_saved_on_one_provider_survives_a_restart_on_that_provider():
    manager = _manager()
    await manager.set_models(provider="strata", default_model="q2_0", overrides={"writer": "big-model"})
    fresh = AppConfig()
    apply_stored_model_choices(fresh, manager.services.store)
    assert fresh.provider == "strata"
    assert fresh.strata.default_model == "q2_0"
    assert fresh.lmstudio.default_model is None, "the other provider must not inherit these ids"
    assert fresh.role("writer").model == "big-model"
