"""Provider profiles: strata (default, the AutoGen reference lane) and lmstudio.

The Strata tests are skipped when the Strata server is not running, so the suite stays offline-safe.
"""

from __future__ import annotations

import httpx
import pytest

from ba_agent.config import AppConfig, load_config
from ba_agent.models import ModelGateway, ModelInfo, no_model_message

# --------------------------------------------------------------------- config layer


def test_the_shipped_default_is_the_strata_lane():
    """The app must default to the same model lane the AutoGen reference runs on, so a fresh
    clone exercises q2_0 without any env juggling."""
    cfg = load_config()
    assert cfg.provider == "strata"
    assert cfg.active_profile is cfg.strata
    assert cfg.active_profile.default_model == "q2_0"


def test_lmstudio_is_still_selectable():
    """A provider default is only safe if the other lane still works when someone asks for it."""
    cfg = AppConfig(provider="lmstudio")
    assert cfg.active_profile is cfg.lmstudio
    assert "LM Studio" in no_model_message(cfg.provider)


def test_strata_profile_selected_by_yaml(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "provider: strata\nstrata:\n  base_url: http://127.0.0.1:47831/v1\n  default_model: q2_0\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert cfg.provider == "strata"
    assert cfg.active_profile is cfg.strata
    assert cfg.active_profile.default_model == "q2_0"
    assert cfg.active_profile.base_url == "http://127.0.0.1:47831/v1"
    assert cfg.active_profile.embedding_model is None  # BM25-only fallback


def test_provider_env_overrides_yaml(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("provider: lmstudio\n", encoding="utf-8")
    monkeypatch.setenv("BA_PROVIDER", "strata")
    monkeypatch.setenv("STRATA_MODEL", "q2_0")
    cfg = load_config(cfg_path)
    assert cfg.provider == "strata"
    assert cfg.active_profile.default_model == "q2_0"


def test_unknown_provider_rejected(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("provider: claude\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lmstudio"):
        load_config(cfg_path)


# --------------------------------------------------------------------- gateway wiring


def test_gateway_uses_active_profile_base_url():
    cfg = load_config()
    cfg.provider = "strata"
    gateway = ModelGateway(cfg)
    kwargs = gateway._client_kwargs
    assert kwargs["base_url"] == cfg.strata.base_url
    assert kwargs["api_key"] == "strata"


def test_no_model_message_names_the_right_backend():
    assert "LM Studio" in no_model_message("lmstudio")
    assert "LM Studio" not in no_model_message("strata")
    assert "Strata" in no_model_message("strata")


@pytest.mark.asyncio
async def test_the_engine_error_names_the_active_provider_not_the_default_one(services):
    """A Strata user with no model must be told to start Strata, not LM Studio."""
    from ba_agent.workflow import WorkflowEngine

    services.config.provider = "strata"

    class _EmptyGateway:
        async def resolve_model(self, requested: str | None) -> str | None:
            return None

        async def list_models(self, refresh: bool = False) -> list:
            return []

        usage = {"calls": 0}

    services.gateway = _EmptyGateway()
    state = services.store.create_session(brief="test brief")
    engine = WorkflowEngine(services)

    async def emit(event_type: str, payload: dict | None = None) -> None:
        pass

    await engine.advance(state, emit=emit)
    assert "Strata" in (state.error or "")
    assert "LM Studio" not in (state.error or "")


def test_catalog_falls_back_without_admin_url(monkeypatch):
    """A strata-style profile has no admin API; the gateway must use /v1/models instead."""
    cfg = load_config()
    cfg.provider = "strata"
    gateway = ModelGateway(cfg)

    async def fake_openai_list(self):
        from ba_agent.models import ModelInfo

        return [ModelInfo(id="q2_0", type="llm", state="unknown", capabilities=["tool_use"])]

    monkeypatch.setattr(ModelGateway, "_list_models_openai_compat", fake_openai_list)
    models = gateway.list_models  # bound coroutine
    result = _run(models(refresh=True))
    assert [m.id for m in result] == ["q2_0"]


def _run(coro):
    import asyncio

    return asyncio.run(coro)


# --------------------------------------------------------------------- live smoke (skipped offline)

STRATA = "http://127.0.0.1:47831/v1"


def _strata_up() -> bool:
    try:
        resp = httpx.get(f"{STRATA}/models", timeout=5.0)
        return resp.status_code == 200
    except Exception:  # noqa: BLE001 - any connection error means the provider is unreachable
        return False


@pytest.mark.live
@pytest.mark.skipif(not _strata_up(), reason="Strata server not running")
async def test_strata_chat_and_structured_ladder():
    cfg = load_config()
    cfg.provider = "strata"
    cfg.strata.base_url = STRATA
    cfg.strata.default_model = "q2_0"
    gateway = ModelGateway(cfg)

    health = await gateway.health()
    assert health["reachable"] and health["provider"] == "strata"

    from ba_agent.schemas import RequirementDraft

    result = await gateway.structured(
        RequirementDraft,
        system_prompt="You are a business analyst. Return JSON only.",
        user_prompt="Write one functional requirement for an order system that sends SMS "
        "confirmations when an order is created. Give a short acceptance test.",
        role="triage",
    )
    assert result.parsed is not None, f"structured ladder failed: {result.error}"
    assert result.parsed.description.strip(), "requirement description is empty"
    await gateway.aclose()


def test_structured_ladder_starts_at_the_rung_that_last_worked():
    """A rung that times out on a quantised model would otherwise burn a full request timeout on
    every one of the ~29 structured calls in a run. Once a rung succeeds, try it first."""
    cfg = AppConfig(provider="lmstudio")
    cfg.lmstudio.structured_output_method = "auto"
    gateway = ModelGateway(cfg)
    gateway._catalog = [ModelInfo(id="q2_0", type="llm", state="loaded", capabilities=["tool_use"])]

    assert gateway._ladder("q2_0") == ["json_schema", "function_calling", "prompt"]

    gateway._learned_rung["q2_0"] = "function_calling"
    assert gateway._ladder("q2_0") == ["function_calling", "json_schema", "prompt"]

    # A stale learning (capability gone after a model reload) must not resurrect a rung. The catalogue
    # has to *know* the capability is absent — LM Studio reports it; a bare /v1/models listing does not.
    gateway._catalog = [ModelInfo(id="q2_0", type="llm", state="loaded", capabilities=[], capabilities_known=True)]
    assert gateway._ladder("q2_0") == ["json_schema", "prompt"]

    # An explicitly declared order still wins — learning is only a hint for auto mode.
    cfg.lmstudio.structured_output_rungs = ["prompt", "json_schema"]
    assert gateway._ladder("q2_0") == ["prompt", "json_schema"]


def test_a_server_that_reports_no_capabilities_does_not_deney_function_calling():
    """Strata's `/v1/models` lists ids only, no capability flags.

    Reading an empty capability list as "cannot call tools" drops `function_calling` from the ladder for
    every Strata model, and q2_0 does in fact support tool calls. Absence of evidence is not evidence of
    absence; LM Studio's admin API reports capabilities, and only there is an empty list meaningful.
    """
    gateway = ModelGateway(AppConfig(provider="strata"))
    gateway._catalog = [ModelInfo(id="q2_0", type="llm", state="unknown", capabilities=[])]
    assert gateway._ladder("q2_0") == ["json_schema", "function_calling", "prompt"]

    reported = ModelInfo(id="q2_0", type="llm", state="loaded", capabilities=[], capabilities_known=True)
    gateway._catalog = [reported]
    assert gateway._ladder("q2_0") == ["json_schema", "prompt"], "a reported absence must be honoured"
