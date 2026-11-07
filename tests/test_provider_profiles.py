"""Provider profiles: lmstudio (default) and strata (local Strata server).

The Strata tests are skipped when the Strata server is not running, so the suite stays offline-safe.
"""

from __future__ import annotations

import httpx
import pytest

from ba_agent.config import load_config
from ba_agent.models import ModelGateway, no_model_message


# --------------------------------------------------------------------- config layer

def test_default_provider_is_lmstudio():
    cfg = load_config()
    assert cfg.provider == "lmstudio"
    assert cfg.active_profile is cfg.lmstudio


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
    except Exception:
        return False


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
