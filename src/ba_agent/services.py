"""Dependency container passed to every role and workflow step."""

from __future__ import annotations

from dataclasses import dataclass

from .config import AppConfig, get_config
from .knowledge import KnowledgeService
from .models import ModelGateway, get_gateway
from .store import Store
from .tools.sandbox import Sandbox, get_sandbox


def session_model(state, role: str) -> str | None:
    """The model picked for this session in the UI, or ``None`` to use config / the global default.

    Without this the per-session dropdown selection would be stored and then ignored by every role.
    """
    overrides = getattr(state, "model_overrides", None) or {}
    value = overrides.get(role) or overrides.get("default")
    return (value or "").strip() or None


@dataclass
class Services:
    config: AppConfig
    store: Store
    gateway: ModelGateway
    knowledge: KnowledgeService
    sandbox: Sandbox

    @classmethod
    def create(cls, config: AppConfig | None = None) -> "Services":
        cfg = config or get_config()
        store = Store(cfg.db_path)
        gateway = get_gateway(cfg)
        knowledge = KnowledgeService(store, gateway, cfg)
        return cls(config=cfg, store=store, gateway=gateway, knowledge=knowledge, sandbox=get_sandbox(cfg))
