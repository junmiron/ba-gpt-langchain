"""CLI and UI must agree: settings saved through the API are applied by the command line too.

Regression: `ba-agent serve` restores stored settings in its lifespan, but `ba-agent index`,
`ba-agent export` and `ba-agent doctor` built their services straight from config.yaml. A user who set
chunk_size or the embedding model in the browser and then ran the CLI got different behaviour from the
same stored state.
"""

from __future__ import annotations

import ast
from pathlib import Path

from ba_agent.cli import apply_saved_settings
from ba_agent.config import AppConfig
from ba_agent.runtime import SessionManager
from ba_agent.services import Services


def _services(tmp_path: Path):
    config = AppConfig()
    config.root = tmp_path
    return config, Services.create(config)


def test_cli_sees_a_saved_budget(tmp_path: Path):
    services = _services(tmp_path)[1]

    manager = SessionManager(services)
    manager.restore_run_settings()
    manager.update_run_settings({"interview": {"max_total_questions": 17}})

    # A freshly built Services object starts from config.yaml, not from the store.
    fresh_config, fresh_services = _services(tmp_path)
    assert fresh_config.interview.max_total_questions != 17, "the test is only meaningful if the default differs"

    apply_saved_settings(fresh_services)
    assert fresh_services.config.interview.max_total_questions == 17


def test_cli_sees_a_saved_embedding_choice(tmp_path: Path):
    config, services = _services(tmp_path)
    config.provider = "lmstudio"  # only LM Studio ships an embedding model by default
    manager = SessionManager(services)
    manager.restore_run_settings()
    manager.services.store.set_setting("embedding_model", "")

    fresh_config, fresh_services = _services(tmp_path)
    fresh_config.provider = "lmstudio"
    assert fresh_config.active_profile.embedding_model, "LM Studio ships an embedding model by default"

    apply_saved_settings(fresh_services)
    assert fresh_services.config.active_profile.embedding_model is None


def test_the_strata_default_ships_without_embeddings_and_says_so(tmp_path: Path):
    """Strata serves no `/v1/embeddings`. Claiming vector search would be a lie the UI can't see through."""
    config, services = _services(tmp_path)
    assert config.provider == "strata"
    assert config.active_profile.embedding_model is None
    apply_saved_settings(services)
    assert services.config.active_profile.embedding_model is None


def test_every_cli_command_applies_saved_settings():
    """Behaviour is only half the fix: the helper has to be called on every path, not just one.

    A forgotten call here is invisible until someone's CLI run disagrees with their browser, so this
    reads the AST rather than eyeballing the file.
    """
    tree = ast.parse(Path("src/ba_agent/cli.py").read_text(encoding="utf-8"))

    builds = 0
    applied = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (
                node.func.attr == "create"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "Services"
            ):
                builds += 1
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "apply_saved_settings":
            applied += 1

    assert builds > 0, "no CLI commands build services"
    assert applied == builds, f"{builds} service-building commands, only {applied} apply saved settings"
