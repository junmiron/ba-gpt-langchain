"""Editing one setting must not freeze every other field in that section.

`update_run_settings` persisted `{key: validated[key] for key in SETTABLE[section]}` — the whole section,
not just the keys the user touched. On the next start, `restore_run_settings` re-applies that snapshot,
so a value the user later changes in config.yaml is silently shadowed by a stale snapshot.
"""

from __future__ import annotations

from pathlib import Path

from ba_agent.config import AppConfig
from ba_agent.runtime import SessionManager
from ba_agent.services import Services
from ba_agent.store import Store


def _manager(config: AppConfig, tmp_path: Path) -> SessionManager:
    services = Services.create(config)
    services.store = Store(tmp_path / "settings.db")
    return SessionManager(services)


def test_only_the_edited_key_is_persisted(tmp_path: Path):
    config = AppConfig()
    config.interview.max_subjects = 7  # pretend config.yaml says 7
    manager = _manager(config, tmp_path)
    manager.update_run_settings({"interview": {"max_total_questions": 12}})

    stored = manager.services.store.get_setting("run_settings", {})
    persisted = stored.get("interview", {})
    assert persisted == {"max_total_questions": 12}, (
        f"expected only the edited key, but the whole section was snapshotted: {persisted}"
    )


def test_a_config_change_to_an_untouched_field_survives_a_restart(tmp_path: Path):
    """User edits one budget in the UI, later changes a different budget in config.yaml. The config edit
    must still take effect after a restart."""
    config = AppConfig()
    manager = _manager(config, tmp_path)
    manager.update_run_settings({"interview": {"max_total_questions": 12}})

    # Later, the user raises the subject ceiling in config.yaml.
    config.interview.max_subjects = 9

    restarted = _manager(config, tmp_path)
    restarted.restore_run_settings()
    assert restarted.config.interview.max_total_questions == 12, "the UI edit should survive"
    assert restarted.config.interview.max_subjects == 9, "config.yaml edit was shadowed by the snapshot"
