"""Settings restore must not poison config with unvalidated values.

`update_settings` validates through `model_validate`, but `restore_run_settings` re-applies the persisted
row with bare `setattr`. Pydantic v2 does not validate on assignment by default, so a corrupt stored value
(e.g. `"not-a-number"` in an int field) silently lands in the config and then breaks the budget comparisons
that read it.
"""

from __future__ import annotations

from pathlib import Path

from ba_agent.config import AppConfig
from ba_agent.runtime import SessionManager
from ba_agent.services import Services
from ba_agent.store import Store


def _manager(tmp_path: Path) -> SessionManager:
    config = AppConfig()
    services = Services.create(config)
    services.store = Store(tmp_path / "settings.db")
    return SessionManager(services)


def test_a_corrupt_stored_setting_is_not_applied(tmp_path: Path):
    manager = _manager(tmp_path)
    manager.services.store.set_setting("run_settings", {"interview": {"max_total_questions": "not-a-number"}})
    manager.restore_run_settings()

    assert manager.config.interview.max_total_questions == AppConfig().interview.max_total_questions, (
        "restore_run_settings accepted a non-integer into an integer budget field"
    )


def test_a_valid_stored_setting_is_applied(tmp_path: Path):
    manager = _manager(tmp_path)
    manager.services.store.set_setting("run_settings", {"interview": {"max_total_questions": 12}})
    manager.restore_run_settings()
    assert manager.config.interview.max_total_questions == 12


def test_round_trip_survives_a_restart(tmp_path: Path):
    """The whole point of persisting settings: what the UI sets is still set after the process restarts."""
    first = _manager(tmp_path)
    first.update_run_settings({"interview": {"max_total_questions": 18, "max_subjects": 4}})

    second = _manager(tmp_path)
    second.restore_run_settings()
    assert second.config.interview.max_total_questions == 18
    assert second.config.interview.max_subjects == 4
    assert isinstance(second.config.interview.max_total_questions, int)
