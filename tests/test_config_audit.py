"""Configuration hygiene gates: no dead knobs, no yaml/default drift, and a real log file.

These are cheap to run and they fail in ways unit tests around roles never catch: a config key a user
sets and silently gets nothing for, or config.yaml disagreeing with the code default.
"""

from __future__ import annotations

import logging

from ba_agent.config import AppConfig, configure_logging, load_config
from ba_agent.config_audit import config_drift, dead_config_fields


def test_no_config_field_is_dead():
    """Every AppConfig field must be read somewhere. A declared-but-unread key is a lie to the user.

    `subject_coverage_threshold` shipped exactly like this: present in config.yaml, honoured by nothing.
    """
    dead = dead_config_fields()
    assert not dead, f"config fields declared but never read: {dead}"


def test_config_yaml_matches_code_defaults():
    """config.yaml is the file users edit; if it disagrees with AppConfig, whichever they trust is wrong."""
    drift = config_drift()
    assert not drift, "config.yaml drift:\n" + "\n".join(f"  {item}" for item in drift)


def _fresh_root_logger():
    """Snapshot the process-global root logger so a logging test cannot leak handlers into others."""
    root = logging.getLogger()
    saved = (list(root.handlers), root.level, getattr(root, "_ba_agent_configured", None))
    root.handlers.clear()
    root._ba_agent_configured = False  # type: ignore[attr-defined]
    return root, saved


def _restore_root_logger(root, saved):
    handlers, level, configured = saved
    root.handlers = handlers
    root.setLevel(level)
    if configured is None:
        delattr(root, "_ba_agent_configured")
    else:
        root._ba_agent_configured = configured  # type: ignore[attr-defined]


def test_configure_logging_writes_to_the_logs_dir(tmp_path):
    """`paths.logs_dir` is created by ensure_dirs(); nothing may create a directory it never writes to."""
    config = AppConfig()
    config.root = tmp_path
    root, saved = _fresh_root_logger()
    try:
        configure_logging(config, "WARNING")
        logging.getLogger("ba_agent.test").warning("probe message")

        log_file = tmp_path / "logs" / "ba_agent.log"
        assert log_file.exists(), "configure_logging must write into paths.logs_dir"
        text = log_file.read_text(encoding="utf-8")
        assert "probe message" in text
        assert "WARNING" in text, "the file line must carry the level, not just the message"
    finally:
        _restore_root_logger(root, saved)


def test_configure_logging_is_idempotent(tmp_path):
    """Calling it twice (CLI, then the server) must not double-print every warning."""
    config = AppConfig()
    config.root = tmp_path
    root, saved = _fresh_root_logger()
    try:
        configure_logging(config)
        configure_logging(config)
        assert len(root.handlers) == 2, f"one console + one file handler, got {len(root.handlers)}"
    finally:
        _restore_root_logger(root, saved)


def test_a_bad_env_var_is_reported_not_silently_dropped(caplog, monkeypatch):
    """`BA_PORT=notanumber` must not quietly fall back to config.yaml with no trace."""
    monkeypatch.setenv("BA_PORT", "notanumber")
    with caplog.at_level(logging.WARNING):
        config = load_config()

    assert config.server.port == 8765, "the invalid override must be ignored, not crash the load"
    assert any("BA_PORT" in record.message for record in caplog.records), (
        "an invalid env value was discarded without a warning"
    )
