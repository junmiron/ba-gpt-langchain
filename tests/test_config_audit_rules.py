"""The config audit must actually fire, not just report zero on the happy path.

A gate that can only ever return "clean" is not a gate. These inject each failure mode and assert the
audit names it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ba_agent import config_audit
from ba_agent.config import AppConfig


@pytest.fixture
def audit_root(tmp_path: Path, monkeypatch):
    """Point the audit at a scratch project root with a src/ haystack and a config.yaml."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "mod.py").write_text("value = config.server.port\n", encoding="utf-8")
    monkeypatch.setattr(config_audit, "PROJECT_ROOT", tmp_path)
    return tmp_path


def _write_yaml(root: Path, payload: dict) -> None:
    (root / "config.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_a_section_absent_from_yaml_is_drift(audit_root: Path):
    _write_yaml(audit_root, {"server": {"host": "127.0.0.1", "port": 8765}})
    drift = config_audit.config_drift()
    assert any("interview" in item for item in drift), drift


def test_a_value_disagreeing_with_the_default_is_drift(audit_root: Path):
    defaults = AppConfig()
    _write_yaml(
        audit_root,
        {
            "server": {"host": defaults.server.host, "port": 9999, "cors_origins": defaults.server.cors_origins},
            "provider": defaults.provider,
        },
    )
    drift = config_audit.config_drift()
    assert any("server.port: yaml=9999" in item for item in drift), drift


def test_a_key_with_no_appconfig_field_is_drift(audit_root: Path):
    _write_yaml(audit_root, {"server": {"host": "127.0.0.1", "port": 8765, "typo_field": 1}})
    drift = config_audit.config_drift()
    assert any("server.typo_field" in item for item in drift), drift


def test_a_missing_config_yaml_is_reported(audit_root: Path):
    assert config_audit.config_drift() == ["config.yaml is missing"]


def test_a_field_nothing_reads_is_dead(audit_root: Path, monkeypatch):
    # Scope the section map to the fields under test; a scratch src/ tree cannot mention the real fields,
    # so running the audit over all of them here would report 55 false positives.
    (audit_root / "src" / "mod.py").write_text("token = config.server.auth_token\n", encoding="utf-8")
    monkeypatch.setattr(config_audit, "config_sections", lambda: {"server": ["auth_token", "unused_knob"]})
    assert config_audit.dead_config_fields() == ["server.unused_knob"]


def test_a_field_somewhere_in_src_is_not_dead(audit_root: Path, monkeypatch):
    (audit_root / "src" / "mod.py").write_text("x = cfg.live_knob\n", encoding="utf-8")
    monkeypatch.setattr(config_audit, "config_sections", lambda: {"server": ["live_knob", "auth_token"]})
    assert config_audit.dead_config_fields() == ["server.auth_token"]


def test_report_shape_and_gate_verdict(audit_root: Path):
    result = config_audit.report()
    assert result["declared_fields"] > 0
    assert isinstance(result["dead"], list)
    assert isinstance(result["drift"], list)
    assert result["ok"] is not None


def test_main_returns_nonzero_when_drift_exists(audit_root: Path, capsys):
    _write_yaml(audit_root, {"server": {"host": "127.0.0.1", "port": 1}})
    assert config_audit.main() == 1
    assert "drift" in capsys.readouterr().out


def test_the_ship_config_is_clean():
    """The real config.yaml in the repo must pass the gate, or CI fails for the wrong reason."""
    result = config_audit.report()
    assert result["ok"], f"dead: {result['dead']}\ndrift: {result['drift'][:10]}"
