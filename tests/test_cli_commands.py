"""`ba-agent` command behaviour: exit codes, output, and the settings each command actually applies.

Regression: the CLI is a user-facing entry point but had no execution tests at all — only an AST check
that `apply_saved_settings` was referenced. A typo in a subcommand or a swapped return code would ship
unnoticed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ba_agent.cli import _format_doctor, _format_ladder, main
from ba_agent.config import AppConfig
from ba_agent.runtime import SessionManager
from ba_agent.schemas import LedgerItem, Phase, RunStatus, SummaryDoc
from ba_agent.services import Services
from conftest import FakeGateway


@pytest.fixture
def cli_env(monkeypatch, tmp_path: Path):
    """Point the CLI at a throwaway project root with a scripted gateway."""
    config = AppConfig()
    config.root = tmp_path
    config.ensure_dirs()
    monkeypatch.setattr("ba_agent.cli.get_config", lambda: config)

    monkeypatch.setattr("ba_agent.services.get_gateway", lambda _config: FakeGateway())
    return config


def _seed_finished_session(config: AppConfig) -> str:
    services = Services.create(config)
    state = services.store.create_session(brief="Claims portal for brokers; today everything is email.")
    state.phase = Phase.DONE
    state.status = RunStatus.DONE
    state.summary = SummaryDoc(
        title="Claims portal", text="A portal for broker claims.", bullets=["Email intake today"]
    )
    state.ledger.append(
        LedgerItem(id="REQ-1", kind="requirement", body="Brokers submit claims through the portal", subject="claims")
    )
    services.store.save_state(state)
    return state.session_id


# --------------------------------------------------------------------------- argument handling


def test_no_subcommand_is_an_error(cli_env):
    with pytest.raises(SystemExit) as exit_code:
        main([])
    assert exit_code.value.code == 2


def test_unknown_subcommand_is_an_error(cli_env):
    with pytest.raises(SystemExit):
        main(["bogus"])


def test_provider_flag_works_after_the_subcommand(cli_env):
    # argparse normally rejects `ba-agent sessions --provider strata`; the shared parent parser allows it.
    main(["sessions", "--provider", "strata"])
    assert cli_env.provider == "strata"
    main(["--provider", "lmstudio", "sessions"])
    assert cli_env.provider == "lmstudio"


# --------------------------------------------------------------------------- sessions


def test_sessions_lists_rows(cli_env, capsys):
    config = cli_env
    services = Services.create(config)
    services.store.create_session(brief="Claims portal for brokers")
    services.store.create_session(brief="Warehouse inventory scan")

    code = main(["sessions"])
    assert code == 0
    printed = capsys.readouterr().out
    assert "Claims portal" in printed
    assert "Warehouse inventory" in printed


def test_sessions_respects_limit(cli_env, capsys):
    config = cli_env
    services = Services.create(config)
    for i in range(5):
        services.store.create_session(brief=f"Project {i}")
    main(["sessions", "--limit", "2"])
    assert len(capsys.readouterr().out.strip().splitlines()) == 2


def test_cli_sessions_honours_a_saved_setting(cli_env, capsys):
    """The CLI must not read a different config than the browser saved."""
    config = cli_env
    services = Services.create(config)
    SessionManager(services).update_run_settings({"interview": {"max_total_questions": 9}})

    fresh = Services.create(config)
    from ba_agent.cli import apply_saved_settings

    apply_saved_settings(fresh)
    assert fresh.config.interview.max_total_questions == 9


# --------------------------------------------------------------------------- export


def test_export_of_unknown_session_returns_2(cli_env, capsys):
    assert main(["export", "does-not-exist"]) == 2
    assert "unknown session" in capsys.readouterr().err


def test_export_rebuilds_deliverables_without_a_model(cli_env, capsys):
    session_id = _seed_finished_session(cli_env)
    code = main(["export", session_id])
    assert code == 0
    printed = capsys.readouterr().out
    assert "functional_specification.md" in printed
    directory = cli_env.session_dir(session_id)
    assert (directory / "functional_specification.md").exists()
    assert (directory / "deliverable.html").exists()


# --------------------------------------------------------------------------- doctor


def test_doctor_reports_an_unreachable_backend_as_failure(cli_env, capsys, monkeypatch):
    config = cli_env
    config.provider = "lmstudio"

    class DeadGateway:
        async def health(self) -> dict:
            return {"reachable": False, "chat_models": [], "embedding_models": [], "default_model": None}

        def _ladder(self, model_id):
            return ["prompt"]

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr("ba_agent.services.get_gateway", lambda _config: DeadGateway())

    code = main(["doctor"])
    captured = capsys.readouterr()
    out = captured.out
    assert code == 1, "doctor must exit non-zero when the backend is down, so CI can gate on it"
    assert "NOT REACHABLE" in out
    assert "BM25 only" in out


def test_doctor_json_flag_emits_machine_readable_output(cli_env, capsys, monkeypatch):
    class LiveGateway:
        async def health(self) -> dict:
            return {
                "reachable": True,
                "chat_models": [{"id": "m1", "state": "loaded", "capabilities": ["tool_use"]}],
                "embedding_models": [],
                "default_model": "m1",
            }

        def _ladder(self, model_id):
            return ["json_schema", "prompt"]

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr("ba_agent.services.get_gateway", lambda _config: LiveGateway())

    assert main(["doctor", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["backend"]["reachable"] is True
    assert report["structured_output"]["m1"] == ["json_schema", "prompt"]
    assert report["pdf"]["engine"]


def test_format_ladder_handles_an_empty_catalogue():
    assert _format_ladder({}) == "(no chat model reported by the backend)"
    assert _format_ladder({"m1": ["json_schema", "prompt"]}) == "m1: json_schema > prompt"


def test_format_doctor_survives_a_missing_pdf_section():
    report = {
        "backend": {
            "provider": "lmstudio",
            "base_url": "http://127.0.0.1:1234/v1",
            "reachable": True,
            "chat_models": 1,
            "loaded": ["m1"],
            "tool_capable": [],
            "default_model": None,
        },
        "embeddings": {"model": None, "available": False, "vector_search_enabled": False},
        "knowledge": {"documents": 0, "chunks": 0, "bm25": True},
        "structured_output": {},
        "paths": {"project_root": ".", "db": "x.db", "outputs": "outputs"},
    }
    text = _format_doctor(report)
    assert "no model reports tool_use" in text
    assert "PDF engine     : none" in text


# --------------------------------------------------------------------------- index


def test_index_ingests_a_directory_and_dedupes_on_rerun(cli_env, capsys):
    """Indexing the same file twice must skip, not duplicate: the hash ledger is the dedupe key."""
    config = cli_env
    docs = config.root / "data"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "notes.md").write_text(
        "# Claims\n\nBrokers submit claims by email. Around 400 per month.", encoding="utf-8"
    )

    assert main(["index"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["directory"]["indexed"] == 1
    assert first["directory"]["scanned"] == 1

    assert main(["index"]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["directory"]["indexed"] == 0
    assert second["directory"]["skipped"] == 1
