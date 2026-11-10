"""Run metrics: the numbers a reader uses to tell a slow model from a runaway loop.

`metrics_summary` is the only place the per-phase timings reach a human. If it drops them, a 25-minute
run looks like "the model is slow" when the real problem is that the interview phase ran twice.
"""

from __future__ import annotations

from ba_agent.export import metrics_summary
from ba_agent.schemas import SessionState


def test_metrics_summary_names_the_slowest_phases():
    state = SessionState(session_id="metrics", title="Claims Portal")
    state.metrics = {
        "calls": 148,
        "model_seconds": 1234,
        "questions": 22,
        "by_phase": {
            "interview": {"wall_seconds": 610, "calls": 44},
            "draft": {"wall_seconds": 420, "calls": 60},
            "review": {"wall_seconds": 205, "calls": 30},
            "diagram": {"wall_seconds": 41, "calls": 4},
        },
    }

    text = metrics_summary(state)["text"]
    assert "Model calls: 148" in text
    assert "Model time: 20m 34s" in text, text
    assert "Questions asked: 22" in text
    assert "Slowest phases: interview 10m 10s (44 calls), draft 7m 00s (60 calls), review 3m 25s (30 calls)" in text, (
        text
    )
    assert "diagram" not in text, "only the three slowest phases are reported"


def test_metrics_summary_stays_useful_when_phases_are_missing():
    state = SessionState(session_id="metrics-min", title="T")
    state.metrics = {"calls": 3, "model_seconds": 7}

    text = metrics_summary(state)["text"]
    assert text == "Model calls: 3 · Model time: 7s", text
    assert "Slowest phases" not in text


def test_the_shipped_metadata_carries_the_summary_and_the_backend():
    from ba_agent.config import AppConfig
    from ba_agent.export import build_metadata

    state = SessionState(session_id="metrics-meta", title="Claims Portal")
    state.metrics = {"calls": 5, "model_seconds": 30}

    class GatewayWithConfig:
        config = AppConfig(provider="strata")
        usage = {"calls": 5, "failures": 0, "prompt_chars": 100, "completion_chars": 50}
        rung_counts: dict[str, int] = {}
        _learned_rung: dict[str, str] = {}

    metadata = build_metadata(state, gateway=GatewayWithConfig())
    assert metadata["run_summary"]["text"] == "Model calls: 5 · Model time: 30s", metadata["run_summary"]
    backend = metadata["backend"]
    assert backend["provider"] == "strata", backend
    assert backend["base_url"] == "http://127.0.0.1:47831/v1", backend
    assert backend["default_model"] == "q2_0", backend
    assert backend["embedding_model"] is None, "the bundle must not imply embeddings that do not exist"

    # A bundle without a gateway makes no backend claim.
    assert "backend" not in build_metadata(state)
