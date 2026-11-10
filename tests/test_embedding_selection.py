"""The embedding-model select is a real control, not decoration.

Choosing an embedding model must reach the knowledge service (enabling vector search), and choosing the
"(none — BM25 keyword search only)" option must persist as an explicit empty value — an absent field cannot
express "I deliberately have no embeddings".
"""

from __future__ import annotations

from pathlib import Path

from ba_agent.config import AppConfig
from ba_agent.runtime import SessionManager
from ba_agent.services import Services


def _manager(tmp_path: Path) -> SessionManager:
    config = AppConfig()
    config.paths.state_dir = str(tmp_path)
    config.lmstudio.embedding_model = "text-embedding-nomic-embed-text-v1.5"
    return SessionManager(Services.create(config))


async def test_picking_an_embedding_model_persists_across_restarts(tmp_path: Path):
    manager = _manager(tmp_path)
    manager.config.provider = "lmstudio"

    await manager.set_models(embedding_model="embedder-v2")
    assert manager.config.active_profile.embedding_model == "embedder-v2"
    assert manager.model_settings()["embedding_model"] == "embedder-v2"

    restarted = _manager(tmp_path)
    restarted.config.provider = "lmstudio"
    restarted.restore_run_settings()
    # The store is the override, so a restart keeps the user's choice.
    assert restarted.config.active_profile.embedding_model == "embedder-v2", (
        "the saved embedding model was stored but never applied"
    )


async def test_empty_embedding_model_is_stored_as_a_deliberate_choice(tmp_path: Path):
    manager = _manager(tmp_path)
    manager.config.provider = "strata"

    await manager.set_models(embedding_model="")
    assert manager.config.active_profile.embedding_model is None
    bucket = manager.services.store.get_setting("model_choices", {})[manager.config.provider]
    assert bucket.get("embedding_model") == "", f"empty selection was lost rather than stored ({bucket!r})"

    restarted = _manager(tmp_path)
    restarted.config.provider = "strata"
    restarted.restore_run_settings()
    assert restarted.config.active_profile.embedding_model is None


async def test_omitting_the_field_does_not_clobber_a_saved_choice(tmp_path: Path):
    manager = _manager(tmp_path)
    manager.config.provider = "lmstudio"
    await manager.set_models(embedding_model="embedder-v2")

    # Changing only the default model must leave the embedding model alone.
    await manager.set_models(default_model="qwen3-32b")
    assert manager.config.active_profile.embedding_model == "embedder-v2"


async def test_status_names_the_retrieval_that_is_actually_running(tmp_path: Path):
    """The UI derives its wording from this status, so it must reflect embeddings, not intent."""
    manager = _manager(tmp_path)
    manager.config.provider = "strata"
    await manager.set_models(embedding_model="")

    status = manager.services.knowledge.status()
    assert status["embedding_model"] is None, "a cleared embedding model must not still be reported"
    assert status["vector_search"] is False, "vector search cannot be on with no embedding model"

    manager.config.provider = "lmstudio"
    await manager.set_models(embedding_model="embedder-v2")
    assert manager.services.knowledge.status()["embedding_model"] == "embedder-v2"
