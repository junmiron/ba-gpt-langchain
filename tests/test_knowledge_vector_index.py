"""Vector search must survive a partially-embedded index.

`_load_index` used to feed `entry.vector or []` into `np.asarray`, which raises on inhomogeneous rows: one
chunk indexed before an embedding model existed was enough to set `_matrix = None` and silently disable
semantic retrieval for the whole corpus. It also required 90% of chunks to be vectorised before enabling
vectors at all, so a mixed index was worse than no index.
"""

from __future__ import annotations

from ba_agent.config import AppConfig
from ba_agent.knowledge import KnowledgeService
from ba_agent.store import Store

# Four headed sections, each long enough to be its own chunk under the default heading-aware splitter.
POLICY = """
## Claims triage
The junior broker opens the claim in the portal and the triage queue assigns it to an adjuster within one
business day, which is the workflow the stakeholders described in the kickoff call.

## Auto and property handling
Auto claims and property claims follow different escalation paths, and the adjuster responsible for each
category is decided by the policy type recorded on the claim when it is opened.

## Parking badges
Parking badges for staff are renewed annually by facilities and have nothing to do with claims processing.

## Legacy spreadsheet
The legacy spreadsheet is still the source of truth for historical claims and is exported monthly by the
finance team into a shared drive folder.
"""


class EmbedGateway:
    """Embeds every chunk except the positions in `skip` — a model downloaded partway through the project."""

    def __init__(self, skip: set[int]):
        self.skip = skip

    async def embed(self, texts):
        out = []
        for position, text in enumerate(texts):
            if position in self.skip:
                out.append(None)
            else:
                # A toy "embedding": first word hashed into a 4-d unit vector.
                seed = sum(ord(c) for c in (text or "")[:12]) % 4
                vector = [0.0] * 4
                vector[seed] = 1.0
                out.append(vector)
        return out


def _knowledge(tmp_path, gateway) -> KnowledgeService:
    config = AppConfig()
    config.root = tmp_path
    config.knowledge.docs_dir = "docs"
    # These tests are about a *configured* embedding model that under-delivers, so they pin the provider
    # that actually declares one. The Strata default has no embeddings at all.
    config.provider = "lmstudio"
    return KnowledgeService(Store(tmp_path / "kb.db"), gateway, config)


async def test_partial_embedding_still_enables_vector_search(tmp_path):
    knowledge = _knowledge(tmp_path, EmbedGateway(skip={2}))
    outcome = await knowledge.index_upload("policy.txt", POLICY.encode())
    assert outcome["ok"], outcome

    status = knowledge.status()
    assert status["document_count"] == 1
    assert status["chunks"] == 4, status
    assert status["vector_search"] is True, (
        f"a partially embedded index disabled vector search: {status['vector_search_reason']}"
    )


async def test_a_vector_score_never_leaks_onto_an_unembedded_chunk(tmp_path):
    knowledge = _knowledge(tmp_path, EmbedGateway(skip={1}))
    await knowledge.index_upload("policy.txt", POLICY.encode())

    hits = knowledge.rank("claims triage workflow", k=4)
    assert hits, "ranking returned nothing"

    # Every hit carrying a vector_score must be one the gateway actually embedded, not a neighbour
    # shifted into place by a misaligned row.
    for hit in hits:
        if hit.vector_score is not None:
            assert "Auto and property handling" not in hit.text, (
                f"row/entry misalignment scored an unembedded chunk: {hit.text[:60]!r}"
            )


class _NoEmbeddings:
    embed_error = None

    async def embed(self, texts):
        return None


class _BrokenEmbeddings:
    """The provider has an embedding model configured but the endpoint 404s."""

    embed_error = None

    async def embed(self, texts):
        raise RuntimeError("404: no such embedding model")
        return None


async def test_index_with_no_embeddings_stays_bm25_only(tmp_path):
    knowledge = _knowledge(tmp_path, _NoEmbeddings())
    await knowledge.index_upload("policy.txt", POLICY.encode())

    status = knowledge.status()
    assert status["vector_search"] is False
    # A model is configured but produced no vectors, so the reason must say exactly that rather than
    # claiming no embedding model exists.
    assert "only 0 of 4 chunk(s) vectorised" in status["vector_search_reason"], status["vector_search_reason"]
    assert knowledge.rank("claims triage"), "BM25 fallback must still return hits"


async def test_a_failed_embedding_call_is_named_in_the_reason(tmp_path):
    """Silent degradation is the bug: the reason must carry the actual failure, not a guess."""
    knowledge = _knowledge(tmp_path, _BrokenEmbeddings())
    await knowledge.index_upload("policy.txt", POLICY.encode())

    reason = knowledge.status()["vector_search_reason"]
    assert "404" in reason, reason
