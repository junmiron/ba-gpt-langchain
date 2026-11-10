"""Retriever adaptation and the context budget.

Regression: `context_block` stopped at the first chunk that did not fit and returned "" when the very
first chunk was longer than the budget. A role then silently received no grounding at all, which looks
like "the knowledge base is empty" rather than "the budget is too small".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ba_agent.config import AppConfig
from ba_agent.knowledge import KnowledgeService, Retrieved, fit_blocks
from ba_agent.retriever import KnowledgeRetriever, document_citation, to_document
from ba_agent.store import Store
from conftest import FakeGateway


async def _indexed(tmp_path: Path, documents: dict[str, str], *, chunk_size: int = 400) -> KnowledgeService:
    config = AppConfig()
    config.root = tmp_path
    config.knowledge.chunk_size = chunk_size
    config.knowledge.chunk_overlap = 0
    config.knowledge.heading_aware = False
    config.ensure_dirs()
    docs_dir = config.root / "data"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for name, body in documents.items():
        (docs_dir / name).write_text(body, encoding="utf-8")
    knowledge = KnowledgeService(Store(config.db_path), FakeGateway(), config)
    await knowledge.index_directory(docs_dir)
    return knowledge


def test_a_retrieved_chunk_becomes_a_document_with_provenance():
    item = Retrieved(text="Claims arrive by email.", source="file:notes.md", title="Notes", score=0.42)
    document = to_document(item)
    assert document.page_content == "Claims arrive by email."
    assert document.metadata["source"] == "file:notes.md"
    assert document.metadata["title"] == "Notes"
    assert document.metadata["score"] == 0.42


def test_document_citation_labels_with_title_then_source():
    document = to_document(Retrieved(text="Body.", source="url:x", title="Policy", score=0.5))
    assert document_citation(document).startswith("[Policy · score 0.500]")

    untitled = to_document(Retrieved(text="Body.", source="url:x", title="", score=0.5))
    assert "[url:x · score 0.500]" in document_citation(untitled)


@pytest.mark.asyncio
async def test_retriever_obeys_top_k(tmp_path: Path):
    knowledge = await _indexed(
        tmp_path,
        {
            "scope.md": "Project scope covers claims only.",
            "inventory.md": "Claim inventory is tracked in SAP alongside claims.",
            "season.md": "Claims peak in December.",
        },
    )
    assert len(KnowledgeRetriever(knowledge=knowledge, top_k=1).retrieve("claims")) == 1
    assert len(KnowledgeRetriever(knowledge=knowledge, top_k=3).retrieve("claims")) == 3
    assert await KnowledgeRetriever(knowledge=knowledge, top_k=2).ainvoke("claims")
    assert KnowledgeRetriever(knowledge=knowledge, top_k=3).retrieve("") == []
    assert KnowledgeRetriever(knowledge=knowledge, top_k=3).retrieve("zzzzneverindexed") == []


@pytest.mark.asyncio
async def test_context_block_is_truncated_not_empty_when_one_chunk_overflows(tmp_path: Path):
    """A budget smaller than the only relevant chunk must still yield that chunk, cut to the budget."""
    long_text = "The claims portal must handle attachments. " * 400  # ~15k chars in one chunk
    knowledge = await _indexed(tmp_path, {"policy.md": long_text}, chunk_size=100000)
    retriever = KnowledgeRetriever(knowledge=knowledge, top_k=3)
    assert len(retriever.retrieve("claims portal attachments")[0].page_content) > 1000

    block = retriever.context_block("claims portal attachments", max_chars=1000)
    assert block, "the budget silently dropped the only relevant chunk"
    assert len(block) <= 1000


@pytest.mark.asyncio
async def test_context_block_keeps_every_chunk_that_fits(tmp_path: Path):
    knowledge = await _indexed(
        tmp_path,
        {"reqs.md": "\n\n".join(f"Requirement {i} concerns claim routing." for i in range(12))},
    )
    block = KnowledgeRetriever(knowledge=knowledge, top_k=12).context_block("requirement", max_chars=600)
    assert "Requirement" in block
    assert len(block) <= 600


def test_fit_blocks_handles_degenerate_budgets():
    assert fit_blocks(["a", "b"], 0) == ""
    assert fit_blocks([], 100) == ""
    assert fit_blocks(["abcdefghij"], 4) == "abcd"
    assert fit_blocks(["ab", "cd"], 5) == "ab"
