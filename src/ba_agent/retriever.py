"""``langchain-core`` retrieval interface over the SQLite knowledge store (Phase 1.1).

The plan's rule is that no LangGraph may be imported; ``langchain-core``'s ``BaseRetriever`` is pure
core, so the knowledge base can be consumed by anything that speaks the Runnable protocol — the chat
agent, a future RAG chain, or a caller that already has a LangChain stack — without reaching into
``KnowledgeService`` internals.

Ranking stays in :class:`ba_agent.knowledge.KnowledgeService` (BM25 + cosine fusion, all in SQLite);
this module only adapts the result to ``Document`` objects.
"""

from __future__ import annotations

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from .knowledge import KnowledgeService, Retrieved, fit_blocks


def to_document(item: Retrieved) -> Document:
    """A retrieved chunk as a ``langchain-core`` document, with provenance in metadata."""
    return Document(
        page_content=item.text,
        metadata={
            "source": item.source,
            "title": item.title,
            "chunk_id": item.chunk_id,
            "score": item.score,
            "bm25_score": item.bm25_score,
            "vector_score": item.vector_score,
        },
    )


def document_citation(document: Document) -> str:
    """Citation line for a retrieved document, matching ``Retrieved.citation()``."""
    meta = document.metadata
    label = meta.get("title") or meta.get("source") or "source"
    score = meta.get("score", 0.0)
    return f"[{label} · score {float(score):.3f}]\n{document.page_content}"


def _block_for(document: Document) -> str:
    meta = document.metadata
    return f"[{meta.get('title') or meta.get('source') or 'source'}] {document.page_content}"


class KnowledgeRetriever(BaseRetriever):
    """Hybrid retriever over the session's indexed material.

    ``ainvoke`` uses the full hybrid ranking (embeddings + BM25). ``invoke`` is the synchronous
    interface and deliberately skips the embedding round-trip — keyword-only — so a sync caller never
    blocks on a model call.
    """

    knowledge: KnowledgeService
    top_k: int = 6

    def retrieve(self, query: str) -> list[Document]:
        """Keyword-only documents — the synchronous path, no embedding round-trip."""
        return [to_document(item) for item in self.knowledge.search_sync(query, k=self.top_k)]

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return self.retrieve(query)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        return [to_document(item) for item in await self.knowledge.search(query, k=self.top_k)]

    def context_block(self, query: str, max_chars: int = 3000) -> str:
        """Prompt-ready citation block built from the retriever's own documents.

        A chunk longer than the budget is cut to fit rather than discarded: dropping it would hand the
        role an empty context and read as "nothing in the knowledge base matches" when in fact the
        single best match was simply too big.
        """
        return fit_blocks([_block_for(document) for document in self.retrieve(query)], max_chars)
