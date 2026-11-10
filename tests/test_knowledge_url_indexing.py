"""URL indexing: dedupe, failure isolation and the embedded flag.

`index_urls` is the path the UI uses when someone pastes a list of pages. A run that silently
re-indexes the same page every time inflates the corpus with duplicates, and one failed URL must
not take the rest of the batch down with it.
"""

from __future__ import annotations

import asyncio

import httpx

from ba_agent.config import AppConfig
from ba_agent.knowledge import KnowledgeService
from ba_agent.store import Store
from conftest import FakeGateway


class _FakeTransport(httpx.AsyncBaseTransport):
    """Serve canned pages so the test never touches the network."""

    def __init__(self, pages: dict[str, tuple[str, str]]):
        self.pages = pages

    async def handle_async_request(self, request):
        url = str(request.url)
        if url not in self.pages:
            return httpx.Response(404, text="missing", request=request)
        title, body = self.pages[url]
        html = f"<html><head><title>{title}</title></head><body><main>{body}</main></body></html>"
        return httpx.Response(200, text=html, headers={"content-type": "text/html"}, request=request)


def _knowledge(tmp_path, pages: dict[str, tuple[str, str]], monkeypatch) -> KnowledgeService:
    import ba_agent.knowledge as knowledge

    config = AppConfig(root=tmp_path)
    config.knowledge.docs_dir = "docs"
    service = KnowledgeService(Store(tmp_path / "kb_test.db"), FakeGateway(), config)

    real_client = httpx.AsyncClient
    transport = _FakeTransport(pages)
    # monkeypatch keeps the swap scoped to this test; a bare assignment would leak into every other
    # test that fetches a URL.
    monkeypatch.setattr(
        knowledge.httpx, "AsyncClient", lambda **kwargs: real_client(**{**kwargs, "transport": transport})
    )
    return service


def test_index_urls_dedupes_by_content_and_reports_failures(tmp_path, monkeypatch):
    pages = {
        "https://example.com/a": ("Claims", "Claim intake is manual today for every broker."),
        "https://example.com/b": ("Policy", "Policy lookups go through a REST API."),
    }
    service = _knowledge(tmp_path, pages, monkeypatch)

    first = asyncio.run(service.index_urls([*pages, "https://example.com/missing"]))
    assert first["scanned"] == 3
    assert first["indexed"] == 2, first
    assert len(first["failed"]) == 1 and "missing" in first["failed"][0], first
    assert first["embedded"] is False, "a gateway with no embeddings must not claim vectors"

    # Re-indexing the same URLs must not add the same text twice.
    again = asyncio.run(service.index_urls(list(pages)))
    assert again["indexed"] == 0, again
    assert again["skipped"] == 2, again

    texts = {hit.text for hit in service.search_sync("claim intake policy", k=20)}
    assert len(texts) == 2, f"the corpus grew on re-index: {sorted(texts)}"
    assert service.status()["document_count"] == 2


def test_index_urls_survives_a_total_fetch_failure(tmp_path, monkeypatch):
    service = _knowledge(tmp_path, {}, monkeypatch)

    result = asyncio.run(service.index_urls(["https://example.com/x", "https://example.com/y"]))
    assert result["indexed"] == 0
    assert len(result["failed"]) == 2, result
    assert not service.search_sync("anything"), "a failed fetch must not leave a half-written document behind"
    assert service.status()["document_count"] == 0
