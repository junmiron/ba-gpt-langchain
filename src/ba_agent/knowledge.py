"""Knowledge base: ingest → chunk → embed (LM Studio) → hybrid retrieval (vector + BM25).

Differences from v1's ``rag_helpers.py``:

* content-hash dedup — re-indexing a folder never duplicates chunks;
* no ``torch``/sentence-transformers download — embeddings come from LM Studio, and if that model is not
  loaded the store still works in BM25-only mode (and says so);
* hybrid retrieval with Reciprocal Rank Fusion, which measurably beats cosine alone for BA vocabulary
  ("NFR", "RACI", "SLA" are exact-token queries).
"""

from __future__ import annotations

import hashlib
import math
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import httpx
from lxml import html as lxml_html

from .config import AppConfig, get_config
from .models import ModelGateway
from .store import Store, sha256_text

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".csv", ".json", ".yaml", ".yml"}
PDF_SUFFIXES = {".pdf"}
DOCX_SUFFIXES = {".docx", ".doc"}

#: Extensions accepted by uploads (anything else is rejected before its bytes are written to disk).
ALLOWED_UPLOAD_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES | DOCX_SUFFIXES
#: Hard cap for one uploaded file (the HTTP layer enforces the same limit on the whole request body).
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
#: Hard cap for one fetched web page body.
MAX_FETCH_BYTES = 10 * 1024 * 1024
MAX_REDIRECTS = 5


# --------------------------------------------------------------------------- chunking

_HEADING_RE = re.compile(r"^(#{1,6}\s.+|\d+(\.\d+)*\s+[A-Z].{2,80})$", re.M)
_PARAGRAPH_RE = re.compile(r"\n\s*\n+")
_SENTENCE_RE = re.compile(r"(?<=[.!?。])\s+|\n")


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 140) -> list[str]:
    """Structure-aware splitter: paragraphs → sentences → hard cut, with tail overlap.

    Overlap is taken from the previous chunk's *sentence* boundary rather than a raw character slice, so
    chunks never start mid-word or mid-sentence.
    """
    text = re.sub(r"\r\n?", "\n", text or "").strip()
    if not text:
        return []

    blocks: list[str] = []
    for paragraph in _PARAGRAPH_RE.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= chunk_size:
            blocks.append(paragraph)
            continue
        # Long paragraph → sentence packing.
        current = ""
        for sentence in _SENTENCE_RE.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current) + len(sentence) + 1 <= chunk_size:
                current = f"{current} {sentence}".strip()
            else:
                if current:
                    blocks.append(current)
                while len(sentence) > chunk_size:
                    blocks.append(sentence[:chunk_size])
                    sentence = sentence[chunk_size:]
                current = sentence
        if current:
            blocks.append(current)

    chunks: list[str] = []
    buffer = ""
    for block in blocks:
        candidate = f"{buffer}\n\n{block}".strip() if buffer else block
        if len(candidate) <= chunk_size:
            buffer = candidate
            continue
        if buffer:
            chunks.append(buffer)
        tail = _overlap_tail(buffer, overlap)
        buffer = f"{tail}\n\n{block}".strip() if tail else block
    if buffer:
        chunks.append(buffer)

    # Drop degenerate duplicates produced by the overlap.
    deduped: list[str] = []
    for chunk in chunks:
        cleaned = re.sub(r"\n{3,}", "\n\n", chunk).strip()
        if len(cleaned) < 40 and deduped:
            continue
        if cleaned and cleaned not in deduped[-2:]:
            deduped.append(cleaned)
    return deduped


def _overlap_tail(text: str, overlap: int) -> str:
    if not text or overlap <= 0:
        return ""
    tail = text[-overlap:]
    sentences = [s for s in re.split(r"(?<=[.!?。])\s+", tail) if s.strip()]
    return " ".join(sentences).strip() if sentences else tail.strip()


# --------------------------------------------------------------------------- loaders


def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_pdf_file(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages[:200]:  # keep memory bounded on monster PDFs
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n\n".join(pages).strip()


def load_docx_file(path: Path) -> str:
    import docx

    document = docx.Document(str(path))
    parts = [p.text for p in document.paragraphs if p.text and p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text and cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts).strip()


def load_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in PDF_SUFFIXES:
        return load_pdf_file(path)
    if suffix in DOCX_SUFFIXES:
        return load_docx_file(path)
    return load_text_file(path)


_STRIP_TAGS = ("script", "style", "noscript", "svg", "canvas", "iframe", "form", "button", "input",
               "select", "textarea", "nav", "header", "footer", "aside")


def extract_main_content(html: str, url: str = "") -> tuple[str, str]:
    """Return ``(title, text)`` for the main content of a web page — no readability dependency."""
    document = lxml_html.fromstring(html)
    title_nodes = document.xpath("//h1/text()") or document.xpath("//title/text()")
    title = (title_nodes[0] or "").strip() if title_nodes else url

    for tag in _STRIP_TAGS:
        for node in document.iter(tag):
            node.getparent().remove(node) if node.getparent() is not None else None

    # (kind, length, text) — semantic containers win as long as they are not tiny.
    candidates: list[tuple[str, int, str]] = []
    for kind, xpath in (("article", "//article"), ("main", "//main"),
                        ("content", "//*[contains(@class,'content') or contains(@id,'content')]"),
                        ("body", "//body")):
        for node in document.xpath(xpath):
            text = " ".join(t.strip() for t in node.itertext() if t and t.strip())
            if len(text) > 200:
                candidates.append((kind, len(text), text))
    if not candidates:
        text = " ".join(t.strip() for t in document.itertext() if t and t.strip())
        return title, re.sub(r"\s{2,}", " ", text).strip()

    biggest = max(length for _, length, _ in candidates)
    semantic = [item for item in candidates if item[0] != "body" and item[1] >= 0.35 * biggest]
    chosen = max(semantic or candidates, key=lambda item: item[1])[2]
    return title, re.sub(r"\s{2,}", " ", chosen).strip()


class UnsafeURLError(ValueError):
    """Raised when a URL would reach a non-public host or uses a non-http scheme (SSRF guard)."""


def _is_public_ip(address: str) -> bool:
    import ipaddress

    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])  # drop IPv6 zone ids
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved
                or ip.is_unspecified)


def _resolve_host(host: str) -> list[str]:
    import socket

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise UnsafeURLError(f"host '{host}' does not resolve: {exc}") from exc
    return [str(info[4][0]) for info in infos]


def check_fetch_url(url: str, resolver: Callable[[str], list[str]] = _resolve_host) -> None:
    """Reject anything that is not a plain http(s) URL to a public host. Raises :class:`UnsafeURLError`.

    Every resolved address must be public, so a name that maps to both a public and a private address is refused.
    """
    from urllib.parse import urlsplit

    parts = urlsplit(url or "")
    if parts.scheme not in ("http", "https"):
        raise UnsafeURLError(f"only http and https URLs can be fetched (got '{parts.scheme or 'none'}')")
    host = (parts.hostname or "").strip().rstrip(".").lower()
    if not host:
        raise UnsafeURLError("URL has no host")
    if host == "localhost" or host.endswith(".localhost"):
        raise UnsafeURLError(f"host '{host}' is loopback")
    addresses = [host] if _looks_like_ip(host) else resolver(host)
    if not addresses:
        raise UnsafeURLError(f"host '{host}' has no addresses")
    for address in addresses:
        if not _is_public_ip(address):
            raise UnsafeURLError(f"host '{host}' resolves to a non-public address ({address})")


def _looks_like_ip(host: str) -> bool:
    import ipaddress

    try:
        ipaddress.ip_address(host.split("%", 1)[0])
        return True
    except ValueError:
        return False


async def fetch_url(url: str, timeout: float = 30.0, *, max_bytes: int = MAX_FETCH_BYTES,
                    resolver: Callable[[str], list[str]] = _resolve_host) -> tuple[str, str]:
    """Fetch a public web page and return ``(title, text)``.

    SSRF guard: the initial URL and every redirect target are checked by :func:`check_fetch_url`, and the body is
    streamed with a hard size cap so an oversized response is never fully read into memory.
    """
    current = url
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False,
                                 headers={"User-Agent": "ba-agent/0.1 (+local research tool)"}) as client:
        for _ in range(MAX_REDIRECTS + 1):
            check_fetch_url(current, resolver)
            async with client.stream("GET", current) as response:
                if response.is_redirect:
                    location = response.headers.get("location", "")
                    if not location:
                        raise UnsafeURLError("redirect without a Location header")
                    current = str(response.url.join(location))
                    continue
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > max_bytes:
                        raise UnsafeURLError(f"response larger than {max_bytes // (1024 * 1024)} MB")
                encoding = response.encoding or "utf-8"
                text = bytes(body).decode(encoding, errors="replace")
                break
        else:
            raise UnsafeURLError(f"more than {MAX_REDIRECTS} redirects")

    if "html" not in content_type and "<html" not in text[:500].lower():
        return url, text.strip()
    return extract_main_content(text, url)



# --------------------------------------------------------------------------- retrieval


@dataclass
class Retrieved:
    text: str
    source: str
    title: str = ""
    score: float = 0.0
    vector_score: float | None = None
    bm25_score: float | None = None
    chunk_id: int = -1

    def citation(self) -> str:
        label = self.title or self.source
        return f"[{label} · score {self.score:.3f}]\n{self.text}"


@dataclass
class _IndexEntry:
    chunk_id: int
    text: str
    source: str
    title: str
    tokens: list[str] = field(default_factory=list)
    vector: list[float] | None = None


_TOKEN_RE = re.compile(r"[a-z0-9à-öø-ÿ_]+", re.I)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > 1]


class KnowledgeService:
    def __init__(self, store: Store, gateway: ModelGateway, config: AppConfig | None = None) -> None:
        self.store = store
        self.gateway = gateway
        self.config = config or get_config()
        self._entries: list[_IndexEntry] | None = None
        self._bm25: Any | None = None
        self._matrix: Any = None  # normalised vectors, rows aligned with _entries

    # -- indexing -----------------------------------------------------------

    async def index_directory(self, directory: str | Path | None = None) -> dict[str, Any]:
        cfg = self.config.knowledge
        base = Path(directory or cfg.docs_dir)
        if not base.is_absolute():
            base = self.config.root / base
        result = {"scanned": 0, "indexed": 0, "skipped": 0, "failed": [], "embedded": False}
        if not base.exists():
            return result

        known = self.store.kb_hashes()
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            suffix = path.suffix.lower()
            if suffix not in TEXT_SUFFIXES | PDF_SUFFIXES | DOCX_SUFFIXES:
                continue
            result["scanned"] += 1
            key = f"file::{path}"
            try:
                raw = load_file(path)
            except Exception as exc:  # noqa: BLE001 — one bad file must not stop the index
                result["failed"].append(f"{path.name}: {type(exc).__name__}: {exc}")
                continue
            digest = sha256_text(raw)
            if known.get(key) == digest:
                result["skipped"] += 1
                continue
            outcome = await self._store_document(
                source="file", uri=str(path), title=path.name, text=raw, key_digest=digest
            )
            result["indexed"] += 1
            result["embedded"] = result["embedded"] or outcome
        self.invalidate()
        return result

    async def index_urls(self, urls: Sequence[str]) -> dict[str, Any]:
        result = {"scanned": len(urls), "indexed": 0, "skipped": 0, "failed": [], "embedded": False}
        known = self.store.kb_hashes()
        for url in urls:
            key = f"url::{url}"
            try:
                title, text = await fetch_url(url)
            except Exception as exc:  # noqa: BLE001
                result["failed"].append(f"{url}: {type(exc).__name__}: {exc}")
                continue
            digest = sha256_text(text)
            if known.get(key) == digest:
                result["skipped"] += 1
                continue
            outcome = await self._store_document(source="url", uri=url, title=title or url, text=text,
                                                 key_digest=digest)
            result["indexed"] += 1
            result["embedded"] = result["embedded"] or outcome
        self.invalidate()
        return result

    async def index_upload(self, filename: str, data: bytes) -> dict[str, Any]:
        """Validate, dedupe by content hash, then store one uploaded file in the knowledge folder."""
        if len(data) > MAX_UPLOAD_BYTES:
            return {"ok": False, "error": f"file larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"}
        suffix = Path(filename or "").suffix.lower()
        if suffix not in ALLOWED_UPLOAD_SUFFIXES:
            allowed = ", ".join(sorted(ALLOWED_UPLOAD_SUFFIXES))
            return {"ok": False, "error": f"file type '{suffix or 'none'}' is not allowed (allowed: {allowed})"}

        # Dedup on the content, before anything is written: the same bytes uploaded twice (under any name) must
        # not create a second knowledge-base entry. Digest on the raw bytes so it is cheap and format-agnostic.
        digest = hashlib.sha256(data).hexdigest()
        existing = self.store.kb_find_doc_by_digest(digest)
        if existing is not None:
            return {"ok": True, "duplicate": True, "path": existing.get("uri", ""), "embedded": None,
                    "existing_title": existing.get("title", "")}

        target_dir = self.config.root / self.config.knowledge.docs_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name) or "upload.txt"
        path = target_dir / safe_name
        counter = 1
        while path.exists():
            path = target_dir / f"{Path(safe_name).stem}_{counter}{path.suffix}"
            counter += 1
        path.write_bytes(data)

        try:
            raw = load_file(path)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "path": str(path)}
        outcome = await self._store_document(source="file", uri=str(path), title=safe_name, text=raw,
                                             key_digest=digest)
        self.invalidate()
        return {"ok": True, "path": str(path), "embedded": outcome}


    async def _store_document(self, *, source: str, uri: str, title: str, text: str, key_digest: str) -> bool:
        cfg = self.config.knowledge
        chunks = chunk_text(text, cfg.chunk_size, cfg.chunk_overlap)
        doc_id = uuid.uuid5(uuid.NAMESPACE_URL, f"{source}::{uri}").hex
        vectors = await self.gateway.embed(chunks) if chunks else None
        embedded = bool(vectors) and len(vectors) == len(chunks)
        payload: list[tuple[str, dict[str, Any], list[float] | None]] = []
        for index, chunk in enumerate(chunks):
            meta = {"source": source, "uri": uri, "title": title, "chunk_index": index}
            payload.append((chunk, meta, vectors[index] if embedded else None))
        # The parent row must exist first: kb_chunks has a foreign key onto kb_docs.
        self.store.kb_upsert_doc(doc_id=doc_id, source=source, uri=uri, sha256=key_digest, title=title,
                                 size_bytes=len(text.encode("utf-8", errors="replace")), chunks=len(chunks),
                                 embedded=embedded)
        self.store.kb_replace_chunks(doc_id, payload)
        return embedded

    def invalidate(self) -> None:
        self._entries = None
        self._bm25 = None
        self._matrix = None

    # -- retrieval ----------------------------------------------------------

    def _load_index(self) -> list[_IndexEntry]:
        if self._entries is not None:
            return self._entries
        rows = self.store.kb_all_chunks()
        entries = [
            _IndexEntry(chunk_id=int(row["id"]), text=row["text"], source=row.get("uri") or row.get("source") or "",
                        title=row.get("title") or "", tokens=tokenize(f"{row.get('title') or ''} {row['text']}"),
                        vector=row["vector"] or None)
            for row in rows
        ]
        self._entries = entries
        if entries:
            try:
                from rank_bm25 import BM25Okapi

                self._bm25 = BM25Okapi([entry.tokens for entry in entries])
            except Exception:
                self._bm25 = None
            vectors = [entry.vector for entry in entries if entry.vector]
            if vectors and len(vectors) >= max(1, int(len(entries) * 0.9)):
                try:
                    import numpy as np

                    matrix = np.asarray([entry.vector or [] for entry in entries], dtype="float32")
                    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    self._matrix = matrix / norms
                except Exception:
                    self._matrix = None
        return entries

    @property
    def has_vectors(self) -> bool:
        self._load_index()
        return self._matrix is not None

    async def search(self, query: str, k: int | None = None) -> list[Retrieved]:
        cfg = self.config.knowledge
        k = k or cfg.top_k
        entries = self._load_index()
        if not entries or not (query or "").strip():
            return []

        fused: dict[int, float] = {}
        vector_scores: dict[int, float] = {}
        bm25_scores: dict[int, float] = {}

        # 1) cosine similarity over cached embeddings
        if self._matrix is not None:
            query_vector = await self.gateway.embed([query])
            if query_vector and len(query_vector[0]) == self._matrix.shape[1]:
                import numpy as np

                q = np.asarray(query_vector[0], dtype="float32")
                norm = float(np.linalg.norm(q)) or 1.0
                scores = (self._matrix @ (q / norm)).tolist()
                ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[: k * 4]
                for rank, (index, score) in enumerate(ranked):
                    if score <= cfg.min_score:
                        continue
                    vector_scores[index] = float(score)
                    fused[index] = fused.get(index, 0.0) + 1.0 / (60 + rank)

        # 2) BM25 keyword search
        if self._bm25 is not None:
            scores = self._bm25.get_scores(tokenize(query))
            ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[: k * 4]
            for rank, (index, score) in enumerate(ranked):
                if score <= 0:
                    continue
                bm25_scores[index] = float(score)
                fused[index] = fused.get(index, 0.0) + 1.0 / (60 + rank)

        # 3) last resort: naive substring scoring so retrieval still works with no model at all
        if not fused:
            tokens = set(tokenize(query))
            for index, entry in enumerate(entries):
                overlap = len(tokens & set(entry.tokens[:400]))
                if overlap:
                    score = overlap / max(1.0, math.sqrt(len(entry.tokens)))
                    bm25_scores[index] = score
                    fused[index] = score

        ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:k]
        results = []
        for index, score in ordered:
            entry = entries[index]
            results.append(Retrieved(text=entry.text, source=entry.source, title=entry.title, score=float(score),
                                     vector_score=vector_scores.get(index), bm25_score=bm25_scores.get(index),
                                     chunk_id=entry.chunk_id))
        return results

    async def context_block(self, query: str, k: int | None = None, max_chars: int = 3000) -> str:
        """Prompt-ready block with citations; empty string when nothing relevant."""
        results = await self.search(query, k=k)
        if not results:
            return ""
        blocks: list[str] = []
        used = 0
        for item in results:
            chunk = item.citation()
            if used + len(chunk) > max_chars:
                break
            blocks.append(chunk)
            used += len(chunk)
        return "\n\n---\n\n".join(blocks)

    def status(self) -> dict[str, Any]:
        entries = self._load_index()
        docs = self.store.kb_docs()
        return {
            "documents": [
                {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in doc.items()} for doc in docs
            ],
            "chunks": len(entries),
            "vector_search": self._matrix is not None,
            "bm25": self._bm25 is not None,
            "embedding_model": self.config.active_profile.embedding_model,
        }
