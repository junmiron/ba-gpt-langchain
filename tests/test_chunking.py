"""Chunking and document loading: the paths an uploaded file actually goes through.

A bad chunk size is not just a quality problem — it can hang the ingest loop, and a table-heavy DOCX or a
page that fails to extract must not abort the whole document.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ba_agent.config import AppConfig, KnowledgeConfig
from ba_agent.knowledge import _paragraph_chunks, load_docx_file, load_pdf_file


def test_paragraph_chunks_terminates_on_a_degenerate_chunk_size():
    """chunk_size <= 0 used to spin forever: sentence[:0] is empty and the remainder never shrinks."""
    text = "\n\n".join(f"Sentence number {i} states a requirement for the claims portal." for i in range(40))
    chunks = _paragraph_chunks(text, 0, 0)
    assert chunks and all(len(c) <= 50 for c in chunks)


def test_overlap_cannot_exceed_the_chunk_size():
    text = "Short sentence here, and another one following it, with a third for length. " * 20
    chunks = _paragraph_chunks(text, 200, 5000)
    assert chunks
    assert all(len(c) <= 200 for c in chunks), "overlap larger than chunk_size inflated a chunk"


def test_chunks_are_stable_and_cover_the_document():
    text = "\n\n".join(
        f"Paragraph {i}: the system must do something specific number {i}, described in enough words to matter."
        for i in range(12)
    )
    chunks = _paragraph_chunks(text, 400, 80)
    joined = "\n".join(chunks)
    assert "number 11" in joined and "number 0" in joined
    assert len(chunks) >= 3
    assert all(len(c) <= 400 for c in chunks)


def test_empty_and_whitespace_documents_produce_no_chunks():
    assert _paragraph_chunks("", 900, 100) == []
    assert _paragraph_chunks("   \n\n\n  \n", 900, 100) == []


def test_config_rejects_a_chunk_size_that_would_hang_the_ingest():
    with pytest.raises(ValidationError):
        AppConfig(knowledge=KnowledgeConfig(chunk_size=0))
    with pytest.raises(ValidationError):
        AppConfig(knowledge=KnowledgeConfig(chunk_overlap=-5))


def test_docx_tables_are_read_as_rows():
    docx = pytest.importorskip("docx")

    document = docx.Document()
    document.add_paragraph("The portal must log every claim.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "SLA"
    table.cell(0, 1).text = "4 hours"
    table.cell(1, 0).text = "Escalation"
    table.cell(1, 1).text = "on-call BA"

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as working:
        file_path = Path(working) / "doc.docx"
        document.save(str(file_path))
        text = load_docx_file(file_path)

    assert "The portal must log every claim." in text
    assert "SLA | 4 hours" in text
    assert "Escalation | on-call BA" in text


def test_pdf_extraction_survives_a_page_without_text():
    pypdf = pytest.importorskip("pypdf")
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as working:
        file_path = Path(working) / "doc.pdf"
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=612, height=792)  # no text layer at all
        with open(file_path, "wb") as handle:
            writer.write(handle)

        text = load_pdf_file(file_path)

    assert isinstance(text, str)


def test_every_chunk_stays_within_the_configured_size():
    prose = "\n\n".join(
        f"Requirement {i}: the claims portal must route, log and escalate a claim within four hours, "
        f"and the on-call analyst receives the audit trail for number {i}."
        for i in range(40)
    )
    for chunk_size, overlap in [(400, 80), (900, 140), (250, 50), (200, 5000)]:
        chunks = _paragraph_chunks(prose, chunk_size, overlap)
        assert chunks, f"{chunk_size}/{overlap} produced nothing"
        oversized = [c for c in chunks if len(c) > chunk_size]
        assert not oversized, f"{chunk_size}/{overlap} produced {len(oversized)} oversized chunk(s)"
