"""ReportLab fallback: the paths that only fire when Typst is absent.

The fallback engine is what a user gets on a machine without the vendored binary, so a fenced code
block and an unparseable SVG have to produce something readable rather than a blank page or a crash.
"""

from __future__ import annotations

from pathlib import Path

from ba_agent.artifacts.qa import _extract_text
from ba_agent.artifacts.reportlab_pdf import render_pdf


def test_fenced_code_reaches_the_pdf_as_text(tmp_path: Path):
    spec = (
        "# Claims Portal\n\n## Interface\n\n"
        'The intake endpoint is:\n\n```json\n{"claim": "C-1", "status": "open"}\n```\n\n'
        "Follow-up paragraph after the block.\n"
    )
    pdf = tmp_path / "spec.pdf"
    result = render_pdf(spec, pdf, title="Claims Portal", working_dir=tmp_path)

    assert result["ok"], result
    text = _extract_text(pdf)
    assert '"claim"' in text, "the code block was dropped from the document"
    assert "Follow-up paragraph" in text, "content after a code fence must still be laid out"


def test_an_unparseable_svg_degrades_to_a_note_not_a_missing_page(tmp_path: Path):
    diagram = {
        "title": "Intake flow",
        # A well-formed envelope with no drawable content: svglib returns a zero-size Drawing, which
        # is neither a crash nor a picture, so the reader must be told the figure is missing.
        "svg": '<svg xmlns="http://www.w3.org/2000/svg"><notAnElement/></svg>',
    }
    result = render_pdf(
        "# Claims Portal\n\n## Process\n\nThe flow below is diagrammed in section 2.\n",
        tmp_path / "spec.pdf",
        title="Claims Portal",
        diagrams=[diagram],
        working_dir=tmp_path,
    )
    assert result["ok"], result
    text = _extract_text(tmp_path / "spec.pdf")
    assert "Intake flow" in text, "the figure's title must survive even when the drawing fails"
    assert "could not be drawn" in text.lower(), "a failed diagram needs an explicit note for the reader"


def test_render_pdf_refuses_to_write_band_svg_into_an_unnamed_directory(tmp_path: Path):
    diagram = {
        "title": "Intake",
        "svg": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40"><rect width="40" height="40"/></svg>',
    }
    try:
        render_pdf("# Spec\n\n## Process\n", tmp_path / "spec.pdf", title="Spec", diagrams=[diagram])
    except ValueError as exc:
        assert "working_dir" in str(exc)
    else:
        raise AssertionError("render_pdf must not write band SVGs into the caller's cwd")
