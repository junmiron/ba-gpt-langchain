"""ReportLab + svglib fallback PDF, used when the Typst binary is not available.

Deliberately a *fallback*, not the primary engine: its outline comes from a two-pass TOC rather than a native
bookmark tree, so the Typst path stays the default. It exists so a machine with no Typst binary still gets a
real PDF deliverable rather than nothing.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

from .typst import CONTENT_HEIGHT_PT, CONTENT_WIDTH_PT, prepare_diagrams


def _available() -> bool:
    try:
        import reportlab  # noqa: F401
        from reportlab.lib.pagesizes import A4  # noqa: F401

        return True
    except Exception:  # noqa: BLE001 - the fallback is optional; Typst covers it when present
        return False


def _paragraph(text: str) -> str:
    escaped = (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<![\w])\*([^*\s][^*]*?)\*(?![\w])", r"<i>\1</i>", escaped)
    return escaped.replace("\\#", "#")


def _split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def render_pdf(
    spec_markdown: str,
    pdf_path: Path,
    *,
    title: str,
    version: str = "0.1",
    date: str = "",
    status: str = "Draft",
    diagrams: list[dict[str, Any]] | None = None,
    working_dir: Path | None = None,
) -> dict[str, Any]:
    """Write a PDF with ReportLab. Returns ``{"ok": bool, "engine": "reportlab", "error": str | None}``."""
    if not _available():
        return {"ok": False, "engine": "reportlab", "error": "reportlab not installed"}

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from reportlab.platypus.tableofcontents import TableOfContents

    styles = getSampleStyleSheet()
    body = ParagraphStyle("spec-body", parent=styles["BodyText"], fontSize=9.5, leading=12.5, spaceAfter=6)
    h1 = ParagraphStyle(
        "spec-h1", parent=styles["Heading1"], fontSize=15, spaceAfter=10, textColor=colors.HexColor("#111827")
    )
    h2 = ParagraphStyle(
        "spec-h2", parent=styles["Heading2"], fontSize=12, spaceAfter=8, textColor=colors.HexColor("#1f2937")
    )
    h3 = ParagraphStyle("spec-h3", parent=styles["Heading3"], fontSize=10.5, spaceAfter=6)
    # The cover title is not a section, so it gets its own style name and never enters the Contents.
    cover_title = ParagraphStyle("spec-title", parent=h1, fontSize=19)
    meta = ParagraphStyle("spec-meta", parent=body, fontSize=9, textColor=colors.HexColor("#6b7280"))

    class SpecDoc(SimpleDocTemplate):
        """ReportLab needs an explicit hook to turn headings into bookmarks + TOC entries.

        Without this the fallback PDF has no outline at all, and the QA gate would report a Contents
        that is not clickable. ReportLab also rejects an outline that jumps a level (H1 straight to
        H3), so the level is clamped to one deeper than the last emitted entry.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._last_outline_level = -1

        def handle_documentBegin(self) -> None:
            # multiBuild re-runs the document with a fresh canvas; the outline stack restarts each pass.
            self._last_outline_level = -1
            super().handle_documentBegin()

        def afterFlowable(self, flowable: Any) -> None:
            if not isinstance(flowable, Paragraph):
                return
            name = flowable.style.name
            level = {"spec-h1": 0, "spec-h2": 1, "spec-h3": 2}.get(name)
            if level is None:
                return
            level = min(level, self._last_outline_level + 1)
            self._last_outline_level = level
            key = f"h-{self.seq.nextf('heading')}"
            # flowable.text is ReportLab markup: tags must be stripped *and* entities decoded, or a
            # heading like "User Roles & Personas" shows up in the bookmark tree as "User Roles &amp; Personas".
            plain = html.unescape(re.sub(r"<[^>]+>", "", flowable.text)).strip()
            # bookmarkPage alone only makes a named destination; addOutlineEntry builds the visible tree,
            # and the TOC only links entries it received through the 'TOCEntry' notification.
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(plain, key, level=level, closed=0)
            self.notify("TOCEntry", (level, plain, self.page, key))

    document = SpecDoc(
        str(pdf_path),
        pagesize=A4,
        leftMargin=22 * mm,
        rightMargin=22 * mm,
        topMargin=24 * mm,
        bottomMargin=24 * mm,
        title=title,
        author="BA agent",
    )

    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("toc-1", fontName="Helvetica-Bold", fontSize=10, leading=13, spaceAfter=3),
        ParagraphStyle("toc-2", fontName="Helvetica", fontSize=9, leading=12, leftIndent=14, spaceAfter=2),
    ]

    story: list[Any] = [
        Paragraph(title, cover_title),
        Spacer(1, 8),
        Paragraph(f"Version {version} · {date} · {status}", meta),
        PageBreak(),
        toc,
        PageBreak(),
    ]

    lines = (spec_markdown or "").replace("\r\n", "\n").split("\n")
    index = 0
    in_code = False
    code: list[str] = []
    while index < len(lines):
        line = lines[index]
        if line.strip().startswith("```"):
            if in_code:
                story.append(Paragraph("<br/>".join(_diagram_svg_text(c) for c in code) or "", body))
                code = []
                in_code = False
            else:
                in_code = True
            index += 1
            continue
        if in_code:
            code.append(line)
            index += 1
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            level = len(heading.group(1))
            story.append(Paragraph(_paragraph(heading.group(2)), {1: h1, 2: h2}.get(level, h3)))
            index += 1
            continue

        if re.match(r"^\s*\|.*\|\s*$", line):
            table_lines = []
            while index < len(lines) and re.match(r"^\s*\|.*\|\s*$", lines[index]):
                table_lines.append(lines[index])
                index += 1
            if len(table_lines) >= 2 and re.match(r"^\s*\|[\s|:-]+\|\s*$", table_lines[1]):
                data = [
                    [Paragraph(_paragraph(cell), body) for cell in _split_row(row)]
                    for row in [table_lines[0], *table_lines[2:]]
                ]
                table = Table(data, repeatRows=1, hAlign="LEFT")
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d0d5dd")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                        ]
                    )
                )
                story.append(table)
            continue

        if re.match(r"^\s*>", line):
            story.append(Paragraph(_paragraph(line.strip().lstrip("> ")), body))
            index += 1
            continue

        if re.match(r"^\s*[-*+]\s+", line) or re.match(r"^\s*\d+\.\s+", line):
            story.append(Paragraph(_paragraph(line.strip()), body))
            index += 1
            continue

        if line.strip():
            story.append(Paragraph(_paragraph(line.strip()), body))
        index += 1

    figure = _diagram_block(diagrams or [], styles, body, h2, working_dir)
    if figure:
        story.append(PageBreak())
        story.extend(figure)

    try:
        # two passes: the first collects the TOC entries, the second lays them out with page numbers
        document.multiBuild(story)
    except Exception as exc:  # noqa: BLE001 - a PDF-render failure must not fail the deliverable
        return {"ok": False, "engine": "reportlab", "error": f"{type(exc).__name__}: {exc}"}

    from .typst import pdf_pages

    return {"ok": pdf_path.exists(), "engine": "reportlab", "error": None, "pages": pdf_pages(pdf_path)}


def _diagram_svg_text(chunk: str) -> str:
    return _paragraph(chunk.replace("<", "&lt;"))


def _diagram_block(
    diagrams: list[dict[str, Any]], styles: Any, body: Any, h2: Any, working_dir: Path | None
) -> list[Any]:
    """Embed diagrams as vector paths, banded to the page box.

    The Drawing svglib returns is itself a Flowable, so no rasterising is needed — which matters
    because `renderPM` (the PNG route) requires rlPyCairo, and without it every diagram would have
    silently degraded into a block of Mermaid source. If svglib cannot parse the SVG at all we print
    the source as code rather than dropping the figure.
    """
    from reportlab.platypus import Paragraph, Spacer
    from svglib.svglib import svg2rlg

    story: list[Any] = []
    if working_dir is None:
        raise ValueError("render_pdf requires a working_dir; band SVGs must not be written to the caller's cwd")
    figures = prepare_diagrams(diagrams or [], working_dir)
    for figure in figures:
        for band_file, caption in zip(figure.files, figure.captions, strict=True):
            story.append(Paragraph(_paragraph(caption), h2))
            drawing = None
            try:
                drawing = svg2rlg(band_file)
            except Exception:  # noqa: BLE001 - a malformed SVG should not fail the deliverable
                drawing = None
            if drawing is not None and drawing.width and drawing.height:
                # Each band is already sized to fit the content box; clamp anyway so a band whose
                # intrinsic size disagrees with its viewBox cannot overflow the page.
                scale = min(CONTENT_WIDTH_PT / drawing.width, (CONTENT_HEIGHT_PT - 60.0) / drawing.height, 1.0)
                drawing.width, drawing.height = drawing.width * scale, drawing.height * scale
                story.append(drawing)
            else:
                story.append(Paragraph(_paragraph(_fallback_note(figure)), body))
            story.append(Spacer(1, 10))
    return story


def _fallback_note(figure: Any) -> str:
    """What a reader sees when the fallback engine cannot draw a band."""
    return f"{figure.title} could not be drawn by the fallback engine; see process_diagrams.md for the source."
