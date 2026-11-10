"""PDF QA gate — Phase 2.

The sibling project's renderer had no assertions on its own output: a bad table conversion or a diagram cut
through a node produced a plausible-looking PDF nobody caught. This module checks the produced PDF against the
spec it came from, and fails loudly enough to show up in the timeline and the deliverable folder.

Checks (all evidence-based, read from the actual file):
  1. file opens and has > 0 pages
  2. no raw Markdown syntax leaked into the text (``|---|``, ``` ``` `, ``**``)
  3. every requirement ID in the spec appears in the PDF text
  4. the linked contents outline exists and covers the top-level headings
  5. diagrams are present, and any banded figure's cut rows contain no dark ink
"""

from __future__ import annotations

import html
import re
import tempfile
from pathlib import Path
from typing import Any

from .typst import cut_ink, prepare_diagrams

MARKDOWN_LEAKS = ("|---|", "```", "**", "| ID | Requirement |")


def _extract_text(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:  # noqa: BLE001 - QA must not crash the deliverable
        return ""


def _outline_titles(pdf_path: Path) -> list[str]:
    try:
        from pypdf import PdfReader

        def walk(items) -> list[str]:
            titles: list[str] = []
            for item in items or []:
                if isinstance(item, (list, tuple)):
                    titles.extend(walk(item))
                    continue
                if isinstance(item, dict):
                    title = item.get("/Title") or item.get("title")
                    if title:
                        titles.append(str(title))
                    nested = item.get("/Kids") or item.get("children")
                    if nested:
                        titles.extend(walk(nested))
                else:
                    title = getattr(item, "title", None)
                    if title:
                        titles.append(str(title))
                    children = getattr(item, "children", None)
                    if callable(children):
                        children = children()
                    if children:
                        titles.extend(walk(children))
            return titles

        return walk(PdfReader(str(pdf_path)).outline)
    except Exception:  # noqa: BLE001 - QA must not crash the deliverable
        return []


def _link_annotations(pdf_path: Path) -> int:
    """Count internal link annotations — a Contents block that is not clickable is not a Contents block.

    Typst emits the outline entries as `/Link` annotations whose `/Dest` is an indirect reference, so the
    check cannot rely on `/Dest` being an array.
    """
    try:
        from pypdf import PdfReader

        count = 0
        for page in PdfReader(str(pdf_path)).pages:
            for annot in page.get("/Annots") or []:
                obj = annot.get_object()
                if obj.get("/Subtype") != "/Link":
                    continue
                # Typst writes /Dest, ReportLab writes an /A GoTo action — both are clickable.
                if obj.get("/Dest") is not None or obj.get("/A") is not None:
                    count += 1
        return count
    except Exception:  # noqa: BLE001 - QA must not crash the deliverable
        return 0


def audit_html(html_path: Path) -> dict:
    """QA gate for the HTML deliverable: every Contents entry must actually reach a heading.

    A TOC whose anchors do not resolve is worse than no TOC, so this checks anchor coverage
    rather than a raw link count.
    """
    try:
        text = html_path.read_text(encoding="utf-8")
    except OSError:
        return {"ok": False, "checks": {}, "warnings": ["HTML deliverable is unreadable"]}

    toc = re.search(r'<nav class="toc">(.*?)</nav>', text, re.S)
    links = re.findall(r'<a href="#([^"]+)"', toc.group(1)) if toc else []
    ids = set(re.findall(r'<h[123] id="([^"]+)"', text))
    unresolved = [link for link in links if link not in ids]
    checks = {
        "toc_present": bool(toc),
        "toc_links": len(links),
        "headings": len(ids),
        "unresolved_links": unresolved[:10],
        "inline_svg": text.count("<svg"),
    }
    warnings = []
    if not toc:
        warnings.append("no Contents navigation block")
    elif not links:
        warnings.append("Contents block is empty")
    elif unresolved:
        warnings.append(f"{len(unresolved)} Contents links point at anchors that do not exist")
    return {"ok": bool(toc and links and not unresolved), "checks": checks, "warnings": warnings}


def audit_pdf(pdf_path: Path, spec_markdown: str, diagrams: list[dict]) -> dict:
    """Return a QA report. ``ok`` is False only for structural failures, not soft warnings."""
    text = _extract_text(pdf_path)
    requirement_ids = sorted(
        {m.group(0) for m in re.finditer(r"\b(?:FR|NFR|INT|ASM|DEC|RISK|Q)-\d{3}\b", spec_markdown or "")}
    )
    missing_ids = [rid for rid in requirement_ids if rid not in text]
    leaks = [marker for marker in MARKDOWN_LEAKS if marker in text]
    headings = [line.lstrip("#").strip() for line in (spec_markdown or "").split("\n") if re.match(r"^#{1,3}\s", line)]
    outline = _outline_titles(pdf_path)

    # Compare on a normalised form: engines may emit "Risks &amp; Dependencies" where the source said
    # "Risks & Dependencies", and a whitespace-only difference must not read as a missing bookmark.
    def _normalise(value: str) -> str:
        return html.unescape(value).replace("\u00a0", " ").strip().lower()

    outline_blob = " ".join(_normalise(title) for title in outline)
    missing_outline = [heading for heading in headings if heading and _normalise(heading) not in outline_blob]

    checks: dict[str, Any] = {
        "pages": 0,
        "text_chars": len(text),
        "requirements_expected": len(requirement_ids),
        "requirements_missing": missing_ids,
        "markdown_leaks": leaks,
        "outline_entries": len(outline),
        "headings_without_outline": missing_outline[:10],
        "contents_links": _link_annotations(pdf_path),
        "diagram_bands": 0,
        "diagram_render_failures": [],
        "band_cut_weight": None,
    }
    try:
        from pypdf import PdfReader

        checks["pages"] = len(PdfReader(str(pdf_path)).pages)
    except Exception:  # noqa: BLE001 - page count is informational
        pass

    # Re-run the banding the renderer used, so QA measures the exact rows the PDF will slice at.
    with tempfile.TemporaryDirectory(prefix="ba_qa_") as working:
        for figure in prepare_diagrams(diagrams or [], Path(working)):
            checks["diagram_bands"] += len(figure.files)
            if figure.cuts:
                checks["band_cut_weight"] = max(checks["band_cut_weight"] or 0, cut_ink(figure.source_svg, figure.cuts))

    # A diagram with code but no SVG means the figure is silently missing from the PDF. Name the real
    # reason: a parse error is a code problem, not a missing renderer, and telling the user to install
    # mermaid-cli when it is already installed sends them hunting the wrong thing.
    for diagram in diagrams or []:
        if (diagram.get("code") or "").strip() and not (diagram.get("svg") or "").strip():
            title = diagram.get("title") or "untitled diagram"
            reason = (diagram.get("render_error") or "").strip()
            checks["diagram_render_failures"].append(
                f"{title}: {reason[:160]}" if reason else f"{title}: no rendered SVG (no Mermaid renderer available)"
            )

    warnings: list[str] = []
    if checks["diagram_render_failures"]:
        warnings.append(
            f"{len(checks['diagram_render_failures'])} diagram(s) are missing from the PDF — "
            f"{'; '.join(checks['diagram_render_failures'])}"
        )
    if outline and checks["contents_links"] < len(outline):
        warnings.append(
            f"Contents has {len(outline)} outline entries but only {checks['contents_links']} clickable links"
        )
    if missing_outline:
        warnings.append(f"{len(missing_outline)} headings have no bookmark entry")
    if (checks["band_cut_weight"] or 0) > 6:
        warnings.append(
            f"diagram band cut slices through ink (weight {checks['band_cut_weight']}) — "
            "it should land in the gap between nodes"
        )

    ok = checks["pages"] > 0 and not leaks and not missing_ids
    return {"ok": ok, "engine_pdf": str(pdf_path), "checks": checks, "warnings": warnings}
