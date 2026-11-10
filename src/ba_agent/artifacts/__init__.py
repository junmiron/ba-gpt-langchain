"""Deliverable rendering and PDF QA (Phase 2 of the redesign).

Primary engine is Typst (vendored binary); ReportLab + svglib is the pure-Python fallback.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from .reportlab_pdf import render_pdf as render_pdf_reportlab
from .typst import find_typst
from .typst import render_pdf as render_pdf_typst


def pdf_engine() -> str:
    """Which engine this machine will actually use, for logging and UI display."""
    return "typst" if find_typst() else ("reportlab" if _reportlab_ready() else "none")


def _reportlab_ready() -> bool:
    try:
        import reportlab  # noqa: F401

        return True
    except Exception:  # noqa: BLE001 - optional dependency
        return False


def render_spec_pdf(
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
    """Typst first, ReportLab fallback. Never raises: a failed PDF must not fail the run.

    Banding a tall diagram writes intermediate SVGs to disk. When the caller does not name a
    directory, they land in a temporary one that is removed on exit — not in the caller's cwd,
    which is how stray `process_diagram_N_band_M.svg` files end up littering the repo root.
    """
    if working_dir is not None:
        return _render_spec_pdf(spec_markdown, pdf_path, title, version, date, status, diagrams, working_dir)

    with tempfile.TemporaryDirectory(prefix="ba-render-") as scratch:
        return _render_spec_pdf(spec_markdown, pdf_path, title, version, date, status, diagrams, Path(scratch))


def _render_spec_pdf(
    spec_markdown: str,
    pdf_path: Path,
    title: str,
    version: str,
    date: str,
    status: str,
    diagrams: list[dict[str, Any]] | None,
    working_dir: Path,
) -> dict[str, Any]:
    fallback_reason = None
    if find_typst():
        result = render_pdf_typst(
            spec_markdown,
            pdf_path,
            title=title,
            version=version,
            date=date,
            status=status,
            diagrams=diagrams,
            working_dir=working_dir,
        )
        if result["ok"]:
            return result
        fallback_reason = result.get("error")
    else:
        fallback_reason = "typst binary not found"

    result = render_pdf_reportlab(
        spec_markdown,
        pdf_path,
        title=title,
        version=version,
        date=date,
        status=status,
        diagrams=diagrams,
        working_dir=working_dir,
    )
    result["fallback_reason"] = fallback_reason
    return result
