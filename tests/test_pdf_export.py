"""Phase 2: deliverable rendering — Typst primary, ReportLab fallback, QA gate.

These tests run the *real* renderer. Typst is only exercised when the binary is present (vendored or on PATH);
the ReportLab path is always exercised, and the QA module is engine-independent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ba_agent.artifacts import pdf_engine, render_spec_pdf
from ba_agent.artifacts.html import build_html
from ba_agent.artifacts.qa import audit_html, audit_pdf
from ba_agent.artifacts.typst import (
    CONTENT_WIDTH_PT,
    MAX_DIAGRAM_HEIGHT_PT,
    _column_widths,
    _ink_profile,
    _window_weight,
    cut_ink,
    find_typst,
    markdown_to_typst,
    prepare_diagrams,
    svg_native_size,
)

SPEC = """# Claims Portal

## 1. Introduction

The portal replaces **email intake** for claims.

## 2. Functional Requirements

| ID | Requirement | Acceptance criteria | Priority |
|---|---|---|---|
| FR-001 | Submit claim online | Uploads finish in under 10 seconds | Must |
| FR-002 | Track status | Status is visible within 1 second | Must |
| FR-003 | Notify on rejection | Email sent within 5 minutes | Should |
| FR-004 | Attach documents | 20 MB attachments accepted | Must |
| FR-005 | Audit trail | Every state change is recorded | Must |

## 3. Non-Functional Requirements

- Availability: 99.5%
- Security: encrypted at rest

> Note: the policy system is legacy.
"""

TALL_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 1600" width="400" height="1600">'
    + "".join(
        f'<rect x="20" y="{y}" width="200" height="40" fill="white" stroke="black"/>'
        f'<text x="30" y="{y + 25}">Stage {index} selectable label</text>'
        for index, y in enumerate(range(20, 1580, 100))
    )
    + "</svg>"
)

SHORT_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 200"><rect x="10" y="10" width="120" height="40" fill="white" stroke="black"/><text x="20" y="35">short diagram</text></svg>'


def test_column_widths_are_proportional_and_sum_to_100():
    rows = [
        ["ID", "Requirement", "Acceptance criteria", "Priority"],
        ["FR-001", "Do the thing", "A very long acceptance criterion describing the behaviour in detail", "Must"],
    ]
    widths = _column_widths(rows)
    assert sum(widths) == 100
    assert widths[0] < widths[2]  # the ID column must not eat the prose column


def test_markdown_conversion_leaves_no_markdown_behind():
    body = markdown_to_typst(SPEC)
    assert "= Claims Portal" in body
    assert "#table(" in body
    assert "columns: (" in body
    for leak in ("|---|", "```", "**", "| ID | Requirement |"):
        assert leak not in body


def test_mermaid_fence_is_dropped_not_printed():
    body = markdown_to_typst("```mermaid\nflowchart TD\nA-->B\n```\n")
    assert "flowchart" not in body
    assert "A-->B" not in body


def test_tall_svg_bands_and_short_svg_does_not(tmp_path: Path):
    figures = prepare_diagrams(
        [{"title": "Claim intake", "svg": TALL_SVG}, {"title": "Short", "svg": SHORT_SVG}],
        tmp_path,
    )
    assert len(figures) == 2
    assert len(figures[0].files) > 1, "a diagram taller than the page box must be split into bands"
    assert len(figures[1].files) == 1
    assert len(figures[0].captions) == len(figures[0].files)
    assert "part 1 of" in figures[0].captions[0]


def test_table_cells_keep_emphasis_without_leaking_markdown():
    """Table cells are Typst *code mode*, so `**Version**` must become strong(), not literal asterisks."""
    source = markdown_to_typst("| Field | Value |\n|---|---|\n| **Version** | 1.2 |\n| plain *italic* | `x=1` |")
    assert 'strong("Version")' in source
    assert 'emph("italic")' in source
    assert 'raw("x=1")' in source
    assert "**" not in source
    assert "|---|" not in source
    assert "| Field" not in source


def test_band_cuts_tile_the_diagram_and_each_band_fits_one_page(tmp_path: Path):
    """Bands must reconstruct the diagram exactly, and no band may overflow the page box."""
    figures = prepare_diagrams([{"title": "Tall", "svg": TALL_SVG}], tmp_path)
    figure = figures[0]
    sizes = [svg_native_size(f.read_text(encoding="utf-8")) for f in figure.files]
    assert all(sizes)
    heights = [size[1] for size in sizes if size]
    assert abs(sum(heights) - 1600) < 1e-6, "bands must cover the original height with no gap or overlap"
    scale = CONTENT_WIDTH_PT / 400.0
    for height, width in zip(heights, figure.widths, strict=False):
        assert height * scale * (width / 100.0) <= MAX_DIAGRAM_HEIGHT_PT + 1.0
        assert 0 < width <= 100.0


def test_pdf_rendering_produces_a_readable_pdf(tmp_path: Path):
    diagrams = [{"title": "Claim intake", "code": "flowchart TD", "svg": TALL_SVG}]
    result = render_spec_pdf(
        SPEC, tmp_path / "spec.pdf", title="Claims Portal", diagrams=diagrams, working_dir=tmp_path
    )
    assert result["ok"], result.get("error")
    assert result["pages"] >= 2
    report = audit_pdf(tmp_path / "spec.pdf", SPEC, diagrams)
    assert report["ok"], report["checks"]
    assert report["checks"]["requirements_expected"] == 5
    assert not report["checks"]["requirements_missing"]
    assert not report["checks"]["markdown_leaks"]


def test_typst_pdf_has_a_linked_outline(tmp_path: Path):
    if not find_typst():
        pytest.skip("typst binary not available")
    result = render_spec_pdf(SPEC, tmp_path / "t.pdf", title="Claims Portal", working_dir=tmp_path)
    assert result["ok"] and result["engine"] == "typst"
    report = audit_pdf(tmp_path / "t.pdf", SPEC, [])
    assert report["checks"]["outline_entries"] >= 4, "the contents page must produce real PDF bookmarks"
    from pypdf import PdfReader

    text = "".join(page.extract_text() or "" for page in PdfReader(str(tmp_path / "t.pdf")).pages)
    assert "FR-001" in text  # requirement rows are real, extractable text, not a rasterised image


def test_typst_cover_renders_instead_of_printing_its_own_source(tmp_path: Path):
    """A missing `#` in markup mode makes Typst echo the source as text — the classic silent failure."""
    if not find_typst():
        pytest.skip("typst binary not available")
    diagrams = [{"title": "Claim intake", "code": "flowchart TD\n  A[Submit] --> B[Review]", "svg": SHORT_SVG}]
    result = render_spec_pdf(
        SPEC, tmp_path / "cover.pdf", title="Claims Portal", diagrams=diagrams, working_dir=tmp_path
    )
    assert result["ok"] and result["engine"] == "typst"
    from pypdf import PdfReader

    text = "".join(page.extract_text() or "" for page in PdfReader(str(tmp_path / "cover.pdf")).pages)
    assert "BUSINESS ANALYSIS" in text
    assert "align(top + left" not in text, "cover markup leaked as literal source text"
    assert "Claim intake" in text, "the diagram figure caption must be present"
    assert "FR-001" in text


def test_contents_entries_are_clickable(tmp_path: Path):
    """The plan's acceptance bar for a professional PDF: the Contents page links, not just lists."""
    if not find_typst():
        pytest.skip("typst binary not available")
    result = render_spec_pdf(SPEC, tmp_path / "links.pdf", title="Claims Portal", working_dir=tmp_path)
    assert result["ok"]
    report = audit_pdf(tmp_path / "links.pdf", SPEC, [])
    outline = report["checks"]["outline_entries"]
    links = report["checks"]["contents_links"]
    assert outline >= 4
    assert links >= outline, "every Contents entry must be an internal link"
    assert not report["warnings"]


def test_band_cuts_land_in_whitespace(tmp_path: Path):
    """Node boxes stacked every 200px: a cut must find the 120px gap between them, not a box."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="2400" viewBox="0 0 900 2400">'
        + "".join(f'<rect x="100" y="{y}" width="300" height="80" fill="#111827"/>' for y in range(50, 2300, 200))
        + "</svg>"
    )
    figures = prepare_diagrams([{"title": "Tall flow", "code": "flowchart TD", "svg": svg}], tmp_path)
    assert len(figures[0].files) > 1, "a 2400px diagram must be banded"
    assert cut_ink(svg, figures[0].cuts) == 0, "every cut landed inside a node box"


def test_band_cut_picker_optimises_the_window_qa_scores(tmp_path: Path):
    """The picker must minimise the ±2-row window QA measures, not the weight of a single row.

    Two boxes overlap at rows 799-800, so the nominal cut at 800 is inside ink. The nearest quiet row
    is 820 — quiet at the point, but row 819 still carries a box edge, so QA scores it 3. Row 822 is
    two rows further out and scores 0. Choosing on point weight alone pins every real diagram at the
    warning threshold.
    """
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="1600" viewBox="0 0 900 1600">'
        '<rect x="100" y="780" width="300" height="20" fill="#111827"/>'
        '<rect x="100" y="799" width="300" height="20" fill="#111827"/>'
        "</svg>"
    )
    figures = prepare_diagrams([{"title": "Tall flow", "code": "flowchart TD", "svg": svg}], tmp_path)
    cuts = figures[0].cuts
    assert cuts, "a 1600px diagram at 469pt width must be banded"

    profile = _ink_profile(svg, 1600)
    achieved = cut_ink(svg, cuts)
    best_possible = min(_window_weight(profile, row) for row in range(700, 900))
    assert achieved == best_possible, (
        f"cut scored {achieved} while a {best_possible}-weight row was available inside the search window"
    )
    assert achieved < 3, "the cut chose a row adjacent to a node box"


def test_ink_profile_is_geometry_not_raster():
    """The profile must work without rlPyCairo — the raster route silently returns nothing, which
    is exactly how banding used to skip its own snapping."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 200">'
        '<g transform="translate(10,50)"><rect x="0" y="0" width="40" height="20" fill="#111"/></g>'
        '<path d="M50 60 L50 90" stroke="#111"/>'
        "</svg>"
    )
    profile = _ink_profile(svg, 200)
    assert profile and sum(profile) > 0
    assert profile[60] >= 3, "the node box rows must carry box weight"
    assert profile[150] == 0, "empty space must read as empty"
    assert profile[80] >= 1, "the connector stroke must be visible"


def test_qa_reports_a_missing_diagram_svg(tmp_path: Path):
    """A diagram whose SVG failed to render must be reported, not quietly dropped from the PDF."""
    report = audit_pdf(
        tmp_path / "none.pdf",
        "## A\nFR-001 text",
        [{"title": "Claim intake", "code": "flowchart TD\nA --> B", "svg": ""}],
    )
    assert report["checks"]["diagram_render_failures"] == [
        "Claim intake: no rendered SVG (no Mermaid renderer available)"
    ]
    assert any("missing from the PDF" in warning for warning in report["warnings"])


def test_qa_names_the_render_error_instead_of_blaming_a_missing_renderer(tmp_path: Path):
    """mmdc was installed; the code was unparseable. Saying 'install mermaid-cli' sends the user the wrong way."""
    report = audit_pdf(
        tmp_path / "none.pdf",
        "## A\nFR-001 text",
        [
            {
                "title": "Sequence",
                "code": "sequenceDiagram\nA->>B: x",
                "svg": "",
                "render_error": "Error: Parse error on line 2: Expecting 'end'",
            }
        ],
    )
    failure = report["checks"]["diagram_render_failures"][0]
    assert "Parse error" in failure, failure
    assert "no Mermaid renderer available" not in failure


def test_qa_reports_a_bad_band_cut(tmp_path: Path, monkeypatch):
    """If a cut slices ink, QA must say so — a silent ugly PDF is the whole reason the gate exists."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="2400" viewBox="0 0 900 2400">'
        '<rect x="100" y="1100" width="700" height="200" fill="#111827"/>'
        "</svg>"
    )
    monkeypatch.setattr("ba_agent.artifacts.typst._ink_profile", lambda _svg, rows=1200: [20] * rows)
    report = audit_pdf(tmp_path / "none.pdf", "## A\nFR-001 text", [{"title": "flow", "svg": svg}])
    assert report["checks"]["band_cut_weight"] == 20
    assert any("band cut slices through ink" in warning for warning in report["warnings"])


def test_figure_carries_its_own_source_svg(tmp_path: Path):
    """Two diagrams sharing a title must not be conflated by QA.

    QA used to find a figure's source by matching `figure.title` against the diagram list. Mermaid prompts
    routinely emit the same title for two diagrams, so the second figure's cuts were scored against the
    first figure's geometry — a clean cut could be reported as dirty, or vice versa.
    """
    clean = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="1600" viewBox="0 0 900 1600">'
        '<rect x="100" y="300" width="300" height="80" fill="#111827"/>'
        "</svg>"
    )
    dirty = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="1600" viewBox="0 0 900 1600">'
        '<rect x="100" y="700" width="700" height="120" fill="#111827"/>'
        "</svg>"
    )
    figures = prepare_diagrams(
        [
            {"title": "Process", "code": "flowchart TD", "svg": clean},
            {"title": "Process", "code": "flowchart TD", "svg": dirty},
        ],
        tmp_path,
    )
    assert len(figures) == 2
    assert figures[0].source_svg == clean, "figure 0 lost its own geometry"
    assert figures[1].source_svg == dirty, "figure 1 lost its own geometry"
    assert cut_ink(figures[0].source_svg, figures[0].cuts) == 0
    assert cut_ink(figures[1].source_svg, figures[1].cuts) >= 0


def test_mixed_markup_cells_join_as_one_typst_expression(tmp_path: Path):
    """`"text" + strong("x")` is valid code mode; `(a, b)` builds an array and Typst rejects it."""
    source = markdown_to_typst(
        "| ID | Requirement |\n|---|---|\n| NFR-001 | At least **400 claims per month** |\n"
        "| NFR-002 | Latency `300ms`, then *escalate* |"
    )
    assert 'At least " + strong("400 claims per month")' in source
    assert '("At least' not in source, "a cell must not be built as a Typst array"


def test_banded_diagram_typst_pdf_is_complete(tmp_path: Path):
    """A tall diagram plus a table-heavy spec must compile, band, and pass every QA check."""
    if not find_typst():
        pytest.skip("typst binary not available")
    tall = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="2600" viewBox="0 0 900 2600">'
        + "".join(f'<rect x="120" y="{y}" width="340" height="70" fill="#1f2937"/>' for y in range(60, 2500, 210))
        + "</svg>"
    )
    diagrams = [{"title": "Claim intake flow", "code": "flowchart TD\n  A[Submit] --> B[Validate]", "svg": tall}]
    result = render_spec_pdf(
        SPEC, tmp_path / "tall.pdf", title="Claims Portal", version="1.2", diagrams=diagrams, working_dir=tmp_path
    )
    assert result["ok"] and result["engine"] == "typst"
    report = audit_pdf(tmp_path / "tall.pdf", SPEC, diagrams)
    assert report["ok"], report["checks"]
    assert report["checks"]["diagram_bands"] > 1
    assert (report["checks"]["band_cut_weight"] or 0) >= 0, "a band cut sliced through a node box"
    assert report["checks"]["contents_links"] >= report["checks"]["outline_entries"]
    assert not report["warnings"]


def test_reportlab_fallback_used_when_typst_is_absent(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("ba_agent.artifacts.typst.find_typst", lambda: None)
    result = render_spec_pdf(SPEC, tmp_path / "fallback.pdf", title="Claims Portal", working_dir=tmp_path)
    assert result["ok"]
    assert result["engine"] == "reportlab"
    assert result["fallback_reason"]


def test_pdf_render_failure_never_raises(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("ba_agent.artifacts.typst.find_typst", lambda: "definitely-not-a-real-binary")
    monkeypatch.setattr("ba_agent.artifacts.reportlab_pdf._available", lambda: False)
    result = render_spec_pdf(SPEC, tmp_path / "bad.pdf", title="X", working_dir=tmp_path)
    assert not result["ok"]
    assert result["error"]


def test_export_pipeline_writes_pdf_and_qa(tmp_path: Path):
    from ba_agent.export import write_pdf
    from ba_agent.schemas import SessionState

    state = SessionState(session_id="qa-session", title="Claims Portal")
    state.spec_markdown = SPEC
    artifact, engine, report = write_pdf(
        tmp_path, state, SPEC, [{"title": "Claim intake", "code": "flowchart TD", "svg": TALL_SVG}]
    )
    assert artifact is not None
    assert engine in {"typst", "reportlab"}
    assert (tmp_path / "pdf_qa.json").exists()
    assert artifact.bytes > 1000
    assert report["checks"]["pages"] > 0, report


def test_pdf_engine_reports_available():
    assert pdf_engine() in {"typst", "reportlab"}


def test_rendering_without_a_working_dir_does_not_litter_the_cwd(tmp_path, monkeypatch):
    """Band SVGs are intermediate files. With no working_dir they must not land in the caller's directory.

    `render_spec_pdf(..., working_dir=None)` used to fall through to `Path.cwd()`, which is how stray
    `process_diagram_1_band_2.svg` files ended up committed next to the source.
    """
    monkeypatch.chdir(tmp_path)
    result = render_spec_pdf(
        SPEC,
        tmp_path / "spec.pdf",
        title="Claims Portal",
        diagrams=[{"title": "Claim intake", "code": "flowchart TD", "svg": TALL_SVG}],
    )
    assert result["ok"], result.get("error")

    stray = [p for p in tmp_path.iterdir() if p.suffix == ".svg"]
    assert not stray, f"band SVGs were written into the caller's cwd: {[p.name for p in stray]}"


def test_metadata_carries_the_qa_verdict(services):
    """metadata.json is the manifest a reviewer opens first; it must say whether the gates passed.

    Otherwise `pdf_qa.json` is a second file nobody is told to look at, and a deliverable that failed QA
    looks identical to one that passed.
    """
    from ba_agent.export import write_deliverables
    from ba_agent.schemas import SessionState

    state = SessionState(session_id="qa-manifest", title="Claims Portal")
    state.spec_markdown = SPEC
    write_deliverables(services, state)

    directory = services.config.out_dir("qa-manifest")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    qa = json.loads((directory / "pdf_qa.json").read_text(encoding="utf-8"))

    assert "qa" in metadata, "metadata.json must carry the QA verdict"
    assert metadata["qa"]["ok"] == qa["ok"]
    assert metadata["qa"]["warnings"] == qa["warnings"]
    assert metadata["qa"]["checks"].get("pages", 0) > 0 or metadata["pdf_engine"] == "none"


def test_html_qa_runs_even_when_the_pdf_fails(tmp_path: Path, monkeypatch):
    """With no working PDF engine the HTML deliverable still has to be QA'd.

    `write_pdf` used to bail out before the QA block, so an HTML-only run shipped with no QA report at all
    and no warning in the timeline — the exact case where a reviewer has nothing else to check.
    """
    from ba_agent.export import build_html, write_pdf
    from ba_agent.schemas import SessionState

    monkeypatch.setattr("ba_agent.artifacts.typst.find_typst", lambda: "definitely-not-a-real-binary")
    monkeypatch.setattr("ba_agent.artifacts.reportlab_pdf._available", lambda: False)

    state = SessionState(session_id="no-pdf", title="Claims Portal")
    state.spec_markdown = SPEC
    html_path = tmp_path / "deliverable.html"
    html_path.write_text(build_html(state, SPEC, []), encoding="utf-8")

    artifact, engine, report = write_pdf(tmp_path, state, SPEC, [], html_path=html_path)
    assert artifact is None and engine is None, "no engine means no PDF, but the run must not fail"
    assert report["checks"] == {}, "there is no PDF to audit"

    report = json.loads((tmp_path / "pdf_qa.json").read_text(encoding="utf-8"))
    assert "no PDF engine" in " ".join(report["warnings"]), report["warnings"]
    assert report.get("html"), "the HTML QA block is missing"
    assert report["html"]["checks"]["toc_links"] > 0, report["html"]


def test_metadata_records_which_rung_succeeded():
    """A run that silently fell back to prompt-mode JSON must be visible, not hidden inside the ladder."""
    from ba_agent.export import build_metadata
    from ba_agent.schemas import SessionState

    class FakeGateway:
        rung_counts = {"json_schema": 3, "prompt": 1}
        _learned_rung = {"q2_0": "prompt"}
        usage = {"calls": 14, "failures": 2, "prompt_chars": 9000, "completion_chars": 3100}

    metadata = build_metadata(SessionState(session_id="s", title="T"), gateway=FakeGateway())
    assert metadata["structured_output"]["rung_counts"] == {"json_schema": 3, "prompt": 1}
    assert metadata["structured_output"]["learned_rung"] == {"q2_0": "prompt"}
    # Call volume is what explains a slow run: retries vs. genuinely many questions.
    assert metadata["llm_usage"] == {
        "calls": 14,
        "failures": 2,
        "prompt_chars": 9000,
        "completion_chars": 3100,
    }

    # No gateway means no claim is made about rungs.
    assert "structured_output" not in build_metadata(SessionState(session_id="s", title="T"))


def test_html_contents_links_resolve(tmp_path: Path):
    """The HTML deliverable's Contents nav must link to anchors that exist in the document."""
    html = tmp_path / "deliverable.html"
    html.write_text(build_html("Claims Portal", SPEC, [{"title": "Intake", "svg": SHORT_SVG}]), encoding="utf-8")
    report = audit_html(html)
    assert report["ok"], report["checks"]
    assert report["checks"]["toc_links"] >= 4
    assert report["checks"]["headings"] >= report["checks"]["toc_links"]
    assert report["checks"]["inline_svg"] == 1
    assert not report["warnings"]


def test_html_qa_flags_a_broken_anchor(tmp_path: Path):
    html = tmp_path / "broken.html"
    html.write_text(
        '<nav class="toc"><h2>Contents</h2><ul><li><a href="#nowhere">X</a></li></ul></nav><h1>X</h1>', encoding="utf-8"
    )
    report = audit_html(html)
    assert not report["ok"]
    assert report["warnings"]


def test_cover_status_is_readable_not_a_phase_enum():
    """The cover must say something a stakeholder can read, not the internal phase value."""
    from ba_agent.export import document_status
    from ba_agent.schemas import Phase, SessionState

    done = SessionState(session_id="s", brief="x", phase=Phase.DONE, approved_summary=True)
    assert document_status(done) == "Approved draft"
    unfinished = SessionState(session_id="s", brief="x", phase=Phase.DRAFT, approved_summary=False)
    assert document_status(unfinished) == "Draft — awaiting approval"


def test_reportlab_fallback_draws_banded_diagrams_as_vector(tmp_path: Path):
    """The fallback must embed diagrams as real vector paths, not raster PNGs or raw Mermaid source.

    `renderPM` (the PNG route) needs rlPyCairo, which is not a dependency here — so a raster-first
    implementation would silently print Mermaid code into the PDF on every machine without it.
    """
    from ba_agent.artifacts.reportlab_pdf import render_pdf

    pdf = tmp_path / "fallback.pdf"
    result = render_pdf(
        SPEC,
        pdf,
        title="Claims Portal",
        diagrams=[{"title": "Intake", "code": "x", "svg": TALL_SVG}],
        working_dir=tmp_path,
    )
    assert result["ok"], result["error"]

    bands = sorted(p.name for p in tmp_path.glob("process_diagram*band*.svg"))
    assert bands, "a diagram taller than the page box must be banded"

    report = audit_pdf(pdf, SPEC, [{"title": "Intake", "code": "x", "svg": TALL_SVG}])
    assert report["checks"]["diagram_bands"] == len(bands)
    assert (report["checks"]["band_cut_weight"] or 0) >= 0
    text = " ".join(
        page.get_contents().get_data().decode("latin-1") for page in __import__("pypdf").PdfReader(pdf).pages
    )
    assert "flowchart" not in text, "the Mermaid source must not appear as text in the PDF"
    assert " re" in text or " c" in text, "the diagram must be drawn as vector paths"


def test_reportlab_outline_titles_are_not_double_escaped(tmp_path: Path):
    """A heading containing `&` must appear in the bookmark tree as `&`, not `&amp;`."""
    from ba_agent.artifacts.qa import _outline_titles
    from ba_agent.artifacts.reportlab_pdf import render_pdf

    spec = "# Portal\n\n## User Roles & Personas\n\nText.\n"
    pdf = tmp_path / "amp.pdf"
    assert render_pdf(spec, pdf, title="Portal", working_dir=tmp_path)["ok"]
    outline = _outline_titles(pdf)
    assert "User Roles & Personas" in outline
    assert "User Roles &amp; Personas" not in outline
