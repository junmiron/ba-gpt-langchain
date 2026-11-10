"""HTML rendering and the PDF engine-selection report.

The HTML deliverable is read by people who will not run JavaScript, so every Markdown shape the
spec generator can emit has to survive the server-side renderer. These cases cover the shapes that
the current tests never exercised: fenced code, tables without a separator row, duplicate headings.
"""

from __future__ import annotations

from pathlib import Path

from ba_agent.artifacts import pdf_engine, render_spec_pdf
from ba_agent.artifacts.html import build_html, markdown_to_html


def test_fenced_code_blocks_render_as_code_and_mermaid_is_marked():
    markdown = "```mermaid\ngraph TD; A-->B;\n```\n\n```python\nx = 1 < 2\n```\n"
    html = markdown_to_html(markdown, [])
    assert '<pre class="mermaid-src">graph TD; A--&gt;B;</pre>' in html
    assert "<pre>x = 1 &lt; 2</pre>" in html, "a plain code block must be escaped, not interpreted"


def test_a_table_without_a_separator_row_stays_plain_text():
    # A single pipe-delimited line is not a table; turning it into <thead><tbody> with no rows would
    # produce a header-only table that reads as a broken document.
    html = markdown_to_html("| one | two |\n", [])
    assert "<table>" not in html
    assert "<p>| one | two |</p>" in html


def test_duplicate_heading_titles_get_distinct_anchors():
    html = build_html("Portal", "## Setup\n\nbody\n\n## Setup\n\nmore\n", [])
    assert 'id="setup"' in html and 'id="setup-2"' in html, html
    assert '<a href="#setup-2">Setup</a>' in html, "the Contents nav must link to every occurrence"


def test_lists_and_quotes_survive_inline_markup():
    html = markdown_to_html(
        "- **bold** item\n  1. nested\n> quoted *text*\n[link](https://example.com)",
        [],
    )
    assert "<strong>bold</strong>" in html
    assert "<em>text</em>" in html
    assert '<a href="https://example.com">link</a>' in html
    assert "<blockquote>" in html


def test_render_spec_pdf_reports_why_it_used_the_fallback(tmp_path: Path, monkeypatch):
    import ba_agent.artifacts as artifacts

    monkeypatch.setattr(artifacts, "find_typst", lambda: None)

    result = render_spec_pdf("## Section\n\nSome body text.", tmp_path / "spec.pdf", title="Spec")
    assert result["engine"] == "reportlab", result
    assert result.get("fallback_reason") == "typst binary not found", result
    assert (tmp_path / "spec.pdf").exists()


def test_pdf_engine_names_the_engine_the_machine_has():
    name = pdf_engine()
    assert name in {"typst", "reportlab", "none"}, name


def test_html_qa_flags_a_contents_block_that_links_nowhere(tmp_path: Path):
    from ba_agent.artifacts.qa import audit_html

    good = build_html("Portal", "## Scope\n\nbody\n\n## Risks\n\nbody\n", [])
    (tmp_path / "good.html").write_text(good, encoding="utf-8")
    verdict = audit_html(tmp_path / "good.html")
    assert verdict["ok"], verdict["warnings"]
    assert verdict["checks"]["toc_links"] == 2

    broken = good.replace('href="#risks"', 'href="#nonexistent"')
    (tmp_path / "broken.html").write_text(broken, encoding="utf-8")
    verdict = audit_html(tmp_path / "broken.html")
    assert not verdict["ok"]
    assert "nonexistent" in verdict["checks"]["unresolved_links"], verdict["checks"]
    assert any("anchors that do not exist" in warning for warning in verdict["warnings"])

    (tmp_path / "empty.html").write_text("<main><h1>Spec</h1></main>", encoding="utf-8")
    verdict = audit_html(tmp_path / "empty.html")
    assert not verdict["ok"]
    assert "no Contents navigation block" in verdict["warnings"]


def test_html_qa_reports_an_unreadable_deliverable_instead_of_raising(tmp_path: Path):
    from ba_agent.artifacts.qa import audit_html

    verdict = audit_html(tmp_path / "does-not-exist.html")
    assert verdict["ok"] is False
    assert verdict["warnings"] == ["HTML deliverable is unreadable"]
