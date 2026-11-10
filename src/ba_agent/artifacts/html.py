"""Self-contained HTML deliverable: rendered Markdown, a linked Contents block, inline SVG diagrams.

The sibling project shipped HTML that relied on client-side ``marked.js`` and had no navigable outline.
This renders server-side so the file is readable without JavaScript, printable, and has real anchor links.
"""

from __future__ import annotations

import re
from html import escape

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_SEP_ROW_RE = re.compile(r"^\s*\|[\s|:-]+\|\s*$")
_LIST_RE = re.compile(r"^(\s*)([-*+]\s+)(.*)$")
_ORDERED_RE = re.compile(r"^(\s*)(\d+\.\s+)(.*)$")
_FENCE_RE = re.compile(r"^\s*```(.*)$")
_QUOTE_RE = re.compile(r"^\s*>\s?(.*)$")

_CSS = """
:root { --fg:#16191d; --muted:#5b6672; --line:#e3e8ee; --accent:#0b6b5e; --band:#f6f8fa; }
* { box-sizing: border-box }
body { margin:0; background:#f6f7f9; color:var(--fg); font:16px/1.6 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
main { max-width:960px; margin:0 auto; padding:48px 32px 96px; background:#fff; box-shadow:0 1px 3px rgba(0,0,0,.08); min-height:100vh; }
h1,h2,h3,h4 { line-height:1.25; margin:1.5em 0 .5em }
h1 { font-size:2rem; border-bottom:3px solid var(--accent); padding-bottom:.3em }
h2 { font-size:1.4rem; border-bottom:1px solid var(--line); padding-bottom:.25em }
h3 { font-size:1.08rem; color:#2c3640 }
table { border-collapse:collapse; width:100%; margin:1em 0; font-size:.94em }
th,td { border:1px solid var(--line); padding:7px 10px; text-align:left; vertical-align:top }
th { background:#111827; color:#fff }
tbody tr:nth-child(even) { background:var(--band) }
code { background:#f2f4f7; padding:.1em .35em; border-radius:4px; font-size:.9em }
pre { background:#0f1115; color:#dfe4ea; padding:14px; border-radius:8px; overflow:auto; font-size:.85em }
blockquote { margin:1em 0; padding:.6em 1em; border-left:4px solid var(--accent); background:#f4faf8; color:#2c3a37 }
.meta { color:var(--muted); font-size:.85rem; margin-bottom:1.5rem }
nav.toc { border:1px solid var(--line); border-radius:8px; padding:14px 18px; background:#fbfcfd; margin-bottom:2rem }
nav.toc h2 { margin:0 0 .4em; font-size:1.1rem; border:none }
nav.toc ul { margin:0; padding-left:1.2em }
nav.toc a { color:var(--accent); text-decoration:none }
nav.toc a:hover { text-decoration:underline }
figure { margin:1.5em 0; }
figure svg, figure img { width:100%; height:auto; max-height:70vh; border:1px solid var(--line); border-radius:6px }
figcaption { color:var(--muted); font-size:.85rem; margin-top:.4em }
footer { margin-top:3rem; padding-top:1rem; border-top:1px solid var(--line); color:var(--muted); font-size:.8rem }
@media print { body { background:#fff } main { box-shadow:none; max-width:none; padding:0 } nav.toc, footer { display:none } }
"""


def _slug(text: str, used: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-") or "section"
    slug, n = base, 2
    while slug in used:
        slug = f"{base}-{n}"
        n += 1
    used.add(slug)
    return slug


def _inline(text: str) -> str:
    out = escape(text)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"(?<![\w])\*([^*\s][^*]*?)\*(?![\w])", r"<em>\1</em>", out)
    out = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', out)
    return out


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def markdown_to_html(markdown: str, headings: list[tuple[int, str, str]]) -> str:
    lines = (markdown or "").replace("\r\n", "\n").split("\n")
    out: list[str] = []
    index = 0
    heading_index = 0
    while index < len(lines):
        line = lines[index]

        if _FENCE_RE.match(line):
            language = line.strip().lstrip("`").strip()
            block: list[str] = []
            index += 1
            while index < len(lines) and not _FENCE_RE.match(lines[index]):
                block.append(lines[index])
                index += 1
            index += 1
            if language.lower() == "mermaid":
                body = "\n".join(block)
                out.append(f'<pre class="mermaid-src">{escape(body)}</pre>')
            else:
                body = "\n".join(block)
                out.append(f"<pre>{escape(body)}</pre>")
            continue

        if _TABLE_ROW_RE.match(line):
            table: list[str] = [line]
            index += 1
            while index < len(lines) and _TABLE_ROW_RE.match(lines[index]):
                table.append(lines[index])
                index += 1
            if len(table) >= 2 and _SEP_ROW_RE.match(table[1]):
                head = "".join(f"<th>{_inline(cell)}</th>" for cell in _cells(table[0]))
                body = "".join(
                    "<tr>" + "".join(f"<td>{_inline(cell)}</td>" for cell in _cells(row)) + "</tr>" for row in table[2:]
                )
                out.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
            else:
                out.extend(f"<p>{_inline(row)}</p>" for row in table)
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()
            anchor = headings[heading_index][1] if heading_index < len(headings) else _slug(title, set())
            heading_index += 1
            out.append(f'<h{level} id="{anchor}">{_inline(title)}</h{level}>')
            index += 1
            continue

        quote = _QUOTE_RE.match(line)
        if quote:
            out.append(f"<blockquote>{_inline(quote.group(1))}</blockquote>")
            index += 1
            continue

        match = _LIST_RE.match(line) or _ORDERED_RE.match(line)
        if match:
            indent = len(match.group(1)) // 2
            content = _inline(match.group(3))
            tag = "ol" if match.group(2)[0].isdigit() else "ul"
            out.append("  " * indent + f"<{tag}><li>{content}</li></{tag}>")
            index += 1
            continue

        if line.strip():
            out.append(f"<p>{_inline(line.strip())}</p>")
        index += 1

    return "\n".join(out)


def build_html(state_title: str, spec_markdown: str, diagrams: list[dict], *, meta: str = "") -> str:
    """Render the spec as a standalone HTML document with a clickable Contents nav and inline SVG figures."""
    used: set[str] = set()
    headings: list[tuple[int, str, str]] = []
    for line in (spec_markdown or "").split("\n"):
        match = _HEADING_RE.match(line)
        if match:
            headings.append((len(match.group(1)), _slug(match.group(2), used), match.group(2).strip()))

    toc_items = "".join(
        f'<li><a href="#{anchor}">{escape(title)}</a></li>' for level, anchor, title in headings if level <= 3
    )
    figures = []
    for position, diagram in enumerate(diagrams or [], start=1):
        svg = diagram.get("svg") if isinstance(diagram, dict) else None
        if not svg:
            continue
        title = diagram.get("title") or f"Process diagram {position}"
        figures.append(f"<figure>{svg}<figcaption>{escape(title)}</figcaption></figure>")

    return (
        f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{escape(state_title)} — Functional Specification</title>\n<style>{_CSS}</style>\n</head>\n<body>\n<main>\n"
        f'<div class="meta">{meta or escape(state_title)}</div>\n'
        f'<nav class="toc"><h2>Contents</h2><ul>{toc_items}</ul></nav>\n'
        f"{markdown_to_html(spec_markdown, headings)}\n"
        f"{chr(10).join(figures)}\n"
        f"<footer>Generated by the BA agent (LangChain implementation). Items marked TBD still need "
        f"stakeholder input.</footer>\n</main>\n</body>\n</html>\n"
    )
