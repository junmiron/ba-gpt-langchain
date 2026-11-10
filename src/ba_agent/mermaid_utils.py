"""Mermaid validation, cleaning and optional server-side rendering.

v1 delegated this to an MCP server plus ``xelatex`` (with absolute Linux paths baked into the YAML), so on Windows
the diagram step simply could not run. Here validation is pure Python — always available — and PNG/SVG rendering
happens in the browser with vendored mermaid.js. An optional HTTP renderer (kroki) can be enabled for a real parse
check; it is **off by default** because it would send your specification to a third party.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

DIAGRAM_KEYWORDS = (
    "flowchart",
    "graph",
    "sequenceDiagram",
    "stateDiagram",
    "classDiagram",
    "erDiagram",
    "journey",
    "gantt",
    "mindmap",
    "timeline",
)

RESERVED_IDS = {"end", "graph", "class", "subgraph", "click", "style", "linkStyle"}
_NODE_DECL_RE = re.compile(r"(?P<id>[A-Za-z0-9_\u00C0-\u024F]+)\s*(?P<open>[\[{\(])")
_ARROW_RE = re.compile(r"(-->|->>|---|--|\\.\\.>|==>)")
# A node label is a run of plain characters, one balanced ``(...)`` group, or one balanced ``[...]`` group, so
# ``A[total (net)]`` and ``A[T[inner]]`` are matched whole. A stray mismatch (``A[x (y]``) is left for the
# validator's balance check to flag.
_UNQUOTED_RISKY_RE = re.compile(
    r"(?P<open>[\[{(])(?P<label>(?:[^\[\]{}()\"]|\([^[\]{}()]*\)|\[[^\[\]{}]*\]|\{[^[\]{}]*\})*?)(?P<close>[\]})])"
)
_CLOSER_FOR = {"[": "]", "{": "}", "(": ")"}
_DIRECTIONS = {"TD", "TB", "BT", "LR", "RL"}
#: Characters an unquoted label cannot contain. Verified against mermaid-cli: a pipe is read as an edge-label
#: separator and a bracket/brace opens a second node, both hard parse errors. Commas, colons and semicolons,
# which an earlier version flagged, parse fine — flagging them quoted every label and burned a repair round
# on diagrams that were already correct.
_RISKY_LABEL_CHARS = re.compile(r"[|()\[\]{}]")

# sequenceDiagram blocks that must be closed with `end`. An unbalanced one is a hard parse error in
# mermaid, and models produce it constantly when they nest an alt inside an alt.
_SEQ_BLOCK_OPENERS = {"alt", "loop", "opt", "par", "rect", "block"}


def _block_balance(text: str) -> tuple[int, list[int], int]:
    """Walk a sequence diagram and return (unclosed_blocks, opener_lines, stray_ends)."""
    stack: list[int] = []
    stray = 0
    for number, line in enumerate(text.splitlines(), start=1):
        head = line.strip().split(maxsplit=1)[0].lower() if line.strip() else ""
        if head in _SEQ_BLOCK_OPENERS:
            stack.append(number)
        elif head == "end":
            if stack:
                stack.pop()
            else:
                stray += 1
    return len(stack), stack, stray


def _label_is_risky(opener: str, label: str) -> bool:
    """True when an *unquoted* node label is likely to break the mermaid parser.

    Measured against mermaid-cli, not guessed: the characters that actually fail inside a label are a pipe
    (read as an edge-label separator) and any nested bracket or brace. Commas, colons and semicolons parse
    fine, so they are no longer flagged — quoting a label that was already valid costs a repair round and
    makes the QA report claim a fix that never happened. Quoting is still applied whenever a label carries a
    character that genuinely breaks the parse.
    """
    if not label or label.startswith('"') or '"' in label:
        return False
    return bool(_RISKY_LABEL_CHARS.search(label))


@dataclass
class MermaidReport:
    code: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def strip_fences(code: str) -> str:
    text = (code or "").strip()
    match = re.match(r"^```(?:mermaid|markdown|md)?\s*\n(?P<body>.*)\n?```\s*$", text, re.S)
    if match:
        text = match.group("body").strip()
    # Models sometimes emit several fenced blocks; keep everything that is not a fence line.
    lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
    return "\n".join(lines).strip()


def clean_mermaid(code: str) -> str:
    """Normalise whitespace, drop comments/emoji that break parsers, keep indentation."""
    cleaned: list[str] = []
    for raw in strip_fences(code).splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("%%"):  # mermaid comments are fine but models emit junk there
            continue
        line = line.replace("\t", "    ")
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def validate_mermaid(code: str, *, max_nodes: int = 40) -> MermaidReport:
    """Static checks for the failure modes local models actually produce."""
    text = clean_mermaid(code)
    errors: list[str] = []
    warnings: list[str] = []

    if not text:
        return MermaidReport(code="", errors=["diagram is empty"])

    lines = text.splitlines()
    first = lines[0].strip()
    if not first.startswith(DIAGRAM_KEYWORDS):
        errors.append(f"first line must be a diagram type keyword ({', '.join(DIAGRAM_KEYWORDS[:5])} …); got '{first}'")
    elif len(lines) > 1:
        head = first.split()
        if head[0] in {"graph", "flowchart"} and len(head) < 2:
            errors.append(f"line 1: '{head[0]}' needs a direction — write '{head[0]} TD' (or LR/BT/RL)")
        elif head[0] in {"graph", "flowchart"} and head[1].upper() not in _DIRECTIONS:
            warnings.append(f"line 1: unknown direction '{head[1]}' — expected one of {'/'.join(sorted(_DIRECTIONS))}")

    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        for opener, closer in (("[", "]"), ("{", "}"), ("(", ")")):
            if stripped.count(opener) != stripped.count(closer):
                errors.append(f"line {number}: unbalanced '{opener}…{closer}' — {stripped[:70]}")
        if stripped.count('"') % 2:
            errors.append(f"line {number}: odd number of double quotes — {stripped[:70]}")

        for match in _NODE_DECL_RE.finditer(stripped):
            node_id = match.group("id")
            if node_id.lower() in RESERVED_IDS and match.group("open") in "[{(":
                errors.append(f"line {number}: '{node_id}' is a reserved Mermaid word, rename the node id")
            if re.match(r"^\d", node_id) and match.group("open") in "[{(":
                warnings.append(f"line {number}: numeric node id '{node_id}' can confuse some renderers")

        # Parentheses/commas inside an unquoted label are the classic parse error.
        for match in _UNQUOTED_RISKY_RE.finditer(stripped):
            label = (match.group("label") or "").strip()
            if _label_is_risky(match.group("open"), label):
                errors.append(
                    f"line {number}: unquoted label '{label[:40]}' is risky — "
                    f'wrap it in double quotes, e.g. A["{label[:40]}"]'
                )
                break

    if len(text.splitlines()) > 1 and not _ARROW_RE.search(text):
        warnings.append("no arrows found: the diagram has nodes but no connections")

    if first.startswith("sequenceDiagram"):
        unclosed, openers, stray = _block_balance(text)
        if unclosed:
            opener_lines = ", ".join(str(number) for number in openers)
            errors.append(
                f"sequenceDiagram has {unclosed} block(s) opened on line(s) {opener_lines} with no matching 'end'"
            )
        if stray:
            errors.append(f"sequenceDiagram has {stray} stray 'end' with no matching block opener")

    node_ids = {m.group("id") for m in _NODE_DECL_RE.finditer(text)} - RESERVED_IDS
    if len(node_ids) > max_nodes:
        warnings.append(f"{len(node_ids)} nodes — consider splitting the diagram (readability)")

    return MermaidReport(code=text, errors=errors, warnings=warnings)


def quote_risky_labels(code: str) -> str:
    """Best-effort auto-fix for the most common error class before asking a model to repair it."""
    fixed_lines: list[str] = []
    for line in clean_mermaid(code).splitlines():

        def _fix(match: re.Match[str]) -> str:
            label = (match.group("label") or "").strip()
            opener, closer = match.group("open"), match.group("close")
            if _CLOSER_FOR.get(opener) != closer or not _label_is_risky(opener, label):
                return match.group(0)
            safe = label.replace('"', "'")
            return f'{opener}"{safe}"{closer}'

        fixed_lines.append(_UNQUOTED_RISKY_RE.sub(_fix, line))
    return "\n".join(fixed_lines).strip()


def close_open_blocks(code: str) -> str:
    """Append the `end`s a sequenceDiagram is missing.

    Deterministic and safe: an unclosed block is always a parse error, and adding the closers at the end
    keeps every statement inside the block it was written in. A stray `end` cannot be fixed this way, so
    the code is returned unchanged in that case and the model gets asked to repair it.
    """
    text = clean_mermaid(code)
    unclosed, _, stray = _block_balance(text)
    if not text.startswith("sequenceDiagram") or unclosed == 0 or stray:
        return text
    return (text + "\n" + "end\n" * unclosed).rstrip()


def mermaid_cli() -> str | None:
    """Resolve ``mmdc``. npm on Windows installs both an extension-less script and a .cmd shim; only the
    shim is runnable via CreateProcess, so prefer it."""
    import shutil

    return shutil.which("mmdc.cmd") or shutil.which("mmdc")


def render_svg_local(
    code: str, working_dir: Path, *, timeout: float = 120.0, want_png: bool = False
) -> tuple[str | None, bytes | None, str | None]:
    """Render with the local mermaid CLI -> ``(svg, png, error)``.

    No network, no third party: the diagram never leaves the machine. Returns (None, None, reason) when
    ``mmdc`` is not installed, which is the normal case and is not an error for the pipeline.

    ``-b`` in mermaid-cli is *background colour*, not a second output file, so the PNG needs its own
    invocation. It is opt-in because each call spins up a headless browser; the PDF and HTML deliverables
    only need the SVG.
    """
    binary = mermaid_cli()
    if not binary:
        return None, None, "mermaid-cli (mmdc) not installed"

    import subprocess

    working_dir.mkdir(parents=True, exist_ok=True)
    source = working_dir / "diagram.mmd"
    svg_path = working_dir / "diagram.svg"
    png_path = working_dir / "diagram.png"
    source.write_text(clean_mermaid(code), encoding="utf-8")
    try:
        result = subprocess.run(
            [binary, "-i", str(source), "-o", str(svg_path), "-q"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001 - a missing/hung renderer must not fail the run
        return None, None, f"mermaid-cli failed: {type(exc).__name__}: {exc}"
    if result.returncode != 0 or not svg_path.exists():
        return None, None, (result.stderr or result.stdout or "mermaid-cli produced no SVG").strip()[:300]
    svg = svg_path.read_text(encoding="utf-8", errors="replace")

    png: bytes | None = None
    if want_png:
        try:
            png_result = subprocess.run(
                [binary, "-i", str(source), "-o", str(png_path), "-q"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if png_result.returncode == 0 and png_path.exists():
                png = png_path.read_bytes()
        except Exception:  # noqa: BLE001 - the PNG is a bonus; the SVG is the deliverable
            png = None
    return svg, png, None


async def render_svg(code: str, endpoint: str | None, *, timeout: float = 20.0) -> tuple[str | None, str | None]:
    """Render via kroki-compatible endpoint → ``(svg_or_None, error_or_None)``.

    Returns ``(None, None)`` when no endpoint is configured (the browser renders instead). Only call this with
    content you are willing to send to that host.
    """
    if not endpoint:
        return None, None
    import httpx

    text = clean_mermaid(code)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{endpoint.rstrip('/')}/mermaid/svg",
                content=text.encode("utf-8"),
                headers={"content-type": "text/plain"},
            )
    except Exception as exc:  # noqa: BLE001 — offline is normal here
        return None, f"renderer unreachable: {type(exc).__name__}: {exc}"

    if response.status_code >= 400:
        detail = response.text[:300].replace("\n", " ")
        return None, f"renderer rejected the diagram (HTTP {response.status_code}): {detail}"
    body = response.text
    if "<svg" not in body.lower():
        return None, "renderer returned a non-SVG payload"
    return body, None
