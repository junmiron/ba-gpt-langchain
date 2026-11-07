"""Mermaid validation, cleaning and optional server-side rendering.

v1 delegated this to an MCP server plus ``xelatex`` (with absolute Linux paths baked into the YAML), so on Windows
the diagram step simply could not run. Here validation is pure Python — always available — and PNG/SVG rendering
happens in the browser with vendored mermaid.js. An optional HTTP renderer (kroki) can be enabled for a real parse
check; it is **off by default** because it would send your specification to a third party.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

DIAGRAM_KEYWORDS = ("flowchart", "graph", "sequenceDiagram", "stateDiagram", "classDiagram",
                    "erDiagram", "journey", "gantt", "mindmap", "timeline")

RESERVED_IDS = {"end", "graph", "class", "subgraph", "click", "style", "linkStyle"}
_NODE_DECL_RE = re.compile(r"(?P<id>[A-Za-z0-9_\u00C0-\u024F]+)\s*(?P<open>[\[{\(])")
_ARROW_RE = re.compile(r"(-->|->>|---|--|\\.\\.>|==>)")
# A node label is a run of plain characters or one balanced ``(...)`` group, so ``A[total (net)]`` is matched
# whole. The closer must pair with the opener; a stray mismatch (``A[x (y]``) is left for the validator to flag.
_UNQUOTED_RISKY_RE = re.compile(
    r"(?P<open>[\[{(])(?P<label>(?:[^\[\]{}()\"]|\([^\[\]{}()]*\))*)(?P<close>[\]})])"
)
_CLOSER_FOR = {"[": "]", "{": "}", "(": ")"}
_DIRECTIONS = {"TD", "TB", "BT", "LR", "RL"}


def _label_is_risky(opener: str, label: str) -> bool:
    """True when an *unquoted* node label is likely to break the mermaid parser.

    Punctuation inside a label is the classic failure (``A[total (net)]``, ``A[a, b]``). Round nodes are also
    where models dump prose, so a multi-word round label gets quoted defensively: quoting is always safe and
    costs nothing visually, while an unparseable diagram costs a whole revision round.
    """
    if not label or label.startswith('"') or '"' in label:
        return False
    if re.search(r"[()\[\]{}:;,]", label):
        return True
    if opener == "(" and re.search(r"\s", label.strip()):
        return True
    return False


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
                errors.append(f"line {number}: unquoted label '{label[:40]}' is risky — "
                              f'wrap it in double quotes, e.g. A["{label[:40]}"]')
                break

    if len(text.splitlines()) > 1 and not _ARROW_RE.search(text):
        warnings.append("no arrows found: the diagram has nodes but no connections")

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
            response = await client.post(f"{endpoint.rstrip('/')}/mermaid/svg",
                                         content=text.encode("utf-8"),
                                         headers={"content-type": "text/plain"})
    except Exception as exc:  # noqa: BLE001 — offline is normal here
        return None, f"renderer unreachable: {type(exc).__name__}: {exc}"

    if response.status_code >= 400:
        detail = response.text[:300].replace("\n", " ")
        return None, f"renderer rejected the diagram (HTTP {response.status_code}): {detail}"
    body = response.text
    if "<svg" not in body.lower():
        return None, "renderer returned a non-SVG payload"
    return body, None
