"""Tolerant JSON extraction for local models.

Local models wrap JSON in prose, fences, trailing commas or Python literals far more often than
hosted ones. ``extract_json`` is the last-resort parser used by the prompt-based rung of the
structured-output ladder (see :mod:`ba_agent.models`).

Strategy, cheapest first, so well-formed output is never rewritten:

1. parse the text as-is, then each fenced block and each top-level block;
2. only if all of that fails, apply repairs (trailing commas, smart quotes, raw newlines in strings, Python
   literals ``True/False/None``, single-quoted strings). Every repair is token aware: it rewrites only text
   outside JSON string literals, so string contents are never corrupted.

Block discovery is one string-aware linear scan (:func:`iter_top_level_blocks`), not a scan from every opener.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from typing import Any

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(?P<body>.*?)```", re.S)
_SMART_QUOTES = {"\u201c": '"', "\u201d": '"'}
_PY_CONSTANTS = {"True": "true", "False": "false", "None": "null"}
_OPENERS = "{["
_CLOSERS = "}]"


# --------------------------------------------------------------------------- scanning


def iter_top_level_blocks(text: str) -> Iterator[str]:
    """Yield each outermost balanced ``{...}`` / ``[...]`` block in *text*, in order, in one linear pass.

    Braces inside quoted strings do not count. An opener that never closes is abandoned at end of input, so
    the whole pass is O(n) regardless of how many braces the text contains.
    """
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for idx, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            if depth:
                in_string = True
            continue
        if ch in _OPENERS:
            if depth == 0:
                start = idx
            depth += 1
        elif ch in _CLOSERS and depth:
            depth -= 1
            if depth == 0:
                yield text[start : idx + 1]


# --------------------------------------------------------------------------- repairs


def _map_outside_strings(text: str, transform: Callable[[str], str]) -> str:
    """Apply *transform* to the non-string segments of *text*; double-quoted literals are copied verbatim."""
    out: list[str] = []
    segment: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            out.append(transform("".join(segment)))
            segment = []
            in_string = True
            out.append(ch)
            continue
        segment.append(ch)
    out.append(transform("".join(segment)))
    return "".join(out)


def _outside_fix(segment: str) -> str:
    """Repairs that are only valid on non-string text: smart quotes, Python literals, trailing commas."""
    for bad, good in _SMART_QUOTES.items():
        segment = segment.replace(bad, good)
    segment = re.sub(r"\b(True|False|None)\b", lambda m: _PY_CONSTANTS[m.group(1)], segment)
    return re.sub(r",(\s*[}\]])", r"\1", segment)


def _fix_common(text: str) -> str:
    """Trailing commas, smart quotes and Python literals, touching only text outside string literals."""
    return _map_outside_strings(text, _outside_fix)


def _escape_inner_newlines(text: str) -> str:
    """Escape raw newlines/tabs inside string literals (a very common local-model mistake)."""
    out: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
                out.append(ch)
                continue
            if ch == "\\":
                escaped = True
                out.append(ch)
                continue
            if ch == '"':
                in_string = False
                out.append(ch)
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
            out.append(ch)
        else:
            if ch == '"':
                in_string = True
            out.append(ch)
    return "".join(out)


def _single_to_double(text: str) -> str:
    """Convert single-quoted string literals to double-quoted ones, escaping inner double quotes.

    One pass. A single quote *inside* a double-quoted string is ordinary content (the apostrophe in
    ``"it's"``) and is left alone; an earlier regex version corrupted exactly that case.
    """
    out: list[str] = []
    mode = ""  # "" outside strings, '"' inside a double-quoted literal, "'" inside a single-quoted one
    escaped = False
    for ch in text:
        if mode == "":
            if ch == '"':
                mode = '"'
                out.append(ch)
            elif ch == "'":
                mode = "'"
                out.append('"')
            else:
                out.append(ch)
            continue
        if escaped:
            escaped = False
            out.append(ch)
            continue
        if ch == "\\":
            escaped = True
            out.append(ch)
            continue
        if mode == '"':
            if ch == '"':
                mode = ""
            out.append(ch)
        elif ch == "'":
            mode = ""
            out.append('"')
        elif ch == '"':
            out.append('\\"')
        else:
            out.append(ch)
    return "".join(out)


# --------------------------------------------------------------------------- public API


def extract_json(text: str | None) -> Any | None:
    """Best-effort parse of the first JSON value inside *text*. ``None`` when hopeless."""
    if not text or not text.strip():
        return None

    raw = text.strip()
    fenced = [m.group("body").strip() for m in _FENCE_RE.finditer(raw)]
    faithful = [raw, *fenced, *iter_top_level_blocks(raw)]

    # Faithful parses first: nothing is rewritten unless the text is genuinely malformed.
    value = _first_parsed(faithful)
    if value is not None:
        return value

    repaired = [_fix_common(c) for c in faithful if c]
    for transform in (
        _identity,
        _escape_inner_newlines,
        _single_to_double,
        lambda t: _escape_inner_newlines(_single_to_double(t)),
    ):
        value = _first_parsed([transform(c) for c in repaired])
        if value is not None:
            return value
    return None


def _identity(text: str) -> str:
    return text


def _first_parsed(candidates: list[str]) -> Any | None:
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except ValueError:
            continue
    return None


def dumps(value: Any, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=False, indent=indent, default=str)
