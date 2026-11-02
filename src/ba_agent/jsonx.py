"""Tolerant JSON extraction for local models.

Local models wrap JSON in prose, fences, trailing commas or Python literals far more often than
hosted ones. ``extract_json`` is the last-resort parser used by the prompt-based rung of the
structured-output ladder (see :mod:`ba_agent.models`).
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(?P<body>.*?)```", re.S)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_PY_CONST_RE = re.compile(r"\b(True|False|None)\b")
_SMART_QUOTES = {"“": '"', "”": '"', "‘": "'", "’": "'"}
# A single-quoted token, tolerating backslash escapes inside it.
_SINGLE_QUOTED_RE = re.compile(r"'((?:[^'\\]|\\.)*)'")


def _fix_common(text: str) -> str:
    out = text
    for bad, good in _SMART_QUOTES.items():
        out = out.replace(bad, good)
    out = _TRAILING_COMMA_RE.sub(r"\1", out)

    def _const(match: re.Match[str]) -> str:
        return {"True": "true", "False": "false", "None": "null"}[match.group(1)]

    # Only outside strings would be ideal; a plain replace is harmless for our schemas.
    return _PY_CONST_RE.sub(_const, out)


def _balanced_candidates(text: str) -> list[str]:
    """Return every balanced ``{...}`` / ``[...]`` block found in *text*, largest first."""
    results: list[str] = []
    openers = "{["
    closers = "}]"
    for start, ch in enumerate(text):
        if ch not in openers:
            continue
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            cur = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif cur == "\\":
                    escaped = True
                elif cur == '"':
                    in_string = False
                continue
            if cur == '"':
                in_string = True
            elif cur in openers:
                depth += 1
            elif cur in closers:
                depth -= 1
                if depth == 0:
                    results.append(text[start : idx + 1])
                    break
        # keep scanning for other starts even if this one never closed
    return sorted(set(results), key=len, reverse=True)


def extract_json(text: str | None) -> Any | None:
    """Best-effort parse of the first JSON value inside *text*. ``None`` when hopeless."""
    if not text or not text.strip():
        return None

    raw = text.strip()
    candidates: list[str] = [raw]
    for match in _FENCE_RE.finditer(raw):
        candidates.insert(0, match.group("body").strip())

    ordered: list[str] = []
    for cand in candidates:
        ordered.extend([cand, _fix_common(cand)])
    for cand in _balanced_candidates(raw):
        ordered.extend([cand, _fix_common(cand)])

    for candidate in ordered:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except Exception:
            continue

    # Escape raw newlines inside strings (a very common local-model mistake).
    for candidate in ordered:
        try:
            return json.loads(_escape_inner_newlines(candidate))
        except Exception:
            continue

    # Last resort: a Python-style literal such as ``{'title': 'x'}``. Only reached when everything else
    # failed, so the risk of mangling an apostrophe inside a value is preferable to losing the step.
    for candidate in ordered:
        converted = _single_to_double(candidate)
        for attempt in (converted, _escape_inner_newlines(converted)):
            try:
                return json.loads(attempt)
            except Exception:
                continue
    return None


def _single_to_double(text: str) -> str:
    def _repl(match: re.Match[str]) -> str:
        inner = match.group(1).replace('\\"', '"').replace('"', '\\"')
        return f'"{inner}"'

    return _SINGLE_QUOTED_RE.sub(_repl, text)


def _escape_inner_newlines(text: str) -> str:
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


def dumps(value: Any, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=False, indent=indent, default=str)
