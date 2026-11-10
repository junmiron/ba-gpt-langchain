"""Sanitising of model-generated SVG before it is embedded in an HTML page.

A model can be coaxed into emitting SVG that runs script (``<script>``, ``<foreignObject>`` with HTML, event
handler attributes, ``javascript:`` links). Embedded inline in a same-origin page that is a stored-XSS vector, so
the deliverable never embeds raw model SVG: :func:`sanitize_svg` removes everything executable and returns the
markup that is safe to put in the page.
"""

from __future__ import annotations

import re
from xml.etree import ElementTree as ET

SVG_NS = "http://www.w3.org/2000/svg"

#: Elements that are removed together with their content.
_DROP_ELEMENTS = {
    "script",
    "style",
    "foreignobject",
    "iframe",
    "object",
    "embed",
    "audio",
    "video",
    "image",
    "use",
    "handler",
    "listener",
    "set",
    "animate",
    "animatemotion",
    "animatetransform",
    "link",
    "meta",
    "base",
    "form",
    "input",
    "textarea",
    "button",
    "select",
}

#: Attributes that may hold a URL; a ``javascript:``/``data:`` value makes them unsafe.
_URL_ATTRIBUTES = {
    "href",
    "src",
    "xlink:href",
    "action",
    "formaction",
    "data",
    "background",
    "poster",
    "to",
    "from",
    "values",
    "by",
}

_UNSAFE_URL_RE = re.compile(r"^\s*(javascript|vbscript|data)\s*:", re.IGNORECASE)
_CSS_EXPR_RE = re.compile(r"(expression\s*\(|javascript\s*:|url\s*\(\s*['\"]?\s*(javascript|data)\s*:)", re.I)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower() if isinstance(tag, str) else ""


def _strip_element(parent: ET.Element) -> None:
    for child in list(parent):
        if _local(child.tag) in _DROP_ELEMENTS:
            parent.remove(child)
            continue
        _sanitize_attrs(child)
        _strip_element(child)


def _sanitize_attrs(element: ET.Element) -> None:
    for name in list(element.attrib):
        lname = name.rsplit("}", 1)[-1].lower()
        value = element.attrib[name]
        if (
            lname.startswith("on")
            or (lname in _URL_ATTRIBUTES and _UNSAFE_URL_RE.match(value))
            or (lname == "style" and _CSS_EXPR_RE.search(value))
        ):  # onload, onclick, onmouseover, …
            del element.attrib[name]
        elif lname in ("href", "xlink:href") and not value.lstrip().startswith("#"):
            # Only in-document fragments are legitimate inside an inline diagram.
            del element.attrib[name]


def sanitize_svg(svg: str | None) -> str:
    """Return a script-free copy of *svg*, or ``""`` when the input is not a well-formed SVG document.

    * ``<script>``, ``<style>``, ``<foreignObject>``, ``<use>``, ``<image>``, animation and form elements are removed;
    * every ``on*`` attribute is removed;
    * URL-bearing attributes survive only when they are in-document fragments (``#id``) or non-executable schemes.
    * ``style`` attributes holding ``expression()`` or ``javascript:`` are removed.

    Unparseable input yields an empty string instead of raw markup, so a broken block can never leak through.
    """
    if not svg or not svg.strip():
        return ""
    text = svg.strip()
    # Drop the XML declaration / doctype: ElementTree rejects DOCTYPE (XXE/billion-laughs surface) outright.
    # Rejecting DOCTYPE before parsing is what makes the stdlib parser safe here: with no DTD there are no
    # external entities to resolve and no entity-expansion bombs, so defusedxml is not needed for this input.
    if re.search(r"<!DOCTYPE", text, re.IGNORECASE):
        return ""
    text = re.sub(r"^<\?xml[^>]*\?>", "", text).strip()
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return ""
    if _local(root.tag) != "svg":
        return ""
    _sanitize_attrs(root)
    _strip_element(root)
    ET.register_namespace("", SVG_NS)
    rendered = ET.tostring(root, encoding="unicode")
    # Defence in depth: nothing executable may survive serialisation.
    if re.search(r"<\s*script|on[a-z]+\s*=|javascript\s*:", rendered, re.IGNORECASE):
        return ""
    return rendered
