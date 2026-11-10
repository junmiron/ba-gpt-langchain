"""The diagram role must ship a *renderable* diagram, not just a syntactically plausible one.

Regression for a real run: a sequence diagram with an `alt` block that never closed passed the old
line-balance validator, then mmdc rejected it with a parse error, so the deliverable carried no SVG and
QA reported a render failure. The repair path has to close the block before rendering.
"""

from __future__ import annotations

import asyncio

import pytest

from ba_agent.mermaid_utils import (
    close_open_blocks,
    mermaid_cli,
    quote_risky_labels,
    render_svg_local,
    validate_mermaid,
)
from ba_agent.roles.diagram import generate_diagrams
from ba_agent.schemas import DiagramDraft, DiagramSet, SessionState

# Verbatim shape of the diagram that broke a real run: an `alt` opened, never closed.
BROKEN_SEQUENCE = """sequenceDiagram
    participant U as User
    participant P as Portal
    participant W as Worker
    U->>P: submit claim
    alt urgent
        P->>W: escalate immediately
        W->>P: confirm triage
    P->>U: acknowledge receipt
"""


def test_unclosed_block_is_caught_then_repaired():
    report = validate_mermaid(BROKEN_SEQUENCE)
    assert not report.ok, "the validator must not wave through an unclosed alt block"
    assert any("no matching 'end'" in error for error in report.errors), report.errors

    repaired = close_open_blocks(quote_risky_labels(report.code))
    final = validate_mermaid(repaired)
    assert final.ok, final.errors


@pytest.mark.skipif(not mermaid_cli(), reason="mermaid-cli not installed")
def test_the_repaired_diagram_actually_renders(tmp_path):
    repaired = close_open_blocks(quote_risky_labels(validate_mermaid(BROKEN_SEQUENCE).code))
    svg, _png, error = render_svg_local(repaired, tmp_path, timeout=120)
    assert svg and not error, error
    assert "<svg" in svg


@pytest.mark.skipif(not mermaid_cli(), reason="mermaid-cli not installed")
def test_generate_diagrams_stores_an_svg_for_a_broken_draft(services):
    """The role's output is what QA and the PDF read; an empty svg there was the original bug."""

    class DiagramGateway:
        async def structured(self, model, *, system_prompt, user_prompt, role, emit, step, model_override=None):
            draft = DiagramDraft(title="Sequence", diagram_type="sequenceDiagram", code=BROKEN_SEQUENCE)
            return type("Result", (), {"parsed": DiagramSet(diagrams=[draft]), "error": None})()

        async def chat(self, *args, **kwargs):
            return type("Result", (), {"ok": True, "reply": ""})()

    services.gateway = DiagramGateway()  # type: ignore[assignment]
    services.config.diagram.local_renderer = True

    state = SessionState(session_id="diag-test", title="Test", brief="test")
    diagrams = asyncio.run(generate_diagrams(services, state))

    assert diagrams, "the role produced nothing"
    for diagram in diagrams:
        assert diagram["code"], "diagram code missing"
        assert diagram["valid"], f"diagram still invalid: {diagram['errors']}"
        assert diagram["svg"], f"diagram has no rendered SVG: {diagram['render_error']}"
