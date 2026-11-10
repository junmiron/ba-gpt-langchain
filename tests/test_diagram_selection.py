"""Which diagram ships, and where the rendered PNG goes.

`primary_diagram` decides the content of `process_diagram.mmd`, and `_save_png` decides whether the
rendered image is available to the PDF. Both are small enough to test directly, and both fail in a
way an end-to-end run would not surface: a sequence diagram shipping as the "process" diagram, or a
PNG written next to the working directory instead of the session folder.
"""

from __future__ import annotations

from ba_agent.config import AppConfig
from ba_agent.roles.diagram import _save_png, primary_diagram
from ba_agent.schemas import SessionState
from ba_agent.services import Services


def test_a_valid_flowchart_beats_a_valid_sequence_diagram():
    diagrams = [
        {"title": "Onboarding", "code": "sequenceDiagram\nA->>B: hi", "valid": True},
        {"title": "Claims", "code": "flowchart TD\nA[claim] --> B[triage]", "valid": True},
    ]
    chosen = primary_diagram(diagrams)
    assert chosen["title"] == "Claims", f"the process diagram shipped as {chosen['title']}"


def test_an_invalid_flowchart_does_not_win_over_a_valid_sequence_diagram():
    diagrams = [
        {"title": "Broken", "code": "flowchart TD\nA[[[", "valid": False},
        {"title": "Sequence", "code": "sequenceDiagram\nA->>B: hi", "valid": True},
    ]
    assert primary_diagram(diagrams)["title"] == "Sequence"


def test_no_diagrams_at_all_yields_none_rather_than_a_crash():
    assert primary_diagram([]) is None


def test_when_nothing_is_valid_the_first_is_still_shipped():
    """Shipping the draft beats shipping nothing: the UI shows the attempt and its errors."""
    diagrams = [{"title": "Draft", "code": "flowchart TD\nA[[[", "valid": False}]
    assert primary_diagram(diagrams)["title"] == "Draft"


def _services(tmp_path) -> Services:
    config = AppConfig()
    config.paths.outputs_dir = str(tmp_path / "outputs")
    return Services.create(config)


def test_the_rendered_png_lands_in_the_session_folder(tmp_path):
    services = _services(tmp_path)
    state = SessionState(session_id="sess-png")

    path = _save_png(services, state, 1, b"\x89PNG\r\n\x1a\nfake")

    assert path, "the renderer reported success but the image was not written"
    target = tmp_path / "outputs" / "sess-png" / "process_diagram_1.png"
    assert path == str(target), f"PNG written to {path}, expected {target}"
    assert target.read_bytes() == b"\x89PNG\r\n\x1a\nfake"


def test_a_png_write_failure_degrades_to_no_image_instead_of_crashing(tmp_path):
    # `outputs` is a regular file, so no session directory can be created under it.
    (tmp_path / "outputs").write_text("not a directory")
    services = _services(tmp_path)
    state = SessionState(session_id="sess-fail")

    assert _save_png(services, state, 0, b"data") == "", "a failed write must not raise out of the phase"
