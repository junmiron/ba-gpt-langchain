"""The settings UI and the settings API must expose the same knobs.

`SessionManager.SETTABLE` is the authoritative allow-list of what may be changed. The UI builds its grid
from that list but labels each field from a hand-written `SETTING_LABELS` map. When someone adds a field
to `SETTABLE` and forgets the label, the grid renders the raw snake_case key — a knob the user cannot read.
When someone adds a label with no matching field, the grid renders nothing for it. Both are silent drift,
so the check belongs in the test suite, not in a review comment.
"""

from pathlib import Path

from ba_agent.runtime import SessionManager

APP_JS = Path("src/ba_agent/web/app.js")


def _label_keys() -> set[str]:
    text = APP_JS.read_text(encoding="utf-8")
    start = text.index("const SETTING_LABELS = {")
    block = text[start : text.index("};", start)]
    keys: set[str] = set()
    for line in block.splitlines():
        line = line.strip()
        if ":" in line and not line.startswith("const"):
            keys.add(line.split(":")[0].strip())
    return keys


def test_every_settable_field_has_a_readable_ui_label():
    settable = {field for fields in SessionManager.SETTABLE.values() for field in fields}
    labels = _label_keys()
    missing = sorted(settable - labels)
    assert not missing, f"settings the API accepts but the UI renders as a raw key: {missing}"


def test_every_ui_label_maps_to_a_settable_field():
    settable = {field for fields in SessionManager.SETTABLE.values() for field in fields}
    orphan = sorted(_label_keys() - settable)
    assert not orphan, f"UI labels for fields the API will not accept: {orphan}"


def test_sections_rendered_by_the_grid_match_the_allowlist():
    """The grid iterates `S.settings`, which the server builds from SETTABLE; a section present in the UI
    but absent from SETTABLE would be a field the server silently drops."""
    text = APP_JS.read_text(encoding="utf-8")
    sections_in_ui = set()
    for line in text.splitlines():
        if 'data-section="' in line:
            sections_in_ui.add(line.split('data-section="')[1].split('"')[0])
    dynamic = {"${section}"}
    declared = sections_in_ui - dynamic
    assert declared <= set(SessionManager.SETTABLE), (
        f"UI references settings sections the API does not accept: {sorted(declared - set(SessionManager.SETTABLE))}"
    )
