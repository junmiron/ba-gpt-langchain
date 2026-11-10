"""The browser only sees a stream event it registered a listener for.

EventSource delivers a frame to the handler registered under that frame's exact `event:` name.
A type the client never registered is dropped in silence: the engine keeps emitting, the page
stays blank, and an interview that asked a question looks like an interview that asked nothing.
So every EventType the server can publish must appear in the client's list.
"""

import re
import subprocess
from pathlib import Path

from ba_agent.events import EventType

APP_JS = Path("src/ba_agent/web/app.js")


def _declared_stream_types() -> set[str]:
    source = APP_JS.read_text(encoding="utf-8")
    block = re.search(r"const STREAM_EVENT_TYPES = \[(.*?)\];", source, re.S)
    assert block, "app.js has no STREAM_EVENT_TYPES list for the SSE stream"
    return set(re.findall(r'"([a-z_]+)"', block.group(1)))


def _publishable_event_types() -> set[str]:
    return {value for name, value in vars(EventType).items() if not name.startswith("_") and isinstance(value, str)}


def test_every_event_the_server_can_emit_has_a_client_listener():
    publishable = _publishable_event_types()
    declared = _declared_stream_types()
    missing = sorted(publishable - declared)
    assert not missing, f"app.js never listens for: {', '.join(missing)} — those events are silently dropped"


def test_the_stream_listener_list_does_not_invent_types():
    # A listener for a type the server never emits is harmless, but it usually means the list drifted.
    extra = sorted(_declared_stream_types() - _publishable_event_types())
    assert not extra, f"app.js listens for unknown event types: {', '.join(extra)}"


def test_app_js_parses():
    """A single bad token in app.js kills the whole script, so the UI goes blank without an error."""
    result = subprocess.run(["node", "--check", str(APP_JS)], capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(f"app.js is not valid JavaScript: {result.stderr.strip()}")
