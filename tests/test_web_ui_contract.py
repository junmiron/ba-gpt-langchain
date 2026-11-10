from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "src" / "ba_agent" / "web"


def test_suggestion_chips_define_readable_foreground_and_focus_state() -> None:
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    chip_rule = css.split(".chip {", 1)[1].split("}", 1)[0]
    assert "color: var(--fg)" in chip_rule
    assert ".chip:focus-visible" in css


def test_timeline_cards_keep_their_height_and_preserve_history_scroll() -> None:
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    app = (WEB / "app.js").read_text(encoding="utf-8")
    card_rule = css.split(".card {", 1)[1].split("}", 1)[0]
    timeline_rule = css.split(".timeline {", 1)[1].split("}", 1)[0]
    assert "flex-shrink: 0" in card_rule or "flex: 0 0 auto" in card_rule
    assert "min-height: 0" in timeline_rule
    assert "timelineIsNearBottom" in app


def test_approval_state_has_distinct_composer_controls() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    app = (WEB / "app.js").read_text(encoding="utf-8")
    assert 'id="approval-card"' in html
    assert "function renderApprovalCard(snapshot)" in app
    assert 'snapshot.awaiting === "approval"' in app
    assert 'snapshot.awaiting === "answer"' in app
    assert "answerForm.hidden = !showQuestion" in app
    assert "Approval required — review the summary below." in app


def test_busy_state_has_an_accessible_live_elapsed_indicator() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    app = (WEB / "app.js").read_text(encoding="utf-8")
    assert 'id="activity-indicator"' in html
    assert 'aria-live="polite"' in html
    assert "startActivityIndicator" in app
    assert "stopActivityIndicator" in app


def test_session_restore_pages_the_entire_event_history() -> None:
    app = (WEB / "app.js").read_text(encoding="utf-8")
    assert "EVENT_PAGE_SIZE" in app
    restore = app.split("async function openSession", 1)[1].split("async function refreshSnapshot", 1)[0]
    assert "while (true)" in restore


def test_trace_displays_per_call_role_step_latency() -> None:
    app = (WEB / "app.js").read_text(encoding="utf-8")
    assert 'payload.category === "model_call"' in app
    assert "payload.latency_ms" in app
    assert "payload.prompt_chars" in app
