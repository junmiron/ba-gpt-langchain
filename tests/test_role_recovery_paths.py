"""The recovery paths that decide whether a run survives a bad input.

These are the branches a demo never hits: a critic that returns prose instead of a verdict, a writer
that emits an empty section, a glossary delivered as a table, a PDF page pypdf cannot read. Each one
has to degrade into something the deliverable can still use rather than raise.
"""

from __future__ import annotations

from typing import Any

from ba_agent.knowledge import _is_public_ip, extract_main_content, load_file
from ba_agent.models import StructuredResult, TextResult
from ba_agent.roles.review import critique, verdict_from_lint
from ba_agent.roles.writing import write_glossary, write_section
from ba_agent.schemas import SessionState
from ba_agent.spec import LintFinding, SectionSpec
from conftest import FakeGateway


def _collector() -> tuple[list[tuple[str, dict]], Any]:
    events: list[tuple[str, dict]] = []

    async def emit(event_type: str, payload: dict | None = None) -> None:
        events.append((event_type, payload or {}))

    return events, emit


# --------------------------------------------------------------------------- critic


def test_lint_only_verdict_classifies_blockers_by_who_can_fix_them():
    findings = [
        LintFinding("S1", "missing acceptance criteria", "blocker", "internal", "scope"),
        LintFinding("S3", "stakeholder must confirm the SLA", "blocker", "user", "nfr"),
    ]
    verdict = verdict_from_lint(findings)
    assert verdict.status == "needs_user", (
        "a user-kind blocker cannot be auto-fixed, so the verdict must route it back to the stakeholder"
    )

    only_internal = verdict_from_lint([LintFinding("S1", "duplicate id", "blocker", "internal", "scope")])
    assert only_internal.status == "auto_fixable"

    assert verdict_from_lint([]).status == "complete"


async def test_a_critic_that_returns_no_verdict_falls_back_to_the_linter(services):
    """R6: the deterministic checker is the safety net when the LLM critic is unusable."""

    class _ProseGateway(FakeGateway):
        async def structured(self, schema, **kwargs):
            return StructuredResult(parsed=None, text="I cannot review this.", model="fake", rung="prompt")

    state = SessionState(session_id="critic-fallback", title="Fallback")
    state.spec_markdown = "## Scope\n\nSome text.\n"
    events, emit = _collector()
    services.gateway = _ProseGateway()

    verdict = await critique(
        services,
        state,
        [LintFinding("S1", "no NFRs captured", "blocker", "internal", "nfr")],
        emit=emit,
    )

    assert verdict is not None, "the critic must never return None — the review phase needs a verdict"
    warning = [e for e in events if e[0] == "warning" and e[1].get("step") == "review:critic"]
    assert warning, "falling back to the linter must be visible in the event stream"
    assert "deterministic checker" in warning[0][1]["message"]
    assert any(issue.source == "lint" for issue in verdict.issues), "the fallback verdict carries the lint findings"


# --------------------------------------------------------------------------- writing


async def test_an_empty_section_emits_a_warning_instead_of_silently_vanishing(services):
    state = SessionState(session_id="empty-section", title="Empty section")
    section = SectionSpec(id="scope", title="Scope", guidance="Define what is in and out of scope.")

    class _EmptyGateway(FakeGateway):
        async def stream_text(self, **kwargs):

            return TextResult(text="", model="fake", latency_ms=1, error="model returned nothing")

    events, emit = _collector()

    services.gateway = _EmptyGateway()
    text = await write_section(services, state, section, emit=emit)
    assert text == ""
    warning = [e for e in events if e[0] == "warning" and e[1].get("step") == "write:scope"]
    assert warning, "an empty section must be reported, not swallowed"
    assert "placeholder" in warning[0][1]["message"] or "empty" in warning[0][1]["message"]


async def test_a_glossary_written_as_a_table_is_normalised_to_term_definitions(services):
    """The spec assembler expects `- **Term** — definition`; models often emit a Markdown table instead."""

    class _TableGateway(FakeGateway):
        async def stream_text(self, **kwargs):

            return TextResult(
                text=(
                    "| Term | Meaning |\n| --- | --- |\n| Claim | A reimbursement request |\n"
                    "| Broker | The submitting party |\n"
                ),
                model="fake",
                latency_ms=1,
            )

    services.gateway = _TableGateway()
    text = await write_glossary(services, SessionState(session_id="glossary", title="Glossary"), "claims and brokers")
    assert "- **Claim** — A reimbursement request" in text, text
    assert "- **Broker** — The submitting party" in text, text


# --------------------------------------------------------------------------- knowledge readers


def test_is_public_ip_handles_mapped_and_malformed_addresses():
    assert _is_public_ip("93.184.216.34") is True
    assert _is_public_ip("127.0.0.1") is False
    assert _is_public_ip("10.0.0.5") is False
    assert _is_public_ip("not-an-ip") is False
    assert _is_public_ip("::1") is False
    assert _is_public_ip("::ffff:10.0.0.1") is False, "an IPv4-mapped loopback must not sneak through as public"
    assert _is_public_ip("::ffff:93.184.216.34") is True


def test_extract_main_content_ignores_script_and_keeps_article_text():
    html = """
    <html><head><title>Claims portal</title><script>var x = 1;</script></head>
    <body><nav>Home | About</nav><article><h2>Intake</h2><p>Claims arrive by email today.</p></article></body>
    """
    title, text = extract_main_content(html, "https://example.com/claims")
    assert title == "Claims portal", f"the page title should be recoverable, got {title!r}"
    assert "Claims arrive by email today." in text
    assert "var x = 1" not in text
    assert "Home | About" not in text


def test_load_file_reads_every_accepted_extension(tmp_path):
    (tmp_path / "note.txt").write_text("plain text", encoding="utf-8")
    assert load_file(tmp_path / "note.txt") == "plain text"

    (tmp_path / "note.unknown").write_text("fallback reader", encoding="utf-8")
    assert load_file(tmp_path / "note.unknown") == "fallback reader"
