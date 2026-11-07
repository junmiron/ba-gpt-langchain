"""Regression tests for the roles-layer defects R1-R12 (see docs/REDESIGN_PLAN.md section 1.2).

Each test pins one defect so it cannot silently return. All run offline; the gateway is never called.
"""

from __future__ import annotations

import asyncio
import re

import pytest

from ba_agent.context import transcript_lines
from ba_agent.events import EventType
from ba_agent.mermaid_utils import clean_mermaid, quote_risky_labels, validate_mermaid
from ba_agent.prompts import render
from ba_agent.roles import analysis, review
from ba_agent.roles.review import issues_from_lint
from ba_agent.schemas import (CriticVerdict, DiagramDraft, Issue, LedgerItem, RequirementCategory, SessionState,
                              SubjectSpec, SummaryDoc)
from ba_agent.spec import LintFinding, deterministic_gaps, lint_spec, nfr_topics_mentioned


def _req(index: int, body: str, category=RequirementCategory.FUNCTIONAL) -> LedgerItem:
    return LedgerItem(id=f"FR-{index:03d}", kind="requirement", title=f"Req {index}", body=body, category=category)


# --------------------------------------------------------------------------- R1 prompts


def test_r1_template_without_user_marker_keeps_its_whole_persona_as_system_prompt(config):
    """chat_agent.md.j2 has no ---USER--- marker; render() used to replace its persona with a generic line."""
    system, user = render(
        "chat_agent", project_name="Claims Portal", phase="interview", status="running", awaiting="answer",
        coverage_summary="Product: asking", requirement_count=3,
    )
    assert "You are the Business Analyst running a requirements-interview session" in system
    assert "helpful assistant" not in system
    assert "Claims Portal" in system and "Requirements captured: 3" in system
    assert user == ""


def test_r1_templates_with_marker_still_split_system_and_user(config):
    system, user = render(
        "triage", subject=type("S", (), {"name": "Scope"})(), question="Q?", answer="A.",
        existing_titles=[],
    )
    assert system and user
    assert "---USER---" not in system and "---USER---" not in user


# --------------------------------------------------------------------------- R2 has_similar


def test_r2_distinct_requirements_sharing_a_prefix_are_not_merged():
    state = SessionState(session_id="s-r2", brief="brief")
    state.ledger.append(_req(1, "The system shall provide invoice export."))
    assert not state.has_similar("The system shall provide claim export."), (
        "R2: a shared 24-char prefix ('The system shall provide') must not merge distinct requirements")


def test_r2_true_paraphrase_and_exact_repeat_are_still_deduplicated():
    state = SessionState(session_id="s-r2b", brief="brief")
    state.ledger.append(_req(1, "Claims are validated manually today by the claims team."))
    assert state.has_similar("Claims are validated manually today by the claims team.")
    assert state.has_similar("claims are validated manually today by the claims team!")


def test_r2_short_text_is_never_considered_duplicate():
    state = SessionState(session_id="s-r2c", brief="brief")
    state.ledger.append(_req(1, "Export data."))
    assert not state.has_similar("Export data.")  # below the 12-char floor, unchanged behaviour


# --------------------------------------------------------------------------- R3 mermaid balanced parens


@pytest.mark.parametrize("line, expected_label", [
    ("A[total (net)] --> B[done]", "total (net)"),
    ("A{Ok (y)?} --> B[done]", "Ok (y)?"),
])
def test_r3_balanced_parens_in_label_stay_valid(line, expected_label):
    fixed = quote_risky_labels(f"flowchart TD\n    {line}")
    report = validate_mermaid(fixed)
    assert report.ok, report.errors
    assert f'"{expected_label}"' in fixed, fixed
    # The label is not double-wrapped or truncated (the old bug produced ["total (net")]).
    assert '"total (net"' not in fixed


def test_r3_the_documented_case_renders_as_a_single_quoted_label():
    fixed = quote_risky_labels("flowchart TD\n    A[total (net)] --> B[x]")
    assert fixed == 'flowchart TD\n    A["total (net)"] --> B[x]'


def test_r3_mismatched_brackets_are_still_reported_not_silently_rewritten():
    report = validate_mermaid("flowchart TD\n    A[x (y] --> B[z")
    assert not report.ok


# --------------------------------------------------------------------------- R4 lint blockers survive truncation


class _FakeServices:
    def __init__(self, verdict: CriticVerdict):
        self.verdict = verdict

        class _Gateway:
            async def structured(inner, schema, **kwargs):  # noqa: N805
                from ba_agent.models import StructuredResult
                return StructuredResult(parsed=self.verdict, raw_text="{}", method="json_schema", attempts=1)

        self.gateway = _Gateway()


def test_r4_all_lint_blockers_survive_when_critic_floods_the_list(monkeypatch):
    critic_issues = [Issue(section_id="scope", kind="internal", severity="blocker",
                           description=f"critic blocker {i}", fix_hint="fix") for i in range(13)]
    verdict = CriticVerdict(status="auto_fixable", summary="x", issues=critic_issues)
    findings = [LintFinding(f"code{i}", f"lint blocker {i}", "blocker", "internal", "scope") for i in range(13)]

    state = SessionState(session_id="s-r4", brief="b")
    services = _FakeServices(verdict)
    result = asyncio.run(review.critique(services, state, findings))

    lint_blockers = [i for i in result.issues if i.source == "lint" and i.severity == "blocker"]
    assert len(lint_blockers) == 13, [i.description for i in result.issues]
    assert len(result.issues) == 13  # all lint blockers kept; the cap is for the rest, not for blockers


def test_r4_cap_still_applies_to_non_blocker_issues():
    critic_issues = [Issue(section_id="s", kind="internal", severity="warning", description=f"w{i}")
                     for i in range(20)]
    verdict = CriticVerdict(status="complete", summary="x", issues=critic_issues)
    services = _FakeServices(verdict)
    result = asyncio.run(review.critique(services, SessionState(session_id="s-r4b", brief="b"), []))
    assert len(result.issues) == review.MAX_REVIEW_ISSUES


# --------------------------------------------------------------------------- R5 lenient enums


def test_r5_subject_priority_accepts_capitalised_values():
    assert SubjectSpec(name="Scope", priority="High").priority == "high"
    assert SubjectSpec(name="Scope", priority=" LOW ").priority == "low"
    assert SubjectSpec(name="Scope", priority="whatever").priority == "medium"


def test_r5_diagram_type_maps_natural_language_to_mermaid_keyword():
    assert DiagramDraft(code="x", diagram_type="sequence diagram").diagram_type == "sequenceDiagram"
    assert DiagramDraft(code="x", diagram_type="State Diagram").diagram_type == "stateDiagram"
    assert DiagramDraft(code="x", diagram_type="Flow Chart").diagram_type == "flowchart"
    assert DiagramDraft(code="x", diagram_type="nonsense").diagram_type == "flowchart"


# --------------------------------------------------------------------------- R6 critic sees the spec tail


def test_r6_critic_prompt_includes_the_tail_of_a_long_spec(monkeypatch):
    head = "H" * 12_000
    middle = "M" * 9_000
    tail = "GLOSSARY-OPEN-QUESTIONS-TAIL-MARKER"
    document = head + middle + tail
    captured: dict[str, str] = {}

    class _Gateway:
        async def structured(self, schema, *, system_prompt="", user_prompt="", **kwargs):
            captured["user"] = user_prompt
            from ba_agent.models import StructuredResult
            return StructuredResult(parsed=CriticVerdict(status="complete", summary="ok", issues=[]),
                                    raw_text="{}", method="json_schema", attempts=1)

    class _Services:
        gateway = _Gateway()

    state = SessionState(session_id="s-r6", brief="b", spec_markdown=document)
    asyncio.run(review.critique(_Services(), state, []))
    assert tail in captured["user"], "R6: the critic must see the end of the document (glossary/open questions)"
    assert "H" * 12_000 in captured["user"]
    assert "omitted from the middle" in captured["user"]


def test_r6_short_spec_is_passed_whole():
    document = "short spec body"
    assert review._spec_for_critic(document) == document


# --------------------------------------------------------------------------- R8 None summary guard


def test_r8_write_use_cases_tolerates_missing_summary(config, services):
    from ba_agent.roles import writing

    state = SessionState(session_id="s-r8", brief="brief", summary=None)
    state.ledger.append(_req(1, "The system shall let a broker submit a claim with attachments."))
    text = asyncio.run(writing.write_use_cases(services, state))
    assert isinstance(text, str)


def test_r8_write_use_cases_reads_scope_when_summary_present(config, services):
    from ba_agent.roles import writing

    seen: dict[str, str] = {}
    original = services.gateway.stream_text

    async def spy(**kwargs):
        seen["user"] = kwargs.get("user_prompt", "")
        return await original(**kwargs)

    services.gateway.stream_text = spy  # type: ignore[assignment]
    state = SessionState(session_id="s-r8b", brief="brief",
                         summary=SummaryDoc(scope_in=["Claim intake"], scope_out=["Payments"]))
    state.ledger.append(_req(1, "The system shall let a broker submit a claim with attachments."))
    asyncio.run(writing.write_use_cases(services, state))
    assert "In scope: Claim intake" in seen["user"]
    assert "Out of scope: Payments" in seen["user"]


# --------------------------------------------------------------------------- R9 fallback summary keeps scope gap


def test_r9_fallback_summary_does_not_stuff_fr_bodies_into_scope_in():
    state = SessionState(session_id="s-r9", brief="brief")
    state.ledger.append(_req(1, "The system shall let a broker submit a claim with attachments."))
    summary = analysis.deterministic_summary(state)
    assert summary.scope_in == []
    assert not any("broker submit a claim" in s for s in summary.scope_in)
    assert any("broker submit a claim" in b for b in [summary.background])


def test_r9_scope_gap_still_fires_on_a_degraded_run():
    state = SessionState(session_id="s-r9b", brief="brief")
    for index in range(6):
        state.ledger.append(_req(index, f"The system shall provide capability number {index} with measurable output."))
    state.summary = analysis.deterministic_summary(state)
    assert any("in and out of scope" in gap for gap in deterministic_gaps(state)), (
        "R9: the scope gap must fire when the fallback summary has no explicit scope")


# --------------------------------------------------------------------------- R10 unified thresholds


def test_r10_deterministic_gaps_uses_configured_fr_floor():
    state = SessionState(session_id="s-r10", brief="brief")
    for index in range(6):
        state.ledger.append(_req(index, f"The system shall provide capability number {index} with output."))
    gaps_default = deterministic_gaps(state)
    assert any("functional requirement" in g and "expected at least 5" in g for g in gaps_default) is False

    gaps_high = deterministic_gaps(state, min_functional_requirements=10)
    assert any("expected at least 10" in g for g in gaps_high)
    gaps_low = deterministic_gaps(state, min_functional_requirements=2)
    assert not any("functional requirement" in g for g in gaps_low)


def test_r10_sla_matches_as_a_whole_word_only():
    assert "availability" not in nfr_topics_mentioned("The island and slate roofs are fine.")
    assert "availability" in nfr_topics_mentioned("The SLA is 99.9 percent.")


def test_r10_nfr_gap_threshold_matches_lint():
    state = SessionState(session_id="s-r10b", brief="brief")
    # Mention exactly one quality attribute: 3 of 4 topics are uncovered, which both lint (>=3) and the gap
    # detector must report. Under the old >=2 gap threshold the gap detector fired a case lint treats as fine.
    state.ledger.append(_req(1, "Response time must stay under two seconds at peak."))
    lint_nfr = [f for f in lint_spec(state, min_functional_requirements=1) if f.code == "nfr_coverage"]
    gap_nfr = [g for g in deterministic_gaps(state) if "quality attributes" in g]
    assert bool(lint_nfr) == bool(gap_nfr)


# --------------------------------------------------------------------------- R11 warnings on fallbacks


def test_r11_gap_analysis_fallback_emits_warning(config, services):
    from ba_agent.models import StructuredResult

    async def failing(schema, **kwargs):
        return StructuredResult(parsed=None, raw_text="", method="failed", attempts=3, degraded=True,
                                error="schema failed")

    services.gateway.structured = failing  # type: ignore[assignment]
    events: list[tuple[str, dict]] = []

    async def emit(event_type, payload=None):
        events.append((event_type, payload or {}))

    state = SessionState(session_id="s-r11", brief="brief")
    gaps = asyncio.run(analysis.gap_analysis(services, state, ["Only 1 functional requirement captured."],
                                             emit=emit))
    assert gaps and gaps[0].subject == "Clarification"
    warnings = [p for t, p in events if t == EventType.WARNING]
    assert warnings, "R11: a degraded gap analysis must surface a WARNING event"
    assert "degraded" in warnings[0]["message"]


def test_r11_diagram_reviewer_parse_failure_emits_warning(config, services):
    from ba_agent.models import StructuredResult
    from ba_agent.roles import diagram

    async def failing(schema, **kwargs):
        return StructuredResult(parsed=None, raw_text="", method="failed", attempts=3, degraded=True,
                                error="no json")

    services.gateway.structured = failing  # type: ignore[assignment]
    events: list[str] = []

    async def emit(event_type, payload=None):
        if event_type == EventType.WARNING:
            events.append((payload or {}).get("message", ""))

    state = SessionState(session_id="s-r11b", brief="brief")
    result = asyncio.run(diagram.review_diagram(services, state, {"title": "t", "code": "flowchart TD\nA-->B"},
                                                ["some error"], emit=emit))
    assert result["summary"] == "reviewer unavailable"
    assert any("reviewer unavailable" in m or "unavailable" in m for m in events)


# --------------------------------------------------------------------------- R12 dead code


def test_r12_transcript_shows_skipped_and_not_applicable_rows():
    rows = [
        {"q_index": 1, "subject": "Scope", "question": "Q1?", "answer": None, "skipped": 1, "na": 0},
        {"q_index": 2, "subject": "Scope", "question": "Q2?", "answer": None, "skipped": 0, "na": 1},
        {"q_index": 3, "subject": "Scope", "question": "Q3?", "answer": "yes", "skipped": 0, "na": 0},
        {"q_index": 4, "subject": "Scope", "question": "Q4?", "answer": None, "skipped": 0, "na": 0},
    ]
    lines = transcript_lines(rows)
    assert any("Q1" in line and "(skipped)" in line for line in lines), lines
    assert any("Q2" in line and "(not applicable)" in line for line in lines), lines
    assert any("Q3" in line and "yes" in line for line in lines), lines
    assert not any(line.startswith("Q4") for line in lines), "unanswered rows are still omitted"


def test_r12_removed_dead_symbols_are_gone():
    import ba_agent.prompts as prompts
    import ba_agent.schemas as schemas
    assert not hasattr(prompts, "available_prompts")
    assert not hasattr(schemas, "SectionDraft")
    assert not hasattr(SessionState, "ledger_lines")
    import ba_agent.spec as spec_module
    assert not hasattr(spec_module, "ledger_lines")


def test_r12_prompts_fallback_points_at_repo_prompts_dir(config):
    from ba_agent import prompts as prompts_module

    directory = prompts_module._prompts_dir()
    assert directory.exists()
    assert (directory / "chat_agent.md.j2").exists()


def test_r11_probe_fail_open_emits_warning(config, services):
    from ba_agent.models import StructuredResult
    from ba_agent.roles import interview

    async def failing(schema, **kwargs):
        return StructuredResult(parsed=None, raw_text="", method="failed", attempts=3, degraded=True,
                                error="schema failed")

    services.gateway.structured = failing  # type: ignore[assignment]
    events: list[tuple[str, dict]] = []

    async def emit(event_type, payload=None):
        events.append((event_type, payload or {}))

    state = SessionState(session_id="s-r11c", brief="brief")
    decision = asyncio.run(interview.probe_coverage(services, state, "Scope", "Q?", "A.", emit=emit))
    assert decision.covered is True
    warnings = [p for t, p in events if t == EventType.WARNING]
    assert warnings, "R11: a fail-open coverage probe must surface a WARNING event"
    assert "degraded" in warnings[0]["message"]


def test_r7_repair_is_judged_against_the_auto_fixed_baseline(config, services):
    """Draft has unquoted-label errors that auto-fix repairs. The model's repair is worse than that baseline.

    The old check compared against the raw draft (more errors) and accepted it. The auto-fixed code must win.
    """
    from ba_agent.models import StructuredResult
    from ba_agent.roles import diagram
    from ba_agent.schemas import DiagramSet, MermaidReviewResult

    original = "\n".join(["flowchart TD", "    A(Order accepted) --> B[Done]", "    C(Ship it) --> D[x"])
    worse = "\n".join(['flowchart TD', '    A["x" --> B["y"', '    C["z" --> D["w"'])

    auto_fixed = quote_risky_labels(original)
    baseline_errors = validate_mermaid(auto_fixed).errors
    original_errors = validate_mermaid(original).errors
    worse_errors = validate_mermaid(worse).errors
    assert len(baseline_errors) < len(original_errors), "premise: auto-fix must improve the draft"
    assert len(worse_errors) >= len(baseline_errors) > 0, "premise: the repair does not beat the baseline"
    assert len(worse_errors) < len(original_errors), "premise: the old raw-draft comparison would accept it"

    async def scripted(schema, *, role="", **kwargs):
        if role == "diagrammer":
            parsed = DiagramSet(diagrams=[{"title": "t", "diagram_type": "flowchart", "code": original}])
        else:
            parsed = MermaidReviewResult(approved=False, summary="fix", issues=[], revised_code=worse)
        return StructuredResult(parsed=parsed, raw_text="{}", method="json_schema", attempts=1)

    services.gateway.structured = scripted  # type: ignore[assignment]
    state = SessionState(session_id="s-r7", brief="brief")
    diagrams = asyncio.run(diagram.generate_diagrams(services, state))

    assert diagrams, "a diagram should still be produced"
    shipped = diagrams[0]["code"]
    assert shipped == clean_mermaid(auto_fixed), shipped
    assert shipped != clean_mermaid(worse)
