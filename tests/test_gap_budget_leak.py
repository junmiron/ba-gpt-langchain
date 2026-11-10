"""Gaps the question budget cannot afford must still reach the deliverable.

The clarify step tells the user "remaining gaps are listed as open questions". If the budget is
exhausted, the only place those gaps can land is the ledger — otherwise the warning is a promise the
document never keeps, and the Open Questions section is silently empty for exactly the sessions where
the interview ran short.
"""

from __future__ import annotations

from ba_agent.schemas import Phase
from ba_agent.spec import deterministic_gaps
from ba_agent.workflow import WorkflowEngine


async def _clarify_with_no_budget(services):
    services.config.interview.max_gap_rounds = 0
    engine = WorkflowEngine(services)
    state = services.store.create_session(brief="We need a claims portal for insurance brokers. Today intake is email.")
    state.phase = Phase.CLARIFY
    events: list[tuple[str, dict]] = []

    async def emit(event_type, payload=None):
        events.append((str(event_type), payload or {}))

    await engine.advance(state, emit=emit)
    return state, events


async def test_exhausted_gap_budget_records_the_gaps_it_says_are_listed(services):
    state, events = await _clarify_with_no_budget(services)

    warnings = [p for kind, p in events if kind == "warning" and p.get("step") == "clarify"]
    assert warnings, "the run should at least say the gap budget ran out"
    claim = warnings[0].get("message", "")
    assert "open questions" in claim.lower(), claim

    gaps = deterministic_gaps(state, min_functional_requirements=services.config.review.min_functional_requirements)
    assert gaps, "the fixture is deliberately thin so deterministic gaps exist"

    recorded = [entry.body for entry in state.ledger if entry.kind == "open_question"]
    missing = [gap for gap in gaps if gap not in recorded]
    assert not missing, f"the warning promised {len(gaps)} open questions, the ledger holds {len(recorded)}: {missing}"
    assert state.phase is not Phase.CLARIFY, "clarify never exited"


async def test_the_gaps_survive_into_the_rendered_open_questions_section(services):
    from ba_agent.spec import render_open_questions

    state, _ = await _clarify_with_no_budget(services)
    section = render_open_questions(state)
    for gap in deterministic_gaps(
        state, min_functional_requirements=services.config.review.min_functional_requirements
    ):
        fragment = gap.split(" — ")[0].strip()
        assert fragment[:40] in section, f"gap not in the deliverable: {gap!r}\n---\n{section}"
