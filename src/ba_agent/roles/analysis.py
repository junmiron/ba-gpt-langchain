"""Analysis roles: answer triage → ledger, gap analysis, stakeholder summary."""

from __future__ import annotations

import re

from ..context import (assumptions_lines, facts_lines, open_questions_lines, render_summary_markdown,
                       requirements_rendered, risks_lines, transcript_lines)
from ..events import Emitter, EventType, null_emitter
from ..prompts import render
from ..schemas import GapAnalysis, GapItem, LedgerItem, SessionState, SummaryDoc, TriageResult
from ..services import Services, session_model


def merge_triage(state: SessionState, result: TriageResult, *, subject_name: str, q_index: int) -> list[LedgerItem]:
    """Deterministically fold a triage result into the ledger (dedup + ids). Returns the added items."""
    added: list[LedgerItem] = []

    for text in result.facts:
        clean = " ".join((text or "").split())
        if not clean or state.has_similar(clean):
            continue
        item = LedgerItem(id=f"FACT-{len(state.ledger) + 1}", kind="fact", body=clean, subject=subject_name,
                          source_q=q_index)
        state.ledger.append(item)
        added.append(item)

    for draft in result.requirements:
        description = " ".join((draft.description or "").split())
        title = " ".join((draft.title or "").split())[:120]
        if not description and not title:
            continue
        body = description or title
        if state.has_similar(body):
            continue
        item = LedgerItem(
            id=state.new_requirement_id(draft.category), kind="requirement", title=title or body[:60], body=body,
            category=draft.category, priority=draft.priority,
            acceptance=[" ".join(a.split()) for a in draft.acceptance_criteria if a and a.strip()],
            subject=subject_name, source_q=q_index,
        )
        state.ledger.append(item)
        added.append(item)

    for kind, values in (("risk", result.risks), ("assumption", result.assumptions),
                         ("decision", result.decisions)):
        for text in values:
            clean = " ".join((text or "").split())
            if not clean or state.has_similar(clean):
                continue
            item = LedgerItem(id=f"{kind.upper()}-{len(state.ledger) + 1}", kind=kind, body=clean,
                              subject=subject_name, source_q=q_index)
            state.ledger.append(item)
            added.append(item)

    note = " ".join((result.coverage_note or "").split())
    if note and len(note) > 20:
        item = LedgerItem(id=f"OPEN-{len(state.ledger) + 1}", kind="open_question", body=note, subject=subject_name,
                          source_q=q_index)
        state.ledger.append(item)
        added.append(item)

    return added


async def triage(services: Services, state: SessionState, subject_name: str, question_text: str, answer_text: str, *,
                 emit: Emitter = null_emitter) -> list[LedgerItem]:
    existing_titles = [item.title for item in state.requirements()][-40:]
    system, user = render(
        "triage", subject=type("S", (), {"name": subject_name})(), question=question_text, answer=answer_text[:3500],
        existing_titles=existing_titles,
    )
    result = await services.gateway.structured(
        TriageResult, system_prompt=system, user_prompt=user, role="triage", emit=emit,
        step=f"triage:{subject_name}", model_override=session_model(state, "triage")
    )
    parsed = result.parsed if isinstance(result.parsed, TriageResult) else None
    if parsed is None:
        await emit(EventType.WARNING, {
            "step": f"triage:{subject_name}", "role": "triage",
            "message": "answer could not be structured — stored verbatim as a fact so nothing is lost",
            "detail": (result.error or "")[:300],
        })
        clean = " ".join((answer_text or "").split())[:900]
        if clean and not state.has_similar(clean):
            item = LedgerItem(id=f"FACT-{len(state.ledger) + 1}", kind="fact", body=clean, subject=subject_name)
            state.ledger.append(item)
            return [item]
        return []

    added = merge_triage(state, parsed, subject_name=subject_name, q_index=len(state.questions))
    if added:
        await emit(EventType.LEDGER, {
            "step": f"triage:{subject_name}", "role": "triage", "subject": subject_name,
            "added": [{"id": i.id, "kind": i.kind, "title": i.title, "body": i.body[:180]} for i in added],
        })
    return added


async def gap_analysis(services: Services, state: SessionState, deterministic_gaps: list[str], *,
                       emit: Emitter = null_emitter) -> list[GapItem]:
    cfg = services.config.interview
    system, user = render(
        "gaps",
        project_name=(state.plan.project_name if state.plan else state.title),
        objective=(state.plan.objective if state.plan else ""),
        brief=state.brief[:2000],
        coverage_table="\n".join(f"- {s.name}: {s.status}" for s in state.subjects) or "(none)",
        deterministic_gaps=deterministic_gaps,
        requirements=requirements_rendered(state),
        facts=facts_lines(state),
        max_questions=cfg.max_gap_questions_per_round,
    )
    result = await services.gateway.structured(
        GapAnalysis, system_prompt=system, user_prompt=user, role="interview", emit=emit, step="gap_analysis",
        model_override=session_model(state, "interview")
    )
    parsed = result.parsed if isinstance(result.parsed, GapAnalysis) else None
    if parsed is None:
        # Deterministic findings are still actionable questions — use them rather than inventing nothing.
        await emit(EventType.WARNING, {
            "step": "gap_analysis", "role": "interview",
            "message": "gap analysis unavailable — using the deterministic checker's gaps only (degraded)",
            "detail": (result.error or "")[:300],
        })
        return [GapItem(subject="Clarification", question=gap, why="detected by the completeness checker")
                for gap in deterministic_gaps[: cfg.max_gap_questions_per_round]]

    gaps = [g for g in parsed.gaps if (g.question or "").strip()][: cfg.max_gap_questions_per_round]
    return gaps


def deterministic_summary(state: SessionState) -> SummaryDoc:
    """Ledger projection used when the model cannot produce a summary — approval can still happen."""
    plan = state.plan
    requirements = state.requirements()
    stakeholders = [line for line in facts_lines(state) if re.search(r"\b(role|user|persona|team|admin)\b", line.lower())]
    return SummaryDoc(
        project_name=(plan.project_name if plan else state.title),
        objective=(plan.objective if plan else ""),
        background="\n".join(filter(None, [
            (plan.summary if plan else state.brief[:400]),
            *[f"Functional requirement {r.id}: {r.body}" for r in requirements
              if r.category and r.category.value == "functional"][:12],
        ])),
        objectives=[(plan.objective if plan and plan.objective else "")][:1],
        stakeholders=stakeholders[:8],
        # Scope comes from explicit in-scope facts only. Stuffing functional requirement bodies here made the
        # scope-gap check pass on a degraded run even though the boundary was never discussed.
        scope_in=[],
        scope_out=[],
        key_requirements=[f"{r.id} {r.title or r.body}" for r in requirements[:12]],
        non_functional=[r.body for r in requirements if r.category and r.category.value == "non_functional"][:8],
        assumptions=assumptions_lines(state)[:8],
        open_questions=open_questions_lines(state)[:8],
    )


async def summarise(services: Services, state: SessionState, *, emit: Emitter = null_emitter) -> SummaryDoc:
    qa_rows = services.store.list_qa(state.session_id)
    system, user = render(
        "summary",
        project_name=(state.plan.project_name if state.plan else state.title),
        objective=(state.plan.objective if state.plan else ""),
        brief=state.brief[:2000],
        requirements=requirements_rendered(state),
        facts=facts_lines(state) + assumptions_lines(state),
        risks=risks_lines(state),
        transcript=transcript_lines(qa_rows, limit=40),
    )
    result = await services.gateway.structured(
        SummaryDoc, system_prompt=system, user_prompt=user, role="summary", emit=emit, step="summary",
        model_override=session_model(state, "summary")
    )
    summary = result.parsed if isinstance(result.parsed, SummaryDoc) else None
    if summary is None or not (summary.key_requirements or summary.background):
        await emit(EventType.WARNING, {
            "step": "summary", "role": "summary",
            "message": "model summary unavailable — generated one directly from the requirements ledger",
            "detail": (result.error or "")[:300],
        })
        return deterministic_summary(state)

    if not summary.project_name:
        summary.project_name = state.plan.project_name if state.plan else state.title
    return summary


__all__ = ["merge_triage", "triage", "gap_analysis", "summarise", "deterministic_summary", "render_summary_markdown"]
