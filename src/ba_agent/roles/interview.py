"""Interview roles: planning the agenda, asking the next question, judging coverage."""

from __future__ import annotations

from ..context import coverage_table, known_facts_lines
from ..events import Emitter, EventType, null_emitter
from ..prompts import render
from ..schemas import PlanResult, ProbeDecision, QuestionDraft, SessionState, SubjectSpec
from ..services import Services, session_model

#: Kept from v1's ``interviewer_agent`` prompt — the 8-subject agenda was its best idea. If the model cannot
#: produce a plan (small quantised model, offline, schema failure) we still run a competent interview.
DEFAULT_SUBJECTS: tuple[str, ...] = (
    "Product Overview",
    "Objectives & KPIs",
    "Scope In and Out",
    "Users, Roles & Personas",
    "Business Processes",
    "Data & Integrations",
    "Non-Functional Requirements",
    "Constraints, Assumptions & Risks",
)


def _fallback_plan(state: SessionState) -> PlanResult:
    return PlanResult(
        project_name=state.title or "Untitled project",
        objective=(state.seed.strip().splitlines() or [""])[0][:200],
        summary=state.seed.strip()[:600],
        subjects=[
            SubjectSpec(name=name, why="Standard elicitation agenda.", priority="high" if index < 4 else "medium")
            for index, name in enumerate(DEFAULT_SUBJECTS)
        ],
        known_facts=[],
        assumptions=[],
    )


async def make_plan(services: Services, state: SessionState, *, emit: Emitter = null_emitter) -> PlanResult:
    cfg = services.config.interview
    knowledge_query = f"{state.seed[:400]}\nfunctional specification interview subjects"
    knowledge_context = await services.knowledge.context_block(knowledge_query, max_chars=2200)

    system, user = render(
        "plan",
        brief=state.seed,
        knowledge_context=knowledge_context,
        max_subjects_min=min(cfg.min_subjects, cfg.max_subjects),
        max_subjects=cfg.max_subjects,
        prior_lessons=[row["text"] for row in services.store.list_lessons(limit=8)],
    )
    result = await services.gateway.structured(
        PlanResult,
        system_prompt=system,
        user_prompt=user,
        role="plan",
        emit=emit,
        step="plan",
        model_override=session_model(state, "plan"),
    )

    plan = result.parsed if isinstance(result.parsed, PlanResult) else None
    if plan is None or not plan.subjects:
        await emit(
            EventType.WARNING,
            {
                "step": "plan",
                "role": "plan",
                "message": "could not build a tailored interview plan — falling back to the standard BA agenda",
                "detail": (result.error or "")[:300],
            },
        )
        return _fallback_plan(state)

    # Clamp the agenda to configured bounds, drop blanks and duplicates.
    cleaned: list[SubjectSpec] = []
    seen: set[str] = set()
    for subject in plan.subjects:
        name = " ".join((subject.name or "").split())[:80]
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        cleaned.append(subject)
    cleaned = cleaned[: cfg.max_subjects]
    # Top a thin agenda up to the configured floor using the standard BA agenda, so a vague brief still gets a
    # competent interview. The floor comes from config — no magic number overriding an explicit min_subjects.
    target = min(cfg.min_subjects, cfg.max_subjects)
    for name in DEFAULT_SUBJECTS:
        if len(cleaned) >= target:
            break
        if name.lower() not in seen:
            cleaned.append(SubjectSpec(name=name, why="Standard elicitation agenda.", priority="medium"))
            seen.add(name.lower())
    plan.subjects = cleaned
    return plan


#: How many previously asked questions the question prompt is allowed to carry.
QUESTION_WINDOW = 14


def previous_questions(state: SessionState, limit: int = QUESTION_WINDOW) -> list[str]:
    """The anti-repetition list for the next question.

    Two things the naive version got wrong. Building it as `[unresolved] + [all]` put every open
    question in the prompt *twice*, and the `[-limit:]` window then cut the open questions off the
    head once the interview exceeded *limit* — precisely the questions a follow-up is most likely to
    re-ask. Open questions are de-duplicated and kept ahead of the window budget; answered questions
    fill the remainder from the most recent backwards.
    """
    seen: set[str] = set()
    open_ones: list[str] = []
    answered: list[str] = []
    for question in state.questions:
        key = " ".join(question.text.split()).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        (answered if question.resolved else open_ones).append(question.text)

    budget = max(0, limit - len(open_ones))
    return (open_ones + answered[-budget:])[:limit]


async def next_question(
    services: Services, state: SessionState, subject_name: str, *, emit: Emitter = null_emitter
) -> QuestionDraft | None:
    subject = next((s for s in state.subjects if s.name == subject_name), None)
    if subject is None:
        return None

    asked = previous_questions(state)
    knowledge_query = " ".join(
        filter(
            None,
            [
                (state.plan.objective if state.plan else ""),
                subject.name,
                " ".join(known_facts_lines(state)[:6]),
            ],
        )
    )[:500]
    knowledge_context = await services.knowledge.context_block(knowledge_query, max_chars=1800)

    system, user = render(
        "question",
        project_name=(state.plan.project_name if state.plan else state.title),
        objective=(state.plan.objective if state.plan else ""),
        brief=state.seed[:2500],
        subject=subject,
        coverage_table=coverage_table(state),
        known_facts=known_facts_lines(state),
        previous_questions=asked,
        asked_count=subject.asked,
        max_questions=services.config.interview.max_questions_per_subject,
        knowledge_context=knowledge_context,
    )
    result = await services.gateway.structured(
        QuestionDraft,
        system_prompt=system,
        user_prompt=user,
        role="interview",
        emit=emit,
        step=f"question:{subject.name}",
        model_override=session_model(state, "interview"),
    )
    draft = result.parsed if isinstance(result.parsed, QuestionDraft) else None
    if draft is None or not draft.question.strip():
        await emit(
            EventType.WARNING,
            {
                "step": f"question:{subject.name}",
                "role": "interview",
                "message": "no question generated — marking the subject as covered instead of stalling",
                "detail": (result.error or "")[:300],
            },
        )
        return None

    # Guard against a model that ignores "one question" and sends a paragraph.
    text = " ".join(draft.question.split())
    if len(text) > 420:
        text = text[:417].rstrip() + "…"
    draft.question = text
    return draft


async def probe_coverage(
    services: Services,
    state: SessionState,
    subject_name: str,
    question_text: str,
    answer_text: str,
    *,
    emit: Emitter = null_emitter,
) -> ProbeDecision:
    """Decide whether to follow up. On any failure we treat the subject as covered — never stall the interview."""
    subject = next((s for s in state.subjects if s.name == subject_name), None)
    max_questions = services.config.interview.max_questions_per_subject
    if subject is not None and subject.asked >= max_questions:
        return ProbeDecision(covered=True, reason="question budget for this subject exhausted")

    subject_ledger = [item.as_line() for item in state.ledger if item.subject == subject_name][-12:]
    next_subject = next((s.name for s in state.subjects if s.name != subject_name and not s.closed), "")
    system, user = render(
        "probe",
        subject=subject or SubjectSpec(name=subject_name),
        question=question_text,
        answer=answer_text[:3000],
        subject_ledger=subject_ledger,
        asked_count=(subject.asked if subject else 1),
        max_questions=max_questions,
        next_subject=next_subject,
        brief=state.seed[:2500],
        objective=state.plan.objective if state.plan else "",
        known_facts=known_facts_lines(state),
        previous_questions=previous_questions(state),
    )
    result = await services.gateway.structured(
        ProbeDecision,
        system_prompt=system,
        user_prompt=user,
        role="interview",
        emit=emit,
        step=f"probe:{subject_name}",
        model_override=session_model(state, "interview"),
    )
    decision = result.parsed if isinstance(result.parsed, ProbeDecision) else None
    if decision is None:
        await emit(
            EventType.WARNING,
            {
                "step": f"probe:{subject_name}",
                "role": "interview",
                "message": "coverage probe unavailable — treating the subject as covered and moving on (degraded)",
                "detail": (result.error or "")[:300],
            },
        )
        return ProbeDecision(covered=True, reason="coverage probe unavailable — moving on")
    return decision
