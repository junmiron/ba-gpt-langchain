"""The deterministic workflow engine — the heart of the rewrite.

v1 let an LLM decide who speaks next and regex-matched its prose (``READY`` / ``QUESTION:`` / ``NEEDS_INFO`` /
``[ASSIGNMENT]``). Here **the code owns every transition**; models only produce content. Consequences:

* no magic tokens anywhere in prompts or parsing;
* every loop has a budget, so a run cannot stall (v1 used ``MaxMessageTermination(100000000)``);
* the engine stops at human-in-the-loop points and resumes from SQLite — restart the server mid-interview and it
  picks up exactly where it paused;
* escalation is real logic: a *user-dependent* gap goes back to the interview, an *internally fixable* one re-runs
  only the affected section.

Phases::

    intake → interview ⇄ clarify → summary → approval → draft → review ⇄ fix → diagram → deliver → done
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from typing import Literal

from .context import known_facts_lines, render_summary_markdown
from .events import Emitter, EventType, null_emitter
from .export import write_deliverables
from .mermaid_utils import validate_mermaid
from .models import no_model_message
from .prompts import render
from .roles import analysis, interview, writing
from .roles import diagram as diagram_roles
from .roles import review as review_role
from .schemas import (
    ApprovalResult,
    ArtifactRef,
    AwaitKind,
    Issue,
    LedgerItem,
    Phase,
    QAPair,
    QuestionKind,
    QuestionPurpose,
    QuestionState,
    RunStatus,
    SessionState,
    StakeholderAnswer,
    SubjectState,
    question_is_duplicate,
    utcnow,
)
from .services import Services, session_model
from .spec import SECTION_BY_ID, LintFinding, assemble_spec, deterministic_gaps, lint_spec, section_title, spec_stats

#: Floor on engine iterations inside one ``advance`` call. The real ceiling is derived from the
#: configured budgets by :meth:`WorkflowEngine._step_budget` — a fixed number silently truncates a
#: run whose budgets legitimately need more steps, and a truncated run is reported as "running" with
#: no task scheduled, which the UI cannot resume.
MAX_STEPS_PER_ADVANCE = 40


class WorkflowHalt(Exception):
    """Raised by a step to stop the run with a user-visible reason."""


StepHandler = Callable[[SessionState, Emitter], Awaitable[None]]


class WorkflowEngine:
    def __init__(self, services: Services) -> None:
        self.services = services
        self.config = services.config

    # ------------------------------------------------------------------ public API

    async def advance(self, state: SessionState, *, emit: Emitter = null_emitter) -> SessionState:
        """Run steps until the workflow needs the human, finishes, or hits the step budget."""
        handlers: dict[Phase, StepHandler] = {
            Phase.INTAKE: self._step_plan,
            Phase.INTERVIEW: self._step_interview,
            Phase.CLARIFY: self._step_clarify,
            Phase.SUMMARY: self._step_summary,
            Phase.APPROVAL: self._step_approval_wait,
            Phase.DRAFT: self._step_draft,
            Phase.REVIEW: self._step_review,
            Phase.DIAGRAM: self._step_diagram,
            Phase.DELIVER: self._step_deliver,
        }

        # Fail fast and legibly when there is nothing to call. Without this the engine grinds through every step
        # on fallbacks and ships a hollow document; with it, the run stops where it can be resumed.
        if (
            state.phase is not Phase.DONE
            and state.awaiting not in (AwaitKind.ANSWER, AwaitKind.APPROVAL)
            and not await self.services.gateway.resolve_model(None)
        ):
            message = no_model_message(self.config.provider)
            state.error = message
            state.status = RunStatus.ERROR
            self.services.store.save_state(state)
            await emit(EventType.ERROR, {"phase": state.phase.value, "message": message, "resumable": True})
            return state

        for _ in range(self._step_budget(state)):
            if state.phase is Phase.DONE:
                state.status = RunStatus.DONE
                state.awaiting = AwaitKind.FINISHED
                break
            if state.awaiting is AwaitKind.ANSWER and self._human_model(state):
                state.status = RunStatus.RUNNING
                started = time.perf_counter()
                before_calls, before_model_ms = self._usage_snapshot()
                await emit(EventType.STEP_START, {"phase": state.phase.value, "status": state.status.value})
                try:
                    answered = await self._simulate_stakeholder_answer(state, emit=emit)
                except Exception as exc:  # noqa: BLE001 - a simulator failure must not invent a response
                    state.error = f"Human simulator failed: {type(exc).__name__}: {exc}"
                    state.status = RunStatus.ERROR
                    await emit(EventType.ERROR, {"phase": state.phase.value, "message": state.error, "resumable": True})
                    answered = False
                wall_ms = int((time.perf_counter() - started) * 1000)
                self._record_metrics(state, before_calls, before_model_ms, wall_ms)
                self.services.store.save_state(state)
                await emit(EventType.STEP_END, {"phase": state.phase.value, "ms": wall_ms})
                if not answered:
                    if state.status is not RunStatus.ERROR:
                        state.status = RunStatus.WAITING_USER
                        self.services.store.save_state(state)
                    break
                continue
            if state.awaiting in (AwaitKind.ANSWER, AwaitKind.APPROVAL):
                state.status = RunStatus.WAITING_USER
                break

            # Invariant: WAITING_USER <=> awaiting is answer/approval. After an answer is submitted the
            # awaiting flag clears but the status still says WAITING_USER until the end of the loop, so
            # every state persisted mid-run (line 139) would tell the UI "waiting for you" while the
            # engine is actually working on a step.
            state.status = RunStatus.RUNNING

            handler = handlers.get(state.phase)
            if handler is None:
                state.error = f"No handler for phase '{state.phase.value}'"
                state.status = RunStatus.ERROR
                await emit(EventType.ERROR, {"message": state.error})
                break

            started = time.perf_counter()
            before_calls, before_model_ms = self._usage_snapshot()
            await emit(EventType.STEP_START, {"phase": state.phase.value, "status": state.status.value})
            try:
                await handler(state, emit)
            except WorkflowHalt as halt:
                state.error = str(halt)
                state.status = RunStatus.ERROR
                await emit(EventType.ERROR, {"phase": state.phase.value, "message": str(halt)})
                break
            except Exception as exc:  # noqa: BLE001 — one failing step must never lose the session
                state.error = f"{type(exc).__name__}: {exc}"
                state.status = RunStatus.ERROR
                await emit(EventType.ERROR, {"phase": state.phase.value, "message": state.error})
                break

            wall_ms = int((time.perf_counter() - started) * 1000)
            self._record_metrics(state, before_calls, before_model_ms, wall_ms)
            self.services.store.save_state(state)
            await emit(EventType.STEP_END, {"phase": state.phase.value, "ms": wall_ms})
        else:
            # The budget ran out without reaching DONE or a human-in-the-loop point. This is the silent-stall
            # case: every inner loop has a budget, so a legitimate run should never get here — but if it does,
            # the session must be resumable rather than stuck. `RUNNING` with no scheduled task shows no Stop
            # button and no Resume button (the UI only offers Resume for idle/cancelled/error), so the user has
            # no way forward and the session looks frozen forever.
            state.status = RunStatus.IDLE
            state.error = (
                f"stalled: the engine used its whole {self._step_budget(state)}-step budget without "
                f"finishing (phase={state.phase.value}) — press Resume to continue"
            )
            self.services.store.save_state(state)
            await emit(
                EventType.WARNING,
                {
                    "step": "advance",
                    "message": state.error,
                    "resumable": True,
                },
            )

        if (
            state.awaiting not in (AwaitKind.ANSWER, AwaitKind.APPROVAL)
            and state.phase is not Phase.DONE
            and state.status is not RunStatus.ERROR
            and state.status is not RunStatus.IDLE
        ):
            state.status = RunStatus.RUNNING
        self.services.store.save_state(state)
        return state

    async def submit_answer(
        self,
        state: SessionState,
        question_id: int,
        *,
        answer: str | None = None,
        skipped: bool = False,
        not_applicable: bool = False,
        emit: Emitter = null_emitter,
    ) -> SessionState:
        """Record a real stakeholder reply to a pending question and continue the run."""
        await self._record_answer(
            state,
            question_id,
            answer=answer,
            skipped=skipped,
            not_applicable=not_applicable,
            answer_source="human",
            emit=emit,
        )
        return await self.advance(state, emit=emit)

    async def _record_answer(
        self,
        state: SessionState,
        question_id: int,
        *,
        answer: str | None,
        skipped: bool = False,
        not_applicable: bool = False,
        answer_source: Literal["human", "simulated"],
        emit: Emitter,
    ) -> None:
        question = next((q for q in state.questions if q.id == question_id), None)
        if question is None:
            raise ValueError(f"Unknown question id {question_id}")
        if question.resolved:
            raise ValueError(f"Question {question_id} was already answered")
        text = (answer or "").strip()
        if not skipped and not not_applicable and not text:
            raise ValueError("Provide an answer, or skip the question.")

        question.answer = None if (skipped or not_applicable) else text
        question.answer_source = answer_source
        question.skipped = skipped
        question.not_applicable = not_applicable
        question.answered_at = utcnow()
        self.services.store.update_qa_answer(state.session_id, question.id, question.answer, skipped, not_applicable)

        await emit(
            EventType.ANSWER,
            {
                "question_id": question.id,
                "subject": question.subject,
                "purpose": question.purpose,
                "answer": question.answer or ("(skipped)" if skipped else "(not applicable)"),
                "source": answer_source,
            },
        )

        state.awaiting = AwaitKind.NONE
        subject = next((s for s in state.subjects if s.name == question.subject), None)
        usable = "" if (skipped or not_applicable) else text

        # 1. Structure the answer into the ledger (nothing is lost even when the model fails here).
        if usable:
            await analysis.triage(self.services, state, question.subject, question.text, usable, emit=emit)

        # 2. Route deterministically based on why the question was asked.
        if question.purpose == "agenda":
            if subject is not None:
                subject.asked += 1
            await self._after_agenda_answer(state, question, subject, usable, emit=emit)
        elif question.purpose == "gap":
            state.phase = Phase.CLARIFY
        else:  # review follow-up → re-draft with the new information, then re-review
            if usable and not state.has_similar(usable):
                state.ledger.append(
                    LedgerItem(id=f"FACT-{len(state.ledger) + 1}", kind="fact", body=usable, subject=question.subject)
                )
            state.phase = Phase.DRAFT

    def _human_model(self, state: SessionState) -> str | None:
        """Return a selected simulator model; no override means a real stakeholder answers."""
        session_override = (state.model_overrides or {}).get("human")
        configured = self.config.role("human").model
        return (session_override or configured or "").strip() or None

    async def _simulate_stakeholder_answer(self, state: SessionState, *, emit: Emitter) -> bool:
        question = state.current_question
        if question is None:
            return False
        system, user = render(
            "human",
            brief=state.seed,
            known_facts="\n".join(known_facts_lines(state)) or "(none captured)",
            subject=question.subject,
            question=question.text,
        )
        result = await self.services.gateway.structured(
            StakeholderAnswer,
            system_prompt=system,
            user_prompt=user,
            role="human",
            model_override=session_model(state, "human"),
            emit=emit,
            step=f"answer:{question.id}",
        )
        parsed = result.parsed if isinstance(result.parsed, StakeholderAnswer) else None
        answer = (parsed.answer if parsed else "").strip()
        if not answer:
            await emit(
                EventType.WARNING,
                {
                    "step": f"answer:{question.id}",
                    "role": "human",
                    "message": "stakeholder simulator returned no usable answer; waiting for a real person instead",
                    "detail": (result.error or "empty structured answer")[:300],
                },
            )
            return False
        await self._record_answer(
            state,
            question.id,
            answer=answer,
            answer_source="simulated",
            emit=emit,
        )
        return True

    async def set_approval(
        self, state: SessionState, approved: bool, notes: str = "", *, emit: Emitter = null_emitter
    ) -> SessionState:
        """Approval gate. A rejection becomes ledger input and re-opens clarification."""
        if state.phase not in (Phase.APPROVAL, Phase.SUMMARY):
            raise ValueError(f"Nothing to approve while in phase '{state.phase.value}'")

        verdict = ApprovalResult(decision="approve" if approved else "changes", notes=(notes or "").strip())
        await emit(EventType.APPROVAL, {"approved": bool(approved), "notes": verdict.notes})

        if approved:
            state.approved_summary = True
            state.approval_notes = verdict.notes
            state.awaiting = AwaitKind.NONE
            state.phase = Phase.DRAFT
        else:
            for line in [chunk.strip() for chunk in verdict.notes.splitlines() if chunk.strip()]:
                if not state.has_similar(line):
                    state.ledger.append(
                        LedgerItem(
                            id=f"DECISION-{len(state.ledger) + 1}", kind="decision", body=line, subject="summary review"
                        )
                    )
            state.approved_summary = False
            state.awaiting = AwaitKind.NONE
            state.summary_rejections += 1
            limit = max(1, self.config.interview.max_summary_rejections)
            if state.summary_rejections >= limit:
                # Bounded like everything else. The notes are already in the ledger, so drafting continues with
                # them instead of ping-ponging between summary and approval forever (v1's stall class of bug).
                await emit(
                    EventType.WARNING,
                    {
                        "step": "approval",
                        "message": f"{state.summary_rejections} change request(s) noted — drafting now. Your notes are "
                        "in the ledger and are reflected in the document; you can still ask for changes "
                        "after the draft.",
                    },
                )
                state.phase = Phase.DRAFT
            else:
                # Re-open clarification (bounded by max_gap_rounds) so the change is actually elicited.
                state.phase = Phase.CLARIFY if state.gap_round < self.config.interview.max_gap_rounds else Phase.SUMMARY

        return await self.advance(state, emit=emit)

    # ------------------------------------------------------------------ steps

    async def _step_approval_wait(self, state: SessionState, emit: Emitter) -> None:
        state.awaiting = AwaitKind.APPROVAL
        state.status = RunStatus.WAITING_USER

    async def _step_plan(self, state: SessionState, emit: Emitter) -> None:
        has_brief = bool((state.brief or "").strip())
        has_transcript = bool((state.transcript or "").strip())
        if not has_brief and not has_transcript:
            raise WorkflowHalt("No project brief or transcript was provided — cannot plan an interview.")

        # Transcript mode (Phase 3): seed the ledger from the uploaded elicitation transcript *before*
        # planning, so the agenda is built from what is already known and the interview only fills gaps.
        transcript_items: list[LedgerItem] = []
        transcript_chunks: list[str] = []
        if has_transcript:
            transcript_items, transcript_chunks = await analysis.ingest_transcript(self.services, state, emit=emit)

        plan = await interview.make_plan(self.services, state, emit=emit)
        state.plan = plan
        state.title = plan.project_name or state.title

        existing = {s.name for s in state.subjects}
        for spec in plan.subjects:
            if spec.name not in existing:
                state.subjects.append(SubjectState(name=spec.name, why=spec.why, priority=spec.priority))

        for fact in plan.known_facts:
            clean = " ".join(fact.split())
            if clean and not state.has_similar(clean):
                state.ledger.append(
                    LedgerItem(id=f"FACT-{len(state.ledger) + 1}", kind="fact", body=clean, subject="brief")
                )
        for assumption in plan.assumptions:
            clean = " ".join(assumption.split())
            if clean and not state.has_similar(clean):
                state.ledger.append(
                    LedgerItem(id=f"ASSUMPTION-{len(state.ledger) + 1}", kind="assumption", body=clean, subject="brief")
                )

        await emit(
            EventType.SESSION,
            {
                "event": "plan_ready",
                "project_name": plan.project_name,
                "objective": plan.objective,
                "summary": plan.summary,
                "subjects": [s.model_dump() for s in state.subjects],
            },
        )

        if (state.transcript or "").strip():
            # Gap-only interview: subjects the transcript actually addressed are closed; subjects it did not
            # are still interviewed. Coverage is decided by evidence (how many excerpts mention the subject),
            # not by the assumption that an uploaded transcript answers everything.
            threshold = max(1, self.config.interview.subject_coverage_threshold)
            covered = set(analysis.transcript_covered_subjects(state.subjects, transcript_chunks, threshold))
            still_open: list[str] = []
            for subject in state.subjects:
                if subject.closed:
                    continue
                if subject.name in covered:
                    subject.status = "covered"
                    subject.notes = f"answered by the uploaded transcript ({threshold}+ excerpts)"
                else:
                    still_open.append(subject.name)

            gaps = deterministic_gaps(state, min_functional_requirements=self.config.review.min_functional_requirements)
            if still_open:
                state.phase = Phase.INTERVIEW
            elif gaps:
                state.phase = Phase.CLARIFY
            else:
                state.phase = Phase.SUMMARY
            await emit(
                EventType.SESSION,
                {
                    "event": "transcript_mode",
                    "message": (
                        f"the transcript covers {len(covered)} of {len(state.subjects)} subjects"
                        + (f"; {len(still_open)} still need questions" if still_open else "; only gaps remain")
                    ),
                    "covered": sorted(covered),
                    "uncovered": still_open,
                    "coverage_threshold": threshold,
                    "excerpts_triaged": len(transcript_chunks),
                    "ledger_entries": len(transcript_items),
                    "gaps": gaps,
                },
            )
            return

        state.phase = Phase.INTERVIEW

    async def _step_interview(self, state: SessionState, emit: Emitter) -> None:
        cfg = self.config.interview

        if len(state.questions) >= cfg.max_total_questions:
            await emit(
                EventType.WARNING,
                {
                    "step": "interview",
                    "message": f"question budget ({cfg.max_total_questions}) reached — closing the interview",
                },
            )
            for subject in state.subjects:
                if not subject.closed:
                    subject.status = "skipped"
                    subject.notes = subject.notes or "budget exhausted"
            state.phase = Phase.CLARIFY
            return

        target = state.next_subject()
        if target is None:
            state.phase = Phase.CLARIFY
            return

        draft = await interview.next_question(self.services, state, target.name, emit=emit)
        if draft is None:
            target.status = "done"
            target.notes = target.notes or "no question could be generated"
            return

        repeat = self._already_asked(state, draft.question)
        if repeat is not None:
            await emit(
                EventType.WARNING,
                {
                    "step": f"question:{target.name}",
                    "message": f"the model re-asked question {repeat.id}; {target.name} is closed instead",
                    "detail": draft.question[:160],
                },
            )
            target.status = "done"
            target.notes = target.notes or "duplicate of an earlier question"
            return

        target.status = "asking"
        asked = await self._ask(
            state,
            subject=target.name,
            text=draft.question,
            rationale=draft.rationale,
            kind=draft.kind,
            options=draft.options,
            purpose="agenda",
            emit=emit,
        )
        if asked is None:  # budget consumed between the check above and here
            target.status = "skipped"
            target.notes = "question budget exhausted"
            state.phase = Phase.CLARIFY

    async def _after_agenda_answer(
        self, state: SessionState, question: QuestionState, subject: SubjectState | None, answer: str, emit: Emitter
    ) -> None:
        if not answer:
            # Skipped / N/A closes the subject instead of nagging (v1 looped on exactly these).
            if subject is not None and subject.status != "done":
                subject.status = "skipped"
                subject.notes = "skipped by stakeholder" if question.skipped else "not applicable"
            state.phase = Phase.INTERVIEW
            return

        decision = await interview.probe_coverage(
            self.services, state, question.subject, question.text, answer, emit=emit
        )
        follow_up = decision.follow_up
        budget_left = subject is not None and subject.asked < self.config.interview.max_questions_per_subject
        if follow_up is not None and self._already_asked(state, follow_up.question) is not None:
            follow_up = None  # the probe suggested a question we have already asked
        if follow_up is not None and budget_left:
            asked = await self._ask(
                state,
                subject=question.subject,
                text=follow_up.question,
                rationale=follow_up.rationale,
                kind="open",
                options=[],
                purpose="agenda",
                emit=emit,
            )
            if asked is not None:
                return

        if subject is not None and subject.status != "skipped":
            subject.status = "done"
            if decision.reason:
                subject.notes = decision.reason[:160]
        state.phase = Phase.INTERVIEW

        next_subject = state.next_subject()
        next_question = " ".join(decision.next_question.split())
        if (
            decision.covered
            and next_subject is not None
            and next_question
            and len(state.questions) < self.config.interview.max_total_questions
        ):
            next_subject.status = "asking"
            asked = await self._ask(
                state,
                subject=next_subject.name,
                text=next_question,
                rationale=decision.next_rationale,
                kind="open",
                options=[],
                purpose="agenda",
                emit=emit,
            )
            if asked is not None:
                return
            next_subject.status = "pending"  # duplicate/budget guard rejected it; use the regular generator below

    async def _step_clarify(self, state: SessionState, emit: Emitter) -> None:
        cfg = self.config.interview

        if state.current_question is not None:  # a clarification question is already on screen
            state.awaiting = AwaitKind.ANSWER
            return
        if await self._drain_queue(state, emit):  # earlier rounds queued blockers they never got to ask
            return

        budget_left = state.gap_round < cfg.max_gap_rounds and len(state.questions) < cfg.max_total_questions
        code_gaps = deterministic_gaps(
            state, min_functional_requirements=self.config.review.min_functional_requirements
        )
        if not budget_left:
            if code_gaps:
                # The gaps can no longer be asked, so they have to be written down. Without this the
                # warning below promises open questions the deliverable never contains.
                for text in code_gaps:
                    if self._already_asked(state, text) is not None:
                        continue
                    state.ledger.append(
                        LedgerItem(
                            id=f"OPEN-{len(state.ledger) + 1}",
                            kind="open_question",
                            body=text,
                            subject="clarify",
                        )
                    )
                await emit(
                    EventType.WARNING,
                    {
                        "step": "clarify",
                        "message": f"gap-round budget ({cfg.max_gap_rounds}) exhausted — "
                        f"{len(code_gaps)} remaining gap(s) recorded as open questions",
                    },
                )
            state.phase = Phase.SUMMARY
            return

        gaps = await analysis.gap_analysis(self.services, state, code_gaps, emit=emit)
        fresh = [gap for gap in gaps if self._already_asked(state, gap.question) is None]

        state.gap_round += 1
        if not fresh:
            state.phase = Phase.SUMMARY
            return

        await emit(
            EventType.REVIEW,
            {
                "step": "clarify",
                "round": state.gap_round,
                "summary": f"{len(fresh)} clarification question(s) needed",
                "gaps": [gap.model_dump() for gap in fresh],
            },
        )
        first = fresh[0]
        asked_first = await self._ask(
            state,
            subject=first.subject or "Clarification",
            text=first.question,
            rationale=first.why,
            kind="open",
            options=[],
            purpose="gap",
            round=state.gap_round,
            emit=emit,
        )
        dropped: list[str] = []
        for extra in fresh[1:]:  # queued; asked one at a time after each answer
            if (
                self._queue_question(
                    state,
                    subject=extra.subject or "Clarification",
                    text=extra.question,
                    rationale=extra.why,
                    purpose="gap",
                    round=state.gap_round,
                )
                is None
            ):
                dropped.append(extra.question)
        if asked_first is None or dropped:
            # Gaps the question budget cannot afford still belong in the deliverable's open-questions list.
            for text in ([first.question] if asked_first is None else []) + dropped:
                if self._already_asked(state, text) is not None:
                    continue
                state.ledger.append(
                    LedgerItem(id=f"OPEN-{len(state.ledger) + 1}", kind="open_question", body=text, subject="clarify")
                )
            if asked_first is None:
                state.phase = Phase.SUMMARY

    async def _step_summary(self, state: SessionState, emit: Emitter) -> None:
        summary = await analysis.summarise(self.services, state, emit=emit)
        state.summary = summary
        await emit(EventType.SPEC, {"kind": "summary", "markdown": render_summary_markdown(summary)})
        await emit(
            EventType.APPROVAL,
            {"requested": True, "project_name": summary.project_name, "objective": summary.objective},
        )
        state.phase = Phase.APPROVAL

    async def _step_draft(self, state: SessionState, emit: Emitter) -> None:
        if not state.approved_summary and state.summary is None:
            raise WorkflowHalt("Draft requested before the requirements summary was approved.")

        from .spec import LLM_SECTIONS

        for section in LLM_SECTIONS:
            hint = self._fix_hint_for(state, section.id)
            if (state.sections.get(section.id) or "").strip() and not hint:
                continue  # keep what we have; regenerate only when the critic complained about it
            markdown = await writing.write_section(self.services, state, section, extra_instructions=hint, emit=emit)
            if markdown:
                state.sections[section.id] = markdown
                await emit(
                    EventType.SPEC,
                    {
                        "kind": "section",
                        "section_id": section.id,
                        "title": section.title,
                        "markdown": markdown,
                        "stats": spec_stats(state),
                    },
                )

        use_case_hint = self._fix_hint_for(state, "use_cases")
        if not (state.sections.get("use_cases") or "").strip() or use_case_hint:
            markdown = await writing.write_use_cases(self.services, state, emit=emit)
            if markdown:
                state.sections["use_cases"] = markdown
                await emit(
                    EventType.SPEC,
                    {
                        "kind": "section",
                        "section_id": "use_cases",
                        "title": SECTION_BY_ID["use_cases"].title,
                        "markdown": markdown,
                    },
                )

        # Assemble once so the glossary sees the whole document, then re-assemble with the glossary included.
        state.spec_markdown = assemble_spec(state)
        glossary_hint = self._fix_hint_for(state, "glossary")
        if not (state.sections.get("glossary") or "").strip() or glossary_hint:
            glossary = await writing.write_glossary(self.services, state, writing.glossary_corpus(state), emit=emit)
            if glossary:
                state.sections["glossary"] = glossary

        state.spec_markdown = assemble_spec(state)
        await emit(EventType.SPEC, {"kind": "document", "markdown": state.spec_markdown, "stats": spec_stats(state)})

        artifact = self._write_text_artifact(state, "functional_specification.md", state.spec_markdown)
        if artifact:
            await emit(
                EventType.ARTIFACT,
                {
                    "name": artifact.name,
                    "bytes": artifact.bytes,
                    "url": f"/api/artifacts/{state.session_id}/{artifact.name}",
                },
            )

        state.phase = Phase.REVIEW

    async def _step_review(self, state: SessionState, emit: Emitter) -> None:
        cfg = self.config.review
        if await self._drain_queue(state, emit):  # blockers queued last round, asked one at a time
            return
        findings: list[LintFinding] = lint_spec(state, min_functional_requirements=cfg.min_functional_requirements)
        verdict = await review_role.critique(self.services, state, findings, emit=emit)

        state.lint_report = [finding.line() for finding in findings]
        state.open_issues = verdict.issues
        await emit(
            EventType.REVIEW,
            {
                "round": state.review_rounds + 1,
                "status": verdict.status,
                "summary": verdict.summary,
                "issues": [issue.model_dump() for issue in verdict.issues],
                "lint": state.lint_report,
                "stats": spec_stats(state),
            },
        )

        if verdict.status == "complete":
            state.phase = Phase.DIAGRAM
            return

        blockers = [issue for issue in verdict.issues if issue.severity == "blocker"]
        budget_left = state.review_rounds < cfg.max_review_rounds

        def already_asked(text: str) -> bool:
            return self._already_asked(state, text) is not None

        # 1) Stakeholder-dependent blockers → ask, then re-draft with the answers. A machine diagnostic is only
        #    put in front of a human when it has stakeholder wording: asking "only 1 functional requirement was
        #    captured" gets an answer that cannot fix it, and burns a review round doing so.
        def askable_text(issue: Issue) -> str:
            question = (issue.question_for_user or "").strip()
            if question:
                return question
            return (issue.description or "").strip() if issue.source == "critic" else ""

        user_issues = [(issue, askable_text(issue)) for issue in blockers if issue.kind == "user"]
        fresh_user = [(issue, text) for issue, text in user_issues if text and not already_asked(text)]
        if fresh_user and budget_left:
            state.review_rounds += 1
            first, first_text = fresh_user[0]
            asked = await self._ask(
                state,
                subject=section_title(first.section_id),
                text=first_text,
                rationale=first.description,
                kind="open",
                options=[],
                purpose="review",
                round=state.review_rounds,
                emit=emit,
            )
            if asked is None:
                state.review_rounds -= 1
            dropped: list[str] = []
            for extra, extra_text in fresh_user[1:]:
                if (
                    self._queue_question(
                        state,
                        subject=section_title(extra.section_id),
                        text=extra_text,
                        rationale=extra.description,
                        purpose="review",
                        round=state.review_rounds,
                    )
                    is None
                ):
                    dropped.append(extra_text)
            if dropped:
                # A blocker the budget will not let us ask must not vanish: it is recorded as an open
                # question so it reaches section 10 of the deliverable and the QA report.
                for text in dropped:
                    if self._already_asked(state, text) is not None:
                        continue
                    state.ledger.append(
                        LedgerItem(
                            id=f"OPEN-{len(state.ledger) + 1}",
                            kind="open_question",
                            body=text,
                            subject="review",
                        )
                    )
                await emit(
                    EventType.WARNING,
                    {
                        "step": "review",
                        "message": f"question budget ({self.config.interview.max_total_questions}) reached — "
                        f"{len(dropped)} blocker(s) recorded as open questions instead of asked",
                    },
                )
            return

        # 2) Internally fixable blockers → regenerate only the offending sections (never a code-rendered one).
        internal = [
            i
            for i in verdict.issues
            if i.kind == "internal" and SECTION_BY_ID.get(i.section_id) and SECTION_BY_ID[i.section_id].kind != "static"
        ]
        if internal and budget_left:
            state.review_rounds += 1
            touched = sorted({i.section_id for i in internal})
            for section_id in touched:
                state.sections.pop(section_id, None)
            await emit(
                EventType.REVIEW,
                {
                    "round": state.review_rounds,
                    "status": "auto_fix",
                    "sections": touched,
                    "summary": f"regenerating {len(touched)} section(s) from critic feedback",
                },
            )
            state.phase = Phase.DRAFT
            return

        # 3) Out of budget (or nothing actionable) → ship, and report what is still open.
        if not budget_left:
            await emit(
                EventType.WARNING,
                {
                    "step": "review",
                    "message": f"review round budget ({cfg.max_review_rounds}) reached — "
                    f"{len(blockers)} unresolved blocker(s) are listed in the deliverable",
                },
            )
        state.phase = Phase.DIAGRAM

    async def _step_diagram(self, state: SessionState, emit: Emitter) -> None:
        cfg = self.config.diagram
        diagrams = await diagram_roles.generate_diagrams(self.services, state, emit=emit)
        if not diagrams:
            await emit(
                EventType.WARNING,
                {"step": "diagram", "message": "no diagram could be produced — continuing without one"},
            )
            state.phase = Phase.DELIVER
            return

        for item in diagrams:
            for _round in range(cfg.max_revision_rounds + 1):
                verdict = await diagram_roles.review_diagram(
                    self.services, state, item, list(item.get("errors") or []), emit=emit
                )
                revised = (verdict.get("revised_code") or "").strip()
                if revised and revised != item["code"]:
                    report = validate_mermaid(revised)
                    item["code"], item["valid"] = report.code, report.ok
                    item["errors"], item["warnings"] = report.errors, report.warnings
                    await emit(
                        EventType.DIAGRAM,
                        {"step": "diagram:revise", "title": item["title"], "valid": report.ok, "code": report.code},
                    )
                if verdict.get("approved") or not revised:
                    break

        state.diagrams = diagrams
        primary = diagram_roles.primary_diagram(diagrams)
        state.mermaid_code = primary["code"] if primary else ""

        if state.mermaid_code:
            artifact = self._write_text_artifact(
                state,
                "process_diagram.mmd",
                f"%% {state.title} — generated by BA agent (LangChain)\n{state.mermaid_code}\n",
            )
            if artifact:
                await emit(
                    EventType.ARTIFACT,
                    {
                        "name": artifact.name,
                        "bytes": artifact.bytes,
                        "url": f"/api/artifacts/{state.session_id}/{artifact.name}",
                    },
                )

        state.phase = Phase.DELIVER

    async def _step_deliver(self, state: SessionState, emit: Emitter) -> None:
        # Reassemble when the spec is missing (a post-delivery ledger edit clears it) so the bundle always
        # reflects the corrected ledger rather than a stale or empty document.
        if not (state.spec_markdown or "").strip():
            state.spec_markdown = assemble_spec(state)
            await emit(
                EventType.SPEC, {"kind": "document", "markdown": state.spec_markdown, "stats": spec_stats(state)}
            )

        written = write_deliverables(self.services, state)
        artifacts: list[ArtifactRef] = []
        for item in written:
            ref = item.to_ref()
            artifacts.append(ref)
            self.services.store.add_artifact(state.session_id, ref)
            await emit(
                EventType.ARTIFACT,
                {"name": item.name, "bytes": item.bytes, "url": f"/api/artifacts/{state.session_id}/{item.name}"},
            )

        lessons = self._record_lessons(state)
        state.artifacts = artifacts
        state.phase = Phase.DONE
        state.status = RunStatus.DONE
        state.awaiting = AwaitKind.FINISHED

        # Surface PDF QA in the timeline: a deliverable that renders but is structurally broken
        # (no bookmarks, leaked Markdown, missing requirement rows) must be visible, not buried in a
        # sidecar file nobody opens.
        qa_path = self.services.config.out_dir(state.session_id) / "pdf_qa.json"
        if qa_path.exists():
            try:
                report = json.loads(qa_path.read_text(encoding="utf-8"))
                for warning in report.get("warnings", []):
                    await emit(EventType.WARNING, {"source": "pdf_qa", "message": warning})
            except Exception:  # noqa: BLE001 - the QA sidecar is informational, never fatal
                pass

        await emit(
            EventType.SESSION,
            {
                "event": "completed",
                "artifacts": [a.model_dump() for a in artifacts],
                "stats": spec_stats(state),
                "lessons_recorded": lessons,
            },
        )

    def _record_lessons(self, state: SessionState) -> int:
        """Feed what this run learned back into the store, so the next interview plans around it (Phase 3).

        Only *generalisable* signals are kept: which kinds of subject stakeholders leave unanswered, which
        requirement categories a run tends to miss. Project-specific text (a particular FR id, a particular
        section length) would poison the next plan, so it is deliberately not recorded.
        """
        store = self.services.store
        recorded = 0

        skipped = [s.name for s in state.subjects if s.status == "skipped"]
        if skipped:
            recorded += bool(
                store.add_lesson(
                    state.session_id,
                    "coverage",
                    f"Subjects stakeholders tend to skip: {', '.join(skipped)}",
                )
            )

        gap_subjects = sorted({q.subject for q in state.questions if q.purpose == "gap"})
        if gap_subjects:
            recorded += bool(
                store.add_lesson(
                    state.session_id,
                    "gap",
                    f"Subjects that needed a clarification round after the first answer: {', '.join(gap_subjects)}",
                )
            )

        categories = {r.category.value for r in state.requirements() if r.category}
        missing = [c for c in ("functional", "non_functional") if c not in categories]
        if missing:
            recorded += bool(
                store.add_lesson(
                    state.session_id,
                    "coverage",
                    f"Runs of this kind often end with no {' or '.join(missing)} requirements — "
                    "ask for them explicitly.",
                )
            )

        coverage = self._coverage(state)
        if coverage["percent"] < 100:
            recorded += bool(
                store.add_lesson(
                    state.session_id,
                    "coverage",
                    f"Interview closed at {coverage['percent']}% subject coverage; "
                    f"{coverage['subjects'] - coverage['covered']} subject(s) never got a full answer.",
                )
            )
        return recorded

    # ------------------------------------------------------------------ helpers

    def _step_budget(self, state: SessionState) -> int:
        """How many engine iterations a single ``advance`` is allowed to run.

        Every loop in the engine is budgeted by config, so the number of steps a *legitimate* run
        can take in one advance is knowable: review ping-pongs draft↔review (two steps a round),
        clarify re-enters once per gap round, and diagram re-enters once per revision. A flat
        ceiling below that does not protect against bugs — it truncates the run, and a truncated
        run is reported as "running" with nothing scheduled, which the UI cannot resume.

        Derived from the budgets, floored at the historical constant, and capped so a runaway
        setting cannot make the guard meaningless.
        """
        cfg = self.config
        review = 2 * cfg.review.max_review_rounds
        clarify = cfg.interview.max_gap_rounds * (1 + cfg.interview.max_gap_questions_per_round)
        diagram = cfg.diagram.max_diagrams * (1 + cfg.diagram.max_revision_rounds)
        interview = cfg.interview.max_total_questions + len(state.subjects) + 2
        return min(MAX_STEPS_PER_ADVANCE + review + clarify + diagram + interview, 500)

    def _already_asked(self, state: SessionState, text: str) -> QuestionState | None:
        """The existing question that asks the same thing, or None.

        The interview prompt says "do not repeat", and local models ignore it: a real LM Studio run
        asked Q1 "Will the portal replace email for claim submission and tracking only, or also
        replace manual review and payment steps?" and then Q2 "For scope, does the portal replace
        email for submission/tracking only, or also replace manual review and payment execution?".
        Asking the stakeholder the same thing twice reads as an unserious analyst and wastes a
        question from a budget that is often the run's scarcest resource.

        The already-asked questions are passed as a corpus so the comparison discounts the domain
        vocabulary that recurs in every question of the session.
        """
        prior = [existing.text for existing in state.questions]
        for existing in state.questions:
            if question_is_duplicate(text, existing.text, prior):
                return existing
        return None

    async def _ask(
        self,
        state: SessionState,
        *,
        subject: str,
        text: str,
        rationale: str = "",
        kind: QuestionKind = "open",
        options: list[str] | None = None,
        purpose: QuestionPurpose = "agenda",
        round: int = 0,
        emit: Emitter = null_emitter,
    ) -> QuestionState | None:
        question = self._queue_question(
            state,
            subject=subject,
            text=text,
            rationale=rationale,
            kind=kind,
            options=options,
            purpose=purpose,
            round=round,
        )
        if question is None:
            return None
        question.emitted = True
        state.awaiting = AwaitKind.ANSWER
        state.status = RunStatus.WAITING_USER
        state.metrics["questions"] = float(len(state.questions))

        await emit(
            EventType.QUESTION,
            {
                "id": question.id,
                "subject": subject,
                "question": question.text,
                "rationale": question.rationale,
                "kind": question.kind,
                "options": question.options,
                "purpose": purpose,
                "round": round,
                "progress": self._coverage(state),
            },
        )
        return question

    async def _drain_queue(self, state: SessionState, emit: Emitter) -> bool:
        """Put the next queued-but-unasked question on screen. True if one was surfaced.

        Review and clarify steps queue several blockers at once and ask them one at a time. Without this
        drain the queued entries stay invisible: they are counted in the question total, block the budget,
        and never reach the stakeholder.
        """
        pending = state.unasked_questions
        if not pending:
            return False
        question = pending[0]
        question.emitted = True
        state.awaiting = AwaitKind.ANSWER
        state.status = RunStatus.WAITING_USER
        state.metrics["questions"] = float(len(state.questions))
        await emit(
            EventType.QUESTION,
            {
                "id": question.id,
                "subject": question.subject,
                "question": question.text,
                "rationale": question.rationale,
                "kind": question.kind,
                "options": question.options,
                "purpose": question.purpose,
                "round": question.round,
                "progress": self._coverage(state),
            },
        )
        return True

    def _queue_question(
        self,
        state: SessionState,
        *,
        subject: str,
        text: str,
        rationale: str = "",
        kind: QuestionKind = "open",
        options: list[str] | None = None,
        purpose: QuestionPurpose = "agenda",
        round: int = 0,
    ) -> QuestionState | None:
        """Record a question, unless the interview budget is already spent.

        Returns None when the ceiling is reached. Callers that queue follow-ups (review blockers, gap
        findings) must not be able to inflate `state.questions` past `max_total_questions`; otherwise the
        budget is advisory only and a critic with many blockers can generate an unbounded queue.
        """
        ceiling = self.config.interview.max_total_questions
        if len(state.questions) >= ceiling:
            return None
        if self._already_asked(state, text) is not None:
            return None  # a rephrased question is the same question
        question = QuestionState(
            id=self._next_question_id(state),
            subject=subject,
            text=" ".join((text or "").split())[:600],
            rationale=" ".join((rationale or "").split())[:300],
            kind=kind if kind in ("open", "closed", "confirm") else "open",
            options=[o for o in (options or []) if o][:6],
            purpose=purpose,
            round=round,
        )
        state.questions.append(question)
        self.services.store.add_qa(
            QAPair(
                session_id=state.session_id,
                q_index=question.id,
                subject=subject,
                question=question.text,
                rationale=question.rationale,
                round=round,
                purpose=purpose,
            )
        )
        return question

    def _next_question_id(self, state: SessionState) -> int:
        state.question_seq += 1
        return state.question_seq

    def _coverage(self, state: SessionState) -> dict:
        total = len(state.subjects) or 1
        closed = sum(1 for s in state.subjects if s.closed)
        return {"subjects": len(state.subjects), "covered": closed, "percent": round(100 * closed / total)}

    def _fix_hint_for(self, state: SessionState, section_id: str) -> str:
        hints = [
            issue.fix_hint or issue.description
            for issue in state.open_issues
            if issue.section_id == section_id and (issue.kind == "internal" or not issue.question_for_user)
        ]
        return " ".join(h.strip() for h in hints if h.strip())[:600]

    def _write_text_artifact(self, state: SessionState, name: str, content: str) -> ArtifactRef | None:
        try:
            target = self.services.config.out_dir(state.session_id) / name
            data = content.encode("utf-8")
            target.write_bytes(data)
        except OSError:
            return None
        from .store import sha256_text

        ref = ArtifactRef(name=name, kind="file", path=str(target), bytes=len(data), sha256=sha256_text(content))
        self.services.store.add_artifact(state.session_id, ref)
        return ref

    def _usage_snapshot(self) -> tuple[int, int]:
        return int(self.services.gateway.usage.get("calls", 0)), int(self.services.gateway.total_model_ms)

    def _record_metrics(self, state: SessionState, calls_before: int, model_ms_before: int, wall_ms: int) -> None:
        calls, model_ms = self._usage_snapshot()
        delta_calls = max(0, calls - calls_before)
        delta_model_ms = max(0, model_ms - model_ms_before)

        state.metrics["calls"] = float(state.metrics.get("calls", 0)) + delta_calls
        state.metrics["model_seconds"] = float(state.metrics.get("model_seconds", 0)) + delta_model_ms / 1000.0
        state.metrics["wall_seconds"] = float(state.metrics.get("wall_seconds", 0)) + wall_ms / 1000.0

        # Per-phase attribution: "the run took 25 minutes" is not actionable, "interview took 19 of
        # them, 40 model calls, and 3 of those calls retried" is.
        phase = state.phase.value
        by_phase = state.metrics.setdefault("by_phase", {})
        entry = by_phase.setdefault(phase, {"calls": 0.0, "model_seconds": 0.0, "wall_seconds": 0.0, "steps": 0})
        entry["calls"] += delta_calls
        entry["model_seconds"] += delta_model_ms / 1000.0
        entry["wall_seconds"] += wall_ms / 1000.0
        entry["steps"] += 1
