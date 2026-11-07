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

import time
from typing import Awaitable, Callable

from .context import render_summary_markdown
from .events import Emitter, EventType, null_emitter
from .export import write_deliverables
from .mermaid_utils import validate_mermaid
from .models import NO_MODEL_MESSAGE
from .roles import analysis, diagram as diagram_roles, interview, review as review_role, writing
from .schemas import (ApprovalResult, ArtifactRef, AwaitKind, Issue, LedgerItem, Phase, QAPair, QuestionState,
                      RunStatus, SessionState, SubjectState, utcnow)
from .services import Services
from .spec import (LintFinding, SECTION_BY_ID, assemble_spec, deterministic_gaps, lint_spec, section_title,
                   spec_stats)

#: Defensive ceiling on engine iterations inside one ``advance`` call (protects against logic bugs).
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
        if state.phase is not Phase.DONE and state.awaiting not in (AwaitKind.ANSWER, AwaitKind.APPROVAL):
            if not await self.services.gateway.resolve_model(None):
                state.error = NO_MODEL_MESSAGE
                state.status = RunStatus.ERROR
                self.services.store.save_state(state)
                await emit(EventType.ERROR, {"phase": state.phase.value, "message": NO_MODEL_MESSAGE,
                                             "resumable": True})
                return state

        for _ in range(MAX_STEPS_PER_ADVANCE):
            if state.phase is Phase.DONE:
                state.status = RunStatus.DONE
                state.awaiting = AwaitKind.FINISHED
                break
            if state.awaiting in (AwaitKind.ANSWER, AwaitKind.APPROVAL):
                state.status = RunStatus.WAITING_USER
                break

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

        if state.awaiting not in (AwaitKind.ANSWER, AwaitKind.APPROVAL) and state.phase is not Phase.DONE:
            state.status = RunStatus.RUNNING
        self.services.store.save_state(state)
        return state

    async def submit_answer(self, state: SessionState, question_id: int, *, answer: str | None = None,
                            skipped: bool = False, not_applicable: bool = False,
                            emit: Emitter = null_emitter) -> SessionState:
        """Record the stakeholder's reply to a pending question and continue the run."""
        question = next((q for q in state.questions if q.id == question_id), None)
        if question is None:
            raise ValueError(f"Unknown question id {question_id}")
        if question.resolved:
            raise ValueError(f"Question {question_id} was already answered")

        text = (answer or "").strip()
        if not skipped and not not_applicable and not text:
            raise ValueError("Provide an answer, or skip the question.")

        question.answer = None if (skipped or not_applicable) else text
        question.skipped = skipped
        question.not_applicable = not_applicable
        question.answered_at = utcnow()
        self.services.store.update_qa_answer(state.session_id, question.id, question.answer, skipped, not_applicable)

        await emit(EventType.ANSWER, {
            "question_id": question.id, "subject": question.subject, "purpose": question.purpose,
            "answer": question.answer or ("(skipped)" if skipped else "(not applicable)"),
        })

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
                state.ledger.append(LedgerItem(id=f"FACT-{len(state.ledger) + 1}", kind="fact", body=usable,
                                               subject=question.subject))
            state.phase = Phase.DRAFT

        return await self.advance(state, emit=emit)

    async def set_approval(self, state: SessionState, approved: bool, notes: str = "", *,
                           emit: Emitter = null_emitter) -> SessionState:
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
                    state.ledger.append(LedgerItem(id=f"DECISION-{len(state.ledger) + 1}", kind="decision",
                                                   body=line, subject="summary review"))
            state.approved_summary = False
            state.awaiting = AwaitKind.NONE
            state.summary_rejections += 1
            limit = max(1, self.config.interview.max_summary_rejections)
            if state.summary_rejections >= limit:
                # Bounded like everything else. The notes are already in the ledger, so drafting continues with
                # them instead of ping-ponging between summary and approval forever (v1's stall class of bug).
                await emit(EventType.WARNING, {
                    "step": "approval",
                    "message": f"{state.summary_rejections} change request(s) noted — drafting now. Your notes are "
                               "in the ledger and are reflected in the document; you can still ask for changes "
                               "after the draft.",
                })
                state.phase = Phase.DRAFT
            else:
                # Re-open clarification (bounded by max_gap_rounds) so the change is actually elicited.
                state.phase = (Phase.CLARIFY if state.gap_round < self.config.interview.max_gap_rounds
                               else Phase.SUMMARY)

        return await self.advance(state, emit=emit)

    # ------------------------------------------------------------------ steps

    async def _step_approval_wait(self, state: SessionState, emit: Emitter) -> None:
        state.awaiting = AwaitKind.APPROVAL
        state.status = RunStatus.WAITING_USER

    async def _step_plan(self, state: SessionState, emit: Emitter) -> None:
        if not (state.brief or "").strip():
            raise WorkflowHalt("No project brief was provided — cannot plan an interview.")

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
                state.ledger.append(LedgerItem(id=f"FACT-{len(state.ledger) + 1}", kind="fact", body=clean,
                                               subject="brief"))
        for assumption in plan.assumptions:
            clean = " ".join(assumption.split())
            if clean and not state.has_similar(clean):
                state.ledger.append(LedgerItem(id=f"ASSUMPTION-{len(state.ledger) + 1}", kind="assumption",
                                               body=clean, subject="brief"))

        await emit(EventType.SESSION, {
            "event": "plan_ready", "project_name": plan.project_name, "objective": plan.objective,
            "summary": plan.summary, "subjects": [s.model_dump() for s in state.subjects],
        })
        state.phase = Phase.INTERVIEW

    async def _step_interview(self, state: SessionState, emit: Emitter) -> None:
        cfg = self.config.interview

        if len(state.questions) >= cfg.max_total_questions:
            await emit(EventType.WARNING, {"step": "interview",
                                           "message": f"question budget ({cfg.max_total_questions}) reached — closing the interview"})
            for subject in state.subjects:
                if not subject.closed:
                    subject.status = "skipped"
                    subject.notes = subject.notes or "budget exhausted"
            state.phase = Phase.CLARIFY
            return

        subject = state.next_subject()
        if subject is None:
            state.phase = Phase.CLARIFY
            return

        draft = await interview.next_question(self.services, state, subject.name, emit=emit)
        if draft is None:
            subject.status = "done"
            subject.notes = subject.notes or "no question could be generated"
            return

        subject.status = "asking"
        await self._ask(state, subject=subject.name, text=draft.question, rationale=draft.rationale,
                        kind=draft.kind, options=draft.options, purpose="agenda", emit=emit)

    async def _after_agenda_answer(self, state: SessionState, question: QuestionState, subject: SubjectState | None,
                                   answer: str, emit: Emitter) -> None:
        if not answer:
            # Skipped / N/A closes the subject instead of nagging (v1 looped on exactly these).
            if subject is not None and subject.status != "done":
                subject.status = "skipped"
                subject.notes = "skipped by stakeholder" if question.skipped else "not applicable"
            state.phase = Phase.INTERVIEW
            return

        decision = await interview.probe_coverage(self.services, state, question.subject, question.text, answer,
                                                  emit=emit)
        follow_up = decision.follow_up
        budget_left = subject is not None and subject.asked < self.config.interview.max_questions_per_subject
        if follow_up is not None and budget_left:
            await self._ask(state, subject=question.subject, text=follow_up.question, rationale=follow_up.rationale,
                            kind="open", options=[], purpose="agenda", emit=emit)
            return

        if subject is not None and subject.status != "skipped":
            subject.status = "done"
            if decision.reason:
                subject.notes = decision.reason[:160]
        state.phase = Phase.INTERVIEW

    async def _step_clarify(self, state: SessionState, emit: Emitter) -> None:
        cfg = self.config.interview

        if state.current_question is not None:  # a clarification question is already on screen
            state.awaiting = AwaitKind.ANSWER
            return

        budget_left = state.gap_round < cfg.max_gap_rounds and len(state.questions) < cfg.max_total_questions
        if not budget_left:
            if deterministic_gaps(state, min_functional_requirements=self.config.review.min_functional_requirements):
                await emit(EventType.WARNING, {"step": "clarify",
                                               "message": f"gap-round budget ({cfg.max_gap_rounds}) exhausted — "
                                                          "remaining gaps are listed as open questions"})
            state.phase = Phase.SUMMARY
            return

        code_gaps = deterministic_gaps(state, min_functional_requirements=self.config.review.min_functional_requirements)
        gaps = await analysis.gap_analysis(self.services, state, code_gaps, emit=emit)
        asked = {q.text.strip().lower() for q in state.questions}
        fresh = [gap for gap in gaps if gap.question.strip().lower() not in asked]

        state.gap_round += 1
        if not fresh:
            state.phase = Phase.SUMMARY
            return

        await emit(EventType.REVIEW, {"step": "clarify", "round": state.gap_round,
                                      "summary": f"{len(fresh)} clarification question(s) needed",
                                      "gaps": [gap.model_dump() for gap in fresh]})
        first = fresh[0]
        await self._ask(state, subject=first.subject or "Clarification", text=first.question, rationale=first.why,
                        kind="open", options=[], purpose="gap", round=state.gap_round, emit=emit)
        for extra in fresh[1:]:  # queued; asked one at a time after each answer
            self._queue_question(state, subject=extra.subject or "Clarification", text=extra.question,
                                 rationale=extra.why, purpose="gap", round=state.gap_round)

    async def _step_summary(self, state: SessionState, emit: Emitter) -> None:
        summary = await analysis.summarise(self.services, state, emit=emit)
        state.summary = summary
        await emit(EventType.SPEC, {"kind": "summary", "markdown": render_summary_markdown(summary)})
        await emit(EventType.APPROVAL, {"requested": True, "project_name": summary.project_name,
                                        "objective": summary.objective})
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
                await emit(EventType.SPEC, {"kind": "section", "section_id": section.id, "title": section.title,
                                            "markdown": markdown, "stats": spec_stats(state)})

        use_case_hint = self._fix_hint_for(state, "use_cases")
        if not (state.sections.get("use_cases") or "").strip() or use_case_hint:
            markdown = await writing.write_use_cases(self.services, state, emit=emit)
            if markdown:
                state.sections["use_cases"] = markdown
                await emit(EventType.SPEC, {"kind": "section", "section_id": "use_cases",
                                            "title": SECTION_BY_ID["use_cases"].title, "markdown": markdown})

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
            await emit(EventType.ARTIFACT, {"name": artifact.name, "bytes": artifact.bytes,
                                            "url": f"/api/artifacts/{state.session_id}/{artifact.name}"})

        state.phase = Phase.REVIEW

    async def _step_review(self, state: SessionState, emit: Emitter) -> None:
        cfg = self.config.review
        findings: list[LintFinding] = lint_spec(state, min_functional_requirements=cfg.min_functional_requirements)
        verdict = await review_role.critique(self.services, state, findings, emit=emit)

        state.lint_report = [finding.line() for finding in findings]
        state.open_issues = verdict.issues
        await emit(EventType.REVIEW, {
            "round": state.review_rounds + 1, "status": verdict.status, "summary": verdict.summary,
            "issues": [issue.model_dump() for issue in verdict.issues], "lint": state.lint_report,
            "stats": spec_stats(state),
        })

        if verdict.status == "complete":
            state.phase = Phase.DIAGRAM
            return

        blockers = [issue for issue in verdict.issues if issue.severity == "blocker"]
        budget_left = state.review_rounds < cfg.max_review_rounds
        asked_texts = {q.text.strip().lower() for q in state.questions}

        # 1) Stakeholder-dependent blockers → ask, then re-draft with the answers. A machine diagnostic is only
        #    put in front of a human when it has stakeholder wording: asking "only 1 functional requirement was
        #    captured" gets an answer that cannot fix it, and burns a review round doing so.
        def askable_text(issue: Issue) -> str:
            question = (issue.question_for_user or "").strip()
            if question:
                return question
            return (issue.description or "").strip() if issue.source == "critic" else ""

        user_issues = [(issue, askable_text(issue)) for issue in blockers if issue.kind == "user"]
        fresh_user = [(issue, text) for issue, text in user_issues if text and text.lower() not in asked_texts]
        if fresh_user and budget_left:
            state.review_rounds += 1
            first, first_text = fresh_user[0]
            await self._ask(state, subject=section_title(first.section_id), text=first_text,
                            rationale=first.description, kind="open", options=[], purpose="review",
                            round=state.review_rounds, emit=emit)
            for extra, extra_text in fresh_user[1:]:
                self._queue_question(state, subject=section_title(extra.section_id),
                                     text=extra_text, rationale=extra.description,
                                     purpose="review", round=state.review_rounds)
            return

        # 2) Internally fixable blockers → regenerate only the offending sections (never a code-rendered one).
        internal = [i for i in verdict.issues if i.kind == "internal" and SECTION_BY_ID.get(i.section_id)
                    and SECTION_BY_ID[i.section_id].kind != "static"]
        if internal and budget_left:
            state.review_rounds += 1
            touched = sorted({i.section_id for i in internal})
            for section_id in touched:
                state.sections.pop(section_id, None)
            await emit(EventType.REVIEW, {"round": state.review_rounds, "status": "auto_fix", "sections": touched,
                                          "summary": f"regenerating {len(touched)} section(s) from critic feedback"})
            state.phase = Phase.DRAFT
            return

        # 3) Out of budget (or nothing actionable) → ship, and report what is still open.
        if not budget_left:
            await emit(EventType.WARNING, {"step": "review",
                                           "message": f"review round budget ({cfg.max_review_rounds}) reached — "
                                                      f"{len(blockers)} unresolved blocker(s) are listed in the deliverable"})
        state.phase = Phase.DIAGRAM

    async def _step_diagram(self, state: SessionState, emit: Emitter) -> None:
        cfg = self.config.diagram
        diagrams = await diagram_roles.generate_diagrams(self.services, state, emit=emit)
        if not diagrams:
            await emit(EventType.WARNING, {"step": "diagram",
                                           "message": "no diagram could be produced — continuing without one"})
            state.phase = Phase.DELIVER
            return

        for item in diagrams:
            for _round in range(cfg.max_revision_rounds + 1):
                verdict = await diagram_roles.review_diagram(self.services, state, item, list(item.get("errors") or []),
                                                             emit=emit)
                revised = (verdict.get("revised_code") or "").strip()
                if revised and revised != item["code"]:
                    report = validate_mermaid(revised)
                    item["code"], item["valid"] = report.code, report.ok
                    item["errors"], item["warnings"] = report.errors, report.warnings
                    await emit(EventType.DIAGRAM, {"step": "diagram:revise", "title": item["title"],
                                                   "valid": report.ok, "code": report.code})
                if verdict.get("approved") or not revised:
                    break

        state.diagrams = diagrams
        primary = diagram_roles.primary_diagram(diagrams)
        state.mermaid_code = primary["code"] if primary else ""

        if state.mermaid_code:
            artifact = self._write_text_artifact(
                state, "process_diagram.mmd", f"%% {state.title} — generated by BA agent (LangChain)\n{state.mermaid_code}\n")
            if artifact:
                await emit(EventType.ARTIFACT, {"name": artifact.name, "bytes": artifact.bytes,
                                                "url": f"/api/artifacts/{state.session_id}/{artifact.name}"})

        state.phase = Phase.DELIVER

    async def _step_deliver(self, state: SessionState, emit: Emitter) -> None:
        written = write_deliverables(self.services, state)
        artifacts: list[ArtifactRef] = []
        for item in written:
            ref = item.to_ref()
            artifacts.append(ref)
            self.services.store.add_artifact(state.session_id, ref)
            await emit(EventType.ARTIFACT, {"name": item.name, "bytes": item.bytes,
                                            "url": f"/api/artifacts/{state.session_id}/{item.name}"})

        state.artifacts = artifacts
        state.phase = Phase.DONE
        state.status = RunStatus.DONE
        state.awaiting = AwaitKind.FINISHED
        await emit(EventType.SESSION, {"event": "completed", "artifacts": [a.model_dump() for a in artifacts],
                                       "stats": spec_stats(state)})

    # ------------------------------------------------------------------ helpers

    async def _ask(self, state: SessionState, *, subject: str, text: str, rationale: str = "", kind: str = "open",
                   options: list[str] | None = None, purpose: str = "agenda", round: int = 0,
                   emit: Emitter = null_emitter) -> QuestionState:
        question = self._queue_question(state, subject=subject, text=text, rationale=rationale, kind=kind,
                                        options=options, purpose=purpose, round=round)
        state.awaiting = AwaitKind.ANSWER
        state.status = RunStatus.WAITING_USER
        state.metrics["questions"] = float(len(state.questions))

        await emit(EventType.QUESTION, {
            "id": question.id, "subject": subject, "question": question.text, "rationale": question.rationale,
            "kind": question.kind, "options": question.options, "purpose": purpose, "round": round,
            "progress": self._coverage(state),
        })
        return question

    def _queue_question(self, state: SessionState, *, subject: str, text: str, rationale: str = "",
                        kind: str = "open", options: list[str] | None = None, purpose: str = "agenda",
                        round: int = 0) -> QuestionState:
        question = QuestionState(
            id=self._next_question_id(state), subject=subject, text=" ".join((text or "").split())[:600],
            rationale=" ".join((rationale or "").split())[:300],
            kind=kind if kind in ("open", "closed", "confirm") else "open",
            options=[o for o in (options or []) if o][:6], purpose=purpose, round=round,  # type: ignore[arg-type]
        )
        state.questions.append(question)
        self.services.store.add_qa(QAPair(session_id=state.session_id, q_index=question.id, subject=subject,
                                          question=question.text, rationale=question.rationale, round=round,
                                          purpose=purpose))
        return question

    def _next_question_id(self, state: SessionState) -> int:
        state.question_seq += 1
        return state.question_seq

    def _coverage(self, state: SessionState) -> dict:
        total = len(state.subjects) or 1
        closed = sum(1 for s in state.subjects if s.closed)
        return {"subjects": len(state.subjects), "covered": closed, "percent": round(100 * closed / total)}

    def _fix_hint_for(self, state: SessionState, section_id: str) -> str:
        hints = [issue.fix_hint or issue.description for issue in state.open_issues
                 if issue.section_id == section_id and (issue.kind == "internal" or not issue.question_for_user)]
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
        state.metrics["calls"] = float(state.metrics.get("calls", 0)) + max(0, calls - calls_before)
        state.metrics["model_seconds"] = float(state.metrics.get("model_seconds", 0)) + max(0, model_ms - model_ms_before) / 1000.0
        state.metrics["wall_seconds"] = float(state.metrics.get("wall_seconds", 0)) + wall_ms / 1000.0
