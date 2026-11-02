"""Prompt context builders — the only way roles see session state.

Centralised on purpose: every step gets a *small, purpose-built* slice of state instead of the whole transcript
(v1 replayed everything through a shared ChromaDB memory). That keeps prompts inside a local model's window and
makes each step reproducible.
"""

from __future__ import annotations

import re
from typing import Iterable

from .schemas import LedgerItem, SessionState, SummaryDoc


def coverage_table(state: SessionState) -> str:
    if not state.subjects:
        return "(no subjects planned yet)"
    lines = []
    for subject in state.subjects:
        note = f" — {subject.notes}" if subject.notes else ""
        lines.append(f"- {subject.name}: {subject.status} (questions asked: {subject.asked}){note}")
    return "\n".join(lines)


def requirements_rendered(state: SessionState, *, limit: int = 60) -> list[str]:
    lines = []
    for item in state.requirements()[-limit:]:
        category = item.category.value if item.category else "functional"
        priority = item.priority.value if item.priority else "-"
        acceptance = f" | acceptance: {'; '.join(item.acceptance)}" if item.acceptance else ""
        body = re.sub(r"\s+", " ", item.body or "").strip()
        lines.append(f"- {item.id} [{category}/{priority}] {item.title}: {body}{acceptance}")
    return lines


def _lines_for(state: SessionState, kinds: tuple[str, ...], *, limit: int = 60) -> list[str]:
    out: list[str] = []
    for item in state.ledger:
        if item.kind not in kinds:
            continue
        text = re.sub(r"\s+", " ", (item.body or item.title or "").strip())
        if text and text not in out:
            out.append(text)
    return out[-limit:]


def facts_lines(state: SessionState) -> list[str]:
    return _lines_for(state, ("fact", "decision"))


def risks_lines(state: SessionState) -> list[str]:
    return _lines_for(state, ("risk",))


def assumptions_lines(state: SessionState) -> list[str]:
    return _lines_for(state, ("assumption",))


def open_questions_lines(state: SessionState) -> list[str]:
    lines = _lines_for(state, ("open_question",))
    if state.summary:
        lines.extend(q for q in state.summary.open_questions if q not in lines)
    return lines[:30]


def known_facts_lines(state: SessionState) -> list[str]:
    facts = list(state.plan.known_facts) if state.plan else []
    facts.extend(_lines_for(state, ("fact", "decision"), limit=25))
    unique: list[str] = []
    for fact in facts:
        text = re.sub(r"\s+", " ", (fact or "").strip())
        if text and text.lower() not in {u.lower() for u in unique}:
            unique.append(text)
    return unique[:40]


def ledger_digest(state: SessionState, *, limit: int = 120) -> str:
    """Compact full-ledger view for steps that need everything (critic, diagrammer)."""
    lines = [item.as_line() for item in state.ledger[-limit:]]
    return "\n".join(lines) if lines else "(empty ledger)"


# --------------------------------------------------------------------------- summary rendering


def render_summary_markdown(summary: SummaryDoc | None) -> str:
    """Deterministic markdown from the structured summary — used by UI, approval card and deliverable."""
    if not summary:
        return ""

    def block(title: str, items: Iterable[str]) -> list[str]:
        values = [re.sub(r"\s+", " ", (i or "").strip()) for i in items if (i or "").strip()]
        if not values:
            return []
        return [f"### {title}", "", *[f"- {value}" for value in values], ""]

    parts = [f"# Requirements Summary — {summary.project_name}".strip(), ""]
    if summary.objective:
        parts += [f"> **Objective.** {summary.objective}", ""]
    if summary.background:
        parts += ["### Background", "", summary.background.strip(), ""]
    parts += block("Objectives", summary.objectives)
    parts += block("Stakeholders & Personas", summary.stakeholders)
    parts += block("In Scope", summary.scope_in)
    parts += block("Out of Scope", summary.scope_out)
    parts += block("Key Requirements", summary.key_requirements)
    parts += block("Non-Functional Requirements", summary.non_functional)
    parts += block("Assumptions", summary.assumptions)
    parts += block("Open Questions", summary.open_questions)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(parts)).strip() + "\n"


# --------------------------------------------------------------------------- transcript / excerpts


def transcript_lines(qa_rows: list[dict], *, limit: int = 40) -> list[str]:
    lines = []
    for row in qa_rows[-limit:]:
        subject = row.get("subject") or ""
        question = re.sub(r"\s+", " ", (row.get("question") or "").strip())
        answer = row.get("answer")
        if answer is None:
            continue
        answer_text = "(skipped)" if row.get("skipped") else re.sub(r"\s+", " ", str(answer).strip())
        if row.get("na"):
            answer_text = "(not applicable)"
        lines.append(f"Q{row.get('q_index')} [{subject}] {question} → {answer_text}")
    return lines


def spec_excerpt(state: SessionState, max_chars: int = 6000) -> str:
    """Trimmed document for diagram/critic prompts: always keeps scope + functional requirements."""
    preferred = ["introduction", "scope", "functional_requirements", "non_functional_requirements"]
    chunks: list[str] = []
    used = 0
    for section_id in preferred:
        body = (state.sections.get(section_id) or "").strip()
        if not body:
            continue
        heading = _heading_for(section_id, state)
        piece = f"{heading}\n{body}"
        remaining = max_chars - used
        if remaining <= 200:
            break
        if len(piece) > remaining:
            piece = piece[:remaining].rsplit("\n", 1)[0] + "\n…"
        chunks.append(piece)
        used += len(piece)
    return "\n\n".join(chunks).strip()


def _heading_for(section_id: str, state: SessionState) -> str:
    from .spec import SECTION_BY_ID

    section = SECTION_BY_ID.get(section_id)
    return f"## {section.title}" if section and section.title else ""
