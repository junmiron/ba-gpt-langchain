"""Tools for the interactive chat channel (see :mod:`ba_agent.agent_loop`).

Each tool is deliberately read-mostly; the two mutating tools go through ``ChatContext.mutate`` so they take the
per-session lock and cannot race with a workflow step that is running at the same time.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from ..retriever import document_citation
from ..schemas import LedgerItem, LedgerKind, SessionState
from .sandbox import Sandbox


@dataclass
class ChatContext:
    session_id: str
    get_state: Callable[[], SessionState]
    #: ``await mutate(fn)`` runs ``fn(state) -> str`` under the session lock and persists the result.
    mutate: Callable[[Callable[[SessionState], str]], Awaitable[str]]
    knowledge: Any
    sandbox: Sandbox
    #: ``langchain-core`` Runnable over the same index (Phase 1.1); falls back to ``knowledge`` if unset.
    retriever: Any = None
    store: Any = None


def build_chat_tools(ctx: ChatContext) -> list[Any]:
    @tool
    async def get_session_state() -> str:
        """Return the current workflow phase, subject coverage and counters for this session."""
        state = ctx.get_state()
        lines = [
            f"Project: {state.plan.project_name if state.plan else state.title}",
            f"Phase: {state.phase.value} | status: {state.status.value} | awaiting: {state.awaiting.value}",
            f"Questions asked: {len(state.questions)} | requirements: {len(state.requirements())}"
            f" | ledger entries: {len(state.ledger)}",
            f"Review rounds used: {state.review_rounds} | gap rounds: {state.gap_round}",
            "Subjects:",
        ]
        for subject in state.subjects:
            lines.append(f"  - {subject.name}: {subject.status} (asked {subject.asked})")
        pending = state.current_question
        if pending is not None:
            lines.append(f"Pending question #{pending.id} [{pending.subject}]: {pending.text}")
        return "\n".join(lines)

    @tool
    async def list_requirements(category: str = "") -> str:
        """List captured requirements. Optional category filter: functional, non_functional, integration, data, reporting."""
        state = ctx.get_state()
        wanted = (category or "").strip().lower()
        items = [i for i in state.requirements() if not wanted or (i.category and i.category.value == wanted)]
        if not items:
            return "No requirements captured yet" + (f" for category '{wanted}'." if wanted else ".")
        lines = []
        for item in items:
            acceptance = f"; acceptance: {' | '.join(item.acceptance)}" if item.acceptance else "; acceptance: MISSING"
            lines.append(
                f"- {item.id} [{item.category.value if item.category else 'n/a'}/{item.priority.value if item.priority else '-'}] "
                f"{item.title}: {item.body}{acceptance}"
            )
        return "\n".join(lines)

    @tool
    async def search_knowledge(query: str, max_results: int = 4) -> str:
        """Search the indexed knowledge base (uploaded documents and reference websites)."""
        limit = max(1, min(int(max_results or 4), 8))
        retriever = ctx.retriever or ctx.knowledge
        documents = await retriever.ainvoke(query)
        documents = list(documents)[:limit]
        if not documents:
            return "No relevant material in the knowledge base."
        return "\n\n---\n\n".join(document_citation(document) for document in documents)

    @tool
    async def read_artifact(name: str) -> str:
        """Read a generated artefact by file name, e.g. 'functional_specification.md' or 'process_diagram.mmd'."""
        # `name` comes from a model, so it must be a bare file name. `Path(...).name` strips any directory
        # component: without it, "../other_session/spec.md" resolves inside the allowed outputs root and
        # leaks another session's artefacts — the sandbox only checks roots, not which session owns the file.
        safe_name = Path(str(name or "")).name
        if not safe_name or safe_name in {".", ".."}:
            available = ", ".join(a["name"] for a in _artifacts(ctx)) or "none yet"
            return f"Invalid artefact name '{name}'. Available artefacts: {available}"
        try:
            return ctx.sandbox.read_text(f"{ctx.session_id}/{safe_name}", max_chars=12_000)
        except Exception as exc:  # noqa: BLE001 — report, don't crash the loop
            available = ", ".join(a["name"] for a in _artifacts(ctx)) or "none yet"
            return f"Could not read '{safe_name}' ({exc}). Available artefacts: {available}"

    @tool
    async def add_ledger_note(kind: str, title: str, body: str) -> str:
        """Record something the stakeholder just decided. kind: fact | requirement | risk | assumption | decision."""
        allowed: dict[str, LedgerKind] = {
            "fact": "fact",
            "requirement": "requirement",
            "risk": "risk",
            "assumption": "assumption",
            "decision": "decision",
            "open_question": "open_question",
        }
        normalised = (kind or "fact").strip().lower()
        if normalised not in allowed:
            return f"Invalid kind '{kind}'. Use one of: {', '.join(sorted(allowed))}."

        def apply(state: SessionState) -> str:
            item_id = (
                state.new_requirement_id()
                if normalised == "requirement"
                else f"{normalised.upper()}-{len(state.ledger) + 1}"
            )
            state.ledger.append(
                LedgerItem(id=item_id, kind=allowed[normalised], title=title.strip(), body=body.strip(), subject="chat")
            )
            return f"Recorded {item_id} ({normalised}): {title.strip() or body.strip()[:60]}"

        return await ctx.mutate(apply)

    @tool
    async def skip_subject(name: str) -> str:
        """Mark an interview subject as not applicable when the stakeholder says it does not apply to them."""

        def apply(state: SessionState) -> str:
            needle = (name or "").strip().lower()
            for subject in state.subjects:
                if needle and (needle in subject.name.lower() or subject.name.lower() in needle):
                    if subject.closed:
                        return f"'{subject.name}' is already {subject.status}."
                    subject.status = "skipped"
                    subject.notes = "marked not applicable via chat"
                    return f"Marked '{subject.name}' as skipped."
            known = ", ".join(s.name for s in state.subjects) or "none planned yet"
            return f"No subject matching '{name}'. Planned subjects: {known}"

        return await ctx.mutate(apply)

    return [get_session_state, list_requirements, search_knowledge, read_artifact, add_ledger_note, skip_subject]


def _artifacts(ctx: ChatContext) -> list[dict[str, Any]]:
    return ctx.store.list_artifacts(ctx.session_id) if ctx.store else []
