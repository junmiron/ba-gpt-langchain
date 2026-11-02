"""Writing roles: one section at a time, plus the use-case and glossary specialists.

v1 asked for the entire FSD in a single generation — which is exactly what small local models are bad at (they
truncate, skip sections and invent ids). Here each call writes ~250 words with the ledger as its only source of
truth, and the assembler (:mod:`ba_agent.spec`) owns document structure.
"""

from __future__ import annotations

import re

from ..context import (assumptions_lines, facts_lines, ledger_digest, open_questions_lines,
                       render_summary_markdown, requirements_rendered, risks_lines)
from ..events import Emitter, EventType, null_emitter
from ..prompts import render
from ..schemas import SessionState, clean_markdown
from ..services import Services, session_model
from ..spec import SectionSpec, outline_text


def _summary_block(state: SessionState) -> str:
    rendered = render_summary_markdown(state.summary).strip()
    if rendered:
        return rendered[:4000]
    return "(no approved summary — work directly from the ledger)"


async def write_section(services: Services, state: SessionState, section: SectionSpec, *,
                        extra_instructions: str = "", emit: Emitter = null_emitter) -> str:
    system, user = render(
        "section",
        section={"id": section.id, "title": section.title, "guidance": section.guidance},
        project_name=(state.plan.project_name if state.plan else state.title),
        objective=(state.plan.objective if state.plan else ""),
        summary_block=_summary_block(state),
        requirements=requirements_rendered(state),
        facts=facts_lines(state) + assumptions_lines(state),
        risks=risks_lines(state),
        open_questions=open_questions_lines(state),
        outline=outline_text(section.id),
        knowledge_context=await services.knowledge.context_block(
            f"{section.title} {state.brief[:200]}", max_chars=1600),
        target_words=section.target_words,
        extra_instructions=extra_instructions,
    )
    result = await services.gateway.stream_text(
        system_prompt=system, user_prompt=user, role="writer", emit=emit, step=f"write:{section.id}",
        model_override=session_model(state, "writer")
    )
    markdown = clean_markdown(result.text)

    if not markdown or len(markdown) < 60:
        await emit(EventType.WARNING, {
            "step": f"write:{section.id}", "role": "writer",
            "message": f"'{section.title}' came back empty — writing a placeholder instead of failing the run",
            "detail": (result.error or "")[:300],
        })
        return ""

    # A model that ignores "no heading" would break the assembler's numbering.
    markdown = re.sub(rf"^\s*#{{1,4}}\s*{re.escape(section.title)}\s*\n+", "", markdown, flags=re.I)
    return markdown.strip()


async def write_use_cases(services: Services, state: SessionState, *, emit: Emitter = null_emitter) -> str:
    stakeholders = []
    if state.summary and state.summary.stakeholders:
        stakeholders = list(state.summary.stakeholders)
    else:
        stakeholders = [line for line in facts_lines(state)
                        if re.search(r"\b(role|user|persona|team|admin|analyst)\b", line.lower())][:8]

    process_context = "\n".join(filter(None, [
        (state.summary.scope_in and "In scope: " + "; ".join(state.summary.scope_in[:8])) or "",
        (state.summary.scope_out and "Out of scope: " + "; ".join(state.summary.scope_out[:8])) or "",
    ])) or "(scope not explicitly captured — derive it from the requirements)"

    system, user = render(
        "use_cases",
        project_name=(state.plan.project_name if state.plan else state.title),
        objective=(state.plan.objective if state.plan else ""),
        stakeholders=stakeholders,
        requirements=requirements_rendered(state),
        process_context=process_context,
        max_use_cases=min(8, max(3, len(state.requirements()))),
    )
    result = await services.gateway.stream_text(
        system_prompt=system, user_prompt=user, role="use_cases", emit=emit, step="write:use_cases",
        model_override=session_model(state, "use_cases")
    )
    markdown = clean_markdown(result.text)
    if "UC-" not in markdown and markdown:
        markdown = "### Use Cases\n\n" + markdown
    return markdown


async def write_glossary(services: Services, state: SessionState, corpus: str, *,
                         emit: Emitter = null_emitter) -> str:
    system, user = render("glossary", project_name=(state.plan.project_name if state.plan else state.title),
                          corpus=corpus[:7000])
    result = await services.gateway.stream_text(
        system_prompt=system, user_prompt=user, role="glossary", emit=emit, step="write:glossary",
        model_override=session_model(state, "glossary")
    )
    markdown = clean_markdown(result.text)
    if not markdown:
        return ""
    if not re.search(r"^-\s+\*\*", markdown, flags=re.M):
        # Model produced a table or plain list — normalise to the "- **Term** — definition" shape.
        lines = []
        for line in markdown.splitlines():
            stripped = line.strip().lstrip("-*•").strip()
            if not stripped or stripped.startswith("#") or "|" in stripped[:3]:
                continue
            parts = re.split(r"\s+[—:-]\s+", stripped, maxsplit=1)
            if len(parts) == 2 and len(parts[0]) <= 60:
                lines.append(f"- **{parts[0].strip()}** — {parts[1].strip()}")
        markdown = "\n".join(lines) or markdown
    return markdown


def glossary_corpus(state: SessionState) -> str:
    """Everything the glossary may define terms from, cheapest-first."""
    parts = [state.spec_markdown[:4000], ledger_digest(state, limit=60)]
    if state.summary:
        parts.append(render_summary_markdown(state.summary)[:1500])
    return "\n\n".join(p for p in parts if p and p.strip())
