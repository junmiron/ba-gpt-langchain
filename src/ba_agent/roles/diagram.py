"""Diagram roles: generate Mermaid, validate in code, review with the model, revise once."""

from __future__ import annotations

import re

from ..context import requirements_rendered, spec_excerpt
from ..events import Emitter, EventType, null_emitter
from ..mermaid_utils import clean_mermaid, quote_risky_labels, render_svg, validate_mermaid
from ..prompts import render
from ..schemas import DiagramSet, MermaidReviewResult, SessionState
from ..services import Services, session_model


async def generate_diagrams(services: Services, state: SessionState, *, emit: Emitter = null_emitter) -> list[dict]:
    cfg = services.config.diagram
    system, user = render(
        "diagram",
        project_name=(state.plan.project_name if state.plan else state.title),
        spec_excerpt=spec_excerpt(state, max_chars=5200),
        requirements=requirements_rendered(state)[:40],
        max_diagrams=cfg.max_diagrams,
    )
    result = await services.gateway.structured(
        DiagramSet, system_prompt=system, user_prompt=user, role="diagrammer", emit=emit,
        step="diagram:generate", model_override=session_model(state, "diagrammer")
    )
    parsed = result.parsed if isinstance(result.parsed, DiagramSet) else None
    if parsed is None or not parsed.diagrams:
        await emit(EventType.WARNING, {"step": "diagram:generate", "role": "diagrammer",
                                       "message": "no diagram produced", "detail": (result.error or "")[:300]})
        return []

    diagrams: list[dict] = []
    for draft in parsed.diagrams[: cfg.max_diagrams]:
        code = clean_mermaid(draft.code)
        report = validate_mermaid(code)
        if not report.ok:
            # Cheap deterministic repair first — most failures are unquoted labels.
            auto_fixed = quote_risky_labels(report.code)
            retry = validate_mermaid(auto_fixed)
            if retry.ok:
                await emit(EventType.WARNING, {"step": "diagram:validate", "role": "diagrammer",
                                               "message": "auto-fixed Mermaid syntax (quoted labels)"})
                code, report = auto_fixed, retry
            else:
                repaired = await _repair(services, state, auto_fixed, retry.errors + report.errors, emit=emit)
                if repaired:
                    candidate_report = validate_mermaid(repaired)
                    if candidate_report.ok or len(candidate_report.errors) < len(report.errors):
                        code, report = candidate_report.code, candidate_report

        svg, render_error = await render_svg(code, cfg.validator_url)
        diagrams.append({
            "title": (draft.title or "Process diagram").strip(),
            "diagram_type": draft.diagram_type or "flowchart",
            "code": code,
            "notes": draft.notes or "",
            "valid": report.ok,
            "errors": report.errors,
            "warnings": report.warnings,
            "svg": svg or "",
            "render_error": render_error or "",
        })
        await emit(EventType.DIAGRAM, {"step": "diagram:generate", "role": "diagrammer", "title": diagrams[-1]["title"],
                                       "valid": report.ok, "errors": report.errors[:6], "code": code})
    return [d for d in diagrams if d["code"]]


async def _repair(services: Services, state: SessionState, code: str, errors: list[str], *,
                  emit: Emitter = null_emitter) -> str:
    review = await review_diagram(services, state, {"title": "Repair pass", "code": code}, errors, emit=emit,
                                  step="diagram:repair")
    return clean_mermaid(review.get("revised_code", "")) if review else ""


async def review_diagram(services: Services, state: SessionState, diagram: dict, validation_errors: list[str], *,
                         emit: Emitter = null_emitter, step: str = "diagram:review") -> dict:
    system, user = render(
        "mermaid_review",
        title=diagram.get("title", "Diagram"),
        validation_errors=validation_errors or [],
        code=diagram.get("code", ""),
        spec_excerpt=spec_excerpt(state, max_chars=3500),
    )
    result = await services.gateway.structured(
        MermaidReviewResult, system_prompt=system, user_prompt=user, role="mermaid_reviewer", emit=emit, step=step,
        model_override=session_model(state, "mermaid_reviewer")
    )
    parsed = result.parsed if isinstance(result.parsed, MermaidReviewResult) else None
    if parsed is None:
        return {"approved": not validation_errors, "summary": "reviewer unavailable", "issues": list(validation_errors),
                "revised_code": ""}

    revised = clean_mermaid(parsed.revised_code) if parsed.revised_code else ""
    if revised:
        check = validate_mermaid(revised)
        if not check.ok and len(check.errors) >= max(1, len(validation_errors)):
            # The "fix" is no better than the original — keep what we had.
            await emit(EventType.WARNING, {"step": step, "role": "mermaid_reviewer",
                                           "message": "reviewer's revision did not validate — keeping previous code"})
            revised = ""
    return {
        "approved": bool(parsed.approved) and (not validation_errors or bool(revised)),
        "summary": parsed.summary,
        "issues": parsed.issues + list(validation_errors),
        "revised_code": revised,
    }


def primary_diagram(diagrams: list[dict]) -> dict | None:
    """The diagram that ships as ``process_diagram.mmd`` — prefer a valid flowchart."""
    if not diagrams:
        return None
    for diagram in diagrams:
        if diagram.get("valid") and re.search(r"flowchart|graph", str(diagram.get("code", "")), re.I):
            return diagram
    for diagram in diagrams:
        if diagram.get("valid"):
            return diagram
    return diagrams[0]
