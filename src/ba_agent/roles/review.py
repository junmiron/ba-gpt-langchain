"""Review roles: the LLM critic that runs *after* the deterministic linter."""

from __future__ import annotations

from ..events import Emitter, EventType, null_emitter
from ..prompts import render
from ..schemas import CriticVerdict, Issue, SessionState
from ..services import Services, session_model
from ..spec import LintFinding, SPEC_TEMPLATE


def issues_from_lint(findings: list[LintFinding]) -> list[Issue]:
    """Turn code-detected problems into the same shape as critic issues."""
    return [Issue(section_id=f.section_id, kind=f.kind, severity=f.severity,  # type: ignore[arg-type]
                  description=f.message, fix_hint="", question_for_user=f.question_for_user,
                  source="lint") for f in findings]


def verdict_from_lint(findings: list[LintFinding]) -> CriticVerdict:
    issues = issues_from_lint(findings)
    blockers = [i for i in issues if i.severity == "blocker"]
    status = "complete"
    if any(i.kind == "user" for i in blockers):
        status = "needs_user"
    elif blockers:
        status = "auto_fixable"
    return CriticVerdict(status=status, summary="Deterministic checker verdict (LLM critic unavailable).",
                         issues=issues)


async def critique(services: Services, state: SessionState, findings: list[LintFinding], *,
                   emit: Emitter = null_emitter) -> CriticVerdict:
    lint_lines = [finding.line() for finding in findings]
    document = (state.spec_markdown or "")[:14_000]

    system, user = render(
        "critic",
        project_name=(state.plan.project_name if state.plan else state.title),
        lint_report=lint_lines,
        section_ids=", ".join(s.id for s in SPEC_TEMPLATE if (state.sections.get(s.id) or "").strip()) or "(none)",
        spec_markdown=document,
    )
    result = await services.gateway.structured(
        CriticVerdict, system_prompt=system, user_prompt=user, role="critic", emit=emit, step="review:critic",
        model_override=session_model(state, "critic")
    )
    verdict = result.parsed if isinstance(result.parsed, CriticVerdict) else None

    if verdict is None:
        await emit(EventType.WARNING, {
            "step": "review:critic", "role": "critic",
            "message": "LLM critic unavailable — using the deterministic checker verdict only",
            "detail": (result.error or "")[:300],
        })
        return verdict_from_lint(findings)

    # The linter is authoritative for structure: merge its findings in, deduplicated by description.
    merged = list(verdict.issues)
    seen = {i.description.strip().lower() for i in merged}
    for finding in findings:
        if finding.message.strip().lower() not in seen:
            merged.extend(issues_from_lint([finding]))
    verdict.issues = merged[:12]

    # Re-derive status so a critic that says "complete" while blockers exist cannot pass the gate.
    blockers = [i for i in verdict.issues if i.severity == "blocker"]
    derived = "needs_user" if any(i.kind == "user" for i in blockers) else ("auto_fixable" if blockers else "complete")
    if verdict.status != derived:
        verdict.status = derived
    return verdict
