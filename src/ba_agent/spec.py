"""The Functional Specification Document: template, deterministic assembly and completeness linter.

Two deliberate choices:

* **Some sections are generated in code** (title page, traceability table, open questions). They are pure
  projections of the ledger — asking a model to render them wastes tokens and invites invented ids.
* **Completeness is checked in code before any LLM critic runs.** v1 asked a model whether the document was
  complete and regex-matched ``READY``/``NEEDS_INFO``; here structure, counts and traceability are verified
  deterministically, and the critic only adds judgement on top.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from .schemas import SessionState

SectionKind = Literal["llm", "use_cases", "glossary", "static"]


@dataclass(frozen=True)
class SectionSpec:
    id: str
    title: str
    guidance: str
    kind: SectionKind = "llm"
    required: bool = True
    target_words: int = 260
    min_chars: int = 180


SPEC_TEMPLATE: tuple[SectionSpec, ...] = (
    SectionSpec(id="title", title="", guidance="", kind="static", required=False),
    SectionSpec(id="introduction", title="1. Introduction",
                guidance="Purpose of the document, project background, business objectives and success measures.",
                target_words=220, min_chars=200),
    SectionSpec(id="scope", title="2. Scope",
                guidance="In scope / out of scope lists, plus explicit boundaries with adjacent systems.",
                target_words=220, min_chars=200),
    SectionSpec(id="functional_requirements", title="3. Functional Requirements",
                guidance="Numbered requirements reusing the ledger ids (FR-001 …). One behaviour per requirement, "
                         "grouped by capability with ### sub-headings.",
                target_words=420, min_chars=400),
    SectionSpec(id="non_functional_requirements", title="4. Non-Functional Requirements",
                guidance="Performance, security, compliance, availability and usability targets as a table with "
                         "measurable values; mark unknown targets TBD.",
                target_words=260, min_chars=220),
    SectionSpec(id="roles_personas", title="5. User Roles & Personas",
                guidance="Who uses the system, their goals, permissions and frequency of use — as a table.",
                target_words=220, min_chars=180),
    SectionSpec(id="assumptions_constraints", title="6. Assumptions & Constraints",
                guidance="Technical, organisational and regulatory constraints; assumptions that must be validated.",
                target_words=180, min_chars=150),
    SectionSpec(id="risks_dependencies", title="7. Risks & Dependencies",
                guidance="Risks with impact/likelihood/mitigation and external dependencies (systems, teams, data).",
                target_words=180, min_chars=140),
    SectionSpec(id="use_cases", title="8. Use Cases & User Stories",
                guidance="Use cases UC-01… plus user stories with acceptance criteria.", kind="use_cases",
                target_words=700, min_chars=500),
    SectionSpec(id="glossary", title="9. Glossary", guidance="Project-specific terms and acronyms.",
                kind="glossary", target_words=260, min_chars=180),
    SectionSpec(id="open_questions", title="10. Open Questions & Gaps", guidance="", kind="static", min_chars=40),
    SectionSpec(id="traceability", title="Appendix A — Requirements Traceability", guidance="", kind="static",
                required=False, min_chars=60),
)

LLM_SECTIONS = [s for s in SPEC_TEMPLATE if s.kind == "llm"]
SECTION_BY_ID = {s.id: s for s in SPEC_TEMPLATE}


def section_title(section_id: str, default: str = "Specification review") -> str:
    section = SECTION_BY_ID.get(section_id)
    return section.title if section and section.title else default


def outline_text(current_id: str | None = None) -> str:
    lines = []
    for section in SPEC_TEMPLATE:
        marker = "  ← you are writing this one" if section.id == current_id else ""
        lines.append(f"- {section.title or '(title page)'}{marker}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- deterministic sections


def render_title(state: SessionState, version: str = "0.1") -> str:
    plan = state.plan
    name = (plan.project_name if plan and plan.project_name else state.title) or "Untitled project"
    objective = (plan.objective if plan else "") or ""
    rows = [
        f"# {name}",
        "",
        "## Functional Specification Document",
        "",
        "| | |",
        "|---|---|",
        f"| **Version** | {version} |",
        f"| **Date** | {date.today().isoformat()} |",
        f"| **Status** | {'Approved' if state.approved_summary else 'Draft — pending stakeholder review'} |",
        "| **Author** | BA agent (LangChain) |",
        "",
    ]
    if objective:
        rows += [f"> **Objective.** {objective}", ""]
    return "\n".join(rows)


def render_open_questions(state: SessionState) -> str:
    items: list[str] = []
    for entry in state.ledger:
        if entry.kind == "open_question":
            items.append(entry.body or entry.title)
    for issue in state.open_issues:
        if issue.kind == "user" and issue.question_for_user:
            items.append(issue.question_for_user)
    if state.summary:
        items.extend(state.summary.open_questions)

    unique: list[str] = []
    for item in items:
        text = re.sub(r"\s+", " ", (item or "").strip())
        if text and text.lower() not in {u.lower() for u in unique}:
            unique.append(text)
    if not unique:
        return "No open questions. All elicited subjects were answered during the interview."
    return "\n".join(f"{index}. {text}" for index, text in enumerate(unique, start=1))


def render_traceability(state: SessionState) -> str:
    requirements = state.requirements()
    if not requirements:
        return "_No requirements captured._"
    rows = ["| ID | Requirement | Category | Priority | Acceptance criteria | Source question |",
            "|---|---|---|---|---|---|"]
    for item in requirements:
        acceptance = "; ".join(item.acceptance) if item.acceptance else "_missing_"
        source = f"Q{item.source_q}" if item.source_q else "—"
        rows.append(
            f"| {item.id} | {_cell(item.title or item.body)} | "
            f"{item.category.value if item.category else '—'} | "
            f"{item.priority.value if item.priority else '—'} | {_cell(acceptance)} | {source} |"
        )
    return "\n".join(rows)


def _cell(text: str, limit: int = 160) -> str:
    clean = re.sub(r"\s+", " ", (text or "").strip()).replace("|", "\\|")
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


# --------------------------------------------------------------------------- assembly


#: Sections whose text is a projection of the ledger — never written by a model.
STATIC_RENDERERS = {
    "open_questions": render_open_questions,
    "traceability": render_traceability,
}


def resolved_section_body(state: SessionState, section_id: str) -> str:
    """The text this section will have in the assembled document.

    Single source of truth for both the assembler and the linter: static sections are projected from the
    ledger here, so a code-rendered section can never be reported as "missing" by the completeness check.
    """
    renderer = STATIC_RENDERERS.get(section_id)
    if renderer is not None:
        return renderer(state).strip()
    return (state.sections.get(section_id) or "").strip()


def assemble_spec(state: SessionState, version: str = "0.1") -> str:
    """Deterministic assembly in template order — the model never controls document structure."""
    parts: list[str] = []
    for section in SPEC_TEMPLATE:
        if section.id == "title":
            parts.append(render_title(state, version=version))
            continue
        body = resolved_section_body(state, section.id)
        if not body:
            if section.required:
                body = "_Section pending — see Open Questions._"
            else:
                continue
        parts.append(f"## {section.title}\n\n{demote_headings(body)}\n")
    document = "\n".join(parts).strip() + "\n"
    return re.sub(r"\n{4,}", "\n\n\n", document)


def demote_headings(markdown: str) -> str:
    """Models keep emitting `#`/`##` inside a section; the assembler owns heading levels."""
    lines = []
    for line in markdown.splitlines():
        if re.match(r"^#\s+", line):
            lines.append("###" + line[1:])
        elif re.match(r"^##\s+", line):
            lines.append("###" + line[2:])
        else:
            lines.append(line)
    return "\n".join(lines).strip()


# --------------------------------------------------------------------------- linter


@dataclass
class LintFinding:
    code: str
    message: str
    severity: Literal["blocker", "warning"] = "warning"
    kind: Literal["internal", "user"] = "internal"
    section_id: str = ""
    #: Stakeholder-facing wording for user-kind findings. Without it, a diagnostic such as
    #  "only 1 functional requirement captured" would be shown to a human as if it were a question.
    question_for_user: str = ""

    def line(self) -> str:
        return f"[{self.severity}/{self.kind}] {self.code}: {self.message}"


NFR_TOPICS = {
    "performance": ("performance", "latency", "throughput", "response time", "speed"),
    "security": ("security", "authentication", "authorisation", "authorization", "encryption", "rbac", "gdpr"),
    "availability": ("availability", "uptime", "sla", "recovery", "backup", "rto", "rpo", "resilience"),
    "scalability": ("scal", "capacity", "concurrent", "volume"),
}

#: Needles that must match as whole words. Plain substring search made "sla" fire on "island" and "slate".
_WHOLE_WORD_NEEDLES = frozenset({"sla", "rto", "rpo", "rbac", "gdpr"})


def _mentions(text: str, needle: str) -> bool:
    if needle in _WHOLE_WORD_NEEDLES:
        return re.search(rf"\b{re.escape(needle)}\b", text) is not None
    return needle in text


def nfr_topics_mentioned(text: str) -> set[str]:
    """Quality-attribute topics named in ``text`` (lower-cased by the caller or here)."""
    lowered = text.lower()
    return {topic for topic, needles in NFR_TOPICS.items() if any(_mentions(lowered, n) for n in needles)}


#: Uncovered-topic count at which both the linter and the gap detector report a missing quality attribute group.
NFR_UNCOVERED_THRESHOLD = 3


def lint_spec(state: SessionState, *, min_functional_requirements: int = 5) -> list[LintFinding]:
    findings: list[LintFinding] = []

    for section in SPEC_TEMPLATE:
        if not section.required:
            continue
        body = resolved_section_body(state, section.id)
        if not body:
            findings.append(LintFinding("missing_section", f"Section '{section.title}' is missing.",
                                        "blocker", "internal", section.id))
        elif len(body) < section.min_chars:
            findings.append(LintFinding("thin_section",
                                        f"Section '{section.title}' has only {len(body)} characters "
                                        f"(expected ≥{section.min_chars}).",
                                        "warning" if len(body) > section.min_chars // 2 else "blocker",
                                        "internal", section.id))

    requirements = state.requirements()
    functional = [r for r in requirements if (r.category and r.category.value == "functional")]

    min_fr = max(1, int(min_functional_requirements))
    if len(functional) < int(min_fr):
        findings.append(LintFinding(
            "too_few_requirements",
            f"Only {len(functional)} functional requirements captured (expected ≥{int(min_fr)}).",
            "blocker", "user",
            question_for_user="Which main capabilities must the system provide? Please list the things users "
                              "actually need to do, one per line — I will turn them into requirements."))

    doc = state.spec_markdown or assemble_spec(state)

    # Traceability is checked against the *narrative* sections only. The appendix lists every id by construction,
    # so including it would make this check vacuous — what matters is whether the prose actually uses them.
    narrative = "\n".join(resolved_section_body(state, section.id) for section in SPEC_TEMPLATE
                          if section.kind != "static")
    missing_ids = [r.id for r in requirements if r.id not in narrative]
    if missing_ids:
        findings.append(LintFinding("untraceable_requirements",
                                    f"Requirements not referenced in the document body: {', '.join(missing_ids[:12])}.",
                                    "warning", "internal"))

    must_without_acceptance = [r.id for r in requirements
                               if (r.priority and r.priority.value == "must") and not r.acceptance]
    if must_without_acceptance:
        findings.append(LintFinding("missing_acceptance_criteria",
                                    f"'must' requirements without acceptance criteria: "
                                    f"{', '.join(must_without_acceptance[:12])}.",
                                    "warning", "internal"))

    covered_topics = nfr_topics_mentioned(doc)
    uncovered = sorted(set(NFR_TOPICS) - covered_topics)
    if len(uncovered) >= NFR_UNCOVERED_THRESHOLD:
        findings.append(LintFinding("nfr_coverage",
                                    f"Non-functional targets not addressed for: {', '.join(uncovered)}.",
                                    "warning", "user"))

    use_cases = state.sections.get("use_cases", "")
    if "UC-" not in use_cases and len(functional) > 0:
        findings.append(LintFinding("no_use_cases", "Use Cases section has no UC- identifiers.",
                                    "blocker", "internal", "use_cases"))
    glossary = state.sections.get("glossary", "")
    if len([line for line in glossary.splitlines() if line.strip().startswith("- **")]) < 5:
        findings.append(LintFinding("thin_glossary", "Glossary defines fewer than 5 terms.",
                                    "warning", "internal", "glossary"))

    tbd_count = len(re.findall(r"\bTBD\b", doc, flags=re.I))
    if tbd_count > 8:
        findings.append(LintFinding("too_many_tbd", f"{tbd_count} 'TBD' placeholders remain in the document.",
                                    "warning", "user"))

    return findings


def deterministic_gaps(state: SessionState, *, min_functional_requirements: int = 5) -> list[str]:
    """Findings that need stakeholder input — feeds the clarify phase (no model needed to detect them).

    ``min_functional_requirements`` must be the same ``config.review.min_functional_requirements`` that
    :func:`lint_spec` uses, otherwise the interview and the linter disagree about how many requirements is enough.
    """
    gaps: list[str] = []
    requirements = state.requirements()
    functional = [r for r in requirements if r.category and r.category.value == "functional"]

    floor = max(1, int(min_functional_requirements))
    if len(functional) < floor:
        gaps.append(f"Only {len(functional)} functional requirement(s) captured (expected at least {floor}) — "
                    f"we need the main capabilities the system must provide.")
    if not any(r.category and r.category.value == "non_functional" for r in requirements):
        gaps.append("No non-functional targets (performance, security, availability) were captured.")
    if not state.summary or not state.summary.scope_in:
        gaps.append("The boundary of the project (what is in and out of scope) is still undefined.")
    if not any(i.kind == "decision" for i in state.ledger):
        gaps.append("No explicit design decisions recorded yet — which options were ruled out matters to delivery.")

    text = " ".join((i.body + " " + i.title) for i in state.ledger)
    missing_topics = sorted(set(NFR_TOPICS) - nfr_topics_mentioned(text))
    if len(missing_topics) >= NFR_UNCOVERED_THRESHOLD:
        gaps.append("These quality attributes are untouched: " + ", ".join(missing_topics) + ".")

    return gaps


def spec_stats(state: SessionState) -> dict[str, Any]:
    doc = state.spec_markdown or assemble_spec(state)
    requirements = state.requirements()
    return {
        "words": len(re.findall(r"\S+", doc)),
        "sections_present": [s.id for s in SPEC_TEMPLATE if resolved_section_body(state, s.id)],
        "requirements_total": len(requirements),
        "functional": sum(1 for r in requirements if r.category and r.category.value == "functional"),
        "non_functional": sum(1 for r in requirements if r.category and r.category.value == "non_functional"),
        "with_acceptance": sum(1 for r in requirements if r.acceptance),
        "tbd_placeholders": len(re.findall(r"\bTBD\b", doc, flags=re.I)),
    }
