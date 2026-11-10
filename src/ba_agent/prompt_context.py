"""Typed prompt-context contracts (``langchain-core``-style: every step has a declared shape).

Every Jinja template under ``prompts/`` gets a matching pydantic model here. Two things follow from that:

* ``render(name, **context)`` validates the context before any model is called, so a typo in a role
  (``knowledge_context`` vs ``knowledge``) is caught in milliseconds offline instead of surfacing as an
  ``UndefinedError`` inside a live interview;
* ``undeclared_variables()`` compares what a template actually reads against what its contract declares,
  which catches a template that grew a new reference while the caller never learned about it.

The contract is intentionally permissive about types (lists/strs are normalised by the callers), strict
about *names*.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment
from jinja2.nodes import Name, Node
from jinja2.parser import Parser
from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Contract(BaseModel):
    """Names only: extra keys are a bug, missing ones are a crash later."""

    model_config = ConfigDict(extra="forbid", from_attributes=True)


class SubjectRef(BaseModel):
    """The interview subject a question is about (accepts ``SubjectState``/``SubjectSpec`` objects)."""

    model_config = ConfigDict(from_attributes=True)

    name: str
    why: str = ""
    priority: str = "medium"
    status: str = "pending"
    asked: int = 0

    @field_validator("priority", "status", mode="before")
    @classmethod
    def _as_text(cls, value: Any) -> str:
        return str(value or "")


class PlanContext(_Contract):
    brief: str
    knowledge_context: str = ""
    max_subjects_min: int = 4
    max_subjects: int = 10
    #: Lessons from previous runs (Phase 3 feedback loop), pinned so the plan avoids known blind spots.
    prior_lessons: list[str] = Field(default_factory=list)


class QuestionContext(_Contract):
    project_name: str
    objective: str = ""
    brief: str = ""
    subject: SubjectRef
    coverage_table: str = ""
    known_facts: list[str] = Field(default_factory=list)
    previous_questions: list[str] = Field(default_factory=list)
    asked_count: int = 0
    max_questions: int = 2
    knowledge_context: str = ""


class ProbeContext(_Contract):
    subject: SubjectRef
    question: str
    answer: str
    subject_ledger: list[str] = Field(default_factory=list)
    asked_count: int = 0
    max_questions: int = 2
    next_subject: str = ""
    brief: str = ""
    objective: str = ""
    known_facts: list[str] = Field(default_factory=list)
    previous_questions: list[str] = Field(default_factory=list)


class HumanContext(_Contract):
    brief: str
    known_facts: str = ""
    subject: str
    question: str


class TriageContext(_Contract):
    subject: SubjectRef
    question: str
    answer: str
    existing_titles: list[str] = Field(default_factory=list)


class GapContext(_Contract):
    project_name: str
    objective: str = ""
    brief: str = ""
    coverage_table: str = ""
    deterministic_gaps: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    max_questions: int = 3


class SummaryContext(_Contract):
    project_name: str
    objective: str = ""
    brief: str = ""
    requirements: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    transcript: list[str] = Field(default_factory=list)


class SectionContext(_Contract):
    section: dict[str, Any]
    project_name: str
    objective: str = ""
    summary_block: str = ""
    requirements: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    outline: str = ""
    knowledge_context: str = ""
    target_words: int = 260
    extra_instructions: str = ""


class UseCasesContext(_Contract):
    project_name: str
    objective: str = ""
    stakeholders: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    process_context: str = ""
    max_use_cases: int = 6


class GlossaryContext(_Contract):
    project_name: str
    corpus: str


class CriticContext(_Contract):
    project_name: str
    lint_report: list[str] = Field(default_factory=list)
    section_ids: str = ""
    spec_markdown: str


class DiagramContext(_Contract):
    project_name: str
    spec_excerpt: str
    requirements: list[str] = Field(default_factory=list)
    max_diagrams: int = 3


class MermaidReviewContext(_Contract):
    title: str
    code: str
    validation_errors: list[str] = Field(default_factory=list)
    spec_excerpt: str = ""


class ChatAgentContext(_Contract):
    project_name: str
    phase: str
    status: str
    awaiting: str = "none"
    coverage_summary: str = ""
    requirement_count: int = 0


#: template file name (without ``.md.j2``) -> contract. A template with no contract is untyped and
#: ``render`` will not validate it; the test suite asserts every shipped template has one.
PROMPT_CONTRACTS: dict[str, type[BaseModel]] = {
    "plan": PlanContext,
    "question": QuestionContext,
    "probe": ProbeContext,
    "human": HumanContext,
    "triage": TriageContext,
    "gaps": GapContext,
    "summary": SummaryContext,
    "section": SectionContext,
    "use_cases": UseCasesContext,
    "glossary": GlossaryContext,
    "critic": CriticContext,
    "diagram": DiagramContext,
    "mermaid_review": MermaidReviewContext,
    "chat_agent": ChatAgentContext,
}


def template_source(name: str, prompts_dir: Path) -> str:
    return (prompts_dir / f"{name}.md.j2").read_text(encoding="utf-8")


def _iter_nodes(node: Node):
    """Depth-first walk of a Jinja AST (works on Jinja 3.1's `jinja2.nodes` tree)."""
    yield node
    for value in node.__dict__.values():
        if isinstance(value, Node):
            yield from _iter_nodes(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, Node):
                    yield from _iter_nodes(item)


def template_variables(name: str, prompts_dir: Path) -> set[str]:
    """Every name the template *reads* (``{{ x }}`` / ``{% if x %}``), excluding loop targets."""
    tree = Parser(Environment(), template_source(name, prompts_dir)).parse()
    nodes = list(_iter_nodes(tree))
    loop_targets = {node.name for node in nodes if isinstance(node, Name) and node.ctx == "store"}
    return {
        node.name for node in nodes if isinstance(node, Name) and node.ctx == "load" and node.name not in loop_targets
    }


def undeclared_variables(name: str, prompts_dir: Path) -> set[str]:
    """Template variables the contract does not declare — the drift the test suite must catch."""
    contract = PROMPT_CONTRACTS.get(name)
    if contract is None:
        return template_variables(name, prompts_dir)
    return template_variables(name, prompts_dir) - set(contract.model_fields)
