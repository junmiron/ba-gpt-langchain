"""Domain + LLM-output schemas.

Two families live here:

* **LLM output schemas** (``PlanResult``, ``QuestionDraft``, …) — deliberately shallow with rich
  field descriptions, because those descriptions *are* the prompt for local models.
* **Session state** (``SubjectState``, ``LedgerItem``, ``SessionState``) — authoritative business
  state. The workflow engine reads/writes these; models never hold state themselves.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def coerce_enum(value: Any, enum_cls: type[Enum], default: Any) -> Any:
    """Local models emit ``'Non-Functional'`` / ``'non functional'`` / ``'MUST'``.

    Normalise instead of failing the whole step; a wrong-but-valid enum is far cheaper than a retry.
    """
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        candidate = value.strip().lower().replace("-", "_").replace(" ", "_")
        for item in enum_cls:
            if item.value == candidate:
                return item
        for item in enum_cls:
            if item.value.replace("_", " ") in candidate or candidate in item.value:
                return item
    return default


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).strip()


def one_of(value: Any, allowed: tuple[str, ...], default: str) -> str:
    """Map a free-form model string onto a small set of literals."""
    text = str(value or "").strip().lower()
    if not text:
        return default
    for option in allowed:
        if text == option:
            return option
    for option in allowed:
        if option in text or text in option:
            return option
    return default


class _Strict(BaseModel):
    """Local models love adding stray keys; ignore them instead of failing the step."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


# --------------------------------------------------------------------------- enums


class Phase(str, Enum):
    INTAKE = "intake"
    PLAN = "plan"
    INTERVIEW = "interview"
    CLARIFY = "clarify"
    SUMMARY = "summary"
    APPROVAL = "approval"
    DRAFT = "draft"
    REVIEW = "review"
    DIAGRAM = "diagram"
    DELIVER = "deliver"
    DONE = "done"


#: Phases that can hand control back to the human.
AWAITING_USER_PHASES = {Phase.INTERVIEW, Phase.CLARIFY, Phase.APPROVAL}


class AwaitKind(str, Enum):
    NONE = "none"
    ANSWER = "answer"          # a question is on screen
    APPROVAL = "approval"      # summary approval card
    FINISHED = "finished"      # nothing left to do


class RunStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING_USER = "waiting_user"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


class RequirementCategory(str, Enum):
    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non_functional"
    INTEGRATION = "integration"
    DATA = "data"
    REPORTING = "reporting"


class Priority(str, Enum):
    MUST = "must"
    SHOULD = "should"
    COULD = "could"
    WONT = "wont"


SubjectStatus = Literal["pending", "asking", "done", "skipped"]


# --------------------------------------------------------------------------- LLM: planning


class SubjectSpec(_Strict):
    name: str = Field(description="Short subject title, e.g. 'Scope In and Out'.")
    why: str = Field(default="", description="One sentence on why this subject matters for THIS project.")
    priority: Literal["high", "medium", "low"] = "medium"


class PlanResult(_Strict):
    """Output of the intake/planning step."""

    project_name: str = Field(description="Short human-readable project name inferred from the brief.")
    objective: str = Field(default="", description="One-sentence statement of what the project must achieve.")
    summary: str = Field(default="", description="2-4 sentence restatement of the brief in BA language.")
    subjects: list[SubjectSpec] = Field(
        default_factory=list,
        description="Ordered interview subjects to cover. 5-8 items, most important first.",
    )
    known_facts: list[str] = Field(
        default_factory=list,
        description="Facts already stated in the brief — do NOT ask the user about these again.",
    )
    assumptions: list[str] = Field(default_factory=list, description="Reasonable assumptions to state in the spec.")


# --------------------------------------------------------------------------- LLM: interview


class QuestionDraft(_Strict):
    question: str = Field(description="Exactly one concise question for the stakeholder. No preamble.")
    rationale: str = Field(default="", description="Why this question is needed (shown as helper text).")
    kind: Literal["open", "closed", "confirm"] = "open"

    @field_validator("kind", mode="before")
    @classmethod
    def _lenient_kind(cls, value: Any) -> str:
        return one_of(value, ("open", "closed", "confirm"), "open")

    options: list[str] = Field(
        default_factory=list,
        description="For closed questions, 2-5 short answer choices the user can click. Empty for open ones.",
    )


class ProbeDecision(_Strict):
    """After an answer: is this subject covered, or do we need one follow-up?

    Deliberately flat (``follow_up_question`` rather than a nested object): small local models handle
    flat schemas dramatically better, and this step runs once per answer.
    """

    covered: bool = Field(description="True when enough was gathered about this subject to move on.")
    reason: str = Field(default="", description="Short justification.")
    follow_up_question: str = Field(default="", description="One follow-up question, only when covered is false.")
    follow_up_rationale: str = ""

    @property
    def follow_up(self) -> QuestionDraft | None:
        text = (self.follow_up_question or "").strip()
        if not text or self.covered:
            return None
        return QuestionDraft(question=text, rationale=self.follow_up_rationale)


class TriageResult(_Strict):
    """Structured extraction from one Q&A pair."""

    facts: list[str] = Field(default_factory=list, description="Concrete facts stated by the user.")
    requirements: list["RequirementDraft"] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list, description="Explicit choices made by the user.")
    coverage_note: str = Field(default="", description="One line on what is still unclear after this answer.")


class RequirementDraft(_Strict):
    title: str = Field(description="Short requirement title (max ~10 words).")
    description: str = Field(description="The requirement itself, in 'The system shall ...' style.")
    category: RequirementCategory = RequirementCategory.FUNCTIONAL
    priority: Priority = Priority.SHOULD
    acceptance_criteria: list[str] = Field(default_factory=list, description="Testable acceptance statements.")

    @field_validator("category", mode="before")
    @classmethod
    def _lenient_category(cls, value: Any) -> RequirementCategory:
        return coerce_enum(value, RequirementCategory, RequirementCategory.FUNCTIONAL)

    @field_validator("priority", mode="before")
    @classmethod
    def _lenient_priority(cls, value: Any) -> Priority:
        return coerce_enum(value, Priority, Priority.SHOULD)

    @field_validator("acceptance_criteria", mode="before")
    @classmethod
    def _listify(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [value] if value.strip() else []
        return value


# --------------------------------------------------------------------------- LLM: gaps / summary


class GapItem(_Strict):
    subject: str = Field(description="Subject the missing information belongs to.")
    question: str = Field(description="A single question that would close this gap.")
    why: str = Field(default="", description="What breaks in the spec if we do not ask.")


class GapAnalysis(_Strict):
    gaps: list[GapItem] = Field(default_factory=list, description="At most 3 questions, highest impact first. Empty when nothing is missing.")


class SummaryDoc(_Strict):
    """Structured summary; markdown is rendered deterministically from these fields."""

    project_name: str = ""
    objective: str = ""
    background: str = Field(default="", description="2-4 sentences of context.")
    objectives: list[str] = Field(default_factory=list)
    stakeholders: list[str] = Field(default_factory=list, description="Roles/personas with a short need each.")
    scope_in: list[str] = Field(default_factory=list)
    scope_out: list[str] = Field(default_factory=list)
    key_requirements: list[str] = Field(default_factory=list, description="The 6-12 most important requirements.")
    non_functional: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class ApprovalResult(_Strict):
    decision: Literal["approve", "changes"]
    notes: str = ""


# --------------------------------------------------------------------------- LLM: spec writing


class SectionDraft(_Strict):
    section_id: str = Field(description="The section id you were asked to write, echoed back.")
    markdown: str = Field(description="Markdown body of the section. No top-level heading, no fenced code wrapper.")


class Issue(_Strict):
    section_id: str = Field(default="", description="Section id the issue belongs to (empty for document-wide).")

    @field_validator("kind", mode="before")
    @classmethod
    def _lenient_kind(cls, value: Any) -> str:
        text = str(value or "").strip().lower()
        if text.startswith("u") or "stakeholder" in text or "user" in text:
            return "user"
        return "internal"

    kind: Literal["internal", "user"] = Field(
        default="internal",
        description="'internal' = the writer can fix it alone; 'user' = only the stakeholder can answer it.",
    )
    severity: Literal["blocker", "warning"] = "warning"

    @field_validator("severity", mode="before")
    @classmethod
    def _lenient_severity(cls, value: Any) -> str:
        return "blocker" if str(value or "").strip().lower().startswith(("b", "crit", "high", "maje")) else "warning"
    description: str = ""
    fix_hint: str = Field(default="", description="For internal issues: what to change.")
    question_for_user: str = Field(default="", description="For user issues: the exact question to ask.")
    #: Who raised it. Lint findings are machine diagnostics; they must not be shown to a stakeholder verbatim.
    source: Literal["critic", "lint"] = "critic"


class CriticVerdict(_Strict):
    #: ``complete`` | ``auto_fixable`` | ``needs_user`` — normalised (and re-derived) in :meth:`_reconcile_status`.
    status: str = "complete"
    summary: str = ""
    issues: list[Issue] = Field(default_factory=list)

    @field_validator("status", mode="before")
    @classmethod
    def _lenient_status(cls, value: Any) -> str:
        text = str(value or "").strip().lower().replace(" ", "_")
        if "need" in text and ("user" in text or "stakeholder" in text or "info" in text):
            return "needs_user"
        if "auto" in text or "fix" in text or "writer" in text:
            return "auto_fixable"
        return one_of(text, ("complete", "auto_fixable", "needs_user"), "")

    @model_validator(mode="after")
    def _reconcile_status(self) -> "CriticVerdict":
        """Trust the issues over the label: 'complete' + blockers means it is not complete."""
        blockers = [i for i in self.issues if i.severity == "blocker"]
        derived = "complete"
        if any(i.kind == "user" for i in blockers):
            derived = "needs_user"
        elif blockers:
            derived = "auto_fixable"
        if self.status not in ("complete", "auto_fixable", "needs_user") or (
            self.status == "complete" and derived != "complete"
        ):
            self.status = derived
        return self


# --------------------------------------------------------------------------- LLM: diagrams


class DiagramDraft(_Strict):
    title: str = ""
    diagram_type: Literal["flowchart", "sequenceDiagram", "stateDiagram", "classDiagram"] = "flowchart"
    code: str = Field(description="Mermaid body only (no ``` fences), starting with the diagram keyword.")
    notes: str = ""


class DiagramSet(_Strict):
    diagrams: list[DiagramDraft] = Field(default_factory=list)


class MermaidReviewResult(_Strict):
    approved: bool
    summary: str = ""
    issues: list[str] = Field(default_factory=list, description="Concrete syntax/logic problems.")
    revised_code: str = Field(default="", description="Full corrected Mermaid code when you can fix it yourself.")


# --------------------------------------------------------------------------- session state


class SubjectState(BaseModel):
    name: str
    why: str = ""
    priority: Literal["high", "medium", "low"] = "medium"
    status: SubjectStatus = "pending"
    asked: int = 0
    notes: str = ""

    @property
    def closed(self) -> bool:
        return self.status in ("done", "skipped")


class LedgerItem(BaseModel):
    """One entry of the requirements ledger — the single source of truth for gathered knowledge."""

    id: str
    kind: Literal["fact", "requirement", "risk", "assumption", "decision", "open_question"] = "fact"
    title: str = ""
    body: str = ""
    category: RequirementCategory | None = None
    priority: Priority | None = None
    acceptance: list[str] = Field(default_factory=list)
    subject: str = ""
    source_q: int | None = None
    created_at: str = Field(default_factory=utcnow)

    def as_line(self) -> str:
        prefix = {"requirement": self.id, "fact": "FACT", "risk": "RISK", "assumption": "ASSUMPTION",
                  "decision": "DECISION", "open_question": "OPEN"}.get(self.kind, "-")
        head = f"{prefix}: {self.title or self.body[:60]}"
        tail = f" — {self.body}" if self.title and self.body else ""
        return head + tail


class QuestionState(BaseModel):
    id: int
    subject: str
    text: str
    rationale: str = ""
    kind: Literal["open", "closed", "confirm"] = "open"
    options: list[str] = Field(default_factory=list)
    #: what the question is for — decides where control returns after the answer.
    purpose: Literal["agenda", "gap", "review"] = "agenda"
    round: int = 0            # 0 = agenda interview, ≥1 = clarification/review round
    answer: str | None = None
    skipped: bool = False
    not_applicable: bool = False
    asked_at: str = Field(default_factory=utcnow)
    answered_at: str | None = None

    @property
    def resolved(self) -> bool:
        return self.answer is not None or self.skipped


class ArtifactRef(BaseModel):
    name: str
    kind: str = "file"
    path: str = ""
    url: str = ""
    bytes: int = 0
    sha256: str = ""
    created_at: str = Field(default_factory=utcnow)


class SessionState(BaseModel):
    """Everything needed to resume a run. Persisted as JSON in SQLite after every step."""

    session_id: str
    title: str = ""
    brief: str = ""
    phase: Phase = Phase.INTAKE
    status: RunStatus = RunStatus.IDLE
    awaiting: AwaitKind = AwaitKind.NONE
    error: str | None = None

    # planning
    plan: PlanResult | None = None
    subjects: list[SubjectState] = Field(default_factory=list)

    # interview
    questions: list[QuestionState] = Field(default_factory=list)
    question_seq: int = 0
    gap_round: int = 0
    ledger: list[LedgerItem] = Field(default_factory=list)
    requirement_seq: int = 0

    # documents
    summary: SummaryDoc | None = None
    approved_summary: bool = False
    approval_notes: str = ""
    summary_rejections: int = 0
    sections: dict[str, str] = Field(default_factory=dict)
    spec_markdown: str = ""
    review_rounds: int = 0
    open_issues: list[Issue] = Field(default_factory=list)
    lint_report: list[str] = Field(default_factory=list)

    # diagrams
    diagrams: list[dict[str, Any]] = Field(default_factory=list)
    mermaid_code: str = ""
    diagram_review: dict[str, Any] | None = None

    artifacts: list[ArtifactRef] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    model_overrides: dict[str, str] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utcnow)
    updated_at: str = Field(default_factory=utcnow)

    # -- derived helpers ----------------------------------------------------

    def next_subject(self) -> SubjectState | None:
        for subject in self.subjects:
            if not subject.closed:
                return subject
        return None

    @property
    def current_question(self) -> QuestionState | None:
        for question in reversed(self.questions):
            if not question.resolved:
                return question
        return None

    def ledger_lines(self, kind: str | None = None, limit: int = 200) -> list[str]:
        items = [i for i in self.ledger if kind is None or i.kind == kind]
        return [i.as_line() for i in items[-limit:]]

    def requirements(self) -> list[LedgerItem]:
        return [i for i in self.ledger if i.kind == "requirement"]

    def new_requirement_id(self, category: RequirementCategory | None = None) -> str:
        """Stable, traceable ids (FR-001 / NFR-002 / INT-003 …) — the spec and diagrams reference these."""
        prefix = {RequirementCategory.FUNCTIONAL: "FR", RequirementCategory.NON_FUNCTIONAL: "NFR",
                  RequirementCategory.INTEGRATION: "INT", RequirementCategory.DATA: "DAT",
                  RequirementCategory.REPORTING: "REP"}.get(category, "REQ")
        existing = {item.id for item in self.ledger}
        number = self.requirement_seq + 1
        while f"{prefix}-{number:03d}" in existing:
            number += 1
        self.requirement_seq = number
        return f"{prefix}-{number:03d}"

    def has_similar(self, text: str) -> bool:
        """Cheap near-duplicate guard so repeated answers do not bloat the ledger."""
        needle = _normalise(text)
        if len(needle) < 12:
            return False
        for item in self.ledger:
            existing = _normalise(item.body or item.title)
            if existing == needle or (len(existing) > 24 and (needle.startswith(existing[:24]) or existing.startswith(needle[:24]))):
                return True
        return False

    def touch(self) -> None:
        self.updated_at = utcnow()


class QAPair(BaseModel):
    session_id: str
    q_index: int
    subject: str
    question: str
    answer: str | None = None
    skipped: bool = False
    not_applicable: bool = False
    rationale: str = ""
    purpose: str = "agenda"
    round: int = 0
    created_at: str = Field(default_factory=utcnow)

    def to_jsonl(self) -> dict[str, Any]:
        return self.model_dump()


# --------------------------------------------------------------------------- helpers


def clean_markdown(text: str) -> str:
    """Strip code fences / stray role prefixes a local model may wrap around real content.

    Also unwraps the JSON envelope some models insist on (``{"section_id": ..., "markdown": ...}``)
even when asked for plain Markdown — but only when the whole reply is such an object, so a section
that merely *quotes* a JSON sample is left alone.
    """
    if not text:
        return ""
    cleaned = text.strip()
    fence = re.match(r"^```[a-zA-Z]*\s*\n(?P<body>.*)\n?```\s*$", cleaned, re.S)
    if fence:
        cleaned = fence.group("body").strip()
    cleaned = re.sub(r"^(Here (is|are)[^\n]{0,60}):\s*\n+", "", cleaned, flags=re.I)
    if cleaned.startswith("{"):
        from .jsonx import extract_json  # local import: jsonx is leaf-level, keep schemas import-light

        data = extract_json(cleaned)
        if isinstance(data, dict):
            for key in ("markdown", "body"):
                value = data.get(key)
                if isinstance(value, str) and len(value.strip()) >= 20:
                    return clean_markdown(value).strip()
    return cleaned


TriageResult.model_rebuild()
