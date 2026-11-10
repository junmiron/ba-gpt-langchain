"""Domain + LLM-output schemas.

Two families live here:

* **LLM output schemas** (``PlanResult``, ``QuestionDraft``, …) — deliberately shallow with rich
  field descriptions, because those descriptions *are* the prompt for local models.
* **Session state** (``SubjectState``, ``LedgerItem``, ``SessionState``) — authoritative business
  state. The workflow engine reads/writes these; models never hold state themselves.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from difflib import SequenceMatcher
from enum import Enum, StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


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


def match_canonical(value: Any, options: tuple[str, ...], default: str) -> str:
    """Map a model string onto a canonical option, ignoring case, spaces, hyphens and underscores.

    ``'High'`` -> ``'high'``, ``'sequence diagram'`` -> ``'sequenceDiagram'``. Unlike :func:`one_of` the canonical
    spelling (camelCase included) is returned. An option matches when its normalised form is contained in the
    model's string, so ``'Sequence Diagram (UML)'`` also resolves.
    """
    key = re.sub(r"[^a-z]", "", str(value or "").lower())
    if not key:
        return default
    keyed = {re.sub(r"[^a-z]", "", option.lower()): option for option in options}
    if key in keyed:
        return keyed[key]
    for norm, option in keyed.items():
        if norm and norm in key:
            return option
    return default


#: Two ledger entries are duplicates only when both the character ratio and the word overlap clear this bar.
#: The ratio alone is not enough: "The system shall provide invoice export" and "...claim export" score 0.87.
NEAR_DUPLICATE_RATIO = 0.85


def _is_near_duplicate(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    # A re-stated requirement whose text contains the existing one verbatim is a duplicate no matter how
    # much extra prose the model added, so containment is checked before the ratio.
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) >= 20 and shorter in longer:
        return True
    if SequenceMatcher(None, a, b, autojunk=False).ratio() < NEAR_DUPLICATE_RATIO:
        return False
    words_a, words_b = set(a.split()), set(b.split())
    union = words_a | words_b
    return bool(union) and len(words_a & words_b) / len(union) >= NEAR_DUPLICATE_RATIO


#: Question boilerplate: words that carry no subject matter, so a paraphrase shares them with every
#: other question and inflates any overlap measure.
QUESTION_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "for",
        "to",
        "in",
        "on",
        "at",
        "by",
        "as",
        "is",
        "are",
        "been",
        "its",
        "this",
        "that",
        "these",
        "those",
        "with",
        "without",
        "from",
        "into",
        "about",
        "also",
        "only",
        "more",
        "than",
        "any",
        "all",
        "our",
        "their",
        "there",
        "you",
        "your",
        "they",
        "them",
        "not",
        "yes",
        "will",
        "would",
        "should",
        "could",
        "does",
        "did",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "how",
        "when",
        "where",
        "because",
        "while",
        "during",
        "after",
        "before",
        "over",
        "under",
        "between",
        "within",
        "needs",
        "need",
        "want",
        "asked",
        "please",
        "mean",
        "have",
        "then",
    }
)


def _question_terms(text: str) -> set[str]:
    """Subject-matter words of a question, ignoring boilerplate and short tokens.

    Plurals are folded (`targets` == `target`) and thousands separators are removed (`1,200` == `1200`)
    so a rephrased question does not look like a different one just because of inflection.
    """
    normalised = re.sub(r"(\d)[,.](\d{3})", r"\1\2", (text or "").lower())
    terms: set[str] = set()
    for word in re.findall(r"[a-z0-9]+", normalised):
        if len(word) < 3 or word in QUESTION_STOPWORDS:
            continue
        if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        terms.add(word)
    return terms


def _term_weights(texts: Iterable[str]) -> dict[str, float] | None:
    """Inverse document frequency for the questions asked so far in one session.

    In a claims project the words "claim", "portal" and "payment" appear in nearly every
    question, so plain word overlap calls "which roles get portal access" and "does the portal
    replace email for claim submission" duplicates — they share the domain nouns and nothing
    else. Down-weighting terms by how often they already appear removes that noise, and the
    weighting is self-calibrating: it comes from the session, not from a hand-written list.
    """
    texts = [t for t in texts if t]
    if len(texts) < 4:
        return None
    document_frequency: Counter[str] = Counter()
    for text in texts:
        for term in _question_terms(text):
            document_frequency[term] += 1
    total = len(texts)
    return {term: 1.0 - count / total for term, count in document_frequency.items()}


def _similarity(a: str, b: str, weights: dict[str, float] | None) -> tuple[float, float]:
    """(weighted containment, weighted novelty) between two questions. (0, 1) if either is empty.

    Containment is how much of the *shorter* question's informative vocabulary the longer one
    already covers - a re-ask covers nearly all of it, a follow-up covers some. Novelty is the
    mirror image: how much each question adds that the other lacks.
    """
    terms_a, terms_b = _question_terms(a), _question_terms(b)
    if not terms_a or not terms_b:
        return 0.0, 1.0
    weight = (weights or {}).get
    shared = sum(max(0.05, weight(term, 1.0)) for term in terms_a & terms_b)
    total_a = sum(max(0.05, weight(term, 1.0)) for term in terms_a)
    total_b = sum(max(0.05, weight(term, 1.0)) for term in terms_b)
    containment = shared / min(total_a, total_b)
    novelty = min((total_a - shared) / total_a, (total_b - shared) / total_b)
    return containment, novelty


#: Tuned against 3,206 question pairs from 16 real interviews (scratch_e2e/dedup_eval.py).
#: Genuine re-asks score containment 0.60-0.92; genuinely different questions that merely share
#: domain vocabulary score containment 0.04-0.57. Containment is the primary signal because
#: Jaccard is inflated by "claim", "portal" and "payment", which recur in every question of a
#: claims project and carry no discriminating information.
QUESTION_DUPLICATE_CONTAINMENT = 0.60
QUESTION_DUPLICATE_NOVELTY = 0.40
QUESTION_DUPLICATE_RATIO = 0.45
#: Without a session corpus there is no way to tell a meaningful "claim" from background "claim",
#: so the unweighted fallback needs a much higher bar: only near-verbatim repeats survive it.
#: The engine always supplies the corpus; this is the conservative default.
QUESTION_DUPLICATE_CONTAINMENT_UNWEIGHTED = 0.80
QUESTION_DUPLICATE_NOVELTY_UNWEIGHTED = 0.30
QUESTION_DUPLICATE_MIN_OVERLAP = 4
#: Near-identical wording needs no second opinion.
QUESTION_DUPLICATE_EXACT_RATIO = 0.85
#: A short question re-asked at length scores a low character ratio even when every one of its
#: informative words reappears. When containment is near-total and novelty is negligible the ratio
#: floor is measuring the length difference, not a different question, so it is skipped.
QUESTION_DUPLICATE_SUBSET_CONTAINMENT = 0.70
QUESTION_DUPLICATE_SUBSET_NOVELTY = 0.30


def question_is_duplicate(a: str, b: str, corpus: Iterable[str] | None = None) -> bool:
    """True when two questions ask the same thing in different words.

    Three signals, all cheap and deterministic:
      * IDF-weighted containment — how much of the shorter question's *informative* vocabulary the
        longer one already covers, with words that recur across the session discounted.
      * Novelty — how much each question adds that the other lacks. A re-ask adds almost nothing;
        a legitimate follow-up does.
      * Character ratio — catches verbatim repeats that word-splitting would miss.

    A local model asked, as Q1 and Q2 of a real run:
        Q1: Will the portal replace email for claim submission and tracking only, or also
            replace manual review and payment steps?
        Q2: For scope, does the portal replace email for submission/tracking only, or also
            replace manual review and payment execution?
    The "do not repeat" instruction in the prompt did not prevent it; this guard does.

    `corpus` is the list of questions already in the session. Without it the weighting collapses
    to raw word counts and the thresholds tighten, so the guard errs toward letting a question
    through rather than suppressing it.
    """
    if not (a or "").strip() or not (b or "").strip():
        return False

    left, right = a.lower().strip(), b.lower().strip()
    ratio = SequenceMatcher(None, left, right, autojunk=False).ratio()
    if ratio >= QUESTION_DUPLICATE_EXACT_RATIO:
        return True

    weights = _term_weights(corpus) if corpus is not None else None
    contain_limit = QUESTION_DUPLICATE_CONTAINMENT if weights else QUESTION_DUPLICATE_CONTAINMENT_UNWEIGHTED
    novel_limit = QUESTION_DUPLICATE_NOVELTY if weights else QUESTION_DUPLICATE_NOVELTY_UNWEIGHTED

    containment, novelty = _similarity(a, b, weights)
    overlap = len(_question_terms(a) & _question_terms(b))
    if overlap < QUESTION_DUPLICATE_MIN_OVERLAP or containment < contain_limit or novelty > novel_limit:
        return False
    if ratio >= QUESTION_DUPLICATE_RATIO:
        return True
    # Ratio fails only because one question is much longer than the other; total containment with
    # negligible novelty is still the same question.
    return containment >= QUESTION_DUPLICATE_SUBSET_CONTAINMENT and novelty <= QUESTION_DUPLICATE_SUBSET_NOVELTY


class _Strict(BaseModel):
    """Local models love adding stray keys; ignore them instead of failing the step."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


# --------------------------------------------------------------------------- enums


class Phase(StrEnum):
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


#: Phases where the engine legitimately hands control back to the human. Kept as documentation of the
#: contract rather than a runtime check: the authoritative signal is `SessionState.awaiting`, which is
#: narrower (it distinguishes "a question is on screen" from "an approval is pending").
AWAITING_USER_PHASES = {Phase.INTERVIEW, Phase.CLARIFY, Phase.APPROVAL}


class AwaitKind(StrEnum):
    NONE = "none"
    ANSWER = "answer"  # a question is on screen
    APPROVAL = "approval"  # summary approval card
    FINISHED = "finished"  # nothing left to do


class RunStatus(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING_USER = "waiting_user"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


class RequirementCategory(StrEnum):
    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non_functional"
    INTEGRATION = "integration"
    DATA = "data"
    REPORTING = "reporting"


class Priority(StrEnum):
    MUST = "must"
    SHOULD = "should"
    COULD = "could"
    WONT = "wont"


SubjectStatus = Literal["pending", "asking", "done", "skipped", "covered"]
LedgerKind = Literal["fact", "requirement", "risk", "assumption", "decision", "open_question"]
QuestionKind = Literal["open", "closed", "confirm"]
QuestionPurpose = Literal["agenda", "gap", "review"]


# --------------------------------------------------------------------------- LLM: planning


class SubjectSpec(_Strict):
    name: str = Field(description="Short subject title, e.g. 'Scope In and Out'.")
    why: str = Field(default="", description="One sentence on why this subject matters for THIS project.")
    priority: Literal["high", "medium", "low"] = "medium"

    @field_validator("priority", mode="before")
    @classmethod
    def _lenient_priority(cls, value: Any) -> str:
        return match_canonical(value, ("high", "medium", "low"), "medium")


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


class StakeholderAnswer(_Strict):
    """A simulated stakeholder reply; only used when the Human lane is model-backed."""

    answer: str = Field(
        description="One concise answer to the stakeholder question; say unknown if the brief lacks facts."
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
    next_question: str = Field(default="", description="When covered, one question for the next subject if provided.")
    next_rationale: str = ""

    @property
    def follow_up(self) -> QuestionDraft | None:
        text = (self.follow_up_question or "").strip()
        if not text or self.covered:
            return None
        return QuestionDraft(question=text, rationale=self.follow_up_rationale)


class TriageResult(_Strict):
    """Structured extraction from one Q&A pair."""

    facts: list[str] = Field(default_factory=list, description="Concrete facts stated by the user.")
    requirements: list[RequirementDraft] = Field(default_factory=list)
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
    gaps: list[GapItem] = Field(
        default_factory=list, description="At most 3 questions, highest impact first. Empty when nothing is missing."
    )


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
    def _reconcile_status(self) -> CriticVerdict:
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

    @field_validator("diagram_type", mode="before")
    @classmethod
    def _lenient_diagram_type(cls, value: Any) -> str:
        return match_canonical(value, ("flowchart", "sequenceDiagram", "stateDiagram", "classDiagram"), "flowchart")


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
        """A subject is finished when it has an answer, was skipped, or was answered by a transcript."""
        return self.status in ("done", "skipped", "covered")


class LedgerItem(BaseModel):
    """One entry of the requirements ledger — the single source of truth for gathered knowledge."""

    id: str
    kind: LedgerKind = "fact"
    title: str = ""
    body: str = ""
    category: RequirementCategory | None = None
    priority: Priority | None = None
    acceptance: list[str] = Field(default_factory=list)
    subject: str = ""
    source_q: int | None = None
    created_at: str = Field(default_factory=utcnow)

    def as_line(self) -> str:
        prefix = {
            "requirement": self.id,
            "fact": "FACT",
            "risk": "RISK",
            "assumption": "ASSUMPTION",
            "decision": "DECISION",
            "open_question": "OPEN",
        }.get(self.kind, "-")
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
    purpose: QuestionPurpose = "agenda"
    round: int = 0  # 0 = agenda interview, ≥1 = clarification/review round
    #: True once the question has been put in front of the stakeholder (a QUESTION event was emitted).
    #: Queued questions are created before they are asked, so this distinguishes "pending" from "queued".
    emitted: bool = False
    answer: str | None = None
    answer_source: Literal["human", "simulated"] = "human"
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

    @field_validator("diagrams")
    @classmethod
    def _diagrams_are_json_safe(cls, diagrams: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Reject binary payloads in diagram dicts.

        The whole state is persisted as JSON (`state_json`), so a `bytes` value here does not fail at the
        point it is set — it fails later, inside `save_state`, and the run stalls with a serialization error
        that is far from its cause. Rendered images belong on disk; the dict carries `png_path`.
        """
        for index, diagram in enumerate(diagrams):
            for key, value in diagram.items():
                if isinstance(value, (bytes, bytearray)):
                    raise ValueError(
                        f"diagram {index} field {key!r} holds {len(value)} raw bytes; "
                        "write the file and store its path instead"
                    )
        return diagrams

    artifacts: list[ArtifactRef] = Field(default_factory=list)
    #: Aggregate counters plus a `by_phase` breakdown. `Any` because the breakdown is a dict of
    #: dicts; every scalar key is still a float.
    metrics: dict[str, Any] = Field(default_factory=dict)
    model_overrides: dict[str, str] = Field(default_factory=dict)
    #: An elicitation transcript (meeting notes, a prior interview, a discovery call log). When present the
    # engine seeds the ledger from it and the interview only asks about what the transcript left open.
    transcript: str = ""
    transcript_source: str = ""
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
        """The question the stakeholder is looking at: the oldest one asked but not yet answered.

        Oldest-first matters because a step can queue follow-ups *before* asking them; returning the newest
        unresolved question would hide the one actually on screen behind questions that have not been asked
        yet.
        """
        for question in self.questions:
            if question.emitted and not question.resolved:
                return question
        return None

    @property
    def unasked_questions(self) -> list[QuestionState]:
        """Queued questions that exist but have never been put in front of the stakeholder."""
        return [q for q in self.questions if not q.emitted and not q.resolved]

    @property
    def seed(self) -> str:
        """What the run is working from: the brief, or a digest of the transcript when there is no brief.

        Roles read this instead of ``brief`` so a transcript-only session behaves like a briefed one
        everywhere (planning, gap analysis, writing) without duplicating the elicitation text.
        """
        brief = (self.brief or "").strip()
        if brief:
            return brief
        text = (self.transcript or "").strip()
        if not text:
            return ""
        lines = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]
        return " ".join(lines[:12])[:1500]

    def requirements(self) -> list[LedgerItem]:
        return [i for i in self.ledger if i.kind == "requirement"]

    def new_requirement_id(self, category: RequirementCategory | None = None) -> str:
        """Stable, traceable ids (FR-001 / NFR-002 / INT-003 …) — the spec and diagrams reference these."""
        prefix_by_category = {
            RequirementCategory.FUNCTIONAL: "FR",
            RequirementCategory.NON_FUNCTIONAL: "NFR",
            RequirementCategory.INTEGRATION: "INT",
            RequirementCategory.DATA: "DAT",
            RequirementCategory.REPORTING: "REP",
        }
        prefix = "REQ" if category is None else prefix_by_category.get(category, "REQ")
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
        return any(_is_near_duplicate(needle, _normalise(item.body or item.title)) for item in self.ledger)

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
