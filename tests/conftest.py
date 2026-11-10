"""Test fixtures: a scripted fake LM Studio gateway so the whole pipeline runs without any model.

The point of these tests is the **state machine**, budgets and artefacts — not model quality. They run in
milliseconds and never touch VRAM, which is exactly what v1 lacked (its behaviour could only be observed by
running the full AutoGen team against Azure).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ba_agent.config import AppConfig  # noqa: E402
from ba_agent.models import StructuredResult, TextResult  # noqa: E402
from ba_agent.schemas import (  # noqa: E402
    CriticVerdict,
    DiagramSet,
    GapAnalysis,
    PlanResult,
    ProbeDecision,
    QuestionDraft,
    RequirementDraft,
    SummaryDoc,
    TriageResult,
)
from ba_agent.services import Services  # noqa: E402
from ba_agent.tools.sandbox import Sandbox  # noqa: E402

SECTION_BODY = (
    "The system shall provide the capability described in the approved summary. "
    "Each requirement below is traceable to an interview answer and carries acceptance criteria. "
    "Actors follow the documented business rules; exceptions are handled by notifying the responsible role."
)


class FakeGateway:
    """Deterministic stand-in for :class:`ba_agent.models.ModelGateway`."""

    def __init__(
        self, *, critic_script: list[str] | None = None, always_probe_followup: bool = False, fail_triage: bool = False
    ) -> None:
        self.usage: dict[str, Any] = {"calls": 0, "failures": 0, "prompt_chars": 0, "completion_chars": 0}
        self.total_model_ms = 0
        self.calls: list[tuple[str, str]] = []
        #: (role, step, model_override) — lets tests assert that a chosen model actually reaches the gateway.
        self.routed: list[tuple[str, str, str | None]] = []
        #: (role, step, system_prompt, user_prompt) — for asserting what a prompt actually contains.
        self.prompts: list[tuple[str, str, str, str]] = []
        self.critic_script = list(critic_script or ["complete"])
        self.always_probe_followup = always_probe_followup
        self.fail_triage = fail_triage

    # -- catalogue ----------------------------------------------------------

    def rebind(self) -> None:
        """Same contract as `ModelGateway.rebind` — a fake has nothing to re-point."""

    async def resolve_model(self, requested: str | None) -> str:
        return requested or "fake-model"

    async def list_models(self, refresh: bool = False) -> list[Any]:
        return []

    async def health(self) -> dict[str, Any]:
        return {"reachable": True}

    # -- unused by the workflow --------------------------------------------

    def chat(self, *args: Any, **kwargs: Any):  # pragma: no cover - agent loop only
        raise NotImplementedError

    def bind_tools(self, *args: Any, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError

    async def embed(self, texts):  # knowledge base stays BM25-only in tests
        return None

    async def aclose(self) -> None:
        self._closed = True

    # -- the two calls roles actually use ----------------------------------

    async def structured(
        self,
        schema,
        *,
        system_prompt: str = "",
        user_prompt: str = "",
        role: str = "",
        model_override: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        emit=None,
        step: str = "",
    ) -> StructuredResult:
        self.calls.append((role, step))
        self.routed.append((role, step, model_override))
        self.prompts.append((role, step, system_prompt, user_prompt))
        self.usage["calls"] += 1

        if role == "plan":
            parsed = PlanResult(
                project_name="Claims Portal",
                objective="Let brokers submit and track claims online.",
                summary="A portal replacing the current email-based claim submission process.",
                subjects=[
                    {"name": "Product Overview", "why": "baseline", "priority": "high"},
                    {"name": "Scope In and Out", "why": "boundaries", "priority": "medium"},
                ],
                known_facts=["Claims arrive by email today"],
                assumptions=["Existing policy system stays"],
            )
        elif role == "interview" and step.startswith("question"):
            parsed = QuestionDraft(
                question=f"What does {step.split(':', 1)[-1]} mean for your team?",
                rationale="needed for scope",
                kind="open",
                options=[],
            )
        elif role == "interview" and step.startswith("probe"):
            if self.always_probe_followup:
                parsed = ProbeDecision(covered=False, reason="want more", follow_up_question="And who approves it?")
            else:
                parsed = ProbeDecision(covered=True, reason="enough detail")
        elif role == "interview" and step == "gap_analysis":
            parsed = GapAnalysis(gaps=[])
        elif role == "triage":
            if self.fail_triage:
                return StructuredResult(
                    parsed=None,
                    raw_text="",
                    method="failed",
                    attempts=3,
                    degraded=True,
                    error="simulated schema failure",
                )
            parsed = TriageResult(
                facts=["Claims are validated manually today"],
                requirements=[
                    RequirementDraft(
                        title="Submit claim online",
                        description="The system shall let a broker submit a claim with attachments.",
                        acceptance_criteria=["A claim with 5 PDFs uploads in under 10 seconds"],
                    )
                ],
                risks=["Policy system API may rate-limit"],
                assumptions=[],
                decisions=["Email intake is retired"],
                coverage_note="Volume per month still unknown",
            )
        elif role == "summary":
            parsed = SummaryDoc(
                project_name="Claims Portal",
                objective="Online claim submission.",
                background="Replaces email intake.",
                objectives=["Reduce handling time"],
                stakeholders=["Broker — submit claims quickly"],
                scope_in=["Claim intake"],
                scope_out=["Payments"],
                key_requirements=["FR-001 Submit claim online"],
                non_functional=["NFR-001 upload under 10 seconds"],
                assumptions=["Policy system stays"],
                open_questions=[],
            )
        elif role == "critic":
            verdict = (
                self.critic_script.pop(0)
                if len(self.critic_script) > 1
                else (self.critic_script[0] if self.critic_script else "complete")
            )
            if verdict == "needs_user":
                parsed = CriticVerdict(
                    status="needs_user",
                    summary="missing volumes",
                    issues=[
                        {
                            "section_id": "non_functional_requirements",
                            "kind": "user",
                            "severity": "blocker",
                            "description": "No monthly volume captured",
                            "question_for_user": "How many claims per month?",
                        }
                    ],
                )
            elif verdict == "auto_fixable":
                parsed = CriticVerdict(
                    status="auto_fixable",
                    summary="glossary too thin",
                    issues=[
                        {
                            "section_id": "glossary",
                            "kind": "internal",
                            "severity": "blocker",
                            "description": "Glossary defines fewer than 5 terms",
                            "fix_hint": "define more project terms",
                        }
                    ],
                )
            else:
                parsed = CriticVerdict(status="complete", summary="good enough", issues=[])
        elif role == "diagrammer":
            parsed = DiagramSet(
                diagrams=[
                    {
                        "title": "Claim submission",
                        "diagram_type": "flowchart",
                        "code": 'flowchart TD\n    A["Broker submits claim"] --> B{"Valid?"}\n'
                        '    B -- "yes" --> C["Claim registered"]\n'
                        '    B -- "no" --> D["Rejection notice"]',
                    }
                ]
            )
        elif role == "mermaid_reviewer":
            parsed = {"approved": True, "summary": "ok", "issues": [], "revised_code": ""}
            from ba_agent.schemas import MermaidReviewResult

            parsed = MermaidReviewResult(**parsed)
        else:  # pragma: no cover - guards against new roles without a script entry
            raise AssertionError(f"FakeGateway has no scripted answer for role={role!r} step={step!r}")

        return StructuredResult(parsed=parsed, raw_text="{}", method="json_schema", attempts=1, model="fake-model")

    async def stream_text(
        self,
        *,
        system_prompt: str = "",
        user_prompt: str = "",
        role: str = "",
        model_override: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        emit=None,
        step: str = "",
    ) -> TextResult:
        self.calls.append((role, step))
        self.routed.append((role, step, model_override))
        self.prompts.append((role, step, system_prompt, user_prompt))
        self.usage["calls"] += 1

        if role == "use_cases":
            text = (
                "### UC-01 · Submit claim\n**Goal:** broker files a claim.\n**Primary actor:** Broker\n"
                "**Preconditions:** broker is authenticated\n**Main flow:**\n1. Broker uploads documents\n"
                "2. System validates\n3. System registers the claim\n**Postconditions:** claim exists\n\n"
                "### User Stories\n- **US-01** As a broker, I want to upload attachments, so that claims are complete.\n"
                "    Acceptance: Given valid files, when submitting, then the claim is registered\n"
            )
        elif role == "glossary":
            text = (
                "### Business terms\n- **Claim** — a request for reimbursement filed by a broker.\n"
                "- **Broker** — an external party submitting claims on behalf of a client.\n"
                "### Technical terms\n- **Policy system** — the legacy system of record.\n"
                "### Acronyms\n- **FSD** — functional specification document.\n- **SLA** — service level agreement."
            )
        else:
            text = SECTION_BODY

        return TextResult(text=text, model="fake-model", latency_ms=5)


@pytest.fixture()
def config(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(root=tmp_path)
    cfg.lmstudio.embedding_model = None  # keep tests offline and deterministic
    cfg.interview.max_questions_per_subject = 2
    cfg.interview.min_subjects = 2
    cfg.interview.max_subjects = 4
    cfg.interview.max_gap_rounds = 1
    cfg.review.max_review_rounds = 2
    # The scripted gateway yields one requirement per answer; keep the linter's count gate out of the way so
    # each pipeline test exercises its own escalation path (the gate itself is covered in test_units.py).
    cfg.review.min_functional_requirements = 1
    cfg.ensure_dirs()
    (tmp_path / "data").mkdir(exist_ok=True)
    return cfg


@pytest.fixture()
def services(config: AppConfig, request) -> Services:
    gateway = getattr(request, "param", None) or FakeGateway()
    from ba_agent.knowledge import KnowledgeService

    store_services = Services.create(config)  # creates dirs + sqlite once
    store_services.gateway = gateway  # type: ignore[assignment]
    store_services.knowledge = KnowledgeService(store_services.store, gateway, config)  # type: ignore[arg-type]
    store_services.sandbox = Sandbox(writable_roots=[config.root / "outputs", config.root / "data"])
    return store_services


@pytest.fixture()
def fake_gateway_cls():
    return FakeGateway
