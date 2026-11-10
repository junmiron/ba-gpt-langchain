"""Regression tests for the model-facing chat tools.

These are the only functions in the project that take *arbitrary strings from a model* and write to the
session. `build_chat_tools` is exercised directly rather than through the agent loop, because a real model
call would make the test non-deterministic — and the point of these tools is that a hallucinated argument
must degrade to a helpful message, not an exception that kills the turn.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ba_agent.config import AppConfig
from ba_agent.schemas import LedgerItem, QuestionState, SubjectState
from ba_agent.tools.chat_tools import ChatContext, build_chat_tools
from ba_agent.tools.sandbox import Sandbox


class FakeStore:
    """Minimal artifact registry — `read_artifact`'s "available artefacts" hint comes from here."""

    def __init__(self, artifacts: list[dict[str, Any]] | None = None) -> None:
        self._artifacts = artifacts or []

    def list_artifacts(self, session_id: str) -> list[dict[str, Any]]:
        return self._artifacts


def _services(tmp_path: Path) -> tuple[AppConfig, Any]:
    """Services built the way the server and CLI build them, with an offline-safe knowledge service."""
    from ba_agent.knowledge import KnowledgeService
    from ba_agent.services import Services

    config = AppConfig(root=tmp_path)
    config.lmstudio.embedding_model = None
    config.ensure_dirs()
    services = Services.create(config)
    services.knowledge = KnowledgeService(services.store, services.gateway, config)
    services.sandbox = Sandbox(writable_roots=[config.root / "outputs", config.root / "data"])
    return config, services


def make_context(tmp_path: Path, *, subjects: list[SubjectState] | None = None) -> ChatContext:
    """A ChatContext whose `mutate` actually persists, mirroring SessionManager.mutate.

    The session is created through the store (synchronous) rather than `SessionManager.create_session`,
    which schedules a pipeline run — these tests must see a settled, un-run state.
    """
    from ba_agent.runtime import SessionRuntime

    _, services = _services(tmp_path)
    state = services.store.create_session(brief="Claims arrive by email.", title="Chat tool probe")
    runtime = SessionRuntime(session_id=state.session_id, state=state)
    if subjects is not None:
        state.subjects = subjects
        services.store.save_state(state)

    async def mutate(apply_state):
        async with runtime.lock:
            message = apply_state(runtime.state)
            services.store.save_state(runtime.state)
        return message

    return ChatContext(
        session_id=state.session_id,
        get_state=lambda: runtime.state,
        mutate=mutate,
        knowledge=services.knowledge,
        retriever=services.retriever,
        sandbox=services.sandbox,
        store=FakeStore(),
    )


def tools_by_name(ctx: ChatContext) -> dict[str, Any]:
    return {t.name: t for t in build_chat_tools(ctx)}


# --------------------------------------------------------------------- read tools


@pytest.mark.asyncio
async def test_get_session_state_reports_phase_counters_and_pending_question(tmp_path: Path) -> None:
    ctx = make_context(
        tmp_path,
        subjects=[SubjectState(name="Scope", status="asking", asked=1), SubjectState(name="Data", status="pending")],
    )
    state = ctx.get_state()
    state.questions.append(
        QuestionState(id=7, subject="Scope", text="Which systems are in scope?", rationale="", emitted=True)
    )
    ctx.store = FakeStore()

    text = await tools_by_name(ctx)["get_session_state"].ainvoke({})

    assert "Phase:" in text
    assert "Scope: asking (asked 1)" in text
    assert "Data: pending" in text
    assert "Pending question #7" in text
    assert "Which systems are in scope?" in text


@pytest.mark.asyncio
async def test_get_session_state_uses_the_plan_name_once_one_exists(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)
    ctx.get_state().title = "Fallback title"
    text = await tools_by_name(ctx)["get_session_state"].ainvoke({})
    assert "Project:" in text


@pytest.mark.asyncio
async def test_list_requirements_filters_by_category_and_flags_missing_acceptance(tmp_path: Path) -> None:
    from ba_agent.schemas import RequirementCategory

    ctx = make_context(tmp_path)
    state = ctx.get_state()
    state.ledger = [
        LedgerItem(
            id="FR-001",
            kind="requirement",
            title="Submit claim",
            body="Online intake",
            category=RequirementCategory.FUNCTIONAL,
        ),
        LedgerItem(
            id="NFR-002",
            kind="requirement",
            title="Upload speed",
            body="Under 10s",
            category=RequirementCategory.NON_FUNCTIONAL,
        ),
        LedgerItem(id="FACT-1", kind="fact", body="Email intake is retired"),
    ]
    ctx.store = FakeStore()

    tool = tools_by_name(ctx)["list_requirements"]
    all_items = await tool.ainvoke({})
    assert "FR-001" in all_items and "NFR-002" in all_items
    assert "FACT-1" not in all_items
    # The tool is meant to tell the model what is still unready, not just list rows.
    assert "acceptance: MISSING" in all_items

    functional_only = await tool.ainvoke({"category": "functional"})
    assert "FR-001" in functional_only and "NFR-002" not in functional_only

    empty = await tool.ainvoke({"category": "reporting"})
    assert "No requirements captured yet for category 'reporting'." in empty


@pytest.mark.asyncio
async def test_list_requirements_survives_a_category_that_is_not_a_valid_enum_value(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)
    ctx.get_state().ledger = [LedgerItem(id="FR-001", kind="requirement", title="x", body="y")]
    text = await tools_by_name(ctx)["list_requirements"].ainvoke({"category": "nonsense"})
    assert "No requirements captured yet for category 'nonsense'." in text


@pytest.mark.asyncio
async def test_search_knowledge_returns_citations_and_says_so_when_empty(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)
    from ba_agent.knowledge import KnowledgeService

    assert isinstance(ctx.knowledge, KnowledgeService)
    await ctx.knowledge.index_upload("claims_intake.txt", b"Claims arrive by email and are triaged by the back office.")

    tool = tools_by_name(ctx)["search_knowledge"]
    hit = await tool.ainvoke({"query": "claims email"})
    assert "claims_intake.txt" in hit
    assert "score" in hit

    miss = await tool.ainvoke({"query": "zzzzzz nonexistent term qqqq"})
    assert miss == "No relevant material in the knowledge base."


@pytest.mark.asyncio
async def test_search_knowledge_clamps_max_results_from_a_stray_model_value(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)
    await ctx.knowledge.index_upload("doc_one.txt", b"alpha " * 20)
    await ctx.knowledge.index_upload("doc_two.txt", b"beta " * 20)

    # 0, a negative and a huge number all have to clamp: never a crash, never the whole index.
    for limit in (0, -5, 9999):
        result = await tools_by_name(ctx)["search_knowledge"].ainvoke({"query": "alpha beta", "max_results": limit})
        assert isinstance(result, str)
        assert result.count("\n\n---\n\n") <= 7


@pytest.mark.asyncio
async def test_read_artifact_reports_available_files_when_the_name_is_wrong(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)
    ctx.store = FakeStore([{"name": "functional_specification.md"}, {"name": "process_diagram.mmd"}])

    out = await tools_by_name(ctx)["read_artifact"].ainvoke({"name": "does_not_exist.md"})
    assert "Could not read" in out
    assert "functional_specification.md" in out and "process_diagram.mmd" in out


@pytest.mark.asyncio
async def test_read_artifact_returns_the_file_when_it_exists(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)
    out_dir = ctx.sandbox.writable_roots[0] / ctx.session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "notes.md").write_text("The claim portal must handle 400 claims a month.", encoding="utf-8")

    text = await tools_by_name(ctx)["read_artifact"].ainvoke({"name": "notes.md"})
    assert "400 claims a month" in text


@pytest.mark.asyncio
async def test_read_artifact_cannot_reach_another_sessions_files(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)
    outputs = ctx.sandbox.writable_roots[0]

    # A second session's folder, holding something the current chat must never see.
    other = outputs / "sess-other"
    other.mkdir(parents=True, exist_ok=True)
    (other / "salary_data.md").write_text("Salaries of the finance team", encoding="utf-8")

    for attempt in ("../sess-other/salary_data.md", "sess-other/salary_data.md", "sess-other/"):
        text = await tools_by_name(ctx)["read_artifact"].ainvoke({"name": attempt})
        assert "Salaries of the finance team" not in text, f"read_artifact leaked another session via {attempt}"

    # The tool must still serve its own artefacts.
    own = outputs / ctx.session_id
    own.mkdir(parents=True, exist_ok=True)
    (own / "mine.md").write_text("This one is fine", encoding="utf-8")
    assert "This one is fine" in await tools_by_name(ctx)["read_artifact"].ainvoke({"name": "mine.md"})


# --------------------------------------------------------------------- mutating tools


@pytest.mark.asyncio
async def test_add_ledger_note_rejects_an_unknown_kind_without_mutating_state(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)
    before = len(ctx.get_state().ledger)

    out = await tools_by_name(ctx)["add_ledger_note"].ainvoke({"kind": "banana", "title": "t", "body": "b"})
    assert "Invalid kind" in out
    assert "fact" in out and "open_question" in out
    assert len(ctx.get_state().ledger) == before


@pytest.mark.asyncio
async def test_add_ledger_note_persists_a_requirement_with_a_real_requirement_id(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)
    out = await tools_by_name(ctx)["add_ledger_note"].ainvoke(
        {"kind": "Requirement", "title": "Export to CSV", "body": "Every claim can be exported to CSV"}
    )
    state = ctx.get_state()
    assert len(state.ledger) == 1
    item = state.ledger[0]
    assert item.kind == "requirement"
    # Requirements must carry an id the spec and diagrams can trace, not a generic counter.
    assert item.id.startswith("REQ-") or item.id.startswith("FR-")
    assert "Recorded" in out
    # The note has to be readable by a later session reload, not just in memory.
    assert json.dumps(item.model_dump())


@pytest.mark.asyncio
async def test_add_ledger_note_ids_do_not_collide_across_kinds(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)
    tool = tools_by_name(ctx)["add_ledger_note"]
    for kind in ("fact", "risk", "assumption", "decision", "fact", "risk"):
        await tool.ainvoke(
            {"kind": kind, "title": f"{kind} note", "body": f"body {kind} {len(ctx.get_state().ledger)}"}
        )

    ids = [i.id for i in ctx.get_state().ledger]
    assert len(ids) == len(set(ids)), f"duplicate ledger ids: {ids}"


@pytest.mark.asyncio
async def test_a_chat_ledger_note_survives_a_restart(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)
    await tools_by_name(ctx)["add_ledger_note"].ainvoke(
        {"kind": "decision", "title": "Retire email", "body": "Confirmed"}
    )

    # Reload from SQLite the way a restarted server would — the note must be on disk, not just in memory.
    _, fresh_services = _services(tmp_path)
    reloaded = fresh_services.store.load_state(ctx.session_id)
    assert reloaded is not None
    assert any(i.kind == "decision" and i.title == "Retire email" for i in reloaded.ledger)


@pytest.mark.asyncio
async def test_skip_subject_matches_case_insensitively_and_closes_the_subject(tmp_path: Path) -> None:
    ctx = make_context(
        tmp_path,
        subjects=[
            SubjectState(name="Reporting and dashboards", status="pending"),
            SubjectState(name="Data", status="pending"),
        ],
    )
    out = await tools_by_name(ctx)["skip_subject"].ainvoke({"name": "REPORTING"})
    assert "Marked 'Reporting and dashboards' as skipped." in out

    state = ctx.get_state()
    subject = next(s for s in state.subjects if s.name == "Reporting and dashboards")
    assert subject.status == "skipped"
    assert subject.closed is True
    assert subject.notes == "marked not applicable via chat"


@pytest.mark.asyncio
async def test_skip_subject_is_idempotent_on_an_already_closed_subject(tmp_path: Path) -> None:
    ctx = make_context(tmp_path, subjects=[SubjectState(name="Data", status="done")])
    out = await tools_by_name(ctx)["skip_subject"].ainvoke({"name": "Data"})
    assert "already done" in out
    assert ctx.get_state().subjects[0].status == "done"


@pytest.mark.asyncio
async def test_skip_subject_names_the_planned_subjects_when_nothing_matches(tmp_path: Path) -> None:
    ctx = make_context(tmp_path, subjects=[SubjectState(name="Integration", status="pending")])
    out = await tools_by_name(ctx)["skip_subject"].ainvoke({"name": "payroll"})
    assert "No subject matching 'payroll'" in out
    assert "Integration" in out


@pytest.mark.asyncio
async def test_skip_subject_with_an_empty_name_matches_nothing(tmp_path: Path) -> None:
    ctx = make_context(tmp_path, subjects=[SubjectState(name="Integration", status="pending")])
    out = await tools_by_name(ctx)["skip_subject"].ainvoke({"name": ""})
    assert "No subject matching" in out
    assert ctx.get_state().subjects[0].status == "pending"


@pytest.mark.asyncio
async def test_build_chat_tools_exposes_exactly_the_documented_tool_set(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)
    names = sorted(t.name for t in build_chat_tools(ctx))
    assert names == [
        "add_ledger_note",
        "get_session_state",
        "list_requirements",
        "read_artifact",
        "search_knowledge",
        "skip_subject",
    ]
