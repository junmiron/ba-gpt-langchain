"""Phase 3 persistence: transcript ingestion, the lessons feedback loop, the settings store and the v2 migration."""

from __future__ import annotations

import asyncio
import sqlite3

from ba_agent.prompts import render
from ba_agent.roles.analysis import ingest_transcript
from ba_agent.schemas import Phase, RunStatus
from ba_agent.store import SCHEMA, SCHEMA_VERSION, Store
from ba_agent.workflow import WorkflowEngine

TRANSCRIPT = """
# Discovery call with the operations lead

## Product overview
The claims portal is the product: one place for brokers to submit and track claims.
The overview screen lists every claim with its status, and the product replaces the shared inbox.

## Scope
In scope: submission, attachments, status tracking, notifications.
Out of scope: adjudication, billing, the legacy spreadsheet. The scope boundaries are still being
negotiated with the policy team.

## Current process
Claims arrive by email and are entered manually into the policy system.
A claim is rejected if the attachments are missing or the policy number is wrong.

## Volumes
Around 400 claims per month, with a peak of 900 in December.
The current process takes about 40 minutes per claim.

## Integration
The policy system exposes a REST API. There is no SSO; users log in with a shared department account.

## Open items
Nobody knows who owns the retention policy for claim attachments.
"""

# A transcript that only talks about volumes — it must not silence the whole agenda.
THIN_TRANSCRIPT = """
# Quick call

## Volumes
Around 400 claims per month, with a peak of 900 in December.
Each claim takes about 40 minutes to process by hand.
Monthly volume is the only thing we discussed.
"""


# -- transcript ingestion ----------------------------------------------------


def test_transcript_is_chunked_and_triaged_into_the_ledger(services):
    state = services.store.create_session(brief="Claims portal", transcript=TRANSCRIPT)
    added, chunks = asyncio.run(ingest_transcript(services, state))

    assert added, "the transcript produced no ledger entries"
    assert chunks, "the transcript was not chunked"
    assert all(item.subject == "transcript" for item in added)
    assert any(item.kind == "requirement" for item in added)
    # The scripted gateway returns the same draft for every chunk; dedup must collapse it.
    assert sum(1 for item in added if item.kind == "requirement") == 1
    assert len(services.gateway.calls) >= 2, "the transcript was not chunked"


def test_transcript_mode_skips_the_agenda_round_for_covered_subjects(services):
    engine = WorkflowEngine(services)
    state = services.store.create_session(brief="Claims portal", transcript=TRANSCRIPT)
    state = asyncio.run(engine.advance(state))

    assert state.phase is not Phase.INTERVIEW
    assert state.subjects and all(subject.status == "covered" for subject in state.subjects)
    assert not [question for question in state.questions if question.purpose == "agenda"]
    assert any(item.subject == "transcript" for item in state.requirements())


def test_a_thin_transcript_does_not_close_subjects_it_never_mentioned(services):
    engine = WorkflowEngine(services)
    state = services.store.create_session(brief="Claims portal", transcript=THIN_TRANSCRIPT)
    state = asyncio.run(engine.advance(state))

    assert state.phase is Phase.INTERVIEW, "a transcript covering one topic must not skip the agenda"
    covered = [subject.name for subject in state.subjects if subject.status == "covered"]
    assert not covered, f"subjects marked covered with no evidence: {covered}"
    # The interview is running, not skipped: an agenda question is pending.
    pending = [question for question in state.questions if question.purpose == "agenda"]
    assert pending and state.current_question is not None


def test_coverage_attribution_is_evidence_based(services):
    from ba_agent.roles.analysis import transcript_covered_subjects
    from ba_agent.schemas import SubjectState

    subjects = [
        SubjectState(name="Scope In and Out", why="boundaries"),
        SubjectState(name="Volumes and Capacity", why="throughput"),
        SubjectState(name="Security", why="access control"),
    ]
    chunks = [
        "The scope is submission only; out of scope is adjudication.",
        "Scope boundaries are agreed with the policy team.",
        "About 400 claims a month, peaking at 900 in December.",
        "Volume per claim is roughly 40 minutes of manual work.",
        "Capacity planning depends on the December volume spike.",
    ]
    covered = transcript_covered_subjects(subjects, chunks, threshold=2)
    assert "Scope In and Out" in covered
    assert "Volumes and Capacity" in covered
    assert "Security" not in covered, "a subject never mentioned cannot be counted as answered"


def test_a_session_needs_a_brief_or_a_transcript(services):
    engine = WorkflowEngine(services)
    state = services.store.create_session(brief="", transcript="")
    state = asyncio.run(engine.advance(state))

    assert state.status is RunStatus.ERROR
    assert "brief" in (state.error or "").lower()


def test_transcript_text_survives_a_round_trip(services):
    state = services.store.create_session(brief="Claims portal", transcript=TRANSCRIPT, transcript_source="call.log")
    reloaded = services.store.load_state(state.session_id)
    assert reloaded is not None
    assert reloaded.transcript == TRANSCRIPT.strip()
    assert reloaded.transcript_source == "call.log"
    assert reloaded.brief == "Claims portal"


def test_an_empty_transcript_changes_nothing(services):
    engine = WorkflowEngine(services)
    state = services.store.create_session(brief="Claims portal", transcript="   ")
    state = asyncio.run(engine.advance(state))
    assert state.phase is Phase.INTERVIEW
    assert state.subjects and not any(subject.status == "covered" for subject in state.subjects)


# -- lessons --------------------------------------------------


def test_lessons_are_recorded_when_a_run_completes(services):
    from test_pipeline import drive

    state, _ = asyncio.run(drive(services))
    assert state.phase is Phase.DONE

    lessons = services.store.list_lessons()
    assert lessons, "a completed run recorded no lessons"
    assert all(lesson["kind"] in ("coverage", "gap") for lesson in lessons)
    assert services.store.count_lessons() >= len(lessons)
    # A lesson is about the process, not this project: it must not be a verbatim requirement.
    assert not any(lesson["text"].startswith("FR-") for lesson in lessons)
    # A lesson must not carry this project's ids or section names into the next run.
    assert not any("FR-00" in lesson["text"] or "Section '" in lesson["text"] for lesson in lessons)


def test_lessons_are_deduplicated_and_capped(services):
    store = services.store
    first = store.create_session(brief="first run").session_id
    second = store.create_session(brief="second run").session_id

    for _ in range(6):
        store.add_lesson(first, "coverage", "Non-functional requirements are usually under-specified.")
    store.add_lesson(second, "critic", "Traceability matrix missing.")

    # Repeats update one row instead of adding rows, so the store cannot grow unbounded.
    assert store.count_lessons() == 2
    assert store.list_lessons(limit=1)[0]["seen_count"] == 6
    assert store.add_lesson(second, "gap", "   ") == 0
    assert store.count_lessons() == 2
    assert [row["text"] for row in store.list_lessons(kinds=["critic"])] == ["Traceability matrix missing."]
    # The same wording under a different kind is a different lesson.
    assert store.count_lessons() == 2
    store.add_lesson(second, "gap", "Non-functional requirements are usually under-specified.")
    assert store.count_lessons() == 3
    # ...and a row is attributed to the run that last reinforced it.
    store.add_lesson(second, "coverage", "Non-functional requirements are usually under-specified.")
    assert store.count_lessons() == 3, "reinforcing an existing lesson adds no row"
    assert store.list_lessons(session_id=second, kinds=["coverage"])[0]["session_id"] == second
    assert store.list_lessons(session_id=first, kinds=["coverage"]) == []


def test_lessons_can_be_filtered_by_session(services):
    store = services.store
    a = store.create_session(brief="run a").session_id
    b = store.create_session(brief="run b").session_id
    store.add_lesson(a, "coverage", "Ask about volumes early.")
    store.add_lesson(b, "gap", "Confirm the integration owner.")

    assert [row["session_id"] for row in store.list_lessons(session_id=a)] == [a]
    assert store.count_lessons() == 2, "the total always counts the whole store"


def test_lessons_reach_the_plan_prompt(services):
    session_id = services.store.create_session(brief="a completed run").session_id
    services.store.add_lesson(session_id, "coverage", "Non-functional requirements are usually under-specified.")
    lessons = [row["text"] for row in services.store.list_lessons(limit=8)]

    system, _ = render("plan", brief="Claims portal", prior_lessons=lessons)
    assert "previous interviews" in system
    assert "under-specified" in system

    # No lessons -> the block is absent rather than an empty heading.
    empty_system, _ = render("plan", brief="Claims portal")
    assert "previous interviews" not in empty_system


# -- settings store -----------------------------------------------------------


def test_run_settings_round_trip_and_persist(services):
    from ba_agent.runtime import SessionManager

    services.store.set_setting("run_settings", {"interview": {"max_gap_rounds": 0}})
    manager = SessionManager(services)
    manager.restore_run_settings()
    assert manager.config.interview.max_gap_rounds == 0

    applied = manager.update_run_settings({"interview": {"max_total_questions": 12}})
    assert applied["interview"]["max_total_questions"] == 12
    assert manager.config.interview.max_total_questions == 12
    assert services.store.get_setting("run_settings")["interview"]["max_total_questions"] == 12


def test_run_settings_refuse_unknown_keys_and_bad_values(services):
    from ba_agent.runtime import SessionManager

    manager = SessionManager(services)
    before = manager.config.interview.max_subjects
    for patch in (
        {"server": {"port": 1}},
        {"interview": {"max_subjects": 0}},
        {"interview": {"not_a_setting": 5}},
    ):
        try:
            manager.update_run_settings(patch)
        except ValueError:
            continue
        raise AssertionError(f"{patch} should have been refused")
    assert manager.config.interview.max_subjects == before


# -- schema migration ---------------------------------------------------------


def test_v1_database_migrates_through_to_the_current_schema(tmp_path):
    """A database from an older build upgrades all the way without losing its sessions."""
    db = tmp_path / "v1.db"
    v1_lessons = """CREATE TABLE IF NOT EXISTS lessons (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL DEFAULT 'gap',
    text        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);"""
    v1_schema = SCHEMA.replace(
        """CREATE TABLE IF NOT EXISTS lessons (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL DEFAULT 'gap',
    text        TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    seen_count  INTEGER NOT NULL DEFAULT 1,
    updated_at  TEXT NOT NULL DEFAULT '',
    UNIQUE (kind, text)
);""",
        v1_lessons,
    )

    with sqlite3.connect(db) as conn:
        conn.executescript(v1_schema)
        conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
        conn.execute("INSERT INTO schema_version (version) VALUES (1)")
        conn.execute(
            "INSERT INTO sessions (id,title,brief,phase,status,awaiting,created_at,updated_at,state_json)"
            " VALUES ('old','Old run','brief','intake','idle','none','2026-01-01','2026-01-01','{}')"
        )
        # Pre-existing duplicates must collapse, not survive as separate rows.
        for i in range(3):
            conn.execute(
                "INSERT INTO lessons (session_id,kind,text,created_at) VALUES (?,?,?,?)",
                ("old", "coverage", "Ask about volumes early.", f"2026-01-0{i + 1}"),
            )

    store = Store(db)
    assert store.list_sessions() and store.list_sessions()[0]["id"] == "old"

    with sqlite3.connect(db) as check:
        assert check.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == SCHEMA_VERSION
        rows = check.execute("SELECT kind, text, seen_count FROM lessons").fetchall()
    assert rows == [("coverage", "Ask about volumes early.", 3)], rows

    assert store.add_lesson("old", "coverage", "Ask about volumes early.") > 0
    with sqlite3.connect(db) as check:
        assert check.execute("SELECT COUNT(*) FROM lessons").fetchone()[0] == 1
        assert check.execute("SELECT seen_count FROM lessons").fetchone()[0] == 4


def test_lessons_are_ranked_by_how_often_they_recurred(services):
    store = services.store
    session = store.create_session(brief="ranking test").session_id
    store.add_lesson(session, "gap", "Rare observation.")
    for _ in range(5):
        store.add_lesson(session, "coverage", "Very common observation.")

    ordered = [row["text"] for row in store.list_lessons(limit=5)]
    assert ordered[0] == "Very common observation."
    assert store.list_lessons(limit=1)[0]["seen_count"] == 5


def test_a_lesson_reinforced_by_a_new_run_is_attributed_to_that_run(services):
    store = services.store
    old = store.create_session(brief="older run").session_id
    new = store.create_session(brief="newer run").session_id
    store.add_lesson(old, "coverage", "Volumes are always under-specified.")
    store.add_lesson(new, "coverage", "Volumes are always under-specified.")

    assert [row["session_id"] for row in store.list_lessons(session_id=new)] == [new]
    assert store.list_lessons(session_id=old) == []
