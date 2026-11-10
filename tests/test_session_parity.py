"""Phase 3 UI-parity API: rename, delete, ledger editing, export bundle, pagination."""

from __future__ import annotations

import io
import json
import socket
import threading
import time
import zipfile
from types import SimpleNamespace

import httpx
import pytest
import uvicorn

from ba_agent import server as server_package
from ba_agent.knowledge import KnowledgeService
from ba_agent.server.app import create_app
from ba_agent.services import Services
from conftest import FakeGateway

TRANSCRIPT_TEXT = """
# Discovery call with the operations lead

Claims arrive by email and are entered manually.
Around 400 claims per month, peak 900 in December.
The policy system exposes a REST API.
"""


BRIEF = (
    "We need a portal for insurance brokers to submit and track claims. Today everything is email. "
    "About 300 brokers, peak in winter, must integrate with the existing policy system over REST."
)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture()
def api(config, monkeypatch):
    """Live uvicorn with the scripted gateway, same harness as test_api.py."""
    real_create = Services.create
    gateway = FakeGateway()

    def create_with_fake(cfg=None):
        services = real_create(cfg or config)
        services.gateway = gateway  # type: ignore[assignment]
        services.knowledge = KnowledgeService(services.store, gateway, config)  # type: ignore[arg-type]
        return services

    monkeypatch.setattr(server_package.app, "Services", SimpleNamespace(create=create_with_fake))

    app = create_app(config)
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", lifespan="on", access_log=False)
    )
    thread = threading.Thread(target=server.run, name="ba-parity-server", daemon=True)
    thread.start()

    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=20.0) as client:
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                if client.get("/api/health").status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.05)
        else:  # pragma: no cover - defensive
            raise AssertionError("test server did not come up")
        # Expose the scripted gateway so a test can read the exact prompt a role received.
        client.gateway = gateway  # type: ignore[attr-defined]
        yield client

    server.should_exit = True
    thread.join(timeout=10)


def _create(
    client, brief: str = "Claims portal for brokers, email today, 300 brokers, integrate with policy REST API"
) -> str:
    response = client.post("/api/sessions", json={"brief": brief})
    assert response.status_code == 200
    return response.json()["session"]["id"]


def test_rename_session_updates_title(api):
    client = api
    session_id = _create(client)
    response = client.patch(f"/api/sessions/{session_id}", json={"title": "Claims Portal v2"})
    assert response.status_code == 200
    listed = client.get("/api/sessions").json()
    assert any(row["title"] == "Claims Portal v2" for row in listed)
    # the in-memory runtime state must match, otherwise the header shows the stale title
    assert client.get(f"/api/sessions/{session_id}").json()["title"] == "Claims Portal v2"


def test_rename_rejects_empty_and_unknown(api):
    client = api
    session_id = _create(client)
    assert client.patch(f"/api/sessions/{session_id}", json={"title": "   "}).status_code == 400
    assert client.patch("/api/sessions/does-not-exist", json={"title": "x"}).status_code == 400


def test_session_list_paginates_with_total_count(api):
    client = api
    for index in range(5):
        _create(client, brief=f"Project {index}: brief with enough detail to plan, brokers and policy system")
    response = client.get("/api/sessions?limit=2&offset=0")
    assert response.status_code == 200
    assert len(response.json()) == 2
    assert int(response.headers["X-Total-Count"]) == 5
    second = client.get("/api/sessions?limit=2&offset=2").json()
    assert len(second) == 2
    ids = {r["id"] for r in response.json()} | {r["id"] for r in second}
    assert len(ids) == 4, "offset must not repeat rows"


def test_ledger_edit_persists_and_invalidates_the_draft_spec(api):
    client = api
    session_id = _create(client)
    snapshot = client.get(f"/api/sessions/{session_id}").json()
    assert snapshot["ledger"], "the scripted engine should have produced ledger entries"
    entry = snapshot["ledger"][0]
    response = client.post(
        f"/api/sessions/{session_id}/ledger/{entry['id']}",
        json={"title": "Corrected title", "priority": "could", "acceptance": ["measurable criterion"]},
    )
    assert response.status_code == 200
    assert response.json()["entry"]["title"] == "Corrected title"
    assert response.json()["entry"]["priority"] == "could"
    refreshed = client.get(f"/api/sessions/{session_id}").json()
    assert any(i["id"] == entry["id"] and i["title"] == "Corrected title" for i in refreshed["ledger"])
    assert refreshed["spec_markdown"] == "", "editing the ledger must force the spec to be regenerated"
    events = client.get(f"/api/sessions/{session_id}/events").json()
    assert any(e["type"] == "ledger" and e["payload"].get("event") == "edited" for e in events)


def test_ledger_edit_rejects_an_invalid_priority(api):
    session_id = _create(api)
    entry = api.get(f"/api/sessions/{session_id}").json()["ledger"][0]
    response = api.post(
        f"/api/sessions/{session_id}/ledger/{entry['id']}", json={"priority": "definitely-not-a-priority"}
    )
    assert response.status_code in (400, 422)
    assert api.get(f"/api/sessions/{session_id}").json()["ledger"][0]["priority"] == entry["priority"]


def test_ledger_edit_unknown_id_is_404(api):
    client = api
    session_id = _create(client)
    assert client.post(f"/api/sessions/{session_id}/ledger/FR-999", json={"title": "x"}).status_code == 404


def test_export_bundle_is_a_zip_of_the_session_outputs(api):
    session_id = _create(api)
    # Drive the interview to completion the way the UI does, so real deliverables exist to bundle.
    for _ in range(40):
        state = api.get(f"/api/sessions/{session_id}").json()
        if state["phase"] == "done":
            break
        if state["busy"]:
            time.sleep(0.2)
            continue
        if state["awaiting"] == "answer" and state["pending_question"]:
            api.post(
                f"/api/sessions/{session_id}/answer",
                json={"question_id": state["pending_question"]["id"], "answer": "Around 400 claims per month."},
            )
        elif state["awaiting"] == "approval":
            api.post(f"/api/sessions/{session_id}/approve", json={"approved": True})
        time.sleep(0.1)
    assert api.get(f"/api/sessions/{session_id}").json()["phase"] == "done"

    response = api.get(f"/api/sessions/{session_id}/export")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    names = set(zipfile.ZipFile(io.BytesIO(response.content)).namelist())
    assert {"functional_specification.md", "metadata.json", "deliverable.html", "interview.jsonl"} <= names
    # the zip is re-derivable: every file is readable and non-empty
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    assert archive.read("functional_specification.md").startswith(b"#")
    assert json.loads(archive.read("metadata.json"))["session_id"] == session_id
    # The README tells users to check metadata.json for which renderer produced the PDF.
    metadata = json.loads(archive.read("metadata.json"))
    assert metadata["pdf_engine"] in {"typst", "reportlab"}, metadata["pdf_engine"]
    assert metadata["pdf_available"] is True


def test_export_unknown_session_is_404(api):
    client = api
    assert client.get("/api/sessions/nope/export").status_code == 404


def test_settings_round_trip_through_the_api(api):
    settings = api.get("/api/settings").json()
    assert "interview" in settings["settings"]
    assert "max_total_questions" in settings["editable"]["interview"]

    response = api.patch(
        "/api/settings",
        json={"interview": {"max_total_questions": 15, "max_gap_rounds": 1}},
    )
    assert response.status_code == 200, response.text
    assert response.json()["settings"]["interview"]["max_total_questions"] == 15
    assert api.get("/api/settings").json()["settings"]["interview"]["max_total_questions"] == 15


def test_settings_reject_out_of_range_and_unknown_keys(api):
    response = api.patch("/api/settings", json={"interview": {"max_total_questions": 0}})
    assert response.status_code == 400
    response = api.patch("/api/settings", json={"interview": {"nope": 1}})
    assert response.status_code == 400
    response = api.patch("/api/settings", json={"server": {"port": 1}})
    assert response.status_code == 400
    # The untouched default is still in effect.
    assert api.get("/api/settings").json()["settings"]["interview"]["max_total_questions"] == 40


def test_settings_patch_needs_a_same_origin_request(api):
    response = api.patch(
        "/api/settings",
        json={"interview": {"max_gap_rounds": 3}},
        headers={"origin": "http://evil.example"},
    )
    assert response.status_code == 403


def test_a_session_can_be_created_from_a_transcript_alone(api):
    response = api.post(
        "/api/sessions",
        json={"transcript": "Claims arrive by email. Around 400 per month. Peak in December."},
    )
    assert response.status_code == 200, response.text
    session = response.json()["session"]
    assert session["id"]
    detail = api.get(f"/api/sessions/{session['id']}").json()
    assert detail["has_transcript"] is True
    assert detail["transcript"].startswith("Claims arrive by email")


def test_a_session_with_neither_brief_nor_transcript_is_rejected(api):
    assert api.post("/api/sessions", json={"brief": "", "transcript": ""}).status_code == 400


def test_a_transcript_run_ships_the_transcript_in_the_bundle(api):
    response = api.post(
        "/api/sessions",
        json={"transcript": TRANSCRIPT_TEXT},
    )
    session_id = response.json()["session"]["id"]

    for _ in range(40):
        state = api.get(f"/api/sessions/{session_id}").json()
        if state["phase"] == "done":
            break
        if state["busy"]:
            time.sleep(0.2)
            continue
        if state["awaiting"] == "answer" and state["pending_question"]:
            api.post(
                f"/api/sessions/{session_id}/answer",
                json={"question_id": state["pending_question"]["id"], "answer": "Around 400 claims per month."},
            )
        elif state["awaiting"] == "approval":
            api.post(f"/api/sessions/{session_id}/approve", json={"approved": True})
        time.sleep(0.1)

    names = set(zipfile.ZipFile(io.BytesIO(api.get(f"/api/sessions/{session_id}/export").content)).namelist())
    assert "transcript.txt" in names
    assert "Discovery call" in zipfile.ZipFile(io.BytesIO(api.get(f"/api/sessions/{session_id}/export").content)).read(
        "transcript.txt"
    ).decode("utf-8")


def test_a_finished_run_reports_what_it_was_built_with(api):
    """The artifacts tab needs the PDF engine, the ladder and the call count — not just a file list."""
    session_id = api.post("/api/sessions", json={"brief": BRIEF}).json()["session"]["id"]

    for _ in range(60):
        state = api.get(f"/api/sessions/{session_id}").json()
        if state["phase"] == "done":
            break
        if state["busy"]:
            time.sleep(0.2)
            continue
        if state["awaiting"] == "answer" and state["pending_question"]:
            api.post(
                f"/api/sessions/{session_id}/answer",
                json={"question_id": state["pending_question"]["id"], "answer": "Around 400 claims per month."},
            )
        elif state["awaiting"] == "approval":
            api.post(f"/api/sessions/{session_id}/approve", json={"approved": True})
        time.sleep(0.1)

    facts = api.get(f"/api/sessions/{session_id}").json()["run_facts"]
    assert facts["pdf_engine"] in {"typst", "reportlab"}, facts
    assert facts["pages"], "the QA report did not record a page count"
    assert facts["qa_ok"] is True, facts.get("qa_warnings")
    # The scripted gateway has no ladder, so the shape must be present and empty rather than absent;
    # tests/test_pdf_export.py covers a gateway that actually fills it.
    assert set(facts["structured_output"]) == {"rung_counts", "learned_rung"}, facts["structured_output"]
    # Where the time went, phase by phase — the only way to tell a slow interview from a slow writer.
    phases = facts["phase_metrics"]
    assert phases, "run_facts must report per-phase timing"
    assert "interview" in phases, phases
    assert phases["interview"]["wall_seconds"] >= 0 and phases["interview"]["steps"] > 0, phases["interview"]


def test_run_facts_survive_a_missing_metadata_file(api):
    """An in-progress session has no deliverable yet, so the panel must simply be empty — not crash."""
    session_id = api.post("/api/sessions", json={"brief": BRIEF}).json()["session"]["id"]
    assert api.get(f"/api/sessions/{session_id}").json()["run_facts"] == {}


def test_knowledge_directory_is_settable_and_contained(api):
    """`docs_dir` is editable, but escaping the project root is refused."""
    body = api.patch("/api/settings", json={"knowledge": {"docs_dir": "data/reference"}}).json()["settings"]
    assert body["knowledge"]["docs_dir"] == "data/reference"

    for attempt in ("../../etc", "/absolute/path", ""):
        response = api.patch("/api/settings", json={"knowledge": {"docs_dir": attempt}})
        assert response.status_code == 400, attempt
    assert api.get("/api/settings").json()["settings"]["knowledge"]["docs_dir"] == "data/reference"


def test_boolean_settings_round_trip(api):
    body = api.patch("/api/settings", json={"diagram": {"local_renderer": False}}).json()["settings"]
    assert body["diagram"]["local_renderer"] is False
    body = api.patch("/api/settings", json={"knowledge": {"heading_aware": True}}).json()["settings"]
    assert body["knowledge"]["heading_aware"] is True


def test_settings_expose_the_model_lanes(api):
    """The settings payload must carry the role lanes, not just the budgets."""
    response = api.get("/api/settings")
    assert response.status_code == 200
    models = response.json()["models"]
    assert "writer" in models["roles"], "model lanes must be listed for every role"
    assert models["provider"], "the active provider must be reported"


def test_a_finished_run_leaves_lessons_and_the_api_reports_them(api):
    """Phase 3 acceptance: run 1 records lessons, run 2 sees them, and the store stays general-purpose."""
    first = _create(api)
    for _ in range(40):
        state = api.get(f"/api/sessions/{first}").json()
        if state["phase"] == "done":
            break
        if state["busy"]:
            time.sleep(0.2)
            continue
        if state["awaiting"] == "answer" and state["pending_question"]:
            api.post(
                f"/api/sessions/{first}/answer",
                json={"question_id": state["pending_question"]["id"], "answer": "About 400 claims a month."},
            )
        elif state["awaiting"] == "approval":
            api.post(f"/api/sessions/{first}/approve", json={"approved": True})
        time.sleep(0.1)

    listing = api.get("/api/lessons")
    assert listing.status_code == 200
    body = listing.json()
    assert body["total"] >= 1, "a completed run must record at least one lesson"
    assert body["total"] == len(body["lessons"]), "duplicate lesson text must not be stored twice"
    assert not any("FR-00" in lesson["text"] for lesson in body["lessons"]), (
        "lessons must be general, not project-specific"
    )
    assert api.get("/api/lessons?kind=nonexistent").json()["lessons"] == []

    # The next run's plan prompt must literally contain the lessons recorded by the first.
    second = _create(api)
    for _ in range(40):
        state = api.get(f"/api/sessions/{second}").json()
        if state["phase"] == "done":
            break
        if state["busy"]:
            time.sleep(0.2)
            continue
        if state["awaiting"] == "answer" and state["pending_question"]:
            api.post(
                f"/api/sessions/{second}/answer",
                json={"question_id": state["pending_question"]["id"], "answer": "About 400 claims a month."},
            )
        elif state["awaiting"] == "approval":
            api.post(f"/api/sessions/{second}/approve", json={"approved": True})
        time.sleep(0.1)

    plan_prompts = [(system + "\n" + user) for role, _step, system, user in api.gateway.prompts if role == "plan"]
    assert plan_prompts, "the second run must run the planner role"
    # The last plan prompt is the second run's: it must carry what the first run taught the store.
    assert any(lesson["text"] in plan_prompts[-1] for lesson in body["lessons"]), (
        "lessons from the previous run must reach the next plan prompt"
    )


def test_editing_the_ledger_of_a_finished_run_rebuilds_the_deliverables(api):
    """A post-delivery correction must regenerate the PDF/HTML, not just blank the spec."""
    created = api.post("/api/sessions", json={"title": "Later fix", "brief": "Claims portal for brokers"}).json()
    session_id = created["session"]["id"]
    for _ in range(40):
        state = api.get(f"/api/sessions/{session_id}").json()
        if state["phase"] == "done":
            break
        if state["busy"]:
            time.sleep(0.2)
            continue
        if state["awaiting"] == "answer" and state["pending_question"]:
            api.post(
                f"/api/sessions/{session_id}/answer",
                json={"question_id": state["pending_question"]["id"], "answer": "About 400 claims per month."},
            )
        elif state["awaiting"] == "approval":
            api.post(f"/api/sessions/{session_id}/approve", json={"approved": True})
        time.sleep(0.1)

    entry = api.get(f"/api/sessions/{session_id}").json()["ledger"][0]
    response = api.post(f"/api/sessions/{session_id}/ledger/{entry['id']}", json={"body": "Corrected by the BA."})
    assert response.status_code == 200
    assert response.json()["deliverables_rebuilt"] is True

    for _ in range(40):
        state = api.get(f"/api/sessions/{session_id}").json()
        if state["phase"] == "done" and not state["busy"]:
            break
        time.sleep(0.15)

    state = api.get(f"/api/sessions/{session_id}").json()
    assert state["phase"] == "done"
    assert "Corrected by the BA." in state["ledger"][0]["body"]
    assert state["spec_markdown"], "the spec must be reassembled from the corrected ledger"
    bundle = api.get(f"/api/sessions/{session_id}/export")
    assert bundle.status_code == 200
    assert len(bundle.content) > 1000


def test_deleting_a_session_removes_its_artefact_directory(api, config):
    """The DB row and the outputs folder must die together, or the outputs tree accumulates orphans."""
    session_id = _create(api)
    for _ in range(40):
        state = api.get(f"/api/sessions/{session_id}").json()
        if state["phase"] == "done":
            break
        if state["busy"]:
            time.sleep(0.2)
            continue
        if state["awaiting"] == "answer" and state["pending_question"]:
            api.post(
                f"/api/sessions/{session_id}/answer",
                json={"question_id": state["pending_question"]["id"], "answer": "About 400 claims per month."},
            )
        elif state["awaiting"] == "approval":
            api.post(f"/api/sessions/{session_id}/approve", json={"approved": True})
        time.sleep(0.1)

    directory = config.session_dir(session_id)
    assert directory.exists(), "a completed run should have written its bundle"

    assert api.delete(f"/api/sessions/{session_id}").status_code == 200
    assert not directory.exists(), "deleting a session must delete its deliverables"
    assert api.get(f"/api/sessions/{session_id}").status_code == 404
    assert api.delete(f"/api/sessions/{session_id}").status_code == 404, "deleting twice must not crash"
