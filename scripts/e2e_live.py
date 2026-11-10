"""Live acceptance pass against a running server: exercise every Phase 3 UI/API surface end to end.

Start a server first (offline demo needs no model)::

    python scripts/demo_offline.py --port 8769
    python scripts/e2e_live.py --base http://127.0.0.1:8769

Works against a live backend too (LM Studio or Strata, whichever `config.yaml` / `BA_PROVIDER` selects):
it drives sessions by answering whatever the engine asks, so it validates the same surfaces with real models.
"""

from __future__ import annotations

import argparse
import io
import json
import time
import urllib.error
import urllib.request
import zipfile

passes: list[tuple[str, str]] = []
fails: list[tuple[str, str]] = []

ANSWER = "About 400 claims a month, mostly from brokers, arriving by email today."


def call(base: str, method: str, path: str, payload=None) -> tuple[int, dict]:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        base + path, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read()
            return response.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw) if raw else {}
        except ValueError:
            return exc.code, {"detail": raw.decode("utf-8", "replace")[:200]}
    except OSError as exc:
        return 0, {"detail": f"network: {exc}"}


def check(condition: bool, label: str, detail: str = "") -> bool:
    (passes if condition else fails).append((label, detail))
    return condition


def drive(base: str, session_id: str, deadline_s: float = 1800.0, answer: str = ANSWER) -> dict:
    """Advance a session to completion over HTTP, answering whatever it asks.

    A real backend (LM Studio or Strata) keeps the session ``busy`` for minutes per phase, so the
    loop is bounded by wall-clock time, not a round count: a fixed number of rounds is enough for
    the offline demo but makes every live run look like a stalled interview.
    """
    deadline = time.monotonic() + deadline_s
    state = {}
    while time.monotonic() < deadline:
        state = call(base, "GET", f"/api/sessions/{session_id}")[1]
        if state.get("phase") == "done" or state.get("status") == "error":
            return state
        if state.get("busy") or state.get("status") == "running":
            time.sleep(1.0)
            continue
        question = state.get("pending_question")
        if state.get("awaiting") == "answer" and question:
            status, _ = call(
                base, "POST", f"/api/sessions/{session_id}/answer", {"question_id": question["id"], "answer": answer}
            )
            if status != 200:
                return state
        elif state.get("awaiting") == "approval":
            call(base, "POST", f"/api/sessions/{session_id}/approve", {"approved": True, "notes": ""})
        else:
            call(base, "POST", f"/api/sessions/{session_id}/resume")
        time.sleep(0.12)
    return call(base, "GET", f"/api/sessions/{session_id}")[1]


def bundle(base: str, session_id: str) -> zipfile.ZipFile:
    raw = urllib.request.urlopen(f"{base}/api/sessions/{session_id}/export", timeout=180).read()
    return zipfile.ZipFile(io.BytesIO(raw))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8769")
    parser.add_argument("--deadline-minutes", type=float, default=30.0, help="how long a live run may take")
    args = parser.parse_args()
    base = args.base.rstrip("/")
    deadline_s = args.deadline_minutes * 60

    code, health = call(base, "GET", "/api/health")
    if code == 0:
        print(f"cannot reach {base}: {health['detail']}")
        return 2
    check(code == 200 and health.get("status") == "ok", "health endpoint", str(health)[:120])

    transcript = (
        "Meeting with the claims team.\n"
        "Product overview: the portal replaces the shared inbox and the tracking spreadsheet; the overview screen "
        "lists every claim with its current status.\n"
        "Scope: in scope is submission, attachments, status tracking and notifications; out of scope is "
        "adjudication, billing and the legacy spreadsheet. The scope boundaries were agreed with the policy team.\n"
        "Intake: claims arrive by email and are re-keyed into a spreadsheet by hand.\n"
        "Volume: roughly 400 claims per month, 90% from brokers.\n"
        "SLA: adjusters want triage within 24 hours.\n"
        "Roles: broker, adjuster, team leader.\n"
    )

    # --- transcript mode ---------------------------------------------------
    created = call(
        base,
        "POST",
        "/api/sessions",
        {
            "brief": "Claims portal for brokers.",
            "title": "Transcript run",
            "transcript": transcript,
            "transcript_source": "kickoff call notes",
        },
    )[1]
    session_id = created.get("session", {}).get("id")
    if not check(bool(session_id), "transcript session created", str(created)[:120]):
        return 1

    check(
        created["session"].get("has_transcript") is True,
        "session reports transcript",
        str(created["session"].get("transcript_source")),
    )
    check(created["session"].get("transcript_source") == "kickoff call notes", "transcript source round-trips")

    detail = call(base, "GET", f"/api/sessions/{session_id}")[1]
    check("Meeting with the claims team" in (detail.get("transcript") or ""), "transcript text returned in detail")

    state = drive(base, session_id, deadline_s)
    check(state.get("phase") == "done", "transcript run reached done", f"{state.get('phase')}/{state.get('status')}")
    check(state.get("status") != "error", "no error status", str(state.get("error")))
    subjects = [subject["status"] for subject in state.get("subjects", [])]
    check("covered" in subjects, "transcript-covered subjects closed", str(subjects))
    check(all(status != "pending" for status in subjects), "no subject left dangling", str(subjects))
    agenda = [question for question in state.get("questions", []) if question.get("purpose") == "agenda"]
    check(
        len(agenda) < len(subjects),
        "transcript mode cut the agenda interview",
        f"{len(agenda)} agenda questions for {len(subjects)} subjects",
    )
    events = call(base, "GET", f"/api/sessions/{session_id}/events")[1]
    mode = next(
        (
            e["payload"]
            for e in events
            if e.get("type") == "session" and e.get("payload", {}).get("event") == "transcript_mode"
        ),
        None,
    )
    check(mode is not None, "coverage report emitted", str(mode)[:120])
    if mode:
        check(
            bool(mode["covered"]) and bool(mode["uncovered"]),
            "coverage report distinguishes answered from unanswered subjects",
            json.dumps(mode)[:200],
        )

    # --- deliverables ------------------------------------------------------
    zip_file = bundle(base, session_id)
    names = sorted(zip_file.namelist())
    check("functional_specification.pdf" in names, "PDF in bundle", str(names[:8]))
    check("deliverable.html" in names, "HTML in bundle")
    check("transcript.txt" in names, "transcript shipped as a deliverable")
    check(not any("_render" in name for name in names), "no render intermediates in bundle", str(names))
    if "pdf_qa.json" in names:
        qa = json.loads(zip_file.read("pdf_qa.json").decode("utf-8"))
        check(qa["ok"], "PDF QA passed", str(qa["warnings"]))
        check(qa["checks"]["pages"] > 0, "PDF has pages", str(qa["checks"]["pages"]))
        check(qa["checks"]["markdown_leaks"] == [], "no Markdown leakage", str(qa["checks"]["markdown_leaks"]))
        check(
            qa["checks"]["requirements_missing"] == [],
            "all requirement rows present",
            str(qa["checks"]["requirements_missing"]),
        )
        check(
            qa["checks"]["contents_links"] >= qa["checks"]["outline_entries"],
            "every outline entry is clickable",
            f"{qa['checks']['contents_links']} links / {qa['checks']['outline_entries']} entries",
        )
        check(
            (qa["checks"].get("band_cut_weight") or 0) <= 6,
            "diagram band cuts land in gaps",
            str(qa["checks"].get("band_cut_weight")),
        )

    # --- ledger editing on a finished run --------------------------------
    requirement = next((entry for entry in state.get("ledger", []) if entry["kind"] == "requirement"), None)
    if requirement:
        code, body = call(
            base,
            "POST",
            f"/api/sessions/{session_id}/ledger/{requirement['id']}",
            {"title": "Triage every claim within 24 hours"},
        )
        check(code == 200, "ledger edit accepted", str(code))
        check(body.get("deliverables_rebuilt") is True, "finished session reopened for rebuild", str(body)[:80])
        rebuilt = drive(base, session_id, deadline_s)
        check(
            "Triage every claim within 24 hours" in (rebuilt.get("spec_markdown") or ""),
            "spec regenerated from the corrected ledger",
        )
        if "pdf_qa.json" in bundle(base, session_id).namelist():
            qa2 = json.loads(bundle(base, session_id).read("pdf_qa.json").decode("utf-8"))
            check(qa2["ok"], "rebuilt deliverables still pass QA", str(qa2["warnings"]))
        code, body = call(
            base, "POST", f"/api/sessions/{session_id}/ledger/{requirement['id']}", {"priority": "banana"}
        )
        check(code in (400, 422), "invalid priority rejected", f"{code} {str(body)[:80]}")

    # --- settings ----------------------------------------------------------
    code, applied = call(
        base,
        "PATCH",
        "/api/settings",
        {
            "interview": {"max_total_questions": 30},
            "diagram": {"emit_png": True},
            "knowledge": {"heading_aware": False},
        },
    )
    check(code == 200, "settings patch accepted", str(code))
    check(applied.get("settings", {}).get("interview", {}).get("max_total_questions") == 30, "budget persisted")
    check(
        applied.get("settings", {}).get("diagram", {}).get("emit_png") is True,
        "emit_png persisted",
        str(applied.get("settings", {}).get("diagram")),
    )
    check(applied.get("settings", {}).get("knowledge", {}).get("heading_aware") is False, "boolean persisted")

    for patch, label in [
        ({"interview": {"max_total_questions": 500}}, "out-of-range budget rejected"),
        ({"knowledge": {"docs_dir": "../escape"}}, "docs_dir escape rejected"),
        ({"knowledge": {"docs_dir": "C:\\Windows"}}, "absolute docs_dir rejected"),
        ({"nonsense": {"x": 1}}, "unknown settings section rejected"),
        ({"interview": {"max_subjects": "banana"}}, "non-numeric budget rejected"),
    ]:
        code, body = call(base, "PATCH", "/api/settings", patch)
        check(code in (400, 422), label, f"{code} {str(body)[:80]}")

    # --- lessons -----------------------------------------------------------
    lessons = call(base, "GET", "/api/lessons?limit=10")[1]
    check(lessons.get("total", 0) > 0, "lessons recorded after completed runs", str(lessons.get("total")))
    scoped = call(base, "GET", f"/api/lessons?limit=10&session_id={session_id}")[1]
    check(bool(scoped.get("lessons")), "lessons can be scoped to the run that produced them")
    check(
        all("FR-00" not in (row["text"] or "") for row in scoped.get("lessons", [])),
        "lessons are general, not project-specific",
        str([row["text"] for row in scoped.get("lessons", [])][:3]),
    )
    check(
        call(base, "GET", "/api/lessons?limit=5&session_id=does-not-exist")[1].get("lessons") == [],
        "unknown session filter returns nothing",
    )

    # --- pagination / busy / rename / delete -----------------------------
    page = call(base, "GET", "/api/sessions?limit=2&offset=0")[1]
    check(len(page) <= 2, "pagination respected", str(len(page)))
    check("busy" in page[0], "busy flag in session rows")

    call(base, "PATCH", f"/api/sessions/{session_id}", {"title": "Renamed claims run"})
    check(call(base, "GET", f"/api/sessions/{session_id}")[1].get("title") == "Renamed claims run", "rename works")
    check(call(base, "PATCH", f"/api/sessions/{session_id}", {"title": ""})[0] in (400, 422), "empty rename rejected")

    throwaway = call(base, "POST", "/api/sessions", {"brief": "Throwaway run", "title": "Delete me"})[1]
    sid2 = throwaway["session"]["id"]
    check(call(base, "DELETE", f"/api/sessions/{sid2}")[0] == 200, "delete succeeds")
    check(call(base, "DELETE", f"/api/sessions/{sid2}")[0] == 404, "second delete is 404")

    print(f"{len(passes)} passed, {len(fails)} failed\n")
    for label, detail in passes:
        print(f"  ok    {label}" + (f"  ({detail})" if detail else ""))
    for label, detail in fails:
        print(f"  FAIL  {label}  {detail}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
