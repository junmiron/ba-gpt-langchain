#!/usr/bin/env python3
"""Real end-to-end test against LM Studio — DESIGN §8's manual checklist, automated.

Starts its own server (the **real** gateway, no fakes) on a free port, drives one full interview to
completion with canned stakeholder answers, then verifies every artifact is downloadable through the API.

It only uses what is *already loaded* in LM Studio — by default it picks the first loaded chat model (for
example the model currently serving Bionic), so no extra VRAM and no model switching are needed::

    python scripts/e2e_lmstudio.py                          # whatever chat model is loaded right now
    python scripts/e2e_lmstudio.py --model qwen3.8-27b      # pin a specific id
    python scripts/e2e_lmstudio.py --request-changes-first  # also exercise the "Request changes" path

Exit code 0 = run reached ``done`` and every artifact downloaded; anything else prints what failed.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import httpx  # noqa: E402
import uvicorn  # noqa: E402

from ba_agent.cli import configure_console  # noqa: E402
from ba_agent.config import get_config  # noqa: E402
from ba_agent.server.app import create_app  # noqa: E402


BRIEF = (
    "We need a portal for insurance brokers to submit and track claims. Today everything is email. "
    "About 300 brokers, peak in winter, must integrate with the existing policy system over REST."
)

# Keyword-routed canned stakeholder answers: specific enough that triage can extract real requirements,
# generic enough to fit whatever the interviewer asks about this domain.
ANSWER_RULES: list[tuple[tuple[str, ...], str]] = [
    (("who", "actor", "user group", "role", "stakeholder"),
     "Three actor groups: brokers (about 300 of them) submit and track their own claims; internal claims "
     "adjusters review, approve or reject claims and assign follow-ups; managers get read-only dashboards. "
     "Brokers sign in with the credentials they already use for the policy system."),
    (("how many", "volume", "peak", "scale", "per month", "concurrent"),
     "Roughly 400 claims per month, spiking to about three times that in winter (around 1,200). We expect "
     "about 50 concurrent users at peak. Each claim can carry up to 10 attachments."),
    (("integrat", "api", "policy system", "rest"),
     "The portal must call the existing policy system over REST: validate the broker's policy number when a "
     "claim is opened, and push approved claims back for payment processing. The API is documented; it "
     "rate-limits to 10 requests per second per client."),
    (("attachment", "upload", "document", "file"),
     "Attachments are PDFs, images (JPG/PNG) and Office documents, up to 25 MB each, max 10 files per claim. "
     "Uploads must survive a dropped connection — brokers often work from poor connections."),
    (("track", "status", "workflow", "state", "lifecycle"),
     "A claim moves through: submitted, under review, approved or rejected, then payment requested and "
     "closed when approved. Brokers need to see the current state plus a history of every change; adjusters "
     "are the ones who move claims between states."),
    (("security", "auth", "access", "permission"),
     "Brokers can only see their own claims; adjusters see all claims in their region; managers see "
     "everything read-only. Sessions must time out after 15 minutes of inactivity, and every state change is "
     "logged with user and timestamp."),
    (("report", "export", "dashboard"),
     "Managers need a monthly report: claims by status, average handling time, rejection reasons — exportable "
     "to CSV. No real-time analytics needed for v1."),
    (("notification", "email", "alert"),
     "Email notifications when a claim changes state (to the broker) and when a new claim arrives (to the "
     "on-duty adjuster). No SMS or push in v1."),
    (("priority", "deadline", "timeline", "mvp", "phase"),
     "MVP target is three months: submission, tracking and the policy-system integration are must-haves. "
     "Reports and dashboards can wait for phase 2. The email intake process runs in parallel until cutover."),
]

DEFAULT_ANSWER = (
    "Keep it simple: brokers submit a claim with attachments, adjusters review it against the policy data "
    "from the existing REST API, and everyone can see the current status at any time. Anything not covered "
    "here can be decided in phase 2."
)

CONFIRM_ANSWER = "Yes, that matches what we do today."


def canned_answer(question: dict) -> str:
    text = (question.get("text") or "").lower()
    if question.get("kind") == "confirm" or text.startswith(("is ", "are ", "do you confirm", "correct?")):
        return CONFIRM_ANSWER
    for keywords, answer in ANSWER_RULES:
        if any(k in text for k in keywords):
            return answer
    return DEFAULT_ANSWER


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="chat model id (default: first loaded chat model)")
    parser.add_argument("--resume", metavar="SESSION_ID",
                        help="drive an existing session instead of creating a new one")
    parser.add_argument("--brief", default=BRIEF)
    parser.add_argument("--request-changes-first", action="store_true",
                        help="send one 'Request changes' at the approval gate before approving")
    parser.add_argument("--deadline-minutes", type=float, default=45.0)
    args = parser.parse_args()

    config = get_config()
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(config), host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, name="e2e-server", daemon=True).start()

    base_url = f"http://127.0.0.1:{port}"
    client = httpx.Client(base_url=base_url, timeout=30.0)
    deadline = time.time() + args.deadline_minutes * 60

    try:
        # -- wait for the server -------------------------------------------------
        while time.time() < deadline:
            try:
                if client.get("/api/health").status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        else:
            log("FAIL: server did not come up")
            return 1

        health = client.get("/api/health").json()
        if health["status"] != "ok":
            log(f"FAIL: LM Studio unreachable ({health['status']})")
            return 1

        # -- pick the model ------------------------------------------------------
        model_id = args.model
        if not model_id:
            catalog = client.get("/api/models").json()
            loaded = [m["id"] for m in catalog["chat_models"] if m.get("state") == "loaded"]
            if not loaded:
                log("FAIL: no chat model is loaded in LM Studio — pass --model or load one first")
                return 1
            model_id = loaded[0]
        log(f"server up on :{port}; using model {model_id!r}")

        # -- create the session (per-session routing, global settings untouched) -
        if args.resume:
            sid = args.resume
            response = client.get(f"/api/sessions/{sid}")
            if response.status_code != 200:
                log(f"FAIL: unknown session {sid}")
                return 1
            state = response.json()
            log(f"resuming session {sid} (phase={state['phase']} status={state['status']}) — "
                f"using its stored model routing")
            response = client.post(f"/api/sessions/{sid}/resume")
            if response.status_code != 200:
                log(f"FAIL: resume -> {response.status_code} {response.text}")
                return 1
        else:
            response = client.post("/api/sessions", json={
                "brief": args.brief, "title": f"E2E {time.strftime('%Y-%m-%d %H:%M')}",
                "model_overrides": {"default": model_id},
            })
            if response.status_code != 200:
                log(f"FAIL: create session -> {response.status_code} {response.text}")
                return 1
            sid = response.json()["session"]["id"]
            log(f"session {sid} created")

        answered: set[int] = set()
        requested_changes = False
        last_phase = ""
        outcome: str | None = None

        # -- drive the run ---------------------------------------------------------
        while time.time() < deadline:
            state = client.get(f"/api/sessions/{sid}").json()
            phase, status, awaiting = state["phase"], state["status"], state["awaiting"]
            if (phase, status) != last_phase:
                log(f"phase={phase} status={status} awaiting={awaiting}")
                last_phase = (phase, status)

            question = state.get("pending_question")
            if awaiting == "answer" and question and question["id"] not in answered:
                answer = canned_answer(question)
                log(f"Q{question['id']} ({question.get('subject', '?')}): {question['text'][:100]}...")
                response = client.post(f"/api/sessions/{sid}/answer",
                                       json={"question_id": question["id"], "answer": answer})
                if response.status_code == 200:
                    answered.add(question["id"])
                    log(f"    A: {answer[:100]}...")
                elif response.status_code != 409:  # 409 = already resolved, harmless race
                    log(f"FAIL: answer -> {response.status_code} {response.text}")
                    return 1

            if awaiting == "approval":
                if args.request_changes_first and not requested_changes:
                    requested_changes = True
                    response = client.post(f"/api/sessions/{sid}/approve", json={
                        "approved": False,
                        "notes": "Please add the winter peak (3x normal volume) to the non-functional requirements."})
                    log("approval gate: sent 'Request changes'")
                else:
                    response = client.post(f"/api/sessions/{sid}/approve", json={"approved": True})
                    log("approval gate: approved summary")
                if response.status_code != 200:
                    log(f"FAIL: approve -> {response.status_code} {response.text}")
                    return 1

            if status == "done":
                outcome = "done"
                break
            if status in ("error", "cancelled"):
                outcome = f"{status}: {state.get('error')}"
                break
            time.sleep(1.0)

        if outcome is None:
            log(f"FAIL: deadline ({args.deadline_minutes} min) reached; last phase={last_phase}")
            return 1
        log(f"run finished -> {outcome}")
        if not str(outcome).startswith("done"):
            events = client.get(f"/api/sessions/{sid}/events", params={"limit": 200}).json()
            for event in events[-15:]:
                log(f"  last event {event.get('type')}: {str(event.get('payload'))[:200]}")
            return 1

        # -- verify artifacts through the API --------------------------------------
        detailed = client.get(f"/api/sessions/{sid}").json()
        artifacts = detailed["artifacts"]
        log(f"artifacts registered: {[a['name'] for a in artifacts]}")
        failures = []
        for artifact in artifacts:
            response = client.get(f"/api/artifacts/{sid}/{artifact['name']}")
            ok = response.status_code == 200 and len(response.content) > 0
            log(f"  download {artifact['name']}: {response.status_code} ({len(response.content)} bytes)")
            if not ok:
                failures.append(artifact["name"])

        # -- event summary (ladder behaviour, warnings) -----------------------------
        events = client.get(f"/api/sessions/{sid}/events", params={"limit": 2000}).json()
        by_type: dict[str, int] = {}
        for event in events:
            by_type[event["type"]] = by_type.get(event["type"], 0) + 1
        log(f"event counts: {by_type}")
        warnings = [e for e in events if e["type"] == "warning"]
        for event in warnings[:20]:
            log(f"  warning [{event.get('step', '')}]: {str(event.get('payload'))[:180]}")

        # -- on-disk deliverables ----------------------------------------------------
        out_dir = ROOT / config.paths.outputs_dir / sid
        spec_path = out_dir / "functional_specification.md"
        if spec_path.exists():
            spec_text = spec_path.read_text(encoding="utf-8")
            log(f"spec: {len(spec_text)} chars, appendix={'Appendix A' in spec_text}, "
                f"sections={spec_text.count(chr(10) + '## ')}")
        else:
            failures.append("functional_specification.md (missing on disk)")

        metadata_path = out_dir / "metadata.json"
        if metadata_path.exists():
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            log(f"metadata: {json.dumps(meta, ensure_ascii=False)[:400]}")

        if failures:
            log(f"E2E RESULT: FAIL — problems: {failures}")
            return 1
        log("E2E RESULT: PASS — full pipeline reached done; all artifacts downloadable")
        return 0
    finally:
        server.should_exit = True
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
