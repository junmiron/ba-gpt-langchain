#!/usr/bin/env python3
"""Offline end-to-end acceptance run: no model, no VRAM, no network.

Drives the real workflow engine through both elicitation modes and asserts the
Phase 2 + Phase 3 acceptance criteria on the files it writes::

    python scripts/e2e_offline.py
    python scripts/e2e_offline.py --keep     # leave the outputs on disk

Exit code 0 means every check passed; anything else prints the failing checks.
This is what CI cannot do (CI has no deliverable QA), so it is the gate before
shipping a bundle to a stakeholder.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from ba_agent.artifacts import pdf_engine  # noqa: E402
from ba_agent.artifacts.qa import audit_html, audit_pdf  # noqa: E402
from ba_agent.artifacts.typst import find_typst, svg_native_size  # noqa: E402
from ba_agent.config import AppConfig  # noqa: E402
from ba_agent.knowledge import KnowledgeService  # noqa: E402
from ba_agent.runtime import SessionManager  # noqa: E402
from ba_agent.services import Services  # noqa: E402
from conftest import FakeGateway  # noqa: E402

TRANSCRIPT = """
# Discovery call with the Head of Claims Operations — 2026-11-04

## Product overview
The product is a claims portal that replaces the shared inbox and the spreadsheet.
The overview screen shows every claim with its status; the product is used by intake
clerks and adjusters only.

## Scope
In scope: submission, attachments, status tracking, notifications.
Out of scope: adjudication, billing, the legacy spreadsheet. The boundaries of scope
were agreed with the policy team.

Brokers email us a PDF plus a spreadsheet. Three intake clerks re-key each claim by
hand, then an adjuster picks it up. Median turnaround today is six days; the target
is 24 hours.

Volumes: about 400 claims a month, 60-80 on the first business day. Peak day is the
3rd. Twelve percent of claims come back for rework.

The policy system (mainframe, SOAP) is the system of record. It exposes a nightly
batch interface; a real-time API is not available. Documents live in SharePoint.

Actors: broker (external), intake clerk, adjuster, policyholder, compliance officer.
Compliance needs an audit trail of every status change.

Rules today: claims above 5,000 EUR need a senior adjuster. Duplicate detection is
manual and misses about one in ten. GDPR applies - claimant data must not leave the EU.
"""

BRIEF = (
    "We need a portal for insurance brokers to submit and track claims. Today everything "
    "is email. About 300 brokers, peak in winter, must integrate with the existing policy "
    "system over REST."
)

EXPECTED_FILES = {
    "functional_specification.md",
    "functional_specification.pdf",
    "functional_specification.typ",
    "deliverable.html",
    "requirements_summary.md",
    "interview.jsonl",
    "metadata.json",
    "pdf_qa.json",
}


class Gate:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passes: list[str] = []

    def check(self, condition: bool, label: str, detail: str = "") -> None:
        if condition:
            self.passes.append(label)
        else:
            self.failures.append(f"{label}{f' — {detail}' if detail else ''}")


def drive(manager: SessionManager, session_id: str, answer: str, rounds: int = 150) -> dict:
    """Run a session to completion the way the UI does: advance, answer, approve, repeat."""

    async def loop() -> dict:
        for _ in range(rounds):
            state = manager.get_state(session_id)
            if state is None or state.phase.value == "done" or state.status.value == "error":
                break
            if state.awaiting.value == "answer" and state.current_question:
                await manager.answer(session_id, state.current_question.id, answer=answer)
            elif state.awaiting.value == "approval":
                await manager.approve(session_id, True)
            else:
                await manager.schedule(session_id, "advance")
            await asyncio.sleep(0.01)
        final = manager.get_state(session_id)
        return final.model_dump() if final else {}

    return asyncio.run(loop())


def collect_diagrams(directory: Path) -> list[dict]:
    diagrams = []
    for svg_path in sorted(directory.glob("*.svg")):
        mmd_path = directory / (svg_path.stem + ".mmd")
        diagrams.append(
            {
                "title": svg_path.stem,
                "code": mmd_path.read_text(encoding="utf-8") if mmd_path.exists() else "",
                "svg": svg_path.read_text(encoding="utf-8"),
            }
        )
    return diagrams


def run_session(manager: SessionManager, gate: Gate, label: str, *, transcript: str = "", brief: str = "") -> str:
    state = manager.services.store.create_session(brief=brief, transcript=transcript)
    session_id = state.session_id
    result = drive(manager, session_id, "Roughly 400 per month, peaking at 80 on the first business day.")
    directory = manager.services.config.session_dir(session_id)

    gate.check(
        result.get("phase") == "done",
        f"{label}: run reached 'done'",
        f"phase={result.get('phase')}, status={result.get('status')}",
    )
    gate.check(result.get("status") != "error", f"{label}: no error status", result.get("error", ""))

    files = {path.name for path in directory.iterdir() if path.is_file()}
    # The `.typ` source only exists when Typst actually typeset the PDF — the ReportLab fallback never
    # writes one. Expecting it unconditionally makes the gate fail on machines without a typst binary.
    expected = EXPECTED_FILES if find_typst() else EXPECTED_FILES - {"functional_specification.typ"}
    missing = expected - files
    gate.check(not missing, f"{label}: deliverable bundle is complete", f"missing: {sorted(missing)}")

    spec = (directory / "functional_specification.md").read_text(encoding="utf-8")
    gate.check(len(spec) > 1200, f"{label}: spec is substantial", f"{len(spec)} chars")

    # The manifest a reviewer opens first must state the QA verdict and where the time went,
    # not just which engine produced the PDF.
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    gate.check("qa" in metadata, f"{label}: metadata.json carries the QA verdict")
    gate.check(
        metadata.get("qa", {}).get("ok") is True, f"{label}: metadata QA verdict is a pass", str(metadata.get("qa"))
    )
    gate.check(
        metadata.get("phase_metrics"),
        f"{label}: metadata.json attributes timing to phases",
        str(metadata.get("phase_metrics")),
    )

    diagrams = collect_diagrams(directory)
    pdf_path = directory / "functional_specification.pdf"
    report = audit_pdf(pdf_path, spec, diagrams)
    checks = report["checks"]
    gate.check(report["ok"], f"{label}: PDF QA passed", json.dumps(report["warnings"]))
    gate.check(checks["pages"] >= 3, f"{label}: PDF has a real page count", f"pages={checks['pages']}")
    gate.check(checks["markdown_leaks"] == [], f"{label}: no raw Markdown in the PDF", str(checks["markdown_leaks"]))
    gate.check(
        not checks["requirements_missing"],
        f"{label}: every requirement row reached the PDF",
        str(checks["requirements_missing"]),
    )
    gate.check(
        checks["contents_links"] >= checks["outline_entries"] > 0,
        f"{label}: Contents entries are clickable",
        f"links={checks['contents_links']} outline={checks['outline_entries']}",
    )
    if checks.get("band_cut_weight") is not None:
        gate.check(
            checks["band_cut_weight"] <= 6,
            f"{label}: diagram band cuts land between nodes, not through them",
            f"weight={checks['band_cut_weight']}",
        )

    html_report = audit_html(directory / "deliverable.html")
    gate.check(html_report["ok"], f"{label}: HTML QA passed", json.dumps(html_report["warnings"]))
    gate.check(html_report["checks"]["toc_links"] > 0, f"{label}: HTML has a Contents block")
    gate.check(
        html_report["checks"]["unresolved_links"] == [],
        f"{label}: every Contents anchor resolves",
        str(html_report["checks"]["unresolved_links"]),
    )

    for diagram in diagrams:
        width, height = svg_native_size(diagram["svg"])
        gate.check(width > 0 and height > 0, f"{label}: diagram SVG has measurable size", f"{width}x{height}")

    if transcript:
        gate.check((directory / "transcript.txt").exists(), f"{label}: transcript shipped as a deliverable")
        events = manager.services.store.get_events(session_id)
        mode = next(
            (
                e["payload"]
                for e in events
                if e.get("type") == "session" and e.get("payload", {}).get("event") == "transcript_mode"
            ),
            None,
        )
        gate.check(mode is not None, f"{label}: transcript mode reported its coverage")
        if mode:
            gate.check(
                bool(mode.get("covered")),
                f"{label}: the transcript covered at least one subject",
                json.dumps(mode),
            )
            gate.check(
                bool(mode.get("uncovered")),
                f"{label}: transcript mode names the subjects it did NOT answer",
                json.dumps(mode.get("uncovered")),
            )
            subjects = result.get("subjects", [])
            gate.check(
                len(mode["covered"]) + len(mode["uncovered"]) == len(subjects),
                f"{label}: every subject is accounted for in the coverage report",
                f"covered={mode['covered']} uncovered={mode['uncovered']} subjects={len(subjects)}",
            )
        subjects = result.get("subjects", [])
        gate.check(
            all(s["status"] != "pending" for s in subjects),
            f"{label}: every subject ended closed",
            str([s["status"] for s in subjects]),
        )
        gate.check(
            any(s["status"] == "covered" for s in subjects),
            f"{label}: covered-by-transcript subjects are marked closed",
            str([s["status"] for s in subjects]),
        )

    return session_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="write outputs into ./e2e_outputs instead of a temp dir")
    args = parser.parse_args()

    root = ROOT / "e2e_outputs" if args.keep else Path(tempfile.mkdtemp(prefix="ba_e2e_"))
    config = AppConfig(root=root)
    gateway = FakeGateway()
    services = Services.create(config)
    services.gateway = gateway
    services.knowledge = KnowledgeService(services.store, gateway, config)
    manager = SessionManager(services)

    gate = Gate()

    print(f"PDF engine: {pdf_engine()}  (typst binary: {'found' if find_typst() else 'absent -> ReportLab fallback'})")
    print(f"output: {root}\n")

    run_session(manager, gate, "brief mode", brief=BRIEF)
    transcript_session = run_session(manager, gate, "transcript mode", transcript=TRANSCRIPT)

    # A finished run must teach the next one.
    lessons = services.store.list_lessons()
    gate.check(len(lessons) >= 1, "lessons recorded after completed runs")
    gate.check(all("FR-00" not in lesson["text"] for lesson in lessons), "lessons are general, not project-specific")

    # A stakeholder correction after delivery must regenerate the bundle, not rot it.
    state = manager.get_state(transcript_session)
    requirement = next((entry for entry in (state.ledger if state else []) if entry.kind == "requirement"), None)
    if requirement is not None:
        manager.edit_ledger(transcript_session, requirement.id, {"title": "Triage every claim within 24 hours"})
        drive(manager, transcript_session, "ok")
        rebuilt = manager.get_state(transcript_session)
        gate.check(
            rebuilt is not None and "Triage every claim within 24 hours" in (rebuilt.spec_markdown or ""),
            "ledger edit on a finished run rebuilds the spec",
        )
        directory = manager.services.config.session_dir(transcript_session)
        rebuilt_report = audit_pdf(
            directory / "functional_specification.pdf",
            rebuilt.spec_markdown if rebuilt else "",
            collect_diagrams(directory),
        )
        gate.check(
            rebuilt_report["ok"], "rebuilt deliverables still pass PDF QA", json.dumps(rebuilt_report["warnings"])
        )
        gate.check(
            (directory / "functional_specification.pdf").stat().st_size > 2000,
            "rebuilt PDF is a real file",
        )

    # Settings precedence: the store is a delta over config.yaml, not a snapshot of every field.
    before = manager.run_settings()["interview"]["max_subjects"]
    manager.update_run_settings({"interview": {"max_total_questions": 25}})
    stored = services.store.get_setting("run_settings", {})
    gate.check(
        set(stored.get("interview", {})) == {"max_total_questions"},
        "a settings edit stores only the changed key",
        f"stored: {stored}",
    )
    config.interview.max_subjects = before + 3
    manager.restore_run_settings()
    gate.check(
        manager.config.interview.max_total_questions == 25,
        "the stored override survives a restart",
        f"got {manager.config.interview.max_total_questions}",
    )
    gate.check(
        manager.config.interview.max_subjects == before + 3,
        "config.yaml edits to untouched fields are not shadowed by an old snapshot",
        f"got {manager.config.interview.max_subjects}, expected {before + 3}",
    )
    manager.reset_run_settings()
    gate.check(
        manager.run_settings()["interview"]["max_total_questions"] == config.interview.max_total_questions,
        "reset returns budgets to the config.yaml defaults",
        f"got {manager.run_settings()['interview']['max_total_questions']}",
    )

    print("PASSED" if not gate.failures else "CHECKS")
    for label in gate.passes:
        print(f"  ok    {label}")
    if gate.failures:
        print("\nFAILED")
        for label in gate.failures:
            print(f"  fail  {label}")
        return 1

    if args.keep:
        print(f"\noutputs kept in {root}")
    else:
        shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
