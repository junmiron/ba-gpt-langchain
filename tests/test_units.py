"""Unit tests for the deterministic parts: chunking, JSON repair, Mermaid validation, linter, sandbox, retrieval."""

from __future__ import annotations

import pytest

from ba_agent.jsonx import extract_json
from ba_agent.knowledge import KnowledgeService, chunk_text, extract_main_content
from ba_agent.mermaid_utils import clean_mermaid, quote_risky_labels, validate_mermaid
from ba_agent.schemas import LedgerItem, RequirementCategory, SessionState
from ba_agent.spec import assemble_spec, deterministic_gaps, lint_spec
from ba_agent.tools.sandbox import Sandbox, SandboxError


# --------------------------------------------------------------------------- chunking

def test_chunk_text_respects_size_and_keeps_sentences_whole():
    text = "\n\n".join([f"Paragraph {i}. " + ("Some reasonably long sentence about requirements. " * 6) for i in range(8)])
    chunks = chunk_text(text, chunk_size=400, overlap=80)

    assert len(chunks) > 1
    assert all(len(chunk) <= 520 for chunk in chunks), [len(c) for c in chunks]
    assert all(chunk.strip() for chunk in chunks)


def test_chunk_text_is_idempotent_on_short_input_and_empty_input():
    assert chunk_text("", 900, 140) == []
    single = chunk_text("Just one short paragraph.", 900, 140)
    assert len(single) <= 1


def test_chunking_prefers_paragraph_boundaries():
    text = "A" * 500 + "\n\n" + "B" * 500 + "\n\n" + "C" * 200
    chunks = chunk_text(text, chunk_size=600, overlap=50)
    assert any(chunk.startswith("A") for chunk in chunks)
    assert not any(("A" in chunk and chunk.startswith("B")) for chunk in chunks)


# --------------------------------------------------------------------------- JSON repair

@pytest.mark.parametrize("raw", [
    '{"a": 1}',
    '```json\n{"a": 1}\n```',
    'Sure! Here is the object:\n{"a": 1, "b": [1,2,],}',
    "prefix noise {'a': true} suffix",
    '{"a": True, "b": None}',
    '{"text": "line one\nline two"}',
])
def test_extract_json_handles_local_model_mess(raw):
    data = extract_json(raw)
    assert isinstance(data, dict)


def test_extract_json_returns_none_for_prose():
    assert extract_json("There is no JSON in this answer at all.") is None


# --------------------------------------------------------------------------- mermaid

VALID = 'flowchart TD\n    A["Start"] --> B{"Ok?"}\n    B -- "yes" --> C["Done"]\n'


def test_validate_accepts_clean_diagram():
    report = validate_mermaid(VALID)
    assert report.ok, report.errors


def test_validate_flags_unquoted_parentheses_and_missing_keyword():
    bad = 'graph\n    A(Order accepted) --> B[Done]'
    report = validate_mermaid(bad)
    assert not report.ok
    assert any("unquoted label" in error for error in report.errors)


def test_validate_flags_reserved_node_id_and_unbalanced_brackets():
    report = validate_mermaid('flowchart TD\n    end["x"] --> B[foo')
    assert any("reserved" in e for e in report.errors)
    assert any("unbalanced" in e for e in report.errors)


def test_auto_fix_quotes_risky_labels():
    fixed = quote_risky_labels('flowchart TD\n    A(Order accepted) --> B[Done]')
    assert validate_mermaid(fixed).ok, validate_mermaid(fixed).errors


def test_clean_mermaid_strips_fences_and_comments():
    cleaned = clean_mermaid("```mermaid\n%% generated\nflowchart TD\nA[x] --> B[y]\n```")
    assert cleaned.startswith("flowchart TD") and "%%" not in cleaned


# --------------------------------------------------------------------------- linter

def _state_with_sections(**sections) -> SessionState:
    state = SessionState(session_id="s-1", brief="brief text here")
    for section_id, body in sections.items():
        state.sections[section_id] = body
    return state


def test_lint_reports_missing_and_thin_sections():
    state = _state_with_sections(introduction="short", scope="x" * 500)
    findings = lint_spec(state, min_functional_requirements=1)

    codes = {finding.code for finding in findings}
    assert "missing_section" in codes
    assert any(f.section_id == "introduction" for f in findings if f.code == "thin_section")


def test_lint_requires_traceability_and_acceptance_criteria():
    state = _state_with_sections()
    item = LedgerItem(id="FR-001", kind="requirement", title="Submit claim", body="The system shall submit claims.",
                      category=RequirementCategory.FUNCTIONAL, priority=None)
    item.priority = None
    state.ledger.append(item)
    findings = lint_spec(state, min_functional_requirements=1)

    assert any(f.code == "untraceable_requirements" for f in findings)


def test_deterministic_gaps_detect_missing_structure():
    state = SessionState(session_id="s-2", brief="brief")
    gaps = deterministic_gaps(state)
    assert gaps, "an empty ledger must produce clarification questions"

    for index in range(6):
        state.ledger.append(LedgerItem(id=f"FR-{index:03d}", kind="requirement", title=f"Cap {index}",
                                       body="The system shall do something useful and measurable with acceptance.",
                                       category=RequirementCategory.FUNCTIONAL))
    state.ledger.append(LedgerItem(id="NFR-900", kind="requirement", title="Performance",
                                   body="Response time under 2 seconds at peak load.",
                                   category=RequirementCategory.NON_FUNCTIONAL))
    state.ledger.append(LedgerItem(id="D-1", kind="decision", body="We will use the existing SSO."))
    state.summary = None
    assert isinstance(deterministic_gaps(state), list)


def test_assemble_spec_owns_heading_levels():
    state = _state_with_sections(introduction="# Not allowed\n## Nor this\ntext " + "y" * 300)
    document = assemble_spec(state)

    assert "\n### Not allowed" in document
    assert "\n# Not allowed" not in document


# --------------------------------------------------------------------------- sandbox

def test_sandbox_blocks_path_escape(tmp_path):
    writable = tmp_path / "outputs"
    writable.mkdir()
    sandbox = Sandbox(writable_roots=[writable])

    sandbox.write_text("ok.md", "hello")
    assert (writable / "ok.md").read_text(encoding="utf-8") == "hello"

    with pytest.raises(SandboxError):
        sandbox.resolve("../../Windows/system32/evil.txt", write=True)
    with pytest.raises(SandboxError):
        sandbox.resolve(str(tmp_path / "outside.txt"), write=False)


# --------------------------------------------------------------------------- knowledge

@pytest.mark.asyncio
async def test_knowledge_search_finds_relevant_chunk_without_embeddings(services):
    service: KnowledgeService = services.knowledge
    (services.config.root / "data").mkdir(exist_ok=True)
    (services.config.root / "data" / "ba-guide.md").write_text(
        "# Interview guide\n\nAsk about non-functional requirements such as latency, availability and GDPR "
        "compliance. Repeat for every stakeholder group.\n\n" + "Unrelated filler text. " * 40, encoding="utf-8")

    result = await service.index_directory()
    assert result["indexed"] == 1

    hits = await service.search("GDPR latency availability", k=3)
    assert hits and "non-functional" in hits[0].text.lower()


@pytest.mark.asyncio
async def test_reindexing_the_same_file_does_not_duplicate_chunks(services):
    service: KnowledgeService = services.knowledge
    data_dir = services.config.root / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "same.md").write_text("Content about scope in and out of the project. " * 30, encoding="utf-8")

    await service.index_directory()
    second = await service.index_directory()

    assert second["skipped"] == 1 and second["indexed"] == 0
    assert service.status()["chunks"] > 0


def test_extract_main_content_prefers_article():
    html = """<html><head><title>Page</title></head><body>
      <nav>" + "navigation junk " * 50 + "</nav>
      <article><h1>Requirements elicitation</h1><p>{}</p></article>
      <footer>{}</footer></body></html>""".format("Real content about interviews. " * 40, "footer noise " * 60)

    title, text = extract_main_content(html, "https://example.com/post")
    assert title == "Requirements elicitation"
    assert "Real content" in text


# --------------------------------------------------------------------------- store

def test_event_log_replay_is_ordered_and_cursorable(services):
    state = services.store.create_session(brief="cursor test")
    first = services.store.log_event(state.session_id, "step_start", {"phase": "intake"})
    second = services.store.log_event(state.session_id, "question", {"id": 1})

    all_events = services.store.get_events(state.session_id)
    assert [event["id"] for event in all_events] == sorted(event["id"] for event in all_events)
    assert [first, second] == [all_events[0]["id"], all_events[-1]["id"]]

    tail = services.store.get_events(state.session_id, after_id=first)
    assert len(tail) == 1 and tail[0]["type"] == "question"


def test_session_state_roundtrips_through_sqlite(services):
    state = services.store.create_session(brief="roundtrip")
    state.ledger.append(LedgerItem(id="FR-001", kind="requirement", title="T", body="B",
                                   category=RequirementCategory.FUNCTIONAL))
    services.store.save_state(state)

    loaded = services.store.load_state(state.session_id)
    assert loaded is not None and loaded.requirements()[0].id == "FR-001"


# --------------------------------------------------------------------------- model routing

def test_session_model_prefers_the_role_override_then_the_default():
    from ba_agent.services import session_model

    state = SessionState(session_id="s", brief="b")
    assert session_model(state, "writer") is None

    state.model_overrides = {"default": "session-model"}
    assert session_model(state, "writer") == "session-model"

    state.model_overrides["writer"] = "big-writer"
    assert session_model(state, "writer") == "big-writer"
    assert session_model(state, "critic") == "session-model"

    state.model_overrides = {"default": "   "}
    assert session_model(state, "writer") is None


def test_sandbox_resolves_relative_reads_against_every_root(tmp_path):
    outputs = tmp_path / "outputs"
    data = tmp_path / "data"
    outputs.mkdir()
    data.mkdir()
    (data / "guide.md").write_text("kb", encoding="utf-8")

    sandbox = Sandbox(writable_roots=[outputs, data])

    assert sandbox.resolve("guide.md").parent == data.resolve()   # found in the second root
    assert sandbox.resolve("new.md", write=True).parent == outputs.resolve()  # writes use the primary root


# --------------------------------------------------------------------------- cli console encoding

def test_cli_reports_survive_a_legacy_windows_console(monkeypatch):
    """A cp1252 console used to raise UnicodeEncodeError while printing the doctor report and the
    serve banner (both contain →). The CLI must reconfigure its streams before printing."""
    import io
    import sys

    from ba_agent import cli

    legacy = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict", newline="")
    monkeypatch.setattr(sys, "stdout", legacy)
    monkeypatch.setattr(sys, "stderr", legacy)

    cli.configure_console()

    report = cli._format_doctor(
        {
            "backend": {
                "provider": "lmstudio",
                "base_url": "http://127.0.0.1:1234/v1",
                "reachable": True,
                "chat_models": 3,
                "loaded": ["qwen3.8-flash-next"],
                "tool_capable": [],
                "default_model": None,
            },
            "embeddings": {"model": "nomic", "available": False, "vector_search_enabled": False},
            "knowledge": {"documents": 0, "chunks": 0, "bm25": False},
            "paths": {"project_root": ".", "db": "state/ba_agent.db", "outputs": "outputs"},
        }
    )
    print(report)  # would raise UnicodeEncodeError before the fix
    legacy.flush()
    assert "→" in report
    assert sys.stdout.encoding.lower().replace("-", "") == "utf8"


# --------------------------------------------------------------------------- thinking-model output starvation

async def test_stream_text_retries_uncapped_when_the_budget_is_spent_on_reasoning(config, monkeypatch):
    """A thinking model (e.g. Qwen3 via LM Studio) can spend its entire max_tokens budget on internal
    reasoning and end the stream with zero visible content and no error. The gateway must retry once
    without a cap instead of handing the caller an empty section."""
    from types import SimpleNamespace

    from ba_agent.events import EventType
    from ba_agent.models import ModelGateway

    config.lmstudio.default_model = "fake-model"  # keep resolve_model offline
    gateway = ModelGateway(config)
    caps_seen: list[int | None] = []

    class _FakeRun:
        def __init__(self, cap):
            self.cap = cap

        async def astream(self, messages):
            caps_seen.append(self.cap)
            if self.cap is not None:
                return  # budget spent on invisible reasoning → empty stream, no error
            yield SimpleNamespace(content="The system shall let brokers submit claims online.",
                                  additional_kwargs={})

    monkeypatch.setattr(gateway, "chat",
                        lambda model, temperature=None, max_output_tokens=None, streaming=False: _FakeRun(max_output_tokens))

    warnings: list[dict] = []

    async def emit(event_type, payload):
        if event_type == EventType.WARNING:
            warnings.append(payload)

    result = await gateway.stream_text(system_prompt="s", user_prompt="u", max_output_tokens=2048, emit=emit)

    assert caps_seen == [2048, None]  # capped attempt first, then the uncapped retry
    assert "submit claims" in result.text
    assert result.error is None
    assert any("internal reasoning" in w.get("message", "") for w in warnings)


async def test_stream_text_does_not_retry_when_output_arrives(config, monkeypatch):
    from types import SimpleNamespace

    from ba_agent.models import ModelGateway

    config.lmstudio.default_model = "fake-model"
    gateway = ModelGateway(config)
    caps_seen: list[int | None] = []

    class _FakeRun:
        def __init__(self, cap):
            self.cap = cap

        async def astream(self, messages):
            caps_seen.append(self.cap)
            yield SimpleNamespace(content="A perfectly good section body.", additional_kwargs={})

    monkeypatch.setattr(gateway, "chat",
                        lambda model, temperature=None, max_output_tokens=None, streaming=False: _FakeRun(max_output_tokens))

    result = await gateway.stream_text(system_prompt="s", user_prompt="u", max_output_tokens=2048)

    assert caps_seen == [2048]  # no retry when the first attempt produced text
    assert "section body" in result.text


# --------------------------------------------------------------------------- markdown cleanup / JSON envelope

def test_clean_markdown_unwraps_a_json_envelope():
    """Some models wrap a plain-Markdown answer in {section_id, markdown} despite instructions."""
    import json as _json

    from ba_agent.schemas import clean_markdown

    payload = _json.dumps({"section_id": "introduction", "markdown": "### Purpose\nThe system shall do things properly."})
    assert clean_markdown(payload) == "### Purpose\nThe system shall do things properly."


def test_clean_markdown_leaves_json_without_a_content_key_alone():
    import json as _json

    from ba_agent.schemas import clean_markdown

    text = _json.dumps({"status": "ok", "note": "not a section wrapper"}) + "\n\nMore prose follows."
    assert clean_markdown(text) == text.strip()


def test_clean_markdown_leaves_plain_prose_alone():
    from ba_agent.schemas import clean_markdown

    text = "### Purpose\nThe system shall do things properly, with tables and lists."
    assert clean_markdown(text) == text
