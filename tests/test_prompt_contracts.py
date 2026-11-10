"""Phase 1.4 — typed prompt contexts, and the drift test the plan asks for."""

from __future__ import annotations

from pathlib import Path

import pytest

from ba_agent.prompt_context import PROMPT_CONTRACTS, template_variables, undeclared_variables
from ba_agent.prompts import PromptContextError, render

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def test_every_shipped_template_has_a_contract():
    on_disk = {path.name.removesuffix(".md.j2") for path in PROMPTS_DIR.glob("*.md.j2")}
    missing = on_disk - set(PROMPT_CONTRACTS)
    assert not missing, f"templates without a contract: {sorted(missing)}"


@pytest.mark.parametrize("name", sorted(PROMPT_CONTRACTS))
def test_no_template_reads_what_its_contract_does_not_declare(name):
    """The plan's 'undeclared Jinja variable' guard: template and contract must agree exactly."""
    extra = undeclared_variables(name, PROMPTS_DIR)
    unused = set(PROMPT_CONTRACTS[name].model_fields) - template_variables(name, PROMPTS_DIR)
    assert not extra, f"{name}.md.j2 reads undeclared variables: {sorted(extra)}"
    assert not unused, f"{name} contract declares variables the template never uses: {sorted(unused)}"


def test_render_rejects_a_misspelled_variable():
    with pytest.raises(PromptContextError) as exc:
        render("plan", brief="x", knowledge="not the right name")
    assert "knowledge" in str(exc.value)


def test_render_rejects_a_missing_required_variable():
    with pytest.raises(PromptContextError) as exc:
        render("glossary", project_name="P")
    assert "corpus" in str(exc.value)


def test_render_normalises_optional_variables_to_defaults():
    system, user = render("plan", brief="Invoice OCR for a bank")
    assert "Invoice OCR" in user
    assert "4" in system  # max_subjects_min default


def test_render_accepts_domain_objects_for_nested_context():
    from ba_agent.schemas import SubjectSpec

    _, user = render(
        "question",
        project_name="P",
        subject=SubjectSpec(name="Ingestion", why="scanning is manual", priority="high"),
    )
    assert "Ingestion" in user
