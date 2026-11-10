"""Mermaid validation accuracy, pinned to what mermaid-cli actually parses.

The validator used to flag commas, colons and semicolons as risky — they are not — and to miss pipes and
nested brackets, which genuinely break the parse. Each case below was cross-checked against `mmdc`; the
skipped test runs the real renderer when it is installed.
"""

from __future__ import annotations

import pytest

from ba_agent.mermaid_utils import (
    clean_mermaid,
    close_open_blocks,
    quote_risky_labels,
    render_svg_local,
    validate_mermaid,
)

# (label, is the raw label a real mermaid parse error?)
LABEL_CASES = [
    ("plain", False),
    ("Total (net)", True),
    ("a, b", False),
    ("a: b", False),
    ("a; b", False),
    ("a | b", True),
    ("T[inner]", True),
    ("if(x)", True),
    ("Order {pending}", True),
    ("50% done", False),
    ("Claim intake", False),
    ("A/B", False),
    ("é", False),
]


@pytest.mark.parametrize("label, breaks", LABEL_CASES)
def test_the_validator_agrees_with_mermaid_on_each_label_shape(label: str, breaks: bool):
    code = f"graph TD\n  A[{label}] --> B[End]"
    report = validate_mermaid(code)
    assert report.ok != breaks, f"label {label!r}: validator said {report.errors} but mmdc parses it"
    # whatever the repair produces must itself validate clean
    assert validate_mermaid(quote_risky_labels(code)).ok


def test_labels_that_need_quoting_are_quoted():
    fixed = quote_risky_labels("graph TD\n  A[Total (net)] --> B[a | b] --> C[End]")
    assert 'A["Total (net)"]' in fixed
    assert 'B["a | b"]' in fixed
    assert validate_mermaid(fixed).ok


def test_labels_that_are_fine_are_left_alone():
    original = "graph TD\n  A[claim, intake] --> B[review: done]"
    assert quote_risky_labels(original) == original


def test_comma_colon_labels_no_longer_cost_a_repair_round():
    """Before the fix these emitted a WARNING and a spurious 'auto-fixed' event on valid diagrams."""
    report = validate_mermaid("graph TD\n  A[claim, intake] --> B[review: done]")
    assert report.ok
    assert report.errors == []


def test_nested_bracket_label_is_caught():
    report = validate_mermaid("graph TD\n  A[T[inner]] --> B[End]")
    assert report.errors


def test_pipe_in_a_node_label_is_caught():
    report = validate_mermaid("graph TD\n  A[a | b] --> B[End]")
    assert report.errors


def test_unclosed_sequence_block_is_caught_and_closed():
    code = "sequenceDiagram\n  A->>B: submit\n  alt valid\n    B->>A: ok\n"
    assert not validate_mermaid(code).ok
    repaired = close_open_blocks(code)
    assert validate_mermaid(repaired).ok
    assert repaired.endswith("end")


def test_stray_end_is_not_repairable_and_stays_an_error():
    code = "sequenceDiagram\n  A->>B: hi\nend"
    report = validate_mermaid(code)
    assert not report.ok
    assert any("stray" in error for error in report.errors)
    # The repairer must not invent an opener; it returns the code as it came in.
    assert close_open_blocks(code) == code


def test_reserved_node_id_is_caught():
    report = validate_mermaid("graph TD\n  end[Stop] --> B[Finish]")
    assert any("reserved" in error for error in report.errors)


def test_missing_direction_is_caught():
    report = validate_mermaid("graph\n  A --> B")
    assert any("direction" in error for error in report.errors)


def test_real_rendering_of_a_previously_false_positive_label(tmp_path):
    """`A[claim, intake]` is valid mermaid; prove it renders rather than trusting the regex."""
    if render_svg_local("graph TD\n  A[Start] --> B[End]", tmp_path)[0] is None:
        pytest.skip("mermaid-cli not installed")
    svg, _png, error = render_svg_local("graph TD\n  A[claim, intake] --> B[review: done]", tmp_path)
    assert svg is not None, error


def test_real_rendering_of_a_pipe_label_after_repair(tmp_path):
    if render_svg_local("graph TD\n  A[Start] --> B[End]", tmp_path)[0] is None:
        pytest.skip("mermaid-cli not installed")
    repaired = quote_risky_labels("graph TD\n  A[claim | intake] --> B[Total (net)]")
    svg, _png, error = render_svg_local(repaired, tmp_path)
    assert svg is not None, error


def test_an_empty_diagram_is_an_error_not_a_silent_pass():
    report = validate_mermaid("")
    assert not report.ok
    assert report.errors == ["diagram is empty"]
    assert report.code == ""


def test_an_unknown_direction_is_a_warning_not_a_hard_error():
    report = validate_mermaid("graph XY\n  A[Start] --> B[End]")
    assert report.ok
    assert any("unknown direction" in warning for warning in report.warnings)


def test_an_odd_number_of_quotes_is_caught():
    report = validate_mermaid('graph TD\n  A["unterminated] --> B[End]')
    assert any("odd number of double quotes" in error for error in report.errors)


def test_a_numeric_node_id_is_a_warning():
    report = validate_mermaid("graph TD\n  1[Start] --> 2[End]")
    assert any("numeric node id" in warning for warning in report.warnings)


def test_nodes_without_connections_are_flagged():
    report = validate_mermaid("graph TD\n  A[Start]\n  B[End]")
    assert any("no arrows" in warning for warning in report.warnings)


def test_an_overgrown_diagram_is_warned_about_readability():
    lines = ["graph TD"]
    lines += [f"  n{i}[step {i}] --> n{i + 1}[step {i + 1}]" for i in range(45)]
    report = validate_mermaid("\n".join(lines), max_nodes=40)
    assert any("consider splitting" in warning for warning in report.warnings)


def test_comment_and_tab_lines_are_normalised_before_validation():
    code = "graph TD\n\t%% a stray comment\n  A[Start] --> B[End]\n\n"
    cleaned = clean_mermaid(code)
    assert "%%" not in cleaned and "\t" not in cleaned
    assert validate_mermaid(code).ok
