"""The JSON repair helpers in `jsonx`: what they may touch, and what must stay verbatim.

`extract_json` is the last line of defence for every structured call, and its repairs are
string-aware on purpose — a smart quote or the word `None` inside a model's own text is content,
not a syntax error. Those guards are the paths most likely to be broken by a later "simplification"
and the least likely to be caught by an end-to-end test, because a corrupted string still parses.
"""

from __future__ import annotations

from ba_agent.jsonx import _escape_inner_newlines, _fix_common, _single_to_double, extract_json


def test_repairs_do_not_touch_text_inside_string_literals():
    fixed = _fix_common('{"note": "the True value, e.g. \u2018yes\u2019, is fine,"}')
    assert fixed == '{"note": "the True value, e.g. \u2018yes\u2019, is fine,"}', (
        "smart quotes and Python literals inside a string are the model's content, not bugs"
    )


def test_repairs_apply_outside_strings():
    # `_fix_common` normalises literals, not bare keys — quoting keys is a separate ladder rung,
    # and a repair that quietly did both would hide which rung actually failed.
    assert _fix_common("{a: True, b: None,}") == "{a: true, b: null}"
    assert _fix_common("[1, 2,]") == "[1, 2]"
    assert _fix_common("\u201ckey\u201d: 1") == '"key": 1'


def test_backslash_escaping_inside_a_string_is_tracked():
    # The escaped quote must not end the literal, so the smart quote after it stays untouched.
    text = '{"a": "say \\"hi\\" \u2018x\u2019"}'
    assert _fix_common(text) == text


def test_inner_newlines_and_tabs_are_escaped_only_inside_strings():
    assert _escape_inner_newlines('{"a": "one\ntwo"}') == '{"a": "one\\ntwo"}'
    assert _escape_inner_newlines('{"a": "one\ttwo"}') == '{"a": "one\\ttwo"}'
    assert _escape_inner_newlines('{"a": "one\\ntwo"}') == '{"a": "one\\ntwo"}', "already escaped stays"
    assert _escape_inner_newlines("[1,\n2]") == "[1,\n2]", "structural whitespace is not a string"


def test_single_quotes_become_double_without_corrupting_apostrophes():
    assert _single_to_double("{'key': 'value'}") == '{"key": "value"}'
    assert _single_to_double('{"text": "it\'s fine"}') == '{"text": "it\'s fine"}'
    assert _single_to_double("{'a': 'say \"hi\"'}") == '{"a": "say \\"hi\\""}'


def test_an_unterminated_string_is_rejected_rather_than_guessed():
    # Guessing a closing quote here would invent content the model never wrote.
    assert extract_json('{"a": "unclosed\n}') is None


def test_extract_json_prefers_a_faithful_parse_over_a_repair():
    # A string containing a trailing comma before `}` would be "fixed" if the fence were parsed
    # naively; the verbatim parse must win.
    fenced = '```json\n{"note": "end of sentence,"}\n```'
    assert extract_json(fenced) == {"note": "end of sentence,"}


def test_extract_json_recovers_from_the_classical_local_model_mistakes():
    assert extract_json("{'a': [1, 2,]}") == {"a": [1, 2]}
    assert extract_json('the answer is: {"ok": true} plus prose') == {"ok": True}
    assert extract_json('{"a": "multi\nline"}') == {"a": "multi\nline"}
    assert extract_json("single-quoted: {'a': 1}") == {"a": 1}
