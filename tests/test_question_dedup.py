"""The 'do not repeat' list handed to the question role.

`next_question` built the anti-repetition list as `[unresolved] + [all]`, which puts every open
question in the prompt twice. Worse, the list is then truncated with `[-14:]`, so once an interview
passes 14 questions the *open* questions at the head are cut off entirely — exactly the questions a
follow-up is most likely to re-ask.
"""

from __future__ import annotations

import asyncio

from ba_agent.roles.interview import next_question
from ba_agent.schemas import QuestionDraft, QuestionState, SessionState, SubjectState
from conftest import FakeGateway


class PromptCapturingGateway(FakeGateway):
    """FakeGateway that keeps the rendered question prompt so the test can inspect it."""

    def __init__(self) -> None:
        super().__init__()
        self.question_prompts: list[str] = []

    async def structured(self, schema, *, system_prompt: str = "", user_prompt: str = "", **kwargs):
        if schema is QuestionDraft:
            self.question_prompts.append(system_prompt)
        return await super().structured(schema, system_prompt=system_prompt, user_prompt=user_prompt, **kwargs)


def _state(open_positions: list[int], total: int) -> SessionState:
    """Questions in ask order; the ones at *open_positions* are still unanswered."""
    state = SessionState(session_id="dedup-probe", title="Claims portal", brief="A portal for claims.")
    state.subjects = [SubjectState(name="Data", why="systems feeding the portal", priority="high")]
    for index in range(total):
        state.questions.append(
            QuestionState(
                id=index, subject="Data", text=f"Question {index}", answer=None if index in open_positions else "yes"
            )
        )
    return state


def _asked_lines(prompt: str) -> list[str]:
    marker = "Questions already asked (do not repeat):"
    block = prompt.split(marker, 1)[1]
    lines: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- ") and stripped != "- (none)":
            lines.append(stripped[2:])
        else:
            break
    return lines


def test_open_questions_are_not_listed_twice(services):
    gateway = PromptCapturingGateway()
    services.gateway = gateway
    state = _state(open_positions=[6, 7, 8], total=9)

    asyncio.run(next_question(services, state, "Data"))

    lines = _asked_lines(gateway.question_prompts[-1])
    assert lines, "the prompt should carry the already-asked list"
    assert len(lines) == len(set(lines)), f"duplicates in the anti-repetition list: {lines}"
    assert len(lines) == 9, f"expected all nine questions once each, got {lines}"


def test_open_questions_survive_a_long_interview(services):
    """Two questions asked early are still unanswered; a 20-question interview must not drop them."""
    gateway = PromptCapturingGateway()
    services.gateway = gateway
    state = _state(open_positions=[0, 1], total=20)

    asyncio.run(next_question(services, state, "Data"))

    lines = set(_asked_lines(gateway.question_prompts[-1]))
    missing = {"Question 0", "Question 1"} - lines
    assert not missing, f"open questions were truncated out of the window; prompt shows {sorted(lines)}"
    assert len(lines) <= 14, "the window is meant to stay bounded"
