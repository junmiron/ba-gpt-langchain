"""The ReAct loop: tool failures, context trimming and the no-tools fallback.

These are the paths that decide what a chat turn does when the model misuses a tool or the transcript
outgrows the window. None of them may raise — a chat turn must answer, even a degraded one.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from ba_agent.agent_loop import (
    context_policy,
    run_agent,
    tool_executor,
)
from ba_agent.config import AppConfig
from ba_agent.models import ModelGateway


class _RaisingTool(BaseTool):
    name: str = "boom"
    description: str = "always fails"

    def _run(self, *args: Any, **kwargs: Any) -> str:  # pragma: no cover - sync path unused
        raise RuntimeError("kaboom")

    async def _arun(self, value: str) -> str:
        raise ValueError(f"cannot handle {value!r}")


class _DictTool(BaseTool):
    name: str = "lookup"
    description: str = "returns a non-string payload"

    def _run(self, *args: Any, **kwargs: Any) -> dict:  # pragma: no cover - sync path unused
        return {"ok": True}

    async def _arun(self, query: str) -> dict:
        return {"found": query, "count": 3}


class _ScriptedGateway(ModelGateway):
    """Gateway whose bound model emits a fixed sequence of replies."""

    def __init__(self, replies: list[Any]):
        config = AppConfig()
        super().__init__(config)
        self._replies = list(replies)

    async def resolve_model(self, model: str | None = None) -> str | None:
        return "scripted"

    def bind_tools(self, model: str | None, tools: list[BaseTool], **_kwargs):
        from langchain_core.runnables import RunnableLambda

        def reply(messages):
            item = self._replies.pop(0) if self._replies else AIMessage(content="done")
            if isinstance(item, AIMessage) and item.tool_calls:
                return item
            return AIMessage(content=item)

        return RunnableLambda(reply)

    async def stream_text(self, *, system_prompt: str, user_prompt: str, **_kwargs):
        from ba_agent.models import TextResult

        return TextResult(text="plain answer", error=None)


class _ExplodingGateway(_ScriptedGateway):
    """The bound model itself raises — not a tool. The turn must degrade, not crash."""

    def bind_tools(self, model: str | None, tools: list[BaseTool], **_kwargs):
        from langchain_core.runnables import RunnableLambda

        def reply(_messages):
            raise ConnectionResetError("model backend went away mid-turn")

        return RunnableLambda(reply)


@pytest.mark.asyncio
async def test_a_model_that_dies_mid_turn_degrades_to_an_error_not_an_exception():
    result = await run_agent(
        system_prompt="you are a BA assistant",
        user_message="what are the requirements?",
        tools=[_DictTool()],
        gateway=_ExplodingGateway([]),
        max_iterations=3,
    )
    assert result.error and "ConnectionResetError" in result.error
    # the user still gets a readable answer rather than a stack trace
    assert result.text == "I ran out of steps while looking that up — try asking in a narrower way."
    assert result.iterations == 1


async def test_a_raising_tool_is_reported_to_the_model_not_raised():
    executor = tool_executor([_RaisingTool()])
    reply = AIMessage(content="", tool_calls=[{"name": "boom", "args": {"value": "x"}, "id": "c1"}])
    turn = await executor.ainvoke({"messages": [HumanMessage(content="hi")], "reply": reply})

    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].ok is False
    assert "ValueError" in turn.tool_calls[0].result
    # the transcript now carries a ToolMessage so the next model call sees the failure
    assert any(isinstance(m, ToolMessage) for m in turn.messages)


async def test_an_unknown_tool_name_is_answered_not_crashed_on():
    executor = tool_executor([_DictTool()])
    reply = AIMessage(content="", tool_calls=[{"name": "nonexistent", "args": {}, "id": "c2"}])
    turn = await executor.ainvoke({"messages": [], "reply": reply})
    assert turn.tool_calls[0].ok is False
    assert "Unknown tool" in turn.tool_calls[0].result


async def test_a_non_string_tool_result_is_serialised_for_the_transcript():
    executor = tool_executor([_DictTool()])
    reply = AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"query": "SLA"}, "id": "c3"}])
    turn = await executor.ainvoke({"messages": [], "reply": reply})
    assert turn.tool_calls[0].ok is True
    assert "'found': 'SLA'" in turn.tool_calls[0].result


async def test_a_long_tool_result_is_truncated_before_entering_the_transcript():
    from ba_agent.agent_loop import MAX_TOOL_RESULT_CHARS

    class _BigTool(BaseTool):
        name: str = "big"
        description: str = "returns a huge payload"

        def _run(self, *args: Any, **kwargs: Any) -> str:  # pragma: no cover
            return "z" * (MAX_TOOL_RESULT_CHARS + 500)

        async def _arun(self) -> str:
            return "z" * (MAX_TOOL_RESULT_CHARS + 500)

    executor = tool_executor([_BigTool()])
    reply = AIMessage(content="", tool_calls=[{"name": "big", "args": {}, "id": "c4"}])
    turn = await executor.ainvoke({"messages": [], "reply": reply})
    tool_message = next(m for m in turn.messages if isinstance(m, ToolMessage))
    assert len(tool_message.content) <= MAX_TOOL_RESULT_CHARS + 20
    assert tool_message.content.endswith("[truncated]")


def test_context_policy_keeps_tool_call_and_result_together():
    """Dropping history must never leave a ToolMessage whose AIMessage call was removed."""
    policy = context_policy("you are a BA", max_history_chars=120)
    messages = [
        HumanMessage(content="a" * 200),
        AIMessage(content="", tool_calls=[{"name": "lookup", "args": {}, "id": "t1"}]),
        ToolMessage(content="tool output here", tool_call_id="t1"),
        HumanMessage(content="and now the real question"),
    ]
    trimmed = policy.invoke(messages)
    tool_messages = [m for m in trimmed if isinstance(m, ToolMessage)]
    call_ids = {c["id"] for m in trimmed if isinstance(m, AIMessage) for c in getattr(m, "tool_calls", [])}
    assert tool_messages, "the tool result should still be present"
    for tool_message in tool_messages:
        assert tool_message.tool_call_id in call_ids, "a tool result was left without its call"
    assert any("trimmed" in m.content for m in trimmed if isinstance(m, SystemMessage))


def test_context_policy_does_not_trim_when_the_transcript_already_fits():
    policy = context_policy("system", max_history_chars=5000)
    trimmed = policy.invoke([HumanMessage(content="short question")])
    assert not any("trimmed" in m.content for m in trimmed if isinstance(m, SystemMessage))


async def test_chat_without_tools_streams_a_plain_answer():
    result = await run_agent(
        system_prompt="ba",
        user_message="hello",
        tools=[],
        gateway=_ScriptedGateway([]),
    )
    assert result.text == "plain answer"
    assert result.error is None
    assert result.iterations == 0


async def test_the_loop_stops_when_the_model_stops_calling_tools():
    gateway = _ScriptedGateway(
        [
            AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"query": "sla"}, "id": "a"}]),
            "Here is what I found.",
        ]
    )
    result = await run_agent(
        system_prompt="ba",
        user_message="what is the SLA?",
        tools=[_DictTool()],
        gateway=gateway,
    )
    assert result.text == "Here is what I found."
    assert result.iterations == 2
    assert len(result.tool_calls) == 1


async def test_an_exhausted_iteration_budget_answers_instead_of_returning_empty():
    forever = [AIMessage(content="", tool_calls=[{"name": "lookup", "args": {}, "id": f"c{i}"}]) for i in range(20)]
    result = await run_agent(
        system_prompt="ba",
        user_message="keep going",
        tools=[_DictTool()],
        gateway=_ScriptedGateway(forever),
        max_iterations=3,
    )
    assert result.iterations == 3
    assert result.text and "ran out of steps" in result.text


def test_blockified_model_content_is_flattened_to_text():
    """Some providers return content as a list of blocks; the loop must not stringify the whole list."""
    from ba_agent.agent_loop import _text_of

    content = [{"type": "text", "text": "Answer: "}, "the SLA is 4 hours", {"type": "image", "url": "x.png"}]
    assert _text_of(content) == "Answer: the SLA is 4 hours"
    assert _text_of("plain") == "plain"
    assert _text_of(None) == ""


async def test_a_model_without_tool_support_falls_back_to_a_plain_answer(monkeypatch):
    gateway = _ScriptedGateway([])

    def explode(*_args, **_kwargs):
        raise NotImplementedError("this backend has no tool calling")

    monkeypatch.setattr(gateway, "bind_tools", explode)
    result = await run_agent(
        system_prompt="ba",
        user_message="what is the SLA?",
        tools=[_DictTool()],
        gateway=gateway,
    )
    assert result.text == "plain answer"
    assert result.error is None
    assert result.iterations == 0
