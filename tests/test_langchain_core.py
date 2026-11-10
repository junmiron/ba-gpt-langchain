"""Phase 1 — langchain-core idioms: the retriever interface, the runnable agent chain,
profile-declared structured-output rungs.

Everything here runs offline against the scripted gateway, like the rest of the suite.
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableSequence
from langchain_core.tools import tool

from ba_agent.agent_loop import (
    AgentTurn,
    build_agent_chain,
    context_policy,
    run_agent,
    tool_executor,
)
from ba_agent.config import AppConfig
from ba_agent.retriever import document_citation
from ba_agent.services import Services

# ---------------------------------------------------------------- retrieval


def test_retriever_is_a_langchain_core_retriever(services: Services) -> None:
    services.knowledge.invalidate()
    retriever = services.retriever
    assert isinstance(retriever, BaseRetriever)


def test_retriever_returns_documents_with_provenance(services: Services) -> None:
    asyncio_run = __import__("asyncio").run
    asyncio_run(
        services.knowledge.index_upload(
            "claims.md", b"# Monthly volume\nClaims arrive by email at about 400 per month.\n"
        )
    )
    documents = services.retriever.invoke("monthly claim volume")
    assert documents, "keyword retrieval found nothing for an indexed document"
    first = documents[0]
    assert isinstance(first, Document)
    assert "400" in first.page_content
    assert "claims.md" in first.metadata["source"]
    assert first.metadata["score"] > 0


def test_sync_retrieval_never_calls_the_embedding_model(services: Services) -> None:
    calls_before = services.gateway.usage["calls"]
    services.retriever.invoke("anything")
    assert services.gateway.usage["calls"] == calls_before


def test_async_retrieval_uses_the_same_ranking(services: Services) -> None:
    import asyncio

    asyncio.run(services.knowledge.index_upload("sla.md", b"# SLA\nUploads must complete in under 10 seconds.\n"))
    docs = asyncio.run(services.retriever.ainvoke("upload sla"))
    assert docs and "10 seconds" in docs[0].page_content
    assert "SLA" in document_citation(docs[0])


# ---------------------------------------------------------------- agent chain


@tool
def note_ledger(text: str) -> str:
    """Record a note."""
    return f"recorded: {text}"


class ScriptedChatGateway:
    """Minimal gateway: hands out a Runnable that replays scripted replies."""

    def __init__(self, replies: list[AIMessage]) -> None:
        self.replies = list(replies)
        self.seen_messages: list[list[Any]] = []

    async def resolve_model(self, requested: str | None) -> str:
        return requested or "fake-chat"

    def bind_tools(self, model: str | None, tools: list[Any], *, role: str = "chat") -> Any:
        def reply(messages: list[Any]) -> AIMessage:
            self.seen_messages.append(list(messages))
            if not self.replies:
                return AIMessage(content="no more scripted replies")
            return self.replies.pop(0)

        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(reply)


def test_agent_chain_is_a_runnable_sequence() -> None:
    gateway = ScriptedChatGateway([AIMessage(content="done")])
    chain = build_agent_chain(gateway, "fake-chat", [note_ledger], system_prompt="be brief")
    assert isinstance(chain, RunnableSequence)


def test_tool_calls_execute_and_feed_back() -> None:
    import asyncio

    gateway = ScriptedChatGateway(
        [
            AIMessage(content="", tool_calls=[{"name": "note_ledger", "args": {"text": "budget is 50k"}, "id": "c1"}]),
            AIMessage(content="Noted: the budget is 50k."),
        ]
    )
    result = asyncio.run(
        run_agent(system_prompt="be brief", user_message="what is the budget?", tools=[note_ledger], gateway=gateway)
    )
    assert result.iterations == 2
    assert result.text == "Noted: the budget is 50k."
    assert result.tool_calls[0].ok and result.tool_calls[0].result == "recorded: budget is 50k"
    # the second model call must have seen the tool result
    second = gateway.seen_messages[1]
    assert any(isinstance(message, ToolMessage) for message in second)


def test_context_policy_keeps_tool_call_and_result_together() -> None:
    policy = context_policy("system prompt", max_history_chars=120)
    messages = [
        HumanMessage(content="x" * 100),
        AIMessage(content="", tool_calls=[{"name": "note_ledger", "args": {}, "id": "c1"}]),
        ToolMessage(content="y" * 100, tool_call_id="c1"),
        HumanMessage(content="latest question"),
    ]
    prepared = policy.invoke(messages)
    assert isinstance(prepared[0], SystemMessage)
    assert prepared[-1].content == "latest question"
    # a dangling ToolMessage with no AIMessage would break the API contract
    tool_messages = [m for m in prepared if isinstance(m, ToolMessage)]
    ai_with_calls = [m for m in prepared if isinstance(m, AIMessage) and m.tool_calls]
    assert len(tool_messages) == len(ai_with_calls)


def test_unknown_tool_is_reported_not_raised() -> None:
    import asyncio

    async def main() -> AgentTurn:
        executor = tool_executor([note_ledger])
        return await executor.ainvoke(
            {
                "messages": [SystemMessage(content="s")],
                "reply": AIMessage(content="", tool_calls=[{"name": "nope", "args": {}, "id": "c9"}]),
            }
        )

    turn = asyncio.run(main())
    assert turn.tool_calls[0].ok is False
    assert "Unknown tool" in turn.tool_calls[0].result


# ---------------------------------------------------------------- ladder config


def test_profile_declared_rungs_win_over_the_probe(config: AppConfig) -> None:
    from ba_agent.models import ModelInfo, get_gateway

    config.provider = "lmstudio"  # the rung ladder is read from whichever profile is active
    config.lmstudio.structured_output_rungs = ["function_calling", "prompt"]
    gateway = get_gateway(config)
    assert gateway._ladder("some-model") == ["function_calling", "prompt"]

    config.lmstudio.structured_output_rungs = []
    config.lmstudio.structured_output_method = "json_schema"
    assert gateway._ladder("some-model") == ["json_schema"]

    config.lmstudio.structured_output_method = "auto"
    # LM Studio reports capabilities, so an empty list is a denial of tool use and the ladder
    # skips function_calling. A server that reports nothing at all (Strata) is unknown, not
    # incapable — that case is covered in test_provider_profiles.py.
    gateway._catalog = [ModelInfo(id="some-model", capabilities=[], capabilities_known=True)]
    assert gateway._ladder("some-model") == ["json_schema", "prompt"]


def test_rungs_are_normalised_and_unknown_values_dropped(config: AppConfig) -> None:
    config.lmstudio.structured_output_rungs = [" JSON_SCHEMA ", "nonsense", "prompt"]
    assert config.lmstudio.structured_output_rungs == ["json_schema", "prompt"]
