"""Tool-calling agent composed from ``langchain-core`` Runnables (no LangGraph).

The loop is a real Runnable pipeline rather than a hand-written ``for``:

    context policy  ->  (model step, transcript passthrough)  ->  tool executor

each iteration produces an :class:`AgentTurn`, and :func:`run_agent` re-runs the chain until a turn comes
back with no tool calls. That makes every stage independently testable (the policy, the executor) and
swappable — a different model, a different tool set, or a stub LLM in a test — without touching the loop.

The workflow itself does **not** depend on this loop (control flow lives in :mod:`ba_agent.workflow`); it
exists for the interactive chat channel, where a stakeholder can ask questions and record decisions while a
run is in flight — something v1 had no answer to.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel
from langchain_core.tools import BaseTool

from .events import Emitter, EventType, null_emitter
from .models import ModelGateway

#: Tool results handed back to the model are capped so one huge document cannot blow the window.
MAX_TOOL_RESULT_CHARS = 4000
#: Rough character budget for the chat transcript sent to the model.
DEFAULT_HISTORY_CHARS = 24000


@dataclass
class ToolCallRecord:
    name: str
    args: dict[str, Any]
    ok: bool = True
    result: str = ""
    latency_ms: int = 0


@dataclass
class AgentTurn:
    """One pass of the chain: the transcript as it now stands plus what the model asked for."""

    messages: list[BaseMessage] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    text: str = ""


@dataclass
class AgentResult:
    text: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    iterations: int = 0
    model: str = ""
    error: str | None = None


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return str(content or "")


def _turn_atoms(body: list[BaseMessage]) -> list[list[BaseMessage]]:
    """Group a transcript into atoms so trimming never splits a tool call from its result."""
    atoms: list[list[BaseMessage]] = []
    current: list[BaseMessage] = []
    for message in body:
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            if current:
                atoms.append(current)
            current = [message]
        elif isinstance(message, ToolMessage) and current:
            current.append(message)
        else:
            if current:
                atoms.append(current)
                current = []
            atoms.append([message])
    if current:
        atoms.append(current)
    return atoms


def context_policy(system_prompt: str, max_history_chars: int = DEFAULT_HISTORY_CHARS) -> Runnable:
    """Prepend the system prompt and drop the oldest turns until the transcript fits the window."""

    def prepare(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
        body = list(messages)
        has_system = bool(body) and isinstance(body[0], SystemMessage)
        system = body[0] if has_system else SystemMessage(content=system_prompt)
        rest = body[1:] if has_system else body

        used = sum(len(_text_of(message.content)) for message in rest)
        atoms = _turn_atoms(rest)
        dropped = 0
        while atoms and used > max_history_chars:
            removed = atoms.pop(0)
            used -= sum(len(_text_of(message.content)) for message in removed)
            dropped += len(removed)
        trimmed = [system, *(message for atom in atoms for message in atom)]
        if dropped:
            trimmed.insert(
                1, SystemMessage(content=f"[{dropped} earlier message(s) trimmed to fit the context window]")
            )
        return trimmed

    return RunnableLambda(prepare)


def tool_executor(
    tools: Sequence[BaseTool], *, emit: Emitter = null_emitter, step: str = "chat", role: str = "chat"
) -> Runnable:
    """Run every tool the model asked for and return the extended transcript."""
    tool_map = {tool.name: tool for tool in tools}

    async def execute(part: dict[str, Any]) -> AgentTurn:
        messages = list(part["messages"])
        reply = part["reply"]
        reply = reply if isinstance(reply, AIMessage) else AIMessage(content=_text_of(getattr(reply, "content", "")))
        messages.append(reply)

        records: list[ToolCallRecord] = []
        for call in list(getattr(reply, "tool_calls", []) or []):
            name = call.get("name", "")
            args = dict(call.get("args") or {})
            tool = tool_map.get(name)
            record = ToolCallRecord(name=name, args=args)
            started = time.perf_counter()
            if tool is None:
                record.ok, record.result = False, f"Unknown tool '{name}'."
            else:
                try:
                    output = await tool.ainvoke(args)
                    record.result = output if isinstance(output, str) else repr(output)
                    record.ok = True
                except Exception as exc:  # noqa: BLE001 — the model must see the failure and adapt
                    record.ok, record.result = False, f"Tool failed: {type(exc).__name__}: {exc}"
            record.latency_ms = int((time.perf_counter() - started) * 1000)
            records.append(record)

            preview = (
                record.result
                if len(record.result) <= MAX_TOOL_RESULT_CHARS
                else record.result[:MAX_TOOL_RESULT_CHARS] + " …[truncated]"
            )
            await emit(
                EventType.TOOL_CALL,
                {
                    "step": step,
                    "role": role,
                    "tool": name,
                    "args": args,
                    "ok": record.ok,
                    "preview": preview[:600],
                    "latency_ms": record.latency_ms,
                },
            )
            messages.append(ToolMessage(content=preview, tool_call_id=call.get("id") or name, name=name))

        return AgentTurn(messages=messages, tool_calls=records, text=_text_of(reply.content).strip())

    return RunnableLambda(execute)


def build_agent_chain(
    gateway: ModelGateway,
    model: str | None,
    tools: Sequence[BaseTool],
    *,
    system_prompt: str,
    max_history_chars: int = DEFAULT_HISTORY_CHARS,
    emit: Emitter = null_emitter,
    step: str = "chat",
    role: str = "chat",
) -> Runnable:
    """The one-iteration pipeline: trim the window, ask the model, run its tools."""
    llm = gateway.bind_tools(model, tools, role=role)
    return (
        context_policy(system_prompt, max_history_chars)
        | RunnableParallel({"reply": llm, "messages": RunnableLambda(lambda messages: messages)})
        | tool_executor(tools, emit=emit, step=step, role=role)
    )


async def run_agent(
    *,
    system_prompt: str,
    user_message: str,
    tools: Sequence[BaseTool],
    gateway: ModelGateway,
    model: str | None = None,
    history: Sequence[BaseMessage] = (),
    role: str = "chat",
    max_iterations: int = 6,
    emit: Emitter = null_emitter,
    step: str = "chat",
    max_history_chars: int = DEFAULT_HISTORY_CHARS,
) -> AgentResult:
    """Run the chain until a turn has no tool calls. Never raises: failures land in ``AgentResult.error``."""
    resolved_model = await gateway.resolve_model(model)
    result = AgentResult(model=resolved_model or "")

    if not tools:
        text_result = await gateway.stream_text(
            system_prompt=system_prompt,
            user_prompt=user_message,
            role=role,
            model_override=model,
            emit=emit,
            step=step,
        )
        result.text = text_result.text
        result.error = text_result.error
        return result

    try:
        chain = build_agent_chain(
            gateway,
            resolved_model,
            tools,
            system_prompt=system_prompt,
            max_history_chars=max_history_chars,
            emit=emit,
            step=step,
            role=role,
        )
    except Exception as exc:  # noqa: BLE001 — model without tool support: degrade to a plain answer
        await emit(
            EventType.WARNING,
            {"step": step, "role": role, "message": f"tool calling unavailable ({exc}); answering without tools"},
        )
        text_result = await gateway.stream_text(
            system_prompt=system_prompt,
            user_prompt=user_message,
            role=role,
            model_override=model,
            emit=emit,
            step=step,
        )
        result.text = text_result.text
        result.error = text_result.error
        return result

    messages: list[BaseMessage] = [SystemMessage(content=system_prompt), *history, HumanMessage(content=user_message)]

    for iteration in range(1, max_iterations + 1):
        result.iterations = iteration
        try:
            turn = await chain.ainvoke(messages)
        except Exception as exc:  # noqa: BLE001 — the chat channel must answer, not crash
            result.error = f"{type(exc).__name__}: {exc}"
            break

        messages = turn.messages
        result.tool_calls.extend(turn.tool_calls)

        if not turn.tool_calls:
            result.text = turn.text
            await emit(EventType.TOKEN, {"step": step, "role": role, "model": resolved_model, "text": turn.text})
            return result

    if not result.text:
        result.text = "I ran out of steps while looking that up — try asking in a narrower way."
    return result
