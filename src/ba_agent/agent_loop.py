"""A minimal tool-calling agent loop on plain LangChain primitives (no LangGraph).

This is the whole "agent runtime": bind tools → stream/collect → execute requested tools → feed
``ToolMessage``s back → repeat until the model answers without calling tools, or the iteration budget runs out.

The workflow itself does **not** depend on this loop (control flow lives in :mod:`ba_agent.workflow`); it exists
for the interactive chat channel, where the stakeholder can ask questions and record decisions while a run is in
flight — something v1 had no answer to.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from .events import Emitter, EventType, null_emitter
from .models import ModelGateway


@dataclass
class ToolCallRecord:
    name: str
    args: dict[str, Any]
    ok: bool = True
    result: str = ""
    latency_ms: int = 0


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
) -> AgentResult:
    """Run the tool loop until a final answer. Never raises: failures come back in ``AgentResult.error``."""
    resolved_model = await gateway.resolve_model(model)
    result = AgentResult(model=resolved_model or "")

    if not tools:
        text_result = await gateway.stream_text(system_prompt=system_prompt, user_prompt=user_message,
                                                role=role, model_override=model, emit=emit, step=step)
        result.text = text_result.text
        result.error = text_result.error
        return result

    try:
        llm = gateway.bind_tools(resolved_model, tools, role=role)
    except Exception as exc:  # noqa: BLE001 — model without tool support: degrade to a plain answer
        await emit(EventType.WARNING, {"step": step, "role": role,
                                       "message": f"tool calling unavailable ({exc}); answering without tools"})
        text_result = await gateway.stream_text(system_prompt=system_prompt, user_prompt=user_message,
                                                role=role, model_override=model, emit=emit, step=step)
        result.text = text_result.text
        result.error = text_result.error
        return result

    messages: list[BaseMessage] = [SystemMessage(content=system_prompt), *history, HumanMessage(content=user_message)]
    tool_map = {tool.name: tool for tool in tools}

    for iteration in range(1, max_iterations + 1):
        result.iterations = iteration
        try:
            reply: AIMessage = await llm.ainvoke(messages)
        except Exception as exc:  # noqa: BLE001
            result.error = f"{type(exc).__name__}: {exc}"
            break

        messages.append(reply if isinstance(reply, AIMessage) else AIMessage(content=_text_of(reply.content)))
        calls = list(getattr(reply, "tool_calls", []) or [])
        text = _text_of(reply.content).strip()

        if not calls:
            result.text = text
            await emit(EventType.TOKEN, {"step": step, "role": role, "model": resolved_model, "text": text})
            return result

        for call in calls:
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
            result.tool_calls.append(record)

            preview = record.result if len(record.result) <= 4000 else record.result[:4000] + " …[truncated]"
            await emit(EventType.TOOL_CALL, {
                "step": step, "role": role, "tool": name, "args": args, "ok": record.ok,
                "preview": preview[:600], "latency_ms": record.latency_ms,
            })
            messages.append(ToolMessage(content=preview, tool_call_id=call.get("id") or name, name=name))

    if not result.text:
        result.text = "I ran out of steps while looking that up — try asking in a narrower way."
    return result
