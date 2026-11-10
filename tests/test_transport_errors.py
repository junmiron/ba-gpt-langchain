"""Distinguishing "the backend is down" from "the model answered badly".

These look identical to the ladder and are not: a refused connection has nothing to do with JSON
schema, and walking the whole ladder against a dead server produces nine warnings that bury the
real cause. See `ModelGateway._is_transport_error` in src/ba_agent/models.py.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from ba_agent.config import AppConfig
from ba_agent.models import ModelGateway, ModelInfo


class Requirement(BaseModel):
    description: str


async def _silent(event, payload):
    return None


class _DeadChat:
    """A chat client whose every call fails at the transport layer."""

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        raise ConnectionRefusedError("LM Studio is not listening on 127.0.0.1:1234")

    def with_structured_output(self, schema, **kwargs):
        return self


class _SloppyChat:
    """A chat client that connects fine but returns garbage — the opposite failure."""

    def __init__(self):
        self.calls = 0
        self.nudges: list[str] = []

    async def ainvoke(self, messages):
        self.calls += 1
        for message in messages:
            content = getattr(message, "content", "")
            if "not valid for the required JSON schema" in content:
                self.nudges.append(content)
        raise ValueError("model returned no valid payload")


def _gateway(chat) -> ModelGateway:
    config = AppConfig(provider="lmstudio")
    # A default model must exist, otherwise `structured` short-circuits with "no model available"
    # before the ladder — and the test would be measuring the wrong thing.
    config.lmstudio.default_model = "q2_0"
    config.lmstudio.structured_output_rungs = ["json_schema", "function_calling", "prompt"]
    config.lmstudio.max_retries = 1
    gateway = ModelGateway(config)
    gateway._catalog = [ModelInfo(id="q2_0", type="llm", state="loaded", capabilities=["tool_use"])]
    gateway.chat = lambda *args, **kwargs: chat  # type: ignore[assignment]
    return gateway


@pytest.mark.asyncio
async def test_a_down_backend_is_reported_as_unreachable_not_as_bad_json():
    chat = _DeadChat()
    gateway = _gateway(chat)
    warnings: list[dict] = []

    async def emit(event, payload):
        warnings.append(payload)

    result = await gateway.structured(
        Requirement,
        system_prompt="Return JSON only.",
        user_prompt="One requirement for a claims portal.",
        emit=emit,
        step="triage",
    )

    assert result.parsed is None
    assert "unreachable" in (result.error or "").lower(), result.error
    assert not any("not valid for the required JSON schema" in str(w.get("message", "")) for w in warnings)
    assert any("backend unreachable" in str(w.get("message", "")) for w in warnings), warnings


@pytest.mark.asyncio
async def test_a_down_backend_does_not_walk_every_rung_of_the_ladder():
    """Nine identical failures against a dead server reads like a model problem and is not one."""
    chat = _DeadChat()
    gateway = _gateway(chat)

    await gateway.structured(
        Requirement,
        system_prompt="Return JSON only.",
        user_prompt="One requirement for a claims portal.",
        emit=_silent,
        step="triage",
    )

    assert chat.calls < 6, f"a dead backend was hit {chat.calls} times across the whole ladder"


@pytest.mark.asyncio
async def test_a_malformed_reply_still_gets_the_repair_nudge():
    """The transport path must not swallow the genuine case: the model answered and it was wrong."""
    chat = _SloppyChat()
    gateway = _gateway(chat)

    result = await gateway.structured(
        Requirement,
        system_prompt="Return JSON only.",
        user_prompt="One requirement for a claims portal.",
        emit=_silent,
        step="triage",
    )

    assert result.parsed is None
    assert chat.nudges, "a malformed reply must still trigger the schema-repair retry"


class _APITimeout(Exception):
    """Shaped like `openai.APITimeoutError`: not a `TimeoutError` subclass, only the name says timeout."""


class _HangingChat:
    """The connection is fine and the model never produces a token — `request_timeout_s` fires."""

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        raise _APITimeout("Request timed out.")

    def with_structured_output(self, schema, **kwargs):
        return self


class _HangThenAnswer:
    """First rung hangs, the next one answers — the backend is slow, not dead."""

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            raise _APITimeout("Request timed out.")
        return {"parsed": Requirement(description="Claims arrive by email."), "raw": None}

    def with_structured_output(self, schema, **kwargs):
        return self


@pytest.mark.asyncio
async def test_a_hung_rung_gets_one_attempt_not_the_whole_retry_budget():
    """Every attempt on a hung request costs request_timeout_s, so a hung rung must not be retried."""
    chat = _HangingChat()
    gateway = _gateway(chat)

    result = await gateway.structured(
        Requirement,
        system_prompt="Return JSON only.",
        user_prompt="One requirement for a claims portal.",
        emit=_silent,
        step="diagram",
    )

    assert result.parsed is None
    assert chat.calls == 3, f"a hung request was retried {chat.calls} times instead of once per rung"
    assert "unresponsive" in (result.error or "").lower(), result.error
    assert "unreachable" not in (result.error or "").lower(), result.error


@pytest.mark.asyncio
async def test_a_timeout_walks_to_the_next_rung_instead_of_giving_up():
    """One hung rung is not a dead backend: the ladder still has rungs worth trying."""
    chat = _HangThenAnswer()
    gateway = _gateway(chat)
    warnings: list[dict] = []

    async def emit(event, payload):
        warnings.append(payload)

    result = await gateway.structured(
        Requirement,
        system_prompt="Return JSON only.",
        user_prompt="One requirement for a claims portal.",
        emit=emit,
        step="diagram",
    )

    assert result.parsed is not None, result.error
    assert result.method == "function_calling", result.method
    assert any("next rung" in str(w.get("message", "")) for w in warnings), warnings
    assert not any("backend unreachable" in str(w.get("message", "")) for w in warnings), warnings
