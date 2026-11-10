"""The ladder's success and last-resort paths.

`structured()` is the single most used gateway method, yet its happy path and its uncapped-retry
fallback were both untested: a model that answers correctly on rung 1 never reached the tests, and
the "ladder exhausted under a token cap" recovery only fires when every rung fails, which the
transport tests short-circuit on transport errors. Both matter because the pipeline silently degrades
to a raw-text draft when `structured` returns `degraded=True`.
"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage
from pydantic import BaseModel

from ba_agent.config import AppConfig
from ba_agent.models import ModelGateway


class Requirement(BaseModel):
    description: str
    priority: str = "must"


async def _silent(event, payload) -> None:
    del event, payload


class _GoodChat:
    """Answers on the first rung, the way a working Strata/LM Studio model does."""

    def __init__(self):
        self.calls = 0
        self.rungs: list[str] = []
        self.system_prompt = ""

    def with_structured_output(self, schema, *, method, include_raw=False):
        self.rungs.append(method)
        self._method = method
        self._include_raw = include_raw
        self._schema = schema
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        self.system_prompt = messages[0].content
        del messages
        payload = {"description": "Claims must be deduplicated by policy number.", "priority": "must"}
        method = getattr(self, "_method", "prompt")
        if method == "prompt":
            return AIMessage(content="Sure:\n" + json.dumps(payload))
        if method == "function_calling":
            return AIMessage(content="", tool_calls=[{"name": "Requirement", "args": payload, "id": "c1"}])
        parsed = self._schema.model_validate(payload)
        if getattr(self, "_include_raw", False):
            return {"raw": AIMessage(content=json.dumps(payload)), "parsed": parsed, "parsing_error": None}
        return parsed


def _gateway(rungs: list[str], max_output_tokens: int | None = None) -> ModelGateway:
    config = AppConfig(provider="strata")
    config.strata.default_model = "q2_0"
    config.strata.structured_output_rungs = rungs
    config.strata.max_retries = 1
    if max_output_tokens is not None:
        config.strata.max_output_tokens = max_output_tokens
    gateway = ModelGateway(config)
    gateway._catalog = []
    return gateway


async def test_a_clean_reply_on_the_first_rung_stops_the_ladder(monkeypatch):
    chat = _GoodChat()
    gateway = _gateway(["json_schema", "function_calling", "prompt"])
    monkeypatch.setattr(gateway, "chat", lambda *args, **kwargs: chat)

    result = await gateway.structured(
        Requirement,
        system_prompt="Return JSON only.",
        user_prompt="One requirement for a claims portal.",
        role="interviewer",
        emit=_silent,
        step="interview",
    )

    assert result.parsed is not None and result.parsed.description
    assert result.method == "json_schema", f"the ladder should stop at the first rung that works, got {result.method}"
    assert result.degraded is False
    assert result.attempts == 1
    assert gateway.rung_counts.get("json_schema") == 1, gateway.rung_counts
    # The learned rung is what makes later calls skip the rungs that already failed.
    assert gateway._learned_rung["q2_0"] == "json_schema"


async def test_successful_structured_calls_emit_timing_metadata_without_prompts(monkeypatch):
    chat = _GoodChat()
    gateway = _gateway(["json_schema"])
    monkeypatch.setattr(gateway, "chat", lambda *args, **kwargs: chat)
    events = []

    async def capture(event, payload):
        events.append((event, payload))

    await gateway.structured(
        Requirement,
        system_prompt="PRIVATE SYSTEM PROMPT",
        user_prompt="PRIVATE USER PROMPT",
        role="interview",
        emit=capture,
        step="question:Scope",
    )

    timing = [payload for event, payload in events if event == "log" and payload.get("category") == "model_call"]
    assert len(timing) == 1
    assert timing[0]["role"] == "interview"
    assert timing[0]["step"] == "question:Scope"
    assert timing[0]["method"] == "json_schema"
    assert timing[0]["attempts"] == 1
    assert timing[0]["latency_ms"] >= 0
    assert timing[0]["prompt_chars"] > 0
    assert "PRIVATE" not in str(timing[0])


async def test_the_prompt_rung_extracts_json_from_prose(monkeypatch):
    """The last rung has no structured-output support at all: it is plain text with JSON inside."""
    chat = _GoodChat()
    gateway = _gateway(["prompt"])
    monkeypatch.setattr(gateway, "chat", lambda *args, **kwargs: chat)

    result = await gateway.structured(
        Requirement,
        system_prompt="You are a business analyst.",
        user_prompt="One requirement.",
        emit=_silent,
        step="draft",
    )

    assert result.method == "prompt"
    assert result.parsed is not None
    assert "Requirement" in chat.system_prompt, chat.system_prompt[:200]


async def test_the_schema_is_injected_only_for_the_prompt_rung(monkeypatch):
    captured: dict[str, str] = {}

    class _CapturingChat(_GoodChat):
        async def ainvoke(self, messages):
            captured["system"] = messages[0].content
            return await super().ainvoke(messages)

    gateway = _gateway(["prompt"])
    monkeypatch.setattr(gateway, "chat", lambda *args, **kwargs: _CapturingChat())
    await gateway.structured(
        Requirement,
        system_prompt="You are a business analyst.",
        user_prompt="One requirement.",
        emit=_silent,
        step="draft",
    )
    assert "You are a business analyst." in captured["system"]
    assert "Requirement" in captured["system"], "the prompt rung must carry the JSON schema"


async def test_a_ladder_exhausted_under_a_token_cap_retries_uncapped(monkeypatch):
    """Thinking models can spend the whole budget on reasoning and return nothing parseable.

    The uncapped retry is the only recovery left, so it must run exactly once and say so.
    """
    events: list[dict] = []
    caps: list[int | None] = []

    class _StarvedChat:
        def with_structured_output(self, schema, **kwargs):
            del schema, kwargs
            return self

        async def ainvoke(self, messages):
            del messages
            raise ValueError("no JSON object found in the model reply")

    async def emit(event, payload):
        events.append(payload)

    def chat_factory(model, *, temperature=None, max_output_tokens=None, streaming=False):
        del model, temperature, streaming
        caps.append(max_output_tokens)
        return _GoodChat() if max_output_tokens is None else _StarvedChat()

    gateway = _gateway(["json_schema", "prompt"])
    monkeypatch.setattr(gateway, "chat", chat_factory)

    result = await gateway.structured(
        Requirement,
        system_prompt="Return JSON only.",
        user_prompt="One requirement.",
        max_output_tokens=256,
        emit=emit,
        step="draft",
    )

    assert any("without a cap" in str(e.get("message", "")) for e in events), events
    assert None in caps, "the recovery call must drop the token cap"
    assert result.parsed is not None, f"the uncapped retry should recover, got {result.error!r}"
    assert result.method == "prompt_uncapped"
    assert any(e.get("category") == "model_call" and e.get("method") == "prompt_uncapped" for e in events)


async def test_no_model_available_fails_once_without_touching_the_chat(monkeypatch):
    calls: list[int] = []

    def chat_factory(*args, **kwargs):
        del args, kwargs
        calls.append(1)
        raise AssertionError("must not be reached")

    gateway = _gateway(["json_schema", "prompt"])
    monkeypatch.setattr(gateway, "chat", chat_factory)
    monkeypatch.setattr(gateway, "resolve_model", lambda requested: _just_none())

    result = await gateway.structured(
        Requirement,
        system_prompt="Return JSON only.",
        user_prompt="One requirement.",
        emit=_silent,
        step="draft",
    )
    assert result.parsed is None
    assert result.method == "no_model"
    assert result.attempts == 0
    assert not calls, "a model that does not exist must not be called"
    assert "strata" in (result.error or "").lower(), result.error


def _just_none():
    async def resolved():
        return None

    return resolved()
