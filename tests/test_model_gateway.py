"""Model catalogue, health and embedding paths of :class:`ModelGateway`.

These are the paths the UI depends on ("is a model loaded?", "which one do I pick?", "is vector
search available?"), and they were the least exercised part of the gateway. Everything here runs
against a mocked transport — no model server is needed.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from ba_agent.config import AppConfig
from ba_agent.models import ModelGateway, _chunk_delta, _message_text

ADMIN_MODELS = {
    "data": [
        {
            "id": "qwen3-8b",
            "type": "llm",
            "state": "loaded",
            "capabilities": ["tool_use"],
            "max_context_length": 8192,
            "loaded_context_length": 4096,
            "quantization": "Q4_K_M",
            "publisher": "Qwen",
        },
        {
            "id": "nomic-embed-text-v1.5",
            "type": "embedding",
            "state": "loaded",
        },
    ]
}

OPENAI_MODELS = {"data": [{"id": "qwen3-8b"}, {"id": "text-embedding-nomic-embed-text-v1.5"}]}


def _gateway(monkeypatch, *, admin_url: str | None = None) -> ModelGateway:
    config = AppConfig()
    config.provider = "lmstudio"
    config.lmstudio.admin_url = admin_url or config.lmstudio.admin_url
    gateway = ModelGateway(config)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            if "api/v0" in request.url.path:
                return httpx.Response(200, json=ADMIN_MODELS)
            return httpx.Response(200, json=OPENAI_MODELS)
        if request.url.path.endswith("/embeddings"):
            n = len(json.loads(request.content).get("input", []))
            return httpx.Response(200, json={"data": [{"index": i, "embedding": [float(i)]} for i in range(n)]})
        return httpx.Response(404, json={"error": "not found"})

    real_client = httpx.AsyncClient

    def client_factory(**kwargs):
        kwargs.pop("transport", None)
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    return gateway


def test_admin_catalogue_exposes_state_and_capabilities(monkeypatch):
    gateway = _gateway(monkeypatch)
    result = asyncio.run(gateway.list_models(refresh=True))
    assert len(result) == 2
    chat = [m for m in result if m.is_chat]
    assert chat[0].id == "qwen3-8b"
    assert chat[0].state == "loaded"
    assert chat[0].quantization == "Q4_K_M"
    assert chat[0].capabilities == ["tool_use"]
    assert chat[0].max_context_length == 8192
    assert not result[1].is_chat


def test_catalogue_is_cached_for_ten_seconds(monkeypatch):
    gateway = _gateway(monkeypatch)
    first = asyncio.run(gateway.list_models(refresh=True))
    # A second, non-refreshing call must not hit the network again.
    calls = []

    def counting_client(**kwargs):
        calls.append(kwargs)
        raise AssertionError("catalogue should have been served from cache")

    monkeypatch.setattr(httpx, "AsyncClient", counting_client)
    assert asyncio.run(gateway.list_models()) is first


def test_catalogue_falls_back_when_the_admin_api_is_down(monkeypatch):
    config = AppConfig()
    config.provider = "lmstudio"
    config.lmstudio.admin_url = "http://127.0.0.1:1/api/v0"  # unreachable
    gateway = ModelGateway(config)

    def handler(request: httpx.Request) -> httpx.Response:
        if "api/v0" in request.url.path:
            raise httpx.ConnectError("admin API down")
        return httpx.Response(200, json=OPENAI_MODELS)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    models = asyncio.run(gateway.list_models(refresh=True))
    assert [m.id for m in models] == ["qwen3-8b", "text-embedding-nomic-embed-text-v1.5"]
    assert all(m.state == "unknown" for m in models)


def test_catalogue_is_empty_not_a_crash_when_nothing_is_reachable(monkeypatch):
    config = AppConfig()
    config.provider = "lmstudio"
    config.lmstudio.admin_url = ""
    gateway = ModelGateway(config)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no server")

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    assert asyncio.run(gateway.list_models(refresh=True)) == []


def test_health_reports_chat_and_embedding_models_separately(monkeypatch):
    gateway = _gateway(monkeypatch)
    health = asyncio.run(gateway.health())
    assert health["reachable"] is True
    assert health["provider"] == "lmstudio"
    assert [m["id"] for m in health["chat_models"]] == ["qwen3-8b"]
    assert [m["id"] for m in health["embedding_models"]] == ["nomic-embed-text-v1.5"]
    assert health["loaded"] == ["qwen3-8b"]
    assert health["default_model"] == "qwen3-8b"


def test_resolve_model_prefers_explicit_then_config_then_loaded(monkeypatch):
    gateway = _gateway(monkeypatch)
    assert asyncio.run(gateway.resolve_model("my-model")) == "my-model"

    gateway.config.lmstudio.default_model = "configured"
    assert asyncio.run(gateway.resolve_model(None)) == "configured"

    gateway.config.lmstudio.default_model = None
    assert asyncio.run(gateway.resolve_model(None)) == "qwen3-8b"


def test_resolve_model_returns_none_when_nothing_is_reachable(monkeypatch):
    config = AppConfig()
    config.provider = "lmstudio"
    config.lmstudio.default_model = None
    config.lmstudio.admin_url = ""
    gateway = ModelGateway(config)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no server")

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    assert asyncio.run(gateway.resolve_model(None)) is None
    assert asyncio.run(gateway.model_info("qwen3-8b")) is None


def test_model_info_looks_up_by_id(monkeypatch):
    gateway = _gateway(monkeypatch)
    assert asyncio.run(gateway.model_info("qwen3-8b")).quantization == "Q4_K_M"
    assert asyncio.run(gateway.model_info("nonexistent")) is None
    assert asyncio.run(gateway.model_info(None)) is None


def test_role_model_uses_override_then_role_then_profile_default():
    config = AppConfig()
    config.provider = "lmstudio"
    config.lmstudio.default_model = "profile-default"
    config.roles["critic"].model = "critic-model"
    gateway = ModelGateway(config)
    assert gateway.role_model("critic") == "critic-model"
    assert gateway.role_model("critic", {"critic": "override"}) == "override"
    assert gateway.role_model("writer") == "profile-default"


def test_chat_client_is_cached_per_configuration_but_not_across_temperature():
    gateway = ModelGateway(AppConfig())
    first = gateway.chat("qwen3-8b", temperature=0.2)
    assert gateway.chat("qwen3-8b", temperature=0.2) is first
    assert gateway.chat("qwen3-8b", temperature=0.7) is not first
    assert gateway.chat("other-model") is not first


def test_chat_without_a_model_says_what_to_do():
    config = AppConfig()
    config.provider = "strata"
    gateway = ModelGateway(config)
    with pytest.raises(RuntimeError) as exc:
        gateway.chat(None)
    assert "strata" in str(exc.value).lower()


def test_embed_batches_and_records_success(monkeypatch):
    gateway = _gateway(monkeypatch)
    gateway.config.knowledge.embed_batch_size = 2
    vectors = asyncio.run(gateway.embed(["a", "b", "c", "d", "e"]))
    assert vectors is not None
    assert len(vectors) == 5
    assert gateway.embed_error is None


def test_embed_out_of_order_rows_are_reordered_by_index(monkeypatch):
    config = AppConfig()
    config.provider = "lmstudio"
    config.lmstudio.embedding_model = "embed"
    gateway = ModelGateway(config)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}]})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    assert asyncio.run(gateway.embed(["a", "b"])) == [[1.0], [2.0]]


def test_embed_size_mismatch_is_reported_not_silently_truncated(monkeypatch):
    config = AppConfig()
    config.provider = "lmstudio"
    config.lmstudio.embedding_model = "embed"
    gateway = ModelGateway(config)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})  # asked for 3

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    assert asyncio.run(gateway.embed(["a", "b", "c"])) is None
    assert "size mismatch" in (gateway.embed_error or "")


def test_embed_returns_none_when_no_embedding_model_is_configured(monkeypatch):
    gateway = _gateway(monkeypatch)
    gateway.config.lmstudio.embedding_model = None
    assert asyncio.run(gateway.embed(["a"])) is None
    gateway.config.lmstudio.embedding_model = "embed"
    assert asyncio.run(gateway.embed([])) is None


# -- content normalisation ------------------------------------------------------


def test_message_text_handles_every_content_shape():
    assert _message_text("plain") == "plain"
    assert _message_text([{"type": "text", "text": "a"}, "b"]) == "ab"
    assert _message_text([{"type": "output_text", "text": "a"}]) == "a"
    assert _message_text([{"type": "text", "content": "a"}]) == "a"
    assert _message_text([{"type": "image_url", "url": "x"}]) == ""
    assert _message_text(None) == ""


def test_chunk_delta_separates_answer_from_reasoning():
    class Chunk:
        content = [{"type": "text", "text": "the answer"}]
        additional_kwargs = {"reasoning_content": "thinking hard"}

    text, reasoning = _chunk_delta(Chunk())
    assert text == "the answer"
    assert reasoning == "thinking hard"

    class NoReasoning:
        content = "answer"
        additional_kwargs = {}

    assert _chunk_delta(NoReasoning()) == ("answer", "")

    class ListReasoning:
        content = "answer"
        additional_kwargs = {"reasoning": [{"type": "text", "text": "step"}]}

    assert _chunk_delta(ListReasoning()) == ("answer", "step")


def _just(value):
    """Return an awaitable for a stubbed `resolve_model`, so the probe path can be tested offline."""

    async def _resolved():
        return value

    return _resolved()


async def test_tool_probe_proves_capability_when_the_catalogue_is_silent(monkeypatch):
    """`probe_tools` is the only way to know whether a plain OpenAI-compatible server can call tools.

    The probe must actually bind the tool before calling, and must cache: a run would otherwise pay one
    round trip per structured call to learn the same fact.
    """
    from langchain_core.messages import AIMessage

    calls: list[int] = []
    bound: list[list] = []

    class StubChat:
        def bind_tools(self, tools):
            bound.append(list(tools))
            return self

        async def ainvoke(self, messages):
            calls.append(1)
            return AIMessage(content="", tool_calls=[{"name": "ping", "args": {}, "id": "c1"}])

    gateway = ModelGateway(AppConfig(provider="strata"))
    monkeypatch.setattr(gateway, "chat", lambda model, **kwargs: StubChat())
    monkeypatch.setattr(gateway, "resolve_model", lambda requested: _just("q2_0"))

    assert await gateway.probe_tools("q2_0") is True
    assert bound and bound[0], "the probe must attach the tool, not just ask in prose"
    assert await gateway.probe_tools("q2_0") is True
    assert len(calls) == 1, "the probe result is cached"


async def test_tool_probe_reports_false_when_the_model_answers_in_prose(monkeypatch):
    from langchain_core.messages import AIMessage

    class StubChat:
        def bind_tools(self, tools):
            del tools
            return self

        async def ainvoke(self, messages):
            del messages
            return AIMessage(content="I cannot call tools.")

    gateway = ModelGateway(AppConfig(provider="lmstudio"))
    monkeypatch.setattr(gateway, "chat", lambda model, **kwargs: StubChat())
    monkeypatch.setattr(gateway, "resolve_model", lambda requested: _just("small-model"))
    assert await gateway.probe_tools("small-model") is False


async def test_tool_probe_is_falsey_when_no_model_is_available(monkeypatch):
    gateway = ModelGateway(AppConfig(provider="strata"))
    monkeypatch.setattr(gateway, "resolve_model", lambda requested: _just(None))
    assert await gateway.probe_tools(None) is False


def test_rebinding_a_provider_clears_the_tool_probe(monkeypatch):
    gateway = ModelGateway(AppConfig(provider="strata"))
    gateway._tool_probe["q2_0"] = True
    gateway.rebind()
    assert not gateway._tool_probe, "a model's tool capability is per-provider and must not be reused"


def test_both_local_lanes_ask_for_one_tool_call_at_a_time():
    """`parallel_tool_calls` has to reach the client, not just sit in the config.

    A local Qwen spends most of the token budget on reasoning before it emits a tool call; when the
    backend batches several calls the payload is truncated mid-flight and the tool never runs. The
    AutoGen reference implementation sets this to false for both lanes, so an app that ignores the
    field would be tested against a different request shape.
    """
    for provider in ("lmstudio", "strata"):
        gateway = ModelGateway(AppConfig(provider=provider))
        assert gateway._client_kwargs["model_kwargs"]["parallel_tool_calls"] is False, provider
        gateway.config.active_profile.parallel_tool_calls = True
        gateway.rebind()
        assert gateway._client_kwargs["model_kwargs"]["parallel_tool_calls"] is True, provider
