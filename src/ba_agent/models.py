"""Model gateway for LM Studio (OpenAI-compatible) built on plain ``langchain-openai``.

Responsibilities:

* **Catalogue** — list models with load state, tool-calling capability and context length so the UI
  dropdown can make informed choices (LM Studio admin API ``/api/v0/models``, falling back to ``/v1/models``).
* **Per-role routing** — one default model plus optional per-role overrides, hot-swappable.
* **Structured output ladder** — ``json_schema`` → ``function_calling`` → prompt+repair, so the same code
  runs on a frontier-class local model and on a 4B that only produces prose.
* **Resource safety** — one global semaphore (never two requests fighting for VRAM), load-aware timeouts,
  bounded retries and usage accounting.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Iterable, Sequence, Type, TypeVar

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel as _PBaseModel
from pydantic import Field

from .config import AppConfig, get_config
from .events import Emitter, EventType, null_emitter
from .jsonx import dumps, extract_json

T = TypeVar("T", bound=_PBaseModel)


# --------------------------------------------------------------------------- catalogue


class ModelInfo(_PBaseModel):
    id: str
    type: str = "llm"
    state: str = "unknown"  # loaded | not-loaded | loading | unknown
    capabilities: list[str] = Field(default_factory=list)
    max_context_length: int | None = None
    loaded_context_length: int | None = None
    quantization: str | None = None
    publisher: str | None = None

    @property
    def is_chat(self) -> bool:
        return self.type in ("llm", "vlm") or self.type == "unknown"

    @property
    def supports_tools(self) -> bool:
        return "tool_use" in self.capabilities

    @property
    def context_length(self) -> int | None:
        return self.loaded_context_length or self.max_context_length


class StructuredResult(_PBaseModel):
    parsed: Any | None = None
    raw_text: str = ""
    method: str = ""
    attempts: int = 0
    latency_ms: int = 0
    model: str = ""
    degraded: bool = False
    error: str | None = None


class TextResult(_PBaseModel):
    text: str = ""
    reasoning: str = ""
    latency_ms: int = 0
    model: str = ""
    error: str | None = None


# --------------------------------------------------------------------------- helpers


def _message_text(content: Any) -> str:
    """Normalise langchain message content (str or list of content blocks) to text."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") in ("text", "output_text") and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block.get("content"), str):
                    parts.append(block["content"])
    return "".join(parts)


def _chunk_delta(chunk: Any) -> tuple[str, str]:
    """Return ``(text_delta, reasoning_delta)`` from a streamed chunk."""
    text = _message_text(getattr(chunk, "content", ""))
    extra = getattr(chunk, "additional_kwargs", {}) or {}
    reasoning = extra.get("reasoning_content") or extra.get("reasoning") or ""
    if not isinstance(reasoning, str):
        reasoning = _message_text(reasoning)
    return text or "", reasoning or ""


def build_messages(system_prompt: str | None, user_prompt: str | None,
                   extra: Sequence[BaseMessage] | None = None) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    if extra:
        messages.extend(extra)
    if user_prompt is not None:
        messages.append(HumanMessage(content=user_prompt))
    return messages


_SCHEMA_INSTRUCTION = """\
ANSWER FORMAT (STRICT)
Return a single JSON object and nothing else — no prose, no code fences.
It must validate against this JSON Schema:
{schema}
"""


# --------------------------------------------------------------------------- gateway

#: Shown verbatim in the UI when nothing can be called — it has to say what to do about it.
NO_MODEL_MESSAGE = (
    "No LM Studio chat model available. Start LM Studio and load a chat model (or pick one in the model "
    "dropdown), then press Resume."
)


class ModelGateway:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        lm = self.config.lmstudio
        self._http_timeout = httpx.Timeout(lm.request_timeout_s, connect=lm.connect_timeout_s)
        self._client_kwargs: dict[str, Any] = {
            "api_key": lm.api_key,
            "base_url": lm.base_url,
            "timeout": self._http_timeout,
            "max_retries": 0,
        }
        self._cache: dict[tuple, ChatOpenAI] = {}
        self._semaphore = asyncio.Semaphore(max(1, lm.max_concurrency))
        self._catalog: list[ModelInfo] = []
        self._catalog_ts: float = 0.0
        self.usage: dict[str, Any] = {"calls": 0, "failures": 0, "prompt_chars": 0,
                                      "completion_chars": 0, "tokens_in": 0, "tokens_out": 0}
        #: Cumulative time actually spent inside model calls (used for per-step metrics in the UI).
        self.total_model_ms: int = 0

    # -- catalogue ----------------------------------------------------------

    async def list_models(self, refresh: bool = False) -> list[ModelInfo]:
        lm = self.config.lmstudio
        now = time.monotonic()
        if not refresh and self._catalog and (now - self._catalog_ts) < 10:
            return self._catalog

        models: list[ModelInfo] = []
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(8.0)) as client:
                resp = await client.get(f"{lm.admin_url}/models")
                resp.raise_for_status()
                payload = resp.json()
            for item in payload.get("data", []):
                models.append(ModelInfo(
                    id=item.get("id", ""),
                    type=item.get("type") or "llm",
                    state=item.get("state") or "unknown",
                    capabilities=item.get("capabilities") or [],
                    max_context_length=item.get("max_context_length"),
                    loaded_context_length=item.get("loaded_context_length"),
                    quantization=item.get("quantization"),
                    publisher=item.get("publisher"),
                ))
        except Exception:
            models = await self._list_models_openai_compat()

        self._catalog = models
        self._catalog_ts = now
        return models

    async def _list_models_openai_compat(self) -> list[ModelInfo]:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(8.0)) as client:
                resp = await client.get(f"{self.config.lmstudio.base_url}/models",
                                        headers={"Authorization": f"Bearer {self.config.lmstudio.api_key}"})
                resp.raise_for_status()
                data = resp.json().get("data", [])
        except Exception:
            return []
        out = []
        for item in data:
            model_id = str(item.get("id", ""))
            is_embed = "embed" in model_id.lower()
            out.append(ModelInfo(id=model_id, type="embeddings" if is_embed else "llm", state="unknown"))
        return out

    async def health(self) -> dict[str, Any]:
        models = await self.list_models(refresh=True)
        chat = [m for m in models if m.is_chat]
        loaded = [m for m in chat if m.state == "loaded"]
        default = await self.resolve_model(None)
        return {
            "reachable": bool(models),
            "base_url": self.config.lmstudio.base_url,
            "chat_models": [m.model_dump() for m in chat],
            "embedding_models": [m.model_dump() for m in models if not m.is_chat],
            "loaded": [m.id for m in loaded],
            "default_model": default,
        }

    async def resolve_model(self, requested: str | None) -> str | None:
        """Pick a usable model id: explicit → config default → first loaded chat → first chat."""
        if requested:
            return requested
        cfg_default = self.config.lmstudio.default_model
        if cfg_default:
            return cfg_default
        models = [m for m in await self.list_models() if m.is_chat]
        for model in models:
            if model.state == "loaded":
                return model.id
        return models[0].id if models else None

    async def model_info(self, model_id: str | None) -> ModelInfo | None:
        if not model_id:
            return None
        for model in await self.list_models():
            if model.id == model_id:
                return model
        return None

    # -- chat clients -------------------------------------------------------

    def role_model(self, role: str, overrides: dict[str, str] | None = None) -> str | None:
        overrides = overrides or {}
        return overrides.get(role) or self.config.role(role).model or self.config.lmstudio.default_model

    def chat(self, model: str | None, *, temperature: float | None = None,
             max_output_tokens: int | None = None, streaming: bool = False) -> ChatOpenAI:
        if not model:
            raise RuntimeError(
                "No LM Studio model selected and none are loaded. Start a chat model in LM Studio "
                "(or pick one in the UI dropdown)."
            )
        key = (model, temperature, max_output_tokens, streaming)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        kwargs: dict[str, Any] = {**self._client_kwargs, "model": model, "streaming": streaming}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_output_tokens:
            kwargs["max_tokens"] = max_output_tokens
        client = ChatOpenAI(**kwargs)
        self._cache[key] = client
        return client

    # -- structured calls ---------------------------------------------------

    def _ladder(self, model_id: str | None) -> list[str]:
        configured = (self.config.lmstudio.structured_output_method or "auto").lower()
        if configured in ("json_schema", "function_calling", "prompt"):
            return [configured]
        ladder = ["json_schema"]
        info = next((m for m in self._catalog if m.id == model_id), None)
        if info is None or info.supports_tools:
            ladder.append("function_calling")
        ladder.append("prompt")
        return ladder

    async def structured(
        self,
        schema: Type[T],
        *,
        system_prompt: str,
        user_prompt: str,
        role: str = "",
        model_override: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        emit: Emitter = null_emitter,
        step: str = "",
    ) -> StructuredResult:
        """Invoke the model until the reply validates against *schema* (or the ladder is exhausted)."""
        params = self.config.role(role) if role else None
        model_id = await self.resolve_model(model_override or (params.model if params else None))
        temperature = params.temperature if (temperature is None and params) else temperature
        max_output_tokens = (params.max_output_tokens if max_output_tokens is None and params else max_output_tokens)

        # Nothing to call: fail once, loudly, instead of failing the whole ladder with retries and burying the
        # real reason ("LM Studio has no chat model") under nine identical warnings.
        if not model_id:
            self.usage["failures"] += 1
            await emit(EventType.WARNING, {"step": step, "role": role,
                                           "message": NO_MODEL_MESSAGE})
            return StructuredResult(parsed=None, raw_text="", method="no_model", attempts=0, degraded=True,
                                    error=NO_MODEL_MESSAGE)

        started = time.perf_counter()
        attempts = 0
        errors: list[str] = []
        raw_text = ""

        for method in self._ladder(model_id):
            messages = build_messages(system_prompt, user_prompt)
            if method == "prompt":
                messages[0].content = (
                    f"{system_prompt}\n\n{_SCHEMA_INSTRUCTION.format(schema=dumps(schema.model_json_schema()))}"
                )
            for attempt in range(1, self.config.lmstudio.max_retries + 2):
                attempts += 1
                try:
                    async with self._semaphore:
                        if method == "prompt":
                            raw_text = await self._complete(model_id, messages, temperature, max_output_tokens)
                            data = extract_json(raw_text)
                            if data is None:
                                raise ValueError("no JSON object found in the model reply")
                            parsed = schema.model_validate(data)
                        else:
                            runnable = self.chat(model_id, temperature=temperature,
                                                 max_output_tokens=max_output_tokens).with_structured_output(
                                schema, method=method, include_raw=True)
                            result = await runnable.ainvoke(messages)
                            raw_message = result.get("raw") if isinstance(result, dict) else None
                            raw_text = _message_text(getattr(raw_message, "content", ""))
                            parsed = result.get("parsed") if isinstance(result, dict) else result
                            if parsed is None:
                                raise ValueError(str((result or {}).get("parsing_error") or "model returned no valid payload"))

                    latency_ms = int((time.perf_counter() - started) * 1000)
                    self.usage["calls"] += 1
                    self.total_model_ms += latency_ms
                    self.usage["prompt_chars"] += sum(len(m.content) for m in messages if isinstance(m.content, str))
                    self.usage["completion_chars"] += len(raw_text)
                    return StructuredResult(
                        parsed=parsed, raw_text=raw_text, method=method, attempts=attempts,
                        latency_ms=int((time.perf_counter() - started) * 1000), model=model_id or "",
                    )
                except Exception as exc:  # noqa: BLE001 — every failure is recoverable via the ladder
                    errors.append(f"{method}#{attempt}: {type(exc).__name__}: {exc}")
                    self.usage["failures"] += 1
                    await emit(EventType.WARNING, {
                        "step": step, "role": role, "model": model_id,
                        "message": f"structured output failed ({method}, attempt {attempt}) — retrying",
                        "detail": str(exc)[:400],
                    })
                    messages = messages + [AIMessage(content=raw_text or ""), HumanMessage(
                        content="Your previous reply was not valid for the required JSON schema.\n"
                                f"Validation problem: {str(exc)[:500]}\n"
                                "Return ONLY the corrected JSON object, nothing else."
                    )]
                    raw_text = ""

        # Thinking models can exhaust a token budget on internal reasoning and fail every rung of the ladder
        # without producing parseable output. One final uncapped prompt attempt recovers what the cap starved.
        if max_output_tokens is not None:
            await emit(EventType.WARNING, {
                "step": step, "role": role, "model": model_id,
                "message": (f"ladder exhausted under a {max_output_tokens}-token cap — trying once more without a cap"),
            })
            messages = build_messages(system_prompt, user_prompt)
            messages[0].content = (
                f"{system_prompt}\n\n{_SCHEMA_INSTRUCTION.format(schema=dumps(schema.model_json_schema()))}"
            )
            try:
                async with self._semaphore:
                    raw_text = await self._complete(model_id, messages, temperature, None)
                data = extract_json(raw_text)
                if data is not None:
                    parsed = schema.model_validate(data)
                    attempts += 1
                    self.usage["calls"] += 1
                    self.total_model_ms += int((time.perf_counter() - started) * 1000)
                    return StructuredResult(
                        parsed=parsed, raw_text=raw_text, method="prompt_uncapped", attempts=attempts,
                        latency_ms=int((time.perf_counter() - started) * 1000), model=model_id or "",
                    )
                errors.append(f"prompt_uncapped: no JSON object found in the model reply")
            except Exception as exc:  # noqa: BLE001
                self.usage["failures"] += 1
                attempts += 1
                errors.append(f"prompt_uncapped: {type(exc).__name__}: {exc}")

        return StructuredResult(
            parsed=None, raw_text="", method="failed", attempts=attempts,
            latency_ms=int((time.perf_counter() - started) * 1000), model=model_id or "",
            degraded=True, error=" | ".join(errors[-3:]),
        )

    # -- text calls ---------------------------------------------------------

    async def stream_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        role: str = "",
        model_override: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        emit: Emitter = null_emitter,
        step: str = "",
    ) -> TextResult:
        params = self.config.role(role) if role else None
        model_id = await self.resolve_model(model_override or (params.model if params else None))
        temperature = params.temperature if (temperature is None and params) else temperature
        max_output_tokens = (params.max_output_tokens if max_output_tokens is None and params else max_output_tokens)

        if not model_id:
            self.usage["failures"] += 1
            await emit(EventType.WARNING, {"step": step, "role": role, "message": NO_MODEL_MESSAGE})
            return TextResult(error=NO_MODEL_MESSAGE)

        started = time.perf_counter()
        model_ms = 0

        async def _stream_once(token_cap: int | None) -> tuple[str, str, str | None]:
            nonlocal model_ms
            chunks: list[str] = []
            reasoning: list[str] = []
            pending: list[str] = []

            async def flush() -> None:
                if not pending:
                    return
                text = "".join(pending)
                pending.clear()
                await emit(EventType.TOKEN, {"step": step, "role": role, "model": model_id, "text": text})

            t0 = time.perf_counter()
            try:
                client = self.chat(model_id, temperature=temperature, max_output_tokens=token_cap, streaming=True)
                async with self._semaphore:
                    async for chunk in client.astream(build_messages(system_prompt, user_prompt)):
                        delta, think = _chunk_delta(chunk)
                        if think:
                            reasoning.append(think)
                            await emit(EventType.REASONING, {"step": step, "role": role, "model": model_id, "text": think})
                        if delta:
                            chunks.append(delta)
                            pending.append(delta)
                            buffer_len = sum(len(p) for p in pending)
                            if "\n" in delta or buffer_len > 80:
                                await flush()
                await flush()
            except Exception as exc:  # noqa: BLE001
                self.usage["failures"] += 1
                model_ms += int((time.perf_counter() - t0) * 1000)
                return "".join(chunks), "".join(reasoning), f"{type(exc).__name__}: {exc}"
            self.usage["calls"] += 1
            model_ms += int((time.perf_counter() - t0) * 1000)
            return "".join(chunks), "".join(reasoning), None

        text, reasoning_text, error = await _stream_once(max_output_tokens)

        # Thinking models (e.g. Qwen3 served by LM Studio) can spend their entire token budget on internal
        # reasoning and emit zero visible content — the stream ends cleanly with finish_reason=length and an
        # empty completion. One uncapped retry recovers the output; request_timeout_s still bounds it.
        if not text.strip() and error is None and max_output_tokens is not None:
            await emit(EventType.WARNING, {
                "step": step, "role": role, "model": model_id,
                "message": (f"no visible output within {max_output_tokens} tokens — the model likely spent its "
                            f"budget on internal reasoning; retrying without a token cap"),
            })
            text2, reasoning_text2, error2 = await _stream_once(None)
            if text2.strip() or error2 is not None:
                text, reasoning_text, error = text2, reasoning_text2, error2

        latency_ms = int((time.perf_counter() - started) * 1000)
        self.total_model_ms += model_ms
        self.usage["completion_chars"] += len(text)
        return TextResult(text=text, reasoning=reasoning_text, model=model_id or "", latency_ms=latency_ms,
                          error=error)

    async def _complete(self, model_id: str | None, messages: list[BaseMessage],
                        temperature: float | None, max_output_tokens: int | None) -> str:
        client = self.chat(model_id, temperature=temperature, max_output_tokens=max_output_tokens)
        message = await client.ainvoke(messages)
        return _message_text(message.content)

    # -- agent loop support -------------------------------------------------

    def bind_tools(self, model_id: str | None, tools: Iterable[Any], *, role: str = ""):
        params = self.config.role(role) if role else None
        return self.chat(model_id, temperature=params.temperature if params else None).bind_tools(list(tools))

    # -- embeddings ---------------------------------------------------------

    async def embed(self, texts: Sequence[str]) -> list[list[float]] | None:
        model = self.config.lmstudio.embedding_model
        if not model or not texts:
            return None
        vectors: list[list[float]] = []
        batch = max(1, self.config.knowledge.embed_batch_size)
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.config.lmstudio.request_timeout_s,
                                                              connect=self.config.lmstudio.connect_timeout_s)) as client:
                for start in range(0, len(texts), batch):
                    slice_ = list(texts[start : start + batch])
                    resp = await client.post(
                        f"{self.config.lmstudio.base_url}/embeddings",
                        headers={"Authorization": f"Bearer {self.config.lmstudio.api_key}"},
                        json={"model": model, "input": slice_},
                    )
                    resp.raise_for_status()
                    data = resp.json().get("data", [])
                    if len(data) != len(slice_):
                        raise RuntimeError("embedding response size mismatch")
                    vectors.extend([item["embedding"] for item in sorted(data, key=lambda d: d.get("index", 0))])
        except Exception:
            return None
        return vectors

    async def aclose(self) -> None:
        self._cache.clear()


_gateway: ModelGateway | None = None


def get_gateway(config: AppConfig | None = None) -> ModelGateway:
    global _gateway
    if _gateway is None or (config is not None and _gateway.config is not config):
        _gateway = ModelGateway(config)
    return _gateway
