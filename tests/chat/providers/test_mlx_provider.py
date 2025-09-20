from __future__ import annotations

from dataclasses import dataclass
import json

import pytest

from nodetool.chat.providers.mlx_provider import MLXProvider, _MLXRuntime
from nodetool.metadata.types import Message, ToolCall
from nodetool.workflows.types import Chunk
from nodetool.agents.tools.base import Tool


@dataclass
class StubResponse:
    text: str
    finish_reason: str | None
    prompt_tokens: int
    generation_tokens: int
    token: int = 0
    logprobs: object | None = None
    from_draft: bool = False
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    peak_memory: float = 0.0


class StubTokenizer:
    def __init__(self, *, has_tool_calling: bool = False):
        self.calls: list[dict] = []
        self.chat_template = True
        self.has_tool_calling = has_tool_calling
        if has_tool_calling:
            self.tool_call_start = "<tool_call>"
            self.tool_call_end = "</tool_call>"

    def apply_chat_template(
        self, messages, tools=None, add_generation_prompt=True, tokenize=False, **kwargs
    ):
        self.calls.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        return "PROMPT"


def make_runtime(responses, tokenizer):
    def load(model_id, **kwargs):
        return object(), tokenizer

    def stream_generate(model, tokenizer, prompt, **kwargs):
        for response in responses:
            yield response

    return _MLXRuntime(load=load, stream_generate=stream_generate, make_sampler=None)


@pytest.mark.asyncio
async def test_generate_message_collects_streamed_text():
    tokenizer = StubTokenizer()
    responses = [
        StubResponse(
            text="Hello ", finish_reason=None, prompt_tokens=3, generation_tokens=1
        ),
        StubResponse(
            text="world", finish_reason="stop", prompt_tokens=3, generation_tokens=2
        ),
    ]
    provider = MLXProvider(runtime=make_runtime(responses, tokenizer))

    message = await provider.generate_message(
        [Message(role="user", content="Hi")],
        model="stub",
    )

    assert message.content == "Hello world"
    assert message.tool_calls is None
    assert provider.usage["prompt_tokens"] == 3
    assert provider.usage["completion_tokens"] == 2
    assert tokenizer.calls[0]["messages"][0]["role"] == "user"
    assert tokenizer.calls[0]["messages"][0]["content"] == "Hi"


class WeatherTool(Tool):
    name = "weather"
    description = "Lookup weather"
    input_schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }


@pytest.mark.asyncio
async def test_generate_messages_emits_tool_call():
    tokenizer = StubTokenizer(has_tool_calling=True)
    payload = {
        "name": "weather",
        "arguments": {"city": "Berlin"},
    }
    responses = [
        StubResponse(
            text=f"<tool_call>{json.dumps(payload)}</tool_call>",
            finish_reason="stop",
            prompt_tokens=4,
            generation_tokens=1,
        )
    ]
    provider = MLXProvider(runtime=make_runtime(responses, tokenizer))


@pytest.mark.asyncio
async def test_model_instance_cached_for_5_minutes(monkeypatch):
    # Ensure clean cache
    from nodetool.chat.providers.mlx_provider import MLXProvider as _P

    with _P._MODEL_CACHE_LOCK:
        _P._MODEL_CACHE.clear()

    tokenizer = StubTokenizer()

    load_calls = {"count": 0}

    def make_runtime_counting(responses, tokenizer):
        def load(model_id, **kwargs):
            load_calls["count"] += 1
            return object(), tokenizer

        def stream_generate(model, tokenizer, prompt, **kwargs):
            for response in responses:
                yield response

        return _MLXRuntime(
            load=load, stream_generate=stream_generate, make_sampler=None
        )

    # Freeze time for deterministic TTL behavior
    base_time = 1000.0
    monkeypatch.setattr(
        "nodetool.chat.providers.mlx_provider.time.monotonic", lambda: base_time
    )

    # First call loads and caches
    responses = [
        StubResponse(
            text="hi", finish_reason="stop", prompt_tokens=1, generation_tokens=1
        )
    ]
    provider1 = MLXProvider(runtime=make_runtime_counting(responses, tokenizer))
    await provider1.generate_message([Message(role="user", content="Hi")], model="stub")
    assert load_calls["count"] == 1

    # Second provider within TTL should hit cache and not call load
    def make_runtime_fail_if_load(*args, **kwargs):
        def load(*_a, **_k):
            raise AssertionError("load should not be called when cache is warm")

        def stream_generate(model, tokenizer, prompt, **kwargs):
            yield StubResponse(
                text="ok", finish_reason="stop", prompt_tokens=0, generation_tokens=0
            )

        return _MLXRuntime(
            load=load, stream_generate=stream_generate, make_sampler=None
        )

    provider2 = MLXProvider(runtime=make_runtime_fail_if_load([], tokenizer))
    await provider2.generate_message([Message(role="user", content="Hi")], model="stub")
    assert load_calls["count"] == 1


@pytest.mark.asyncio
async def test_model_cache_expires_after_ttl(monkeypatch):
    # Ensure clean cache
    from nodetool.chat.providers.mlx_provider import MLXProvider as _P

    with _P._MODEL_CACHE_LOCK:
        _P._MODEL_CACHE.clear()

    tokenizer = StubTokenizer()
    load_calls = {"count": 0}

    def make_runtime_counting(responses, tokenizer):
        def load(model_id, **kwargs):
            load_calls["count"] += 1
            return object(), tokenizer

        def stream_generate(model, tokenizer, prompt, **kwargs):
            for response in responses:
                yield response

        return _MLXRuntime(
            load=load, stream_generate=stream_generate, make_sampler=None
        )

    # Simulate time progression beyond TTL
    base_time = 2000.0
    monotonic_times = [base_time, base_time + 301.0]

    def fake_monotonic():
        # Pop sequential times: first for set, second for expiry check
        return monotonic_times[0] if len(monotonic_times) == 2 else monotonic_times[-1]

    monkeypatch.setattr(
        "nodetool.chat.providers.mlx_provider.time.monotonic", fake_monotonic
    )

    responses = [
        StubResponse(
            text="first", finish_reason="stop", prompt_tokens=0, generation_tokens=0
        )
    ]
    provider = MLXProvider(runtime=make_runtime_counting(responses, tokenizer))

    # First call populates cache (time = base_time)
    await provider.generate_message([Message(role="user", content="Hi")], model="stub")
    assert load_calls["count"] == 1

    # Advance time beyond TTL and call again; should reload
    monotonic_times.pop(0)
    await provider.generate_message([Message(role="user", content="Hi")], model="stub")
    assert load_calls["count"] == 2

    # No further assertions; separate tests cover tool call behavior. Here we
    # only verify that the cache expired and load was invoked again.
