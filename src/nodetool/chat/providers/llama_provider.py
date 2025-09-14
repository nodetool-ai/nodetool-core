"""
Llama.cpp OpenAI-compatible provider.

Uses openai.AsyncClient against a llama-server base_url. Automatically starts a
background llama-server (per model) via LlamaServerManager and keeps it alive
with a TTL.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Sequence
import asyncio

import httpx
import openai
from openai.types.chat import ChatCompletionChunk

from nodetool.agents.tools.base import Tool
from nodetool.chat.providers.base import ChatProvider
from nodetool.chat.providers.openai_compat import OpenAICompat
from nodetool.chat.providers.llama_server_manager import LlamaServerManager
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import Message, Provider, ToolCall
from nodetool.workflows.types import Chunk
from nodetool.workflows.processing_context import ProcessingContext

log = get_logger(__name__)


class LlamaProvider(ChatProvider, OpenAICompat):
    provider: Provider = Provider.LlamaCpp

    def __init__(self, ttl_seconds: int = 300):
        super().__init__()
        self._manager = LlamaServerManager(ttl_seconds=ttl_seconds)
        self._usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_prompt_tokens": 0,
            "reasoning_tokens": 0,
        }

    def _normalize_messages_for_llama(
        self, messages: Sequence[Message]
    ) -> list[Message]:
        """
        llama.cpp's jinja templates require roles to alternate user/assistant
        (optionally preceded by a single system message). The OpenAI-compatible
        "tool" role breaks this alternation. We normalize by:

        - Merging multiple system messages into a single system message
        - Converting tool messages into user messages that include the tool result
        - Inserting an empty assistant message before a tool result when needed
        - Leaving other roles unchanged

        This preserves semantics while satisfying the alternation constraint:
        user → assistant(tool_calls) → user(tool result) → assistant …
        """

        system_parts: list[str] = []
        normalized: list[Message] = []

        for msg in messages:
            if msg.role == "system":
                system_parts.append(str(msg.content) if msg.content is not None else "")
                continue
            if msg.role == "tool":
                # Represent tool output as a user turn for alternation.
                content_str: str
                if isinstance(msg.content, str):
                    content_str = msg.content
                else:
                    try:
                        content_str = json.dumps(msg.content)  # type: ignore[arg-type]
                    except Exception:
                        content_str = str(msg.content)

                # For Gemma models, avoid adding empty assistant messages as they break the chat template
                # Only add empty assistant message if the normalized list is empty or the last message was a system message
                if (
                    normalized
                    and normalized[-1].role != "assistant"
                    and normalized[-1].role != "system"
                ):
                    # Skip adding empty assistant message - let Gemma handle it naturally
                    pass

                prefix = (
                    f"Tool {msg.name or ''} result:\n" if msg.name else "Tool result:\n"
                )
                normalized.append(
                    Message(role="user", content=f"{prefix}{content_str}")
                )
                continue
            normalized.append(msg)

        if system_parts:
            system = Message(
                role="system", content="\n".join(p for p in system_parts if p)
            )
            normalized = [system] + normalized

        # Enforce strict alternation after optional system: user, assistant, user, ...
        if not normalized:
            return []

        start_index = 1 if normalized[0].role == "system" else 0
        fixed: list[Message] = []
        if start_index == 1:
            fixed.append(normalized[0])

        expected_user_turn = True  # After system or at start, expect a user
        for msg in normalized[start_index:]:
            # Map any non-standard roles (just in case) to user
            role = msg.role if msg.role in ("user", "assistant") else "user"

            # Insert blanks until this message fits the expected parity
            while (role == "user") != expected_user_turn:
                fixed.append(
                    Message(
                        role=("user" if expected_user_turn else "assistant"), content=""
                    )
                )
                expected_user_turn = not expected_user_turn

            # Append the actual message (preserve original content)
            if role != msg.role:
                fixed.append(
                    Message(role=role, content=msg.content, tool_calls=msg.tool_calls)
                )
            else:
                fixed.append(msg)

            expected_user_turn = not expected_user_turn

        return fixed

    def get_container_env(self) -> dict[str, str]:
        return {}

    def get_client(self, base_url: str) -> openai.AsyncClient:
        # llama-server accepts any API key; None is fine when auth is disabled
        return openai.AsyncClient(
            base_url=f"{base_url}/v1",
            api_key="sk-no-key-required",
            http_client=httpx.AsyncClient(
                follow_redirects=True, timeout=600, verify=False
            ),
        )

    def get_context_length(self, model: str) -> int:
        # Defer to server; commonly 4k-128k. Return a safe default.
        return 128000

    async def generate_messages(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        max_tokens: int = 1024,
        context_window: int = 128000,
        response_format: dict | None = None,
        **kwargs,
    ) -> AsyncIterator[Chunk | ToolCall]:
        base_url = await self._manager.ensure_server(model)
        _kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if response_format is not None:
            _kwargs["response_format"] = response_format
        if len(tools) > 0:
            _kwargs["tools"] = self.format_tools(tools)

        # Normalize messages to satisfy llama.cpp alternation constraints
        messages_normalized = self._normalize_messages_for_llama(messages)
        # llama.cpp is sensitive to unsupported fields; pass only necessary ones
        openai_messages = [await self.convert_message(m) for m in messages_normalized]

        self._log_api_request("chat_stream", messages_normalized, **_kwargs)

        client = self.get_client(base_url)
        completion = await client.chat.completions.create(
            messages=openai_messages, **_kwargs
        )

        delta_tool_calls: dict[int, dict[str, Any]] = {}
        current_chunk = ""

        async for chunk in completion:
            chunk = chunk  # type: ignore
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if (
                getattr(delta, "content", None)
                or chunk.choices[0].finish_reason == "stop"
            ):
                current_chunk += delta.content or ""
                finish_reason = chunk.choices[0].finish_reason
                if finish_reason == "stop":
                    self._log_api_response(
                        "chat_stream",
                        Message(role="assistant", content=current_chunk),
                    )
                yield Chunk(content=delta.content or "", done=finish_reason == "stop")

            if chunk.choices[0].finish_reason == "tool_calls":
                if delta_tool_calls:
                    for tc in delta_tool_calls.values():
                        tool_call = ToolCall(
                            id=tc["id"],
                            name=tc["name"],
                            args=json.loads(tc["function"]["arguments"]),
                        )
                        self._log_tool_call(tool_call)
                        yield tool_call

            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    tc = delta_tool_calls.get(tool_call.index) or {"id": tool_call.id}
                    delta_tool_calls[tool_call.index] = tc
                    if tool_call.id:
                        tc["id"] = tool_call.id
                    if tool_call.function and tool_call.function.name:
                        tc["name"] = tool_call.function.name
                    if tool_call.function and tool_call.function.arguments:
                        if "function" not in tc:
                            tc["function"] = {}
                        if "arguments" not in tc["function"]:
                            tc["function"]["arguments"] = ""
                        tc["function"]["arguments"] += tool_call.function.arguments

    async def generate_message(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        max_tokens: int = 1024,
        context_window: int = 128000,
        response_format: dict | None = None,
        **kwargs,
    ) -> Message:
        base_url = await self._manager.ensure_server(model)
        _kwargs: dict[str, Any] = {
            "max_tokens": max_tokens,
            "stream": False,
        }
        if response_format is not None:
            _kwargs["response_format"] = response_format
        if len(tools) > 0:
            _kwargs["tools"] = self.format_tools(tools)
        # Pass through additional sampling/params
        if kwargs:
            _kwargs.update(kwargs)

        messages_normalized = self._normalize_messages_for_llama(messages)
        openai_messages = [await self.convert_message(m) for m in messages_normalized]

        # Debug: print the processed messages
        # print("DEBUG: Normalized messages:", [{"role": m.role, "content": m.content} for m in messages_normalized])
        # print("DEBUG: OpenAI messages:", openai_messages)
        # print("DEBUG: Tools:", len(tools) if tools else 0)

        self._log_api_request("chat", messages_normalized, model=model, **_kwargs)

        client = self.get_client(base_url)
        completion = await client.chat.completions.create(
            model=model, messages=openai_messages, **_kwargs
        )

        choice = completion.choices[0]
        response_message = choice.message

        def try_parse_args(args: Any) -> Any:
            try:
                return json.loads(args)
            except Exception:
                return {}

        tool_calls = None
        if response_message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,  # type: ignore
                    args=try_parse_args(tool_call.function.arguments),  # type: ignore
                )
                for tool_call in response_message.tool_calls
            ]

        message = Message(
            role="assistant", content=response_message.content, tool_calls=tool_calls
        )
        self._log_api_response("chat", message)
        return message

    def get_usage(self) -> dict:
        return self._usage.copy()

    def reset_usage(self) -> None:
        self._usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_prompt_tokens": 0,
            "reasoning_tokens": 0,
        }


async def _smoke_test() -> None:
    """Simple smoke test that spins up llama-server and requests a short reply."""
    provider = LlamaProvider(ttl_seconds=120)
    model = "Qwen/Qwen2.5-7B-Instruct-GGUF"
    messages: list[Message] = [
        Message(role="user", content="Tell me what model is this?"),
    ]
    try:
        reply = await provider.generate_message(
            messages=messages, model=model, max_tokens=32
        )
        content_str = reply.content if isinstance(reply.content, str) else ""
        print("Response:", content_str.strip())
    except Exception as e:
        # Keep errors visible for quick diagnostics
        print("Smoke test failed:", e)


if __name__ == "__main__":

    async def _run_all():
        # Tool call test
        print("--- Tool call test ---")

        class EchoTool(Tool):
            name = "echo"
            description = "Echo back the provided text."
            input_schema = {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            }

            async def process(self, context: ProcessingContext, params: dict[str, Any]) -> Any:  # type: ignore[override]
                return {"echo": params.get("text", "")}

        provider = LlamaProvider(ttl_seconds=120)
        model = "Qwen/Qwen2.5-7B-Instruct-GGUF"
        tools: list[Tool] = [EchoTool()]
        context = ProcessingContext()

        # First try without tools to confirm basic functionality
        messages = [
            Message(role="user", content="Just say 'Hello World'"),
        ]
        print("\nTesting without tools first...")
        resp = await provider.generate_message(
            messages=messages, model=model, max_tokens=64
        )
        print("Basic test successful:", resp.content)

        # Now try with tools
        messages = [
            Message(
                role="user",
                content="Use the echo tool with text 'llama', then say done.",
            ),
        ]
        try:
            # Now test with tools only
            print("\nTesting with tools...")
            resp = await provider.generate_message(
                messages=messages, model=model, tools=tools, max_tokens=64
            )
            print("First response successful:", resp.content)

            if resp.tool_calls:
                print(
                    f"✅ Tool call test PASSED! Got {len(resp.tool_calls)} tool calls:"
                )
                for tc in resp.tool_calls:
                    print(f"  - {tc.name}: {tc.args}")
                    tool = next((t for t in tools if t.name == tc.name), None)
                    if not tool:
                        continue
                    result = await tool.process(context, tc.args or {})
                    print(f"  - Tool result: {result}")

                # Use standard OpenAI approach only
                print("🔄 Testing standard OpenAI tool calling approach...")

                try:
                    messages.append(
                        Message(
                            role="tool",
                            name=tool.name,
                            tool_call_id=tc.id,
                            content=json.dumps(result),
                        )
                    )
                    final = await provider.generate_message(
                        messages=messages, model=model, tools=tools, max_tokens=64
                    )
                    content_str = (
                        final.content if isinstance(final.content, str) else ""
                    )
                    print("✅ OpenAI approach: SUCCESS -", content_str.strip())
                except Exception as e:
                    print("❌ OpenAI approach failed:", e)
                    print(
                        "Tool execution successful but conversation continuation failed."
                    )
            else:
                content_str = resp.content if isinstance(resp.content, str) else ""
                print("❌ No tool call returned. Model said:", content_str.strip())
        except Exception as e:
            print("Tool call test failed:", e)

    asyncio.run(_run_all())
