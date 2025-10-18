"""
Llama.cpp OpenAI-compatible provider.

Uses openai.AsyncClient against a llama-server base_url. Automatically starts a
background llama-server (per model) via LlamaServerManager and keeps it alive
with a TTL.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, List, Sequence

import httpx
import openai
from huggingface_hub import hf_hub_download

from nodetool.agents.tools.base import Tool
from nodetool.providers.base import BaseProvider, register_provider
from nodetool.providers.openai_compat import OpenAICompat
from nodetool.providers.llama_server_manager import LlamaServerManager
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import Message, Provider, ToolCall, LanguageModel
from nodetool.workflows.types import Chunk
from nodetool.workflows.processing_context import ProcessingContext

log = get_logger(__name__)


@register_provider(Provider.LlamaCpp)
class LlamaProvider(BaseProvider, OpenAICompat):
    """OpenAI-compatible chat provider backed by a local llama.cpp server.

    This provider automatically manages a background ``llama-server`` process per
    model via ``LlamaServerManager`` and exposes a familiar OpenAI client
    interface. It normalizes messages to conform to llama.cpp's chat template
    alternation rules (user/assistant), and supports tool calls.

    Attributes:
        provider: Provider identifier used by the application.
    """

    provider_name: str = "llama_cpp"

    @classmethod
    def get_llama_server_manager(cls) -> LlamaServerManager:
        if not hasattr(cls, "_manager"):
            cls._manager = LlamaServerManager()
        return cls._manager

    def __init__(self, ttl_seconds: int = 300):
        """Initialize the provider and its server manager.

        Args:
            ttl_seconds: Inactivity time-to-live for each managed llama-server.
        """
        super().__init__()
        self._manager = self.get_llama_server_manager()
        self._usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_prompt_tokens": 0,
            "reasoning_tokens": 0,
        }

    def _normalize_messages_for_llama(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] = [],
        use_tool_emulation: bool = False,
    ) -> list[Message]:
        """Normalize messages to satisfy llama.cpp alternation constraints.

        llama.cpp's chat templates require strict role alternation between
        user and assistant turns, optionally preceded by a single system
        message. The OpenAI-compatible "tool" role violates this alternation.

        The normalization performs:
        - Merge multiple system messages into a single system message
        - Convert tool messages into user messages that embed tool output
        - Insert blank turns as necessary to maintain alternation
        - For tool emulation: add tool definitions to system message

        Args:
            messages: Original message sequence.
            tools: Optional tools for emulation.
            use_tool_emulation: Whether to use tool calling emulation.

        Returns:
            A list of messages compatible with llama.cpp chat templates.
        """

        system_parts: list[str] = []
        normalized: list[Message] = []

        # Add tool definitions for emulation
        if use_tool_emulation and tools:
            tool_definitions = self._format_tools_as_python(tools)
            tool_instruction = (
                "\n\n=== AVAILABLE FUNCTIONS ===\n"
                "You can call these functions by writing a function call on a single line.\n"
                "DO NOT write function definitions - only write function CALLS.\n\n"
                f"{tool_definitions}\n\n"
                "=== INSTRUCTIONS ===\n"
                "When you need to use a function:\n"
                "1. Write ONLY the function call, nothing else\n"
                "2. Use this exact format: function_name(param='value')\n"
                "3. Do NOT write 'def', 'return', or any other Python keywords\n"
                "4. After calling a function, wait for the result\n"
                "5. Once you receive a function result, use it in your final answer\n"
                "6. Do NOT call the same function twice\n\n"
                "Example conversation:\n"
                "User: What is 5 + 3?\n"
                "You: calculator(expression='5 + 3')\n"
                "[System returns: {'result': 8}]\n"
                "You: The answer is 8."
            )
            system_parts.append(tool_instruction)

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

                # Use emulation-friendly format if needed
                if use_tool_emulation:
                    # Ensure content is a plain string for strict chat templates
                    emulated_content = (
                        f"The function {msg.name}() returned the following result:\n{content_str}\n\n"
                        f"Use this result to answer the user's question. Do NOT call the function again."
                    )
                    # Explicitly create a message with string content only
                    normalized.append(
                        Message(role="user", content=str(emulated_content))
                    )
                else:
                    prefix = (
                        f"Tool {msg.name or ''} result:\n"
                        if msg.name
                        else "Tool result:\n"
                    )
                    normalized.append(
                        Message(role="user", content=str(f"{prefix}{content_str}"))
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
        """Return environment variables for containerized execution.

        Returns:
            A mapping of environment variables to inject. Empty for llama.cpp
            since the local server is spawned by this process.
        """
        return {}

    def get_client(self, base_url: str) -> openai.AsyncClient:
        """Create an OpenAI-compatible async client targeting llama-server.

        Args:
            base_url: Base URL returned by ``LlamaServerManager.ensure_server``.

        Returns:
            Configured ``openai.AsyncClient`` instance.
        """
        # llama-server accepts any API key; None is fine when auth is disabled
        return openai.AsyncClient(
            base_url=f"{base_url}/v1",
            api_key="sk-no-key-required",
            http_client=httpx.AsyncClient(
                follow_redirects=True, timeout=600, verify=False
            ),
        )

    def get_context_length(self, model: str) -> int:
        """Return an approximate context window for the provided model.

        Args:
            model: Model identifier passed to llama.cpp.

        Returns:
            A conservative default context size; server may support more.
        """
        # Defer to server; commonly 4k-128k. Return a safe default.
        return 128000

    def has_tool_support(self, model: str) -> bool:
        """Return True if the given model supports tools/function calling.

        The server manager queries model capabilities after server startup
        and caches them. This method simply checks if "tools" is in the
        cached capabilities list.

        Args:
            model: Model identifier passed to llama.cpp.

        Returns:
            True if the model has "tools" in its capabilities, False otherwise.
            Returns False if capabilities have not been queried yet (server not started).
        """
        log.debug(f"Checking tool support for model: {model}")

        capabilities = self._manager.get_model_capabilities(model)
        has_tools = "tools" in capabilities

        log.debug(
            f"Model {model} capabilities: {capabilities}, tools support: {has_tools}"
        )
        return has_tools

    async def get_available_language_models(self) -> List[LanguageModel]:
        """
        Get available Llama.cpp models.

        Returns GGUF models available in the local HuggingFace cache.
        Always returns models (doesn't check if llama.cpp is available).

        Returns:
            List of LanguageModel instances for Llama.cpp
        """
        try:
            # Import the function to get locally cached GGUF models
            from nodetool.integrations.huggingface.huggingface_models import (
                get_llamacpp_language_models_from_hf_cache,
            )

            models = await get_llamacpp_language_models_from_hf_cache()
            log.debug(f"Found {len(models)} Llama.cpp (GGUF) models in HF cache")
            return models
        except Exception as e:
            log.error(f"Error getting Llama.cpp models: {e}")
            return []

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
        """Stream assistant deltas and tool calls from llama.cpp.

        Args:
            messages: Conversation history to send.
            model: Model spec (GGUF path or HF repo/tag) to run.
            tools: Optional tool definitions.
            max_tokens: Maximum new tokens to generate.
            context_window: Unused hint; present for interface parity.
            response_format: Optional response schema.
            **kwargs: Additional OpenAI-compatible parameters.

        Yields:
            ``Chunk`` objects for text deltas and ``ToolCall`` entries when
            the model requests tool execution.
        """
        # Determine if we need tool emulation
        use_tool_emulation = len(tools) > 0 and not self.has_tool_support(model)
        if use_tool_emulation:
            log.info(f"Using tool emulation for model {model}")

        base_url = await self._manager.ensure_server(model)
        _kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if response_format is not None:
            _kwargs["response_format"] = response_format
        # Only add tools if native support is available
        if len(tools) > 0 and not use_tool_emulation:
            _kwargs["tools"] = self.format_tools(tools)

        # Normalize messages to satisfy llama.cpp alternation constraints
        messages_normalized = self._normalize_messages_for_llama(
            messages, tools, use_tool_emulation
        )
        # llama.cpp is sensitive to unsupported fields; pass only necessary ones
        openai_messages = [await self.convert_message(m) for m in messages_normalized]

        self._log_api_request("chat_stream", messages_normalized, **_kwargs)

        client = self.get_client(base_url)
        completion = await client.chat.completions.create(
            messages=openai_messages, **_kwargs
        )

        delta_tool_calls: dict[int, dict[str, Any]] = {}
        current_chunk = ""
        accumulated_content = ""  # For tool emulation parsing

        async for chunk in completion:
            chunk = chunk  # type: ignore
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if (
                getattr(delta, "content", None)
                or chunk.choices[0].finish_reason == "stop"
            ):
                content = delta.content or ""
                current_chunk += content
                # Accumulate content for emulation parsing
                if use_tool_emulation:
                    accumulated_content += content

                finish_reason = chunk.choices[0].finish_reason
                if finish_reason == "stop":
                    self._log_api_response(
                        "chat_stream",
                        Message(role="assistant", content=current_chunk),
                    )
                    # Parse emulated tool calls from accumulated content
                    if use_tool_emulation and accumulated_content:
                        log.debug("Parsing emulated tool calls from streaming response")
                        emulated_calls = self._parse_function_calls(
                            accumulated_content, tools
                        )
                        for tool_call in emulated_calls:
                            log.debug(f"Yielding emulated tool call: {tool_call.name}")
                            yield tool_call

                yield Chunk(content=content, done=finish_reason == "stop")

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
        """Return a single, non-streaming assistant message.

        Args:
            messages: Conversation history to send.
            model: Model spec (GGUF path or HF repo/tag) to run.
            tools: Optional tool definitions.
            max_tokens: Maximum new tokens to generate.
            context_window: Unused hint; present for interface parity.
            response_format: Optional response schema.
            **kwargs: Additional OpenAI-compatible parameters.

        Returns:
            Final assistant ``Message`` with optional ``tool_calls``.
        """
        # Determine if we need tool emulation
        use_tool_emulation = len(tools) > 0 and not self.has_tool_support(model)
        if use_tool_emulation:
            log.info(f"Using tool emulation for model {model}")

        base_url = await self._manager.ensure_server(model)
        _kwargs: dict[str, Any] = {
            "max_tokens": max_tokens,
            "stream": False,
        }
        if response_format is not None:
            _kwargs["response_format"] = response_format
        # Only add tools if native support is available
        if len(tools) > 0 and not use_tool_emulation:
            _kwargs["tools"] = self.format_tools(tools)
        # Pass through additional sampling/params
        if kwargs:
            _kwargs.update(kwargs)

        messages_normalized = self._normalize_messages_for_llama(
            messages, tools, use_tool_emulation
        )
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
        # Check for native tool calls
        if response_message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,  # type: ignore
                    args=try_parse_args(tool_call.function.arguments),  # type: ignore
                )
                for tool_call in response_message.tool_calls
            ]
        # Parse emulated tool calls if needed
        elif use_tool_emulation and response_message.content:
            log.debug("Parsing emulated tool calls from response")
            emulated_calls = self._parse_function_calls(response_message.content, tools)
            if emulated_calls:
                tool_calls = emulated_calls
                log.debug(f"Parsed {len(emulated_calls)} emulated tool calls")

        message = Message(
            role="assistant", content=response_message.content, tool_calls=tool_calls
        )
        self._log_api_response("chat", message)
        return message

    def get_usage(self) -> dict:
        """Return a shallow copy of accumulated usage counters."""
        return self._usage.copy()

    def reset_usage(self) -> None:
        """Reset all usage counters to zero."""
        self._usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_prompt_tokens": 0,
            "reasoning_tokens": 0,
        }


if __name__ == "__main__":

    async def _run_all():
        from nodetool.agents.tools.math_tools import CalculatorTool

        provider = LlamaProvider()
        context = ProcessingContext()

        # =====================================================================
        # TEST 1: Qwen model with native tool support
        # =====================================================================
        print("\n" + "=" * 60)
        print("TEST 1: Qwen Model (Native Tool Support)")
        print("=" * 60 + "\n")

        class EchoTool(Tool):
            name = "echo"
            description = "Echo back the provided text."
            input_schema = {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            }

            async def process(
                self, context: ProcessingContext, params: dict[str, Any]
            ) -> Any:  # type: ignore[override]
                return {"echo": params.get("text", "")}

        model = "ggml-org/Qwen2.5-Coder-0.5B-Q8_0-GGUF"
        tools: list[Tool] = [EchoTool()]

        print(f"Model: {model}")

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
                    selected_tool = next((t for t in tools if t.name == tc.name), None)
                    if selected_tool is None:
                        continue

                    result = await selected_tool.process(context, tc.args or {})
                    print(f"  - Tool result: {result}")

                    print("🔄 Testing standard OpenAI tool calling approach...")

                    try:
                        messages.append(
                            Message(
                                role="tool",
                                name=selected_tool.name,
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

        # =====================================================================
        # TEST 2: Gemma model with tool emulation
        # =====================================================================
        print("\n" + "=" * 60)
        print("TEST 2: Gemma Model (Tool Emulation)")
        print("=" * 60 + "\n")

        # Download Gemma model
        hf_hub_download(
            "ggml-org/gemma-3-1b-it-GGUF", filename="gemma-3-1b-it-Q4_K_M.gguf"
        )
        gemma_model = "ggml-org/gemma-3-1b-it-GGUF"
        calculator_tools: list[Tool] = [CalculatorTool()]

        # Check if model supports native tool calling
        print(f"Model: {gemma_model}")

        messages_calc: list[Message] = [
            Message(
                role="system",
                content="You are a helpful assistant. Use tools when calculations are needed.",
            ),
            Message(
                role="user",
                content="Please compute 12.3 * (7 - 5) + 10 / 2 and provide the numeric result.",
            ),
        ]

        response = await provider.generate_message(
            messages=messages_calc,
            model=gemma_model,
            tools=calculator_tools,
            max_tokens=512,
        )
        print("--- Initial Response ---")
        print(f"Content: {response.content}")
        print(f"Tool calls: {response.tool_calls}\n")

        iteration = 1
        max_iterations = 5
        while response.tool_calls and iteration <= max_iterations:
            print(f"--- Tool Execution (Iteration {iteration}/{max_iterations}) ---")
            for tool_call in response.tool_calls:
                print(f"Tool: {tool_call.name}")
                print(f"Args: {tool_call.args}")

                matching_tool = next(
                    (t for t in calculator_tools if t.name == tool_call.name), None
                )
                if matching_tool is None:
                    print(f"Warning: Tool {tool_call.name} not found")
                    continue

                tool_result = await matching_tool.process(context, tool_call.args or {})
                print(f"Result: {tool_result}")

                messages_calc.append(
                    Message(
                        role="user",
                        name=matching_tool.name,
                        content=json.dumps(tool_result),
                    )
                )

            response = await provider.generate_message(
                messages=messages_calc,
                model=gemma_model,
                tools=calculator_tools,
                max_tokens=512,
            )
            print("\n--- Response After Tool Call ---")
            print(f"Content: {response.content}")
            print(f"Tool calls: {response.tool_calls}")
            iteration += 1

        if iteration > max_iterations:
            print(
                f"\n⚠️  WARNING: Reached max iterations ({max_iterations}). Stopping.\n"
            )

        print(f"\n{'='*60}")
        print(f"Final Answer: {response.content}")
        print(f"{'='*60}\n")

    asyncio.run(_run_all())
