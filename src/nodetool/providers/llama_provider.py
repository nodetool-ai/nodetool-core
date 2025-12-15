"""
Llama.cpp OpenAI-compatible provider.

Uses openai.AsyncClient against a llama-server base_url. Requires an externally
managed llama-server (e.g., started by Electron on application startup) via the
LLAMA_CPP_URL environment variable.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, AsyncIterator, ClassVar, List, Sequence

import httpx
import openai
import tiktoken
from huggingface_hub import hf_hub_download

from nodetool.agents.tools.base import Tool
from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.integrations.huggingface.huggingface_models import (
    get_llamacpp_language_models_from_hf_cache,
)
from nodetool.metadata.types import LanguageModel, Message, Provider, ToolCall
from nodetool.providers.base import BaseProvider, register_provider
from nodetool.providers.openai_compat import OpenAICompat
from nodetool.runtime.resources import require_scope
from nodetool.security.secret_helper import get_secret_sync
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk

log = get_logger(__name__)

# Only register the provider if LLAMA_CPP_URL is explicitly set
_llama_cpp_url = Environment.get("LLAMA_CPP_URL")


class LlamaProvider(BaseProvider, OpenAICompat):
    """OpenAI-compatible chat provider backed by an external llama.cpp server.

    This provider connects to an externally managed llama-server specified via
    the ``LLAMA_CPP_URL`` environment variable. It exposes a familiar OpenAI
    client interface, normalizes messages to conform to llama.cpp's chat template
    alternation rules (user/assistant), and supports tool call emulation.

    The llama-server is expected to be started by Electron on application startup.
    If ``LLAMA_CPP_URL`` is not set, this provider is not available.

    Attributes:
        provider: Provider identifier used by the application.
        _base_url: The llama-server URL from LLAMA_CPP_URL environment variable.
    """

    provider_name: str = "llama_cpp"
    _base_url: str = ""

    def __init__(self, secrets: dict[str, str], ttl_seconds: int = 300):
        """Initialize the provider.

        Args:
            secrets: Dictionary of secrets for the provider.
            ttl_seconds: Unused, kept for API compatibility.

        Environment:
            LLAMA_CPP_URL: Required. URL of the external llama-server
                (e.g., http://127.0.0.1:8080).
        """
        super().__init__()
        self._base_url = Environment.get("LLAMA_CPP_URL", "")
        assert self._base_url, "LLAMA_CPP_URL not set"
        log.info(f"Using llama-server at: {self._base_url}")
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_prompt_tokens": 0,
            "reasoning_tokens": 0,
        }
        # Initialize tiktoken encoding for token counting
        # Using cl100k_base which is used by gpt-4 and modern models
        try:
            self._encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            log.warning(f"Failed to load tiktoken encoding: {e}. Token counting may be inaccurate.")
            self._encoding = None

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
            normalized = [system, *normalized]

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

    def _count_tokens_in_text(self, text: str) -> int:
        """Count tokens in a text string using tiktoken.

        Args:
            text: Text to count tokens for.

        Returns:
            Number of tokens in the text. Returns 0 if encoding is not available.
        """
        if not self._encoding or not text:
            return 0
        try:
            return len(self._encoding.encode(text))
        except Exception as e:
            log.debug(f"Error counting tokens: {e}")
            return 0

    def _count_tokens_in_messages(self, messages: Sequence[Message]) -> int:
        """Count tokens in a sequence of messages.

        This approximates the token count for messages as they would be formatted
        for the chat completion API. The actual token count may vary slightly
        depending on the specific chat template used by the model.

        Args:
            messages: Messages to count tokens for.

        Returns:
            Approximate number of tokens in the messages.
        """
        if not self._encoding:
            return 0

        num_tokens = 0
        for message in messages:
            # Account for message formatting tokens (role, separators, etc.)
            # This is an approximation based on OpenAI's format
            num_tokens += 4  # Every message has ~4 tokens of overhead

            # Count role tokens
            if message.role:
                num_tokens += self._count_tokens_in_text(message.role)

            # Count content tokens
            if message.content:
                content_str = message.content if isinstance(message.content, str) else str(message.content)
                num_tokens += self._count_tokens_in_text(content_str)

            # Count tool call tokens if present
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.name:
                        num_tokens += self._count_tokens_in_text(tool_call.name)
                    if tool_call.args:
                        try:
                            args_str = json.dumps(tool_call.args)
                            num_tokens += self._count_tokens_in_text(args_str)
                        except Exception:
                            pass

        # Add tokens for the assistant reply primer
        num_tokens += 2

        return num_tokens

    def _update_usage(self, prompt_tokens: int = 0, completion_tokens: int = 0):
        """Update the usage statistics.

        Args:
            prompt_tokens: Number of tokens in the prompt.
            completion_tokens: Number of tokens in the completion.
        """
        self.usage["prompt_tokens"] += prompt_tokens
        self.usage["completion_tokens"] += completion_tokens
        self.usage["total_tokens"] += prompt_tokens + completion_tokens

    def get_container_env(self, context: ProcessingContext) -> dict[str, str]:
        """Return environment variables for containerized execution.

        Returns:
            A mapping of environment variables to inject. Empty for llama.cpp
            since the local server is spawned by this process.
        """
        return {}

    def get_client(self, base_url: str) -> openai.AsyncClient:
        """Create an OpenAI-compatible async client targeting llama-server.

        Uses ResourceScope's HTTP client to ensure correct event loop binding.

        Args:
            base_url: Base URL returned by ``LlamaServerManager.ensure_server``.

        Returns:
            Configured ``openai.AsyncClient`` instance.
        """
        # Use ResourceScope's HTTP client if available, otherwise create a new one
        try:
            http_client = require_scope().get_http_client()
            log.debug("Using ResourceScope HTTP client for Llama")
        except RuntimeError:
            # Fallback if no scope is bound (shouldn't happen in normal operation)
            log.warning("No ResourceScope bound, creating fallback HTTP client for Llama")
            http_client = httpx.AsyncClient(
                follow_redirects=True, timeout=600, verify=False
            )

        # llama-server accepts any API key; None is fine when auth is disabled
        return openai.AsyncClient(
            base_url=f"{base_url}/v1",
            api_key="sk-no-key-required",
            http_client=http_client,
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

        Since we're using an external llama-server, we cannot query model
        capabilities ahead of time. Tool call emulation is used instead.

        Args:
            model: Model identifier passed to llama.cpp.

        Returns:
            False always - tool emulation is used for external servers.
        """
        log.debug(f"has_tool_support called for {model}, returning False (using emulation)")
        return False

    async def get_available_language_models(self) -> List[LanguageModel]:
        """
        Get available Llama.cpp models.

        Queries the llama.cpp server's OpenAI-compatible /v1/models endpoint
        to get the list of available models.

        Returns:
            List of LanguageModel instances for Llama.cpp
        """
        import httpx

        models: List[LanguageModel] = []

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self._base_url}/v1/models")
                response.raise_for_status()
                data = response.json()

                for model_data in data.get("data", []):
                    model_id = model_data.get("id", "")
                    models.append(LanguageModel(
                        id=model_id,
                        name=model_id,
                        provider=Provider.LlamaCpp,
                    ))
                log.debug(f"Found {len(models)} models from llama.cpp server")
        except Exception as e:
            log.warning(f"Error querying llama.cpp server: {e}")

        return models

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

        # Use the external llama-server URL
        base_url = self._base_url
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

        # Count prompt tokens before sending
        prompt_tokens = self._count_tokens_in_messages(messages_normalized)

        self._log_api_request("chat_stream", messages_normalized, **_kwargs)

        client = self.get_client(base_url)
        completion = await client.chat.completions.create(
            messages=openai_messages, **_kwargs
        )

        delta_tool_calls: dict[int, dict[str, Any]] = {}
        current_chunk = ""
        accumulated_content = ""  # For tool emulation parsing
        completion_text = ""  # Track all generated text for token counting

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
                completion_text += content  # Accumulate for token counting
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

                    # Count completion tokens and update usage
                    completion_tokens = self._count_tokens_in_text(completion_text)
                    self._update_usage(prompt_tokens, completion_tokens)
                    log.debug(f"Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {prompt_tokens + completion_tokens}")

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

                    # Count completion tokens and update usage for tool calls
                    completion_tokens = self._count_tokens_in_text(completion_text)
                    # Add tokens for tool call formatting
                    for tc in delta_tool_calls.values():
                        if "function" in tc and "arguments" in tc["function"]:
                            completion_tokens += self._count_tokens_in_text(tc["function"]["arguments"])
                    self._update_usage(prompt_tokens, completion_tokens)
                    log.debug(f"Token usage (tool calls) - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {prompt_tokens + completion_tokens}")

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

        # Use the external llama-server URL
        base_url = self._base_url
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

        # Count prompt tokens before sending
        prompt_tokens = self._count_tokens_in_messages(messages_normalized)

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

        # Count completion tokens
        completion_tokens = 0
        if response_message.content:
            completion_tokens = self._count_tokens_in_text(response_message.content)

        # Add tokens for tool calls
        if tool_calls:
            for tc in tool_calls:
                if tc.name:
                    completion_tokens += self._count_tokens_in_text(tc.name)
                if tc.args:
                    try:
                        args_str = json.dumps(tc.args)
                        completion_tokens += self._count_tokens_in_text(args_str)
                    except Exception:
                        pass

        # Update usage statistics
        self._update_usage(prompt_tokens, completion_tokens)
        log.debug(f"Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {prompt_tokens + completion_tokens}")

        message = Message(
            role="assistant", content=response_message.content, tool_calls=tool_calls
        )
        self._log_api_response("chat", message)
        return message

    def get_usage(self) -> dict:
        """Return a shallow copy of accumulated usage counters."""
        return self.usage.copy()

    def reset_usage(self) -> None:
        """Reset all usage counters to zero."""
        self.usage = {
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
            input_schema: ClassVar[dict[str, Any]] = {
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

        # Download Gemma model - use HF_TOKEN from secrets if available for gated model downloads
        token = get_secret_sync("HF_TOKEN") or os.environ.get("HF_TOKEN")
        hf_hub_download(
            "ggml-org/gemma-3-1b-it-GGUF", filename="gemma-3-1b-it-Q4_K_M.gguf", token=token
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


# Conditionally register the provider only if LLAMA_CPP_URL is set
if _llama_cpp_url:
    register_provider(Provider.LlamaCpp)(LlamaProvider)
else:
    log.debug("LlamaCpp provider not registered: LLAMA_CPP_URL not set")
