"""
xllamacpp OpenAI-compatible provider.

Uses xllamacpp's built-in Server class to provide local LLM inference with OpenAI-compatible API.
xllamacpp is a high-performance Cython wrapper around llama.cpp with built-in optimizations
for CPU, CUDA, Vulkan (AMD/Intel), and Metal (Apple Silicon).

Key advantages over external llama-server:
- Pythonic API with direct library integration
- No external process management required
- Built-in memory estimation for GPU offloading
- Optimized builds for multiple platforms
- Thread-safe continuous batching server

Environment variables:
- XLLAMACPP_N_CTX: Context window size (default: 8192)
- XLLAMACPP_N_GPU_LAYERS: Number of layers to offload to GPU (default: auto-detect)
- XLLAMACPP_N_THREADS: Number of CPU threads (default: auto)
- XLLAMACPP_PARALLEL: Number of parallel sequences (default: 1)
- XLLAMACPP_CACHE_RAM_MIB: RAM cache size in MiB (default: 2048)
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, AsyncIterator, Sequence

import httpx
import openai
import tiktoken

from nodetool.agents.tools.base import Tool
from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import LanguageModel, Message, Provider, ToolCall
from nodetool.providers.base import BaseProvider, register_provider
from nodetool.providers.openai_compat import OpenAICompat
from nodetool.runtime.resources import require_scope
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk

log = get_logger(__name__)

# Lazy import of xllamacpp - only when provider is actually used
_xllamacpp_available = None


def _check_xllamacpp_available() -> bool:
    """Check if xllamacpp is available for import."""
    global _xllamacpp_available
    if _xllamacpp_available is not None:
        return _xllamacpp_available

    try:
        import xllamacpp  # noqa: F401

        _xllamacpp_available = True
        log.info("xllamacpp is available")
    except ImportError:
        _xllamacpp_available = False
        log.debug("xllamacpp is not available - install with: pip install xllamacpp")
    return _xllamacpp_available


class XLlamaCppProvider(BaseProvider, OpenAICompat):
    """OpenAI-compatible chat provider backed by xllamacpp.

    This provider uses xllamacpp's built-in Server class to provide local LLM inference.
    The server exposes an OpenAI-compatible API at a local URL that can be accessed
    via the standard OpenAI client.

    xllamacpp provides:
    - High-performance Cython-based llama.cpp wrapper
    - Optimized builds for CPU, CUDA, Vulkan, and Metal
    - Thread-safe continuous batching
    - Built-in memory estimation for GPU layer offloading
    - No external process management required

    Attributes:
        provider_name: Provider identifier ("xllamacpp")
        _server: The xllamacpp Server instance (created on demand per model)
        _model_servers: Cache of Server instances by model path
        _default_context_length: Default context window size
    """

    provider_name: str = "xllamacpp"
    _server: Any = None
    _model_servers: dict[str, Any] = {}
    _default_context_length: int = 8192

    def __init__(self, secrets: dict[str, str], ttl_seconds: int = 300):
        """Initialize the provider.

        Args:
            secrets: Dictionary of secrets for the provider (not used).
            ttl_seconds: Unused, kept for API compatibility.

        Environment:
            XLLAMACPP_N_CTX: Context window size (default: 8192)
            XLLAMACPP_N_GPU_LAYERS: GPU layers to offload (default: auto)
            XLLAMACPP_N_THREADS: CPU threads (default: auto)
            XLLAMACPP_PARALLEL: Parallel sequences (default: 1)
            XLLAMACPP_CACHE_RAM_MIB: RAM cache in MiB (default: 2048)
        """
        super().__init__()

        if not _check_xllamacpp_available():
            raise ImportError(
                "xllamacpp is not installed. "
                "Install it with: pip install xllamacpp\n"
                "For GPU acceleration:\n"
                "  CUDA: pip install xllamacpp --index-url https://xorbitsai.github.io/xllamacpp/whl/cu128\n"
                "  Vulkan: pip install xllamacpp --index-url https://xorbitsai.github.io/xllamacpp/whl/vulkan"
            )

        # Get context length from settings
        context_length_str = Environment.get("XLLAMACPP_N_CTX")
        self._default_context_length = int(context_length_str) if context_length_str else 8192

        # Initialize tiktoken encoding for token counting
        try:
            self._encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            log.warning(f"Failed to load tiktoken encoding: {e}. Token counting may be inaccurate.")
            self._encoding = None

        log.debug(f"XLlamaCppProvider initialized with default_context_length: {self._default_context_length}")

    def _ensure_server(self, model: str) -> tuple[Any, str]:
        """Ensure a server is running for the given model.

        Creates and caches xllamacpp Server instances per model.

        Args:
            model: Model path or HuggingFace repo (e.g., "path/to/model.gguf" or "org/repo")

        Returns:
            Tuple of (Server instance, base_url)
        """
        import xllamacpp as xlc

        if model in self._model_servers:
            server = self._model_servers[model]
            return server, server.listening_address

        # Create server parameters
        params = xlc.CommonParams()
        params.model.path = model

        # Configure from environment
        params.n_ctx = int(Environment.get("XLLAMACPP_N_CTX", str(self._default_context_length)))
        params.n_parallel = int(Environment.get("XLLAMACPP_PARALLEL", "1"))

        # GPU offloading - auto-detect if not specified
        n_gpu_layers_str = Environment.get("XLLAMACPP_N_GPU_LAYERS")
        if n_gpu_layers_str:
            params.n_gpu_layers = int(n_gpu_layers_str)
        else:
            # Auto-detect optimal GPU layers using memory estimator
            try:
                devices = xlc.get_device_info()
                if any(d["name"] != "CPU" for d in devices):
                    # Estimate optimal GPU layers
                    estimate = xlc.estimate_gpu_layers(
                        devices,
                        model,
                        [],  # No LoRA adapters
                        context_length=params.n_ctx,
                        batch_size=512,
                        num_parallel=params.n_parallel,
                        kv_cache_type="",
                    )
                    params.n_gpu_layers = estimate.layers
                    log.info(
                        f"Auto-detected {params.n_gpu_layers} GPU layers for {model} "
                        f"(VRAM: {estimate.vram_size / (1024**3):.2f}GB / "
                        f"Total: {estimate.total_size / (1024**3):.2f}GB)"
                    )
            except Exception as e:
                log.warning(f"Failed to estimate GPU layers: {e}. Using CPU-only mode.")
                params.n_gpu_layers = 0

        # Threading
        n_threads_str = Environment.get("XLLAMACPP_N_THREADS")
        if n_threads_str:
            params.cpuparams.n_threads = int(n_threads_str)
            params.cpuparams_batch.n_threads = int(n_threads_str)

        # Cache configuration
        cache_ram_str = Environment.get("XLLAMACPP_CACHE_RAM_MIB", "2048")
        params.cache_ram_mib = int(cache_ram_str)

        # Disable warmup for faster startup
        params.warmup = False

        # Create server - this starts the HTTP server automatically
        log.info(f"Starting xllamacpp server for model: {model}")
        server = xlc.Server(params)

        # Cache the server instance
        self._model_servers[model] = server

        log.info(f"xllamacpp server started at: {server.listening_address}")
        return server, server.listening_address

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
                "6. Do NOT call the same function twice\n"
            )
            system_parts.append(tool_instruction)

        for msg in messages:
            if msg.role == "system":
                system_parts.append(str(msg.content) if msg.content is not None else "")
                continue
            if msg.role == "tool":
                # Represent tool output as a user turn for alternation
                content_str: str
                if isinstance(msg.content, str):
                    content_str = msg.content
                else:
                    try:
                        content_str = json.dumps(msg.content)  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        content_str = str(msg.content)

                if use_tool_emulation:
                    emulated_content = (
                        f"The function {msg.name}() returned the following result:\n{content_str}\n\n"
                        f"Use this result to answer the user's question. Do NOT call the function again."
                    )
                    normalized.append(Message(role="user", content=str(emulated_content)))
                else:
                    prefix = f"Tool {msg.name or ''} result:\n" if msg.name else "Tool result:\n"
                    normalized.append(Message(role="user", content=str(f"{prefix}{content_str}")))
                continue
            normalized.append(msg)

        if system_parts:
            system = Message(role="system", content="\n".join(p for p in system_parts if p))
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
            # Map any non-standard roles to user
            role = msg.role if msg.role in ("user", "assistant") else "user"

            # Insert blanks until this message fits the expected parity
            while (role == "user") != expected_user_turn:
                fixed.append(Message(role=("user" if expected_user_turn else "assistant"), content=""))
                expected_user_turn = not expected_user_turn

            # Append the actual message
            if role != msg.role:
                fixed.append(Message(role=role, content=msg.content, tool_calls=msg.tool_calls))
            else:
                fixed.append(msg)

            expected_user_turn = not expected_user_turn

        return fixed

    def _count_tokens_in_text(self, text: str) -> int:
        """Count tokens in a text string using tiktoken."""
        if not self._encoding or not text:
            return 0
        try:
            return len(self._encoding.encode(text))
        except (TypeError, ValueError) as e:
            log.debug(f"Error counting tokens: {e}")
            return 0

    def _count_tokens_in_messages(self, messages: Sequence[Message]) -> int:
        """Count tokens in a sequence of messages."""
        if not self._encoding:
            return 0

        num_tokens = 0
        for message in messages:
            num_tokens += 4  # Message overhead
            if message.role:
                num_tokens += self._count_tokens_in_text(message.role)
            if message.content:
                content_str = message.content if isinstance(message.content, str) else str(message.content)
                num_tokens += self._count_tokens_in_text(content_str)
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.name:
                        num_tokens += self._count_tokens_in_text(tool_call.name)
                    if tool_call.args:
                        try:
                            args_str = json.dumps(tool_call.args)
                            num_tokens += self._count_tokens_in_text(args_str)
                        except (TypeError, ValueError):
                            pass
        num_tokens += 2  # Assistant reply primer
        return num_tokens

    def get_container_env(self, context: ProcessingContext) -> dict[str, str]:
        """Return environment variables for containerized execution."""
        return {}

    def get_client(self, base_url: str) -> openai.AsyncClient:
        """Create an OpenAI-compatible async client targeting xllamacpp server."""
        try:
            http_client = require_scope().get_http_client()
            log.debug("Using ResourceScope HTTP client for xllamacpp")
        except RuntimeError:
            log.warning("No ResourceScope bound, creating fallback HTTP client for xllamacpp")
            http_client = httpx.AsyncClient(follow_redirects=True, timeout=600, verify=False)

        return openai.AsyncClient(
            base_url=f"{base_url}/v1",
            api_key="not-required",  # xllamacpp server doesn't require auth
            http_client=http_client,
        )

    def _as_service_unavailable(self, error: Exception, base_url: str) -> httpx.HTTPStatusError:
        """Convert errors to service unavailable."""
        request = httpx.Request("POST", f"{base_url}/v1/chat/completions")
        response = httpx.Response(status_code=503, request=request, text=str(error))
        return httpx.HTTPStatusError(
            message="503 Service Unavailable",
            request=request,
            response=response,
        )

    def has_tool_support(self, model: str) -> bool:
        """Return True if the model supports native tool calling.

        Since we're using xllamacpp dynamically, we use tool emulation by default.
        """
        log.debug(f"has_tool_support called for {model}, returning False (using emulation)")
        return False

    async def get_available_language_models(self) -> list[LanguageModel]:
        """Get available xllamacpp models.

        Returns an empty list as models are loaded on demand.
        """
        return []

    async def generate_messages(  # type: ignore[override]
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        max_tokens: int = 1024,
        response_format: dict | None = None,
        **kwargs,
    ) -> AsyncIterator[Chunk | ToolCall]:
        """Stream assistant deltas and tool calls from xllamacpp.

        Args:
            messages: Conversation history to send.
            model: Model path or HuggingFace repo to run.
            tools: Optional tool definitions.
            max_tokens: Maximum new tokens to generate.
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

        # Ensure server is running for this model
        _, base_url = self._ensure_server(model)

        _kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if response_format is not None:
            _kwargs["response_format"] = response_format
        if len(tools) > 0 and not use_tool_emulation:
            _kwargs["tools"] = self.format_tools(tools)

        # Normalize messages
        messages_normalized = self._normalize_messages_for_llama(messages, tools, use_tool_emulation)
        openai_messages = [await self.convert_message(m) for m in messages_normalized]

        # Count prompt tokens
        prompt_tokens = self._count_tokens_in_messages(messages_normalized)

        self._log_api_request("chat_stream", messages_normalized, **_kwargs)

        client = self.get_client(base_url)
        try:
            completion = await client.chat.completions.create(messages=openai_messages, **_kwargs)
        except httpx.ConnectError as e:
            raise self._as_service_unavailable(e, base_url) from e

        delta_tool_calls: dict[int, dict[str, Any]] = {}
        current_chunk = ""
        accumulated_content = ""
        completion_text = ""

        async for chunk in completion:
            chunk = chunk  # type: ignore
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if getattr(delta, "content", None) or chunk.choices[0].finish_reason == "stop":
                content = delta.content or ""
                current_chunk += content
                completion_text += content
                if use_tool_emulation:
                    accumulated_content += content

                finish_reason = chunk.choices[0].finish_reason
                if finish_reason == "stop":
                    self._log_api_response(
                        "chat_stream",
                        Message(role="assistant", content=current_chunk),
                    )
                    if use_tool_emulation and accumulated_content:
                        log.debug("Parsing emulated tool calls from streaming response")
                        emulated_calls, _ = self._parse_function_calls(accumulated_content, tools)
                        for tool_call in emulated_calls:
                            log.debug(f"Yielding emulated tool call: {tool_call.name}")
                            yield tool_call

                    completion_tokens = self._count_tokens_in_text(completion_text)
                    log.debug(
                        f"Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, "
                        f"Total: {prompt_tokens + completion_tokens}"
                    )

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

                    completion_tokens = self._count_tokens_in_text(completion_text)
                    for tc in delta_tool_calls.values():
                        if "function" in tc and "arguments" in tc["function"]:
                            completion_tokens += self._count_tokens_in_text(tc["function"]["arguments"])
                    log.debug(
                        f"Token usage (tool calls) - Prompt: {prompt_tokens}, Completion: {completion_tokens}, "
                        f"Total: {prompt_tokens + completion_tokens}"
                    )

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

    async def generate_message(  # type: ignore[override]
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        max_tokens: int = 1024,
        response_format: dict | None = None,
        **kwargs,
    ) -> Message:
        """Return a single, non-streaming assistant message.

        Args:
            messages: Conversation history to send.
            model: Model path or HuggingFace repo to run.
            tools: Optional tool definitions.
            max_tokens: Maximum new tokens to generate.
            response_format: Optional response schema.
            **kwargs: Additional OpenAI-compatible parameters.

        Returns:
            Final assistant ``Message`` with optional ``tool_calls``.
        """
        # Determine if we need tool emulation
        use_tool_emulation = len(tools) > 0 and not self.has_tool_support(model)
        if use_tool_emulation:
            log.info(f"Using tool emulation for model {model}")

        # Ensure server is running
        _, base_url = self._ensure_server(model)

        _kwargs: dict[str, Any] = {
            "max_tokens": max_tokens,
            "stream": False,
        }
        if response_format is not None:
            _kwargs["response_format"] = response_format
        if len(tools) > 0 and not use_tool_emulation:
            _kwargs["tools"] = self.format_tools(tools)
        if kwargs:
            _kwargs.update(kwargs)

        messages_normalized = self._normalize_messages_for_llama(messages, tools, use_tool_emulation)
        openai_messages = [await self.convert_message(m) for m in messages_normalized]

        # Count prompt tokens
        prompt_tokens = self._count_tokens_in_messages(messages_normalized)

        self._log_api_request("chat", messages_normalized, model=model, **_kwargs)

        client = self.get_client(base_url)
        try:
            completion = await client.chat.completions.create(model=model, messages=openai_messages, **_kwargs)
        except httpx.ConnectError as e:
            raise self._as_service_unavailable(e, base_url) from e

        choice = completion.choices[0]
        response_message = choice.message

        def try_parse_args(args: Any) -> Any:
            try:
                return json.loads(args)
            except (json.JSONDecodeError, ValueError, TypeError):
                return {}

        tool_calls = None
        final_content = response_message.content

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
            emulated_calls, cleaned_content = self._parse_function_calls(response_message.content, tools)
            if emulated_calls:
                tool_calls = emulated_calls
                final_content = cleaned_content
                log.debug(f"Parsed {len(emulated_calls)} emulated tool calls")

        # Count completion tokens
        completion_tokens = 0
        if response_message.content:
            completion_tokens = self._count_tokens_in_text(response_message.content)

        if tool_calls:
            for tc in tool_calls:
                if tc.name:
                    completion_tokens += self._count_tokens_in_text(tc.name)
                if tc.args:
                    try:
                        args_str = json.dumps(tc.args)
                        completion_tokens += self._count_tokens_in_text(args_str)
                    except (TypeError, ValueError):
                        pass

        log.debug(
            f"Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, "
            f"Total: {prompt_tokens + completion_tokens}"
        )

        message = Message(role="assistant", content=final_content, tool_calls=tool_calls)
        self._log_api_response("chat", message)
        return message


# Register the provider
if _check_xllamacpp_available():
    register_provider(Provider.XLlamaCpp)(XLlamaCppProvider)
    log.debug("XLlamaCpp provider registered successfully")
else:
    log.debug("XLlamaCpp provider not registered: xllamacpp not available")
