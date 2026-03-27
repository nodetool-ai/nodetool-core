"""
Llama.cpp OpenAI-compatible provider.

Uses openai.AsyncClient against a llama-server base_url. Requires an externally
managed llama-server (e.g., started by Electron on application startup) via the
LLAMA_CPP_URL environment variable.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, AsyncIterator, ClassVar, Sequence

import httpx
import openai
import tiktoken
from huggingface_hub import hf_hub_download

from nodetool.agents.tools.base import Tool
from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    LanguageModel,
    Message,
    MessageTextContent,
    Provider,
    ToolCall,
)
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
    _gemma_placeholder_name_re: ClassVar[re.Pattern[str]] = re.compile(
        r"^[0-9a-f]{8}(?:[_-][0-9a-f]{4}){3}[_-][0-9a-f]{12}$",
        re.IGNORECASE,
    )

    def __init__(self, secrets: dict[str, str], ttl_seconds: int = 300):
        """Initialize the provider.

        Args:
            secrets: Dictionary of secrets for the provider.
            ttl_seconds: Unused, kept for API compatibility.

        Environment:
            LLAMA_CPP_URL: Required. URL of the external llama-server
                (e.g., http://127.0.0.1:8080).
            LLAMA_CPP_CONTEXT_LENGTH: Optional. Context window size in tokens.
                If not set, defaults to 128000.
        """
        super().__init__()
        self._base_url = Environment.get("LLAMA_CPP_URL", "")

        # Get context length from settings, with fallback to 128000
        context_length_str = Environment.get("LLAMA_CPP_CONTEXT_LENGTH")
        self.default_context_length = int(context_length_str) if context_length_str else 128000

        if self._base_url:
            log.info(f"Using llama-server at: {self._base_url}")
        else:
            log.warning("LLAMA_CPP_URL not set; LlamaCpp provider requires an external llama-server URL")
        # Initialize tiktoken encoding for token counting
        # Using cl100k_base which is used by gpt-4 and modern models
        try:
            self._encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            log.warning(f"Failed to load tiktoken encoding: {e}. Token counting may be inaccurate.")
            self._encoding = None

        log.debug(f"LlamaProvider initialized with default_context_length: {self.default_context_length}")

    def _normalize_messages_for_llama(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None = None,
        use_tool_emulation: bool = False,
        model: str | None = None,
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
            model: Optional model identifier for model-specific prompt tweaks.

        Returns:
            A list of messages compatible with llama.cpp chat templates.
        """

        # Convert None to empty tuple to avoid mutable default argument
        if tools is None:
            tools = ()

        system_parts: list[str] = []
        normalized: list[Message] = []

        # Add tool definitions for emulation
        if use_tool_emulation and tools:
            if model and self._is_gemma_model(model):
                tool_definitions = self._format_tools_as_gemma_json(tools)
                tool_instruction = (
                    "\n\nYou have access to functions. If you decide to invoke any of the function(s), "
                    "you MUST put it in the format of\n"
                    "[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]\n\n"
                    "You SHOULD NOT include any other text in the response if you call a function\n\n"
                    f"{tool_definitions}"
                )
            else:
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
            if msg.role == "assistant" and use_tool_emulation:
                # In emulation mode, don't send synthetic tool_calls metadata back to llama.cpp.
                # Keep assistant text only, with any function-call line removed when we know
                # this turn represented a tool request.
                if isinstance(msg.content, str):
                    assistant_content = msg.content
                elif isinstance(msg.content, list):
                    text_parts = [
                        part.text
                        for part in msg.content
                        if isinstance(part, MessageTextContent)
                    ]
                    assistant_content = "\n".join(text_parts)
                elif msg.content is None:
                    assistant_content = ""
                else:
                    assistant_content = str(msg.content)

                if msg.tool_calls:
                    _calls, cleaned_content = self._parse_function_calls(
                        assistant_content, tools
                    )
                    if not _calls:
                        _calls, cleaned_content = self._parse_non_python_function_calls(
                            assistant_content, tools
                        )
                    assistant_content = cleaned_content

                normalized.append(
                    Message(
                        role="assistant",
                        content=assistant_content,
                    )
                )
                continue
            if msg.role == "tool":
                # Represent tool output as a user turn for alternation.
                content_str: str
                if isinstance(msg.content, str):
                    content_str = msg.content
                else:
                    try:
                        content_str = json.dumps(msg.content)  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        content_str = str(msg.content)

                # For Gemma models, avoid adding empty assistant messages as they break the chat template
                # Only add empty assistant message if the normalized list is empty or the last message was a system message
                if normalized and normalized[-1].role != "assistant" and normalized[-1].role != "system":
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
            # Map any non-standard roles (just in case) to user
            role = msg.role if msg.role in ("user", "assistant") else "user"

            # Insert blanks until this message fits the expected parity
            while (role == "user") != expected_user_turn:
                fixed.append(Message(role=("user" if expected_user_turn else "assistant"), content=""))
                expected_user_turn = not expected_user_turn

            # Append the actual message (preserve original content)
            if role != msg.role:
                fixed.append(Message(role=role, content=msg.content, tool_calls=msg.tool_calls))
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
        except (TypeError, ValueError) as e:
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
                        except (TypeError, ValueError):
                            pass

        # Add tokens for the assistant reply primer
        num_tokens += 2

        return num_tokens

    def _resolve_emulated_tool_name(self, raw_name: str, args: dict[str, Any], tools: Sequence[Tool]) -> str:
        """Resolve Gemma-emulated tool names to known tool names when possible."""
        if not tools:
            return raw_name

        # Fast path: already valid
        for tool in tools:
            if tool.name == raw_name:
                return raw_name

        # Some Gemma variants emit UUID-like placeholder names for function calls.
        # If there is only one available tool, map it directly.
        if len(tools) == 1:
            only_tool = tools[0].name
            log.debug(f"Remapping unknown emulated tool '{raw_name}' -> '{only_tool}' (single tool available)")
            return only_tool

        # Fuzzy string containment can recover mild name drift.
        lowered = raw_name.lower()
        for tool in tools:
            name = tool.name.lower()
            if name in lowered or lowered in name:
                log.debug(f"Remapping emulated tool '{raw_name}' -> '{tool.name}' (name similarity)")
                return tool.name

        # Schema-based arg matching as a tie-breaker.
        arg_keys = set(args.keys())
        if arg_keys:
            best_name: str | None = None
            best_score: tuple[int, int] = (-1, -1)
            tie = False

            for tool in tools:
                try:
                    params = tool.tool_param().get("function", {}).get("parameters", {})
                    properties = params.get("properties", {}) if isinstance(params, dict) else {}
                    tool_keys = set(properties.keys()) if isinstance(properties, dict) else set()
                except Exception:
                    tool_keys = set()

                if not tool_keys:
                    continue

                overlap = len(arg_keys & tool_keys)
                precision = -len(arg_keys - tool_keys)
                score = (overlap, precision)

                if score > best_score:
                    best_name = tool.name
                    best_score = score
                    tie = False
                elif score == best_score and score != (-1, -1):
                    tie = True

            if best_name and best_score[0] > 0 and not tie:
                log.debug(f"Remapping emulated tool '{raw_name}' -> '{best_name}' (schema arg match)")
                return best_name

        # UUID-like placeholders should not be surfaced as final tool names.
        if self._gemma_placeholder_name_re.match(raw_name):
            log.warning(f"Could not confidently map UUID-like emulated tool name '{raw_name}' to a known tool")

        return raw_name

    def _parse_non_python_function_calls(
        self, text: str, tools: Sequence[Tool] | None = None
    ) -> tuple[list[ToolCall], str]:
        """Parse function-call-like lines with non-Python identifiers (Gemma fallback)."""
        tool_calls: list[ToolCall] = []
        cleaned_lines: list[str] = []
        lines = text.split("\n")

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("```"):
                cleaned_lines.append(line)
                continue

            bracket_match = re.match(r"^\[(.+)\]$", stripped, re.DOTALL)
            if bracket_match:
                stripped = bracket_match.group(1).strip()

            match = re.match(r"^([A-Za-z0-9][A-Za-z0-9_-]*)\((.*)\)$", stripped, re.DOTALL)
            if not match:
                cleaned_lines.append(line)
                continue

            raw_name = match.group(1)
            args_src = match.group(2)

            try:
                tree = ast.parse(f"_f({args_src})", mode="eval")
                if not isinstance(tree.body, ast.Call):
                    cleaned_lines.append(line)
                    continue
                call = tree.body

                parsed_args: dict[str, Any] = {}

                # Keyword args
                for keyword in call.keywords:
                    if keyword.arg is None:
                        continue
                    parsed_args[keyword.arg] = self._ast_to_value(keyword.value)

                # Positional args (best-effort param mapping when determinable)
                param_names: list[str] = []
                if tools:
                    # Prefer schema from exact name; otherwise single-tool fallback.
                    tool_for_mapping: Tool | None = next((t for t in tools if t.name == raw_name), None)
                    if tool_for_mapping is None and len(tools) == 1:
                        tool_for_mapping = tools[0]

                    if tool_for_mapping is not None:
                        try:
                            params = tool_for_mapping.tool_param().get("function", {}).get("parameters", {})
                            properties = params.get("properties", {}) if isinstance(params, dict) else {}
                            if isinstance(properties, dict):
                                param_names = list(properties.keys())
                        except Exception:
                            param_names = []

                for i, arg in enumerate(call.args):
                    arg_name = param_names[i] if i < len(param_names) else f"arg{i}"
                    parsed_args[arg_name] = self._ast_to_value(arg)

                resolved_name = self._resolve_emulated_tool_name(raw_name, parsed_args, tools or [])
                tool_calls.append(ToolCall(id=f"call_{len(tool_calls)}", name=resolved_name, args=parsed_args))
            except Exception as e:
                log.debug(f"Failed fallback parse for potential function call '{stripped[:80]}': {e}")
                cleaned_lines.append(line)

        return tool_calls, "\n".join(cleaned_lines)

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
            base_url: Base URL of the external llama-server.

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
            http_client = httpx.AsyncClient(follow_redirects=True, timeout=600, verify=False)

        # llama-server accepts any API key; None is fine when auth is disabled
        return openai.AsyncClient(
            base_url=f"{base_url}/v1",
            api_key="sk-no-key-required",
            http_client=http_client,
        )

    def _as_service_unavailable(self, error: Exception, base_url: str) -> httpx.HTTPStatusError:
        request = httpx.Request("POST", f"{base_url}/v1/chat/completions")
        response = httpx.Response(status_code=503, request=request, text=str(error))
        return httpx.HTTPStatusError(
            message="503 Service Unavailable",
            request=request,
            response=response,
        )

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

    @staticmethod
    def _is_gemma_model(model: str) -> bool:
        """Check if the model is a Gemma model by name pattern."""
        return model.lower().startswith("gemma")

    async def _fetch_server_model_ids(self) -> list[str]:
        """Fetch model IDs currently exposed by the configured llama-server."""
        if not self._base_url:
            return []

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(2.0, connect=0.5), verify=False) as client:
                response = await client.get(f"{self._base_url}/v1/models")
                response.raise_for_status()
                payload = response.json()
        except Exception as e:
            log.debug(f"Could not fetch llama.cpp server models from {self._base_url}/v1/models: {e}")
            return []

        raw_models = payload.get("data")
        if not isinstance(raw_models, list):
            raw_models = payload.get("models")
        if not isinstance(raw_models, list):
            return []

        model_ids: list[str] = []
        for entry in raw_models:
            if isinstance(entry, dict):
                model_id = entry.get("id")
                if isinstance(model_id, str) and model_id.strip():
                    model_ids.append(model_id.strip())
            elif isinstance(entry, str) and entry.strip():
                model_ids.append(entry.strip())

        return model_ids

    @staticmethod
    def _model_tokens(model_id: str) -> set[str]:
        """Build normalized comparison tokens for a model identifier."""
        lowered = model_id.strip().lower()
        if not lowered:
            return set()

        tail = lowered.rsplit(":", 1)[-1]
        basename = Path(tail).name.lower()
        stem = Path(basename).stem.lower() if basename else ""
        repo = lowered.split(":", 1)[0]

        return {token for token in (lowered, tail, basename, stem, repo) if token}

    def _choose_server_model(self, requested_model: str, server_model_ids: Sequence[str]) -> str:
        """Resolve requested model to one currently returned by llama-server."""
        if not server_model_ids:
            return requested_model

        requested = requested_model.strip()
        if not requested:
            return server_model_ids[0]

        server_by_lower = {model_id.lower(): model_id for model_id in server_model_ids}
        requested_lower = requested.lower()
        if requested in server_model_ids:
            return requested
        if requested_lower in server_by_lower:
            return server_by_lower[requested_lower]

        requested_tokens = self._model_tokens(requested)
        if requested_tokens:
            for server_model_id in server_model_ids:
                if requested_tokens & self._model_tokens(server_model_id):
                    return server_model_id

        # Final fallback: always use a model that the server confirms is available.
        return server_model_ids[0]

    async def _resolve_server_model(self, requested_model: str) -> str:
        """Resolve the requested model to a model ID advertised by llama-server."""
        server_model_ids = await self._fetch_server_model_ids()
        if not server_model_ids:
            return requested_model

        resolved = self._choose_server_model(requested_model, server_model_ids)
        if resolved != requested_model:
            log.warning(
                f"Requested llama.cpp model '{requested_model}' is not server-advertised; "
                f"using '{resolved}' from /v1/models"
            )
        return resolved

    async def get_available_language_models(self) -> list[LanguageModel]:
        """
        Get available Llama.cpp models.

        Combines:
        - locally cached llama.cpp-compatible GGUF models (downloaded), and
        - models exposed by the running llama-server via /v1/models.

        Cache-backed models are listed first so downloaded models are visible
        even when the server endpoint is temporarily unavailable.

        Returns:
            List of LanguageModel instances for Llama.cpp
        """
        import httpx

        from nodetool.integrations.huggingface.huggingface_models import (
            get_llamacpp_language_models_from_hf_cache,
            get_llamacpp_language_models_from_llama_cache,
        )

        merged: dict[str, LanguageModel] = {}

        # 1) Discover downloaded GGUF models from local caches.
        cache_results = await asyncio.gather(
            get_llamacpp_language_models_from_hf_cache(),
            get_llamacpp_language_models_from_llama_cache(),
            return_exceptions=True,
        )
        for source, result in (
            ("hf cache", cache_results[0]),
            ("llama.cpp cache", cache_results[1]),
        ):
            if isinstance(result, Exception):
                log.warning(f"Error discovering {source} models: {result}")
                continue
            for model in result:
                if model.id and model.id not in merged:
                    merged[model.id] = model

        # 2) Add currently exposed server models (may include aliases/non-cached specs).
        if self._base_url:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{self._base_url}/v1/models")
                    response.raise_for_status()
                    data = response.json()

                server_count = 0
                for model_data in data.get("data", []):
                    model_id = model_data.get("id", "")
                    if not model_id:
                        continue
                    server_count += 1
                    if model_id not in merged:
                        merged[model_id] = LanguageModel(
                            id=model_id,
                            name=model_id,
                            provider=Provider.LlamaCpp,
                        )
                log.debug(
                    f"Found {server_count} models from llama.cpp server; "
                    f"returning {len(merged)} merged models"
                )
            except Exception as e:
                log.warning(f"Error querying llama.cpp server: {e}")
        else:
            log.debug("LLAMA_CPP_URL not set while listing models; returning cache-discovered models only")

        return list(merged.values())

    async def generate_messages(  # type: ignore[override]
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        max_tokens: int = 1024,
        response_format: dict | None = None,
        **kwargs,
    ) -> AsyncIterator[Chunk | ToolCall]:
        """Stream assistant deltas and tool calls from llama.cpp.

        Args:
            messages: Conversation history to send.
            model: Model spec (GGUF path or HF repo/tag) to run.
            tools: Optional tool definitions.
            max_tokens: Maximum new tokens to generate.
            response_format: Optional response schema.
            **kwargs: Additional OpenAI-compatible parameters.

        Yields:
            ``Chunk`` objects for text deltas and ``ToolCall`` entries when
            the model requests tool execution.
        """
        if not messages:
            raise ValueError("messages must not be empty")

        # Determine if we need tool emulation
        resolved_model = await self._resolve_server_model(model)
        use_tool_emulation = len(tools) > 0 and not self.has_tool_support(resolved_model)
        if use_tool_emulation:
            log.info(f"Using tool emulation for model {resolved_model}")

        if not self._base_url:
            raise RuntimeError("LLAMA_CPP_URL is required for LlamaCpp provider")
        base_url = self._base_url
        _kwargs: dict[str, Any] = {
            "model": resolved_model,
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
            messages, tools, use_tool_emulation, model=resolved_model
        )
        # llama.cpp is sensitive to unsupported fields; pass only necessary ones
        openai_messages = [await self.convert_message(m) for m in messages_normalized]

        # Count prompt tokens before sending
        prompt_tokens = self._count_tokens_in_messages(messages_normalized)

        self._log_api_request("chat_stream", messages_normalized, **_kwargs)

        client = self.get_client(base_url)
        try:
            completion = await client.chat.completions.create(messages=openai_messages, **_kwargs)
        except httpx.ConnectError as e:
            raise self._as_service_unavailable(e, base_url) from e

        delta_tool_calls: dict[int, dict[str, Any]] = {}
        current_chunk = ""
        accumulated_content = ""  # For tool emulation parsing
        completion_text = ""  # Track all generated text for token counting

        async for chunk in completion:
            chunk = chunk  # type: ignore
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if getattr(delta, "content", None) or chunk.choices[0].finish_reason == "stop":
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
                        emulated_calls, _ = self._parse_function_calls(accumulated_content, tools)
                        if not emulated_calls:
                            emulated_calls, _ = self._parse_non_python_function_calls(accumulated_content, tools)
                        else:
                            for call in emulated_calls:
                                call.name = self._resolve_emulated_tool_name(call.name, call.args, tools)
                        for tool_call in emulated_calls:
                            log.debug(f"Yielding emulated tool call: {tool_call.name}")
                            yield tool_call

                    # Count completion tokens
                    completion_tokens = self._count_tokens_in_text(completion_text)
                    log.debug(
                        f"Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {prompt_tokens + completion_tokens}"
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

                    # Count completion tokens and update usage for tool calls
                    completion_tokens = self._count_tokens_in_text(completion_text)
                    # Add tokens for tool call formatting
                    for tc in delta_tool_calls.values():
                        if "function" in tc and "arguments" in tc["function"]:
                            completion_tokens += self._count_tokens_in_text(tc["function"]["arguments"])
                    log.debug(
                        f"Token usage (tool calls) - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {prompt_tokens + completion_tokens}"
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
            model: Model spec (GGUF path or HF repo/tag) to run.
            tools: Optional tool definitions.
            max_tokens: Maximum new tokens to generate.
            response_format: Optional response schema.
            **kwargs: Additional OpenAI-compatible parameters.

        Returns:
            Final assistant ``Message`` with optional ``tool_calls``.
        """
        if not messages:
            raise ValueError("messages must not be empty")

        # Determine if we need tool emulation
        resolved_model = await self._resolve_server_model(model)
        use_tool_emulation = len(tools) > 0 and not self.has_tool_support(resolved_model)
        if use_tool_emulation:
            log.info(f"Using tool emulation for model {resolved_model}")

        if not self._base_url:
            raise RuntimeError("LLAMA_CPP_URL is required for LlamaCpp provider")
        base_url = self._base_url
        _kwargs: dict[str, Any] = {
            "model": resolved_model,
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
            messages, tools, use_tool_emulation, model=resolved_model
        )
        openai_messages = [await self.convert_message(m) for m in messages_normalized]

        # Count prompt tokens before sending
        prompt_tokens = self._count_tokens_in_messages(messages_normalized)

        # Debug: print the processed messages
        # print("DEBUG: Normalized messages:", [{"role": m.role, "content": m.content} for m in messages_normalized])
        # print("DEBUG: OpenAI messages:", openai_messages)
        # print("DEBUG: Tools:", len(tools) if tools else 0)

        self._log_api_request("chat", messages_normalized, **_kwargs)

        client = self.get_client(base_url)
        try:
            completion = await client.chat.completions.create(messages=openai_messages, **_kwargs)
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
            if not emulated_calls:
                emulated_calls, cleaned_content = self._parse_non_python_function_calls(response_message.content, tools)
            else:
                for call in emulated_calls:
                    call.name = self._resolve_emulated_tool_name(call.name, call.args, tools)
            if emulated_calls:
                tool_calls = emulated_calls
                final_content = cleaned_content
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
                    except (TypeError, ValueError):
                        pass

        log.debug(
            f"Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {prompt_tokens + completion_tokens}"
        )

        message = Message(role="assistant", content=final_content, tool_calls=tool_calls)
        self._log_api_response("chat", message)
        return message


if __name__ == "__main__":

    async def _run_all():
        from nodetool.agents.tools.math_tools import CalculatorTool

        provider = LlamaProvider({})  # type: ignore[call-arg]
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

            async def process(self, context: ProcessingContext, params: dict[str, Any]) -> Any:  # type: ignore[override]
                return {"echo": params.get("text", "")}

        model = "ggml-org/Qwen2.5-Coder-0.5B-Q8_0-GGUF"
        tools: list[Tool] = [EchoTool()]

        print(f"Model: {model}")

        # First try without tools to confirm basic functionality
        messages = [
            Message(role="user", content="Just say 'Hello World'"),
        ]
        print("\nTesting without tools first...")
        resp = await provider.generate_message(messages=messages, model=model, max_tokens=64)
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
            resp = await provider.generate_message(messages=messages, model=model, tools=tools, max_tokens=64)
            print("First response successful:", resp.content)

            if resp.tool_calls:
                print(f"✅ Tool call test PASSED! Got {len(resp.tool_calls)} tool calls:")
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
                        content_str = final.content if isinstance(final.content, str) else ""
                        print("✅ OpenAI approach: SUCCESS -", content_str.strip())
                    except Exception as e:
                        print("❌ OpenAI approach failed:", e)
                        print("Tool execution successful but conversation continuation failed.")
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
        hf_hub_download("ggml-org/gemma-3-1b-it-GGUF", filename="gemma-3-1b-it-Q4_K_M.gguf", token=token)
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

                matching_tool = next((t for t in calculator_tools if t.name == tool_call.name), None)
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
            print(f"\n⚠️  WARNING: Reached max iterations ({max_iterations}). Stopping.\n")

        print(f"\n{'=' * 60}")
        print(f"Final Answer: {response.content}")
        print(f"{'=' * 60}\n")

    asyncio.run(_run_all())


# Conditionally register the provider only if LLAMA_CPP_URL is set
if _llama_cpp_url:
    register_provider(Provider.LlamaCpp)(LlamaProvider)
else:
    log.debug("LlamaCpp provider not registered: LLAMA_CPP_URL not set")
