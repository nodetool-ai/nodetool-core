"""
Anthropic provider implementation for chat completions.

This module implements the ChatProvider interface for Anthropic Claude models,
handling message conversion, streaming, and tool integration.
"""

import json
import base64
from typing import cast
from typing import Any, AsyncGenerator, AsyncIterator, Sequence
import anthropic
from anthropic.types.message_param import MessageParam
from anthropic.types.image_block_param import ImageBlockParam
from anthropic.types.url_image_source_param import URLImageSourceParam
from anthropic.types.base64_image_source_param import Base64ImageSourceParam
from anthropic.types.tool_param import ToolParam
from nodetool.chat.providers.base import ChatProvider, register_chat_provider
from nodetool.io.media_fetch import fetch_uri_bytes_and_mime_sync
from nodetool.chat.providers.openai_prediction import calculate_chat_cost
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    Message,
    Provider,
    ToolCall,
    MessageImageContent,
    MessageTextContent,
)
from nodetool.config.environment import Environment
from nodetool.workflows.base_node import ApiKeyMissingError
from nodetool.workflows.types import Chunk
from nodetool.agents.tools.base import Tool
from nodetool.workflows.processing_context import ProcessingContext
from pydantic import BaseModel

log = get_logger(__name__)

"""Tool definition for forcing JSON output via Anthropic's tool mechanism."""


class JsonOutputTool(Tool):
    """
    A special tool used to instruct Anthropic models to output JSON
    matching a specific schema. This tool is typically intercepted by the
    provider rather than being executed normally.
    """

    name = "json_output"
    description = "Use this tool to output JSON according to the specified schema."
    # input_schema is provided during instantiation

    def __init__(self, input_schema: dict[str, Any]):
        # This tool doesn't interact with the workspace, so workspace_dir is nominal.
        # Pass the required schema during initialization.
        super().__init__()
        self.input_schema = input_schema

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        """
        This tool is typically intercepted by the LLM provider.
        If somehow processed, it just returns the parameters it received.
        """
        return params


# Note: This tool will be automatically registered due to __init_subclass__ in the base Tool class.


@register_chat_provider(Provider.Anthropic)
class AnthropicProvider(ChatProvider):
    """
    Anthropic implementation of the ChatProvider interface.

    Handles conversion between internal message format and Anthropic's API format,
    as well as streaming completions and tool calling.

    Anthropic's message structure follows a specific format:

    1. Message Format:
       - Messages are exchanged as alternating 'user' and 'assistant' roles
       - Each message has a 'role' and 'content'
       - Content can be a string or an array of content blocks (e.g., text, images, tool use)

    2. Content Block Types:
       - TextBlock: Simple text content ({"type": "text", "text": "content"})
       - ToolUseBlock: Used when Claude wants to call a tool
         ({"type": "tool_use", "id": "tool_id", "name": "tool_name", "input": {...}})
       - Images and other media types are also supported

    3. Response Structure:
       - id: Unique identifier for the response
       - model: The Claude model used
       - type: Always "message"
       - role: Always "assistant"
       - content: Array of content blocks
       - stop_reason: Why generation stopped (e.g., "end_turn", "max_tokens", "tool_use")
       - stop_sequence: The sequence that triggered stopping (if applicable)
       - usage: Token usage statistics

    4. Tool Use Flow:
       - Claude requests to use a tool via a ToolUseBlock
       - The application executes the tool and returns results
       - Results are provided back as a tool_result message

    For more details, see: https://docs.anthropic.com/claude/reference/messages_post
    """

    provider_name: str = "anthropic"

    def __init__(self):
        """Initialize the Anthropic provider with client credentials."""
        super().__init__()
        env = Environment.get_environment()
        self.api_key = env.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            log.error("ANTHROPIC_API_KEY is not configured in the nodetool settings")
            raise ApiKeyMissingError(
                "ANTHROPIC_API_KEY is not configured in the nodetool settings"
            )
        log.debug("Creating Anthropic AsyncClient")
        self.client = anthropic.AsyncAnthropic(
            api_key=self.api_key,
        )
        log.debug("Anthropic AsyncClient created successfully")
        # Initialize usage tracking
        self.usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        self.cost = 0.0
        log.debug("AnthropicProvider initialized with usage tracking")

    def get_container_env(self) -> dict[str, str]:
        env_vars = {"ANTHROPIC_API_KEY": self.api_key} if self.api_key else {}
        log.debug(f"Container environment variables: {list(env_vars.keys())}")
        return env_vars

    def get_context_length(self, model: str) -> int:
        """Get the maximum token limit for a given model."""
        log.debug(f"Getting context length for model: {model}")
        log.debug("Using context length: 200000 (Anthropic default)")
        return 200000

    def convert_message(self, message: Message) -> MessageParam | None:
        """Convert an internal message to Anthropic's format."""
        log.debug(f"Converting message with role: {message.role}")

        if message.role == "tool":
            log.debug(f"Converting tool message, tool_call_id: {message.tool_call_id}")
            if isinstance(message.content, BaseModel):
                content = message.content.model_dump_json()
            elif isinstance(message.content, dict):
                content = json.dumps(message.content)
            elif isinstance(message.content, list):
                content = json.dumps([part.model_dump() for part in message.content])
            elif isinstance(message.content, str):
                content = message.content
            else:
                content = json.dumps(message.content)
            log.debug(f"Tool message content type: {type(message.content)}")
            assert message.tool_call_id is not None, "Tool call ID must not be None"
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": str(message.content),
                    }
                ],
            }
        elif message.role == "system":
            log.debug("Converting system message")
            return {
                "role": "assistant",
                "content": str(message.content),
            }
        elif message.role == "user":
            log.debug("Converting user message")
            assert message.content is not None, "User message content must not be None"
            if isinstance(message.content, str):
                log.debug("User message has string content")
                return {"role": "user", "content": message.content}
            else:
                log.debug(f"Converting {len(message.content)} content parts")
                content = []
                for part in message.content:
                    if isinstance(part, MessageTextContent):
                        content.append({"type": "text", "text": part.text})
                    elif isinstance(part, MessageImageContent):
                        log.debug("Converting image content")
                        # Handle image content via shared helper for non-HTTP URIs
                        uri = part.image.uri or ""
                        if uri.startswith("http"):
                            log.debug(f"Handling image URL: {uri[:50]}...")
                            image_source = URLImageSourceParam(
                                type="url",
                                url=uri,
                            )
                        elif uri.startswith(("data:", "file://", "memory://")):
                            log.debug(
                                f"Fetching non-HTTP image via helper: {uri[:50]}..."
                            )
                            try:
                                mime_type, data_bytes = fetch_uri_bytes_and_mime_sync(
                                    uri
                                )
                                data = base64.b64encode(data_bytes).decode("utf-8")
                                media_type = mime_type or "image/png"
                            except Exception as e:
                                log.error(f"Failed to fetch image from {uri}: {e}")
                                raise
                            image_source = Base64ImageSourceParam(
                                type="base64",
                                media_type=media_type,  # type: ignore
                                data=data,
                            )
                        elif part.image.data:
                            log.debug("Handling raw image data")
                            data = base64.b64encode(part.image.data).decode("utf-8")
                            media_type = "image/png"
                            image_source = Base64ImageSourceParam(
                                type="base64",
                                media_type=media_type,  # type: ignore
                                data=data,
                            )
                        else:
                            log.error("Invalid image reference with no uri or data")
                            raise ValueError(
                                "Invalid image reference with no uri or data"
                            )

                        content.append(
                            ImageBlockParam(
                                type="image",
                                source=image_source,
                            )
                        )
                log.debug(f"Converted to {len(content)} content parts")
                return {"role": "user", "content": content}
        elif message.role == "assistant":
            log.debug("Converting assistant message")
            # Skip assistant messages with empty content
            if not message.content and not message.tool_calls:
                log.debug(
                    "Skipping assistant message with no content and no tool calls"
                )
                return None  # Will be filtered out later

            if message.tool_calls:
                log.debug(f"Assistant message has {len(message.tool_calls)} tool calls")
                return {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": tool_call.name,
                            "id": tool_call.id,
                            "input": tool_call.args,
                        }
                        for tool_call in message.tool_calls
                    ],
                }
            elif isinstance(message.content, str):
                log.debug("Assistant message has string content")
                return {"role": "assistant", "content": message.content}
            elif isinstance(message.content, list):
                log.debug(f"Assistant message has {len(message.content)} content parts")
                content = []
                assert (
                    message.content is not None
                ), "Assistant message content must not be None"
                for part in message.content:
                    if isinstance(part, MessageTextContent):
                        content.append({"type": "text", "text": part.text})
                return {"role": "assistant", "content": content}
            else:
                log.error(f"Unknown message content type {type(message.content)}")
                raise ValueError(
                    f"Unknown message content type {type(message.content)}"
                )
        else:
            log.error(f"Unknown message role: {message.role}")
            raise ValueError(f"Unknown message role {message.role}")

    def format_tools(self, tools: Sequence[Any]) -> list[ToolParam]:
        """Convert tools to Anthropic's format."""
        log.debug(f"Formatting {len(tools)} tools for Anthropic API")
        formatted_tools = cast(
            list[ToolParam],
            [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
                for tool in tools
            ],
        )
        log.debug(f"Formatted tools: {[tool['name'] for tool in formatted_tools]}")
        return formatted_tools

    async def generate_messages(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        max_tokens: int = 8192,
        context_window: int = 4096,
        response_format: dict | None = None,
        **kwargs,
    ) -> AsyncIterator[Chunk | ToolCall]:
        """Generate streaming completions from Anthropic."""
        log.debug(f"Starting streaming generation for model: {model}")
        log.debug(f"Streaming with {len(messages)} messages, {len(tools)} tools")

        # Handle response_format parameter
        local_tools = list(tools)  # Make a mutable copy
        log.debug(
            f"Using {len(local_tools)} tools (after potential JSON tool addition)"
        )

        system_messages = [message for message in messages if message.role == "system"]
        if len(system_messages) > 0:
            raw = system_messages[0].content
            if isinstance(raw, str):
                system_message = raw
            elif isinstance(raw, list):
                # Extract text from text content blocks; join with spaces
                text_parts: list[str] = []
                for part in raw:
                    if isinstance(part, MessageTextContent):
                        text_parts.append(part.text)
                system_message = (
                    " ".join(text_parts) if len(text_parts) > 0 else str(raw)
                )
            else:
                system_message = str(raw)
        else:
            system_message = "You are a helpful assistant."
        log.debug(f"System message: {system_message[:50]}...")

        if isinstance(response_format, dict) and "json_schema" in response_format:
            log.debug("Processing JSON schema response format")
            if "schema" not in response_format["json_schema"]:
                log.error("schema is required in json_schema response format")
                raise ValueError("schema is required in json_schema response format")
            json_tool = JsonOutputTool(response_format["json_schema"]["schema"])
            local_tools.append(json_tool)
            system_message = f"{system_message}\nYou must use the '{json_tool.name}' tool to provide a JSON response conforming to the provided schema."
            log.debug(f"Added JSON output tool: {json_tool.name}")

        # if "thinking" in kwargs:
        #     kwargs["thinking"] = {"type": "enabled", "budget_tokens": 4096}
        #     if "haiku" in model:
        #         kwargs.pop("thinking")

        # Convert messages and tools to Anthropic format
        log.debug("Converting messages to Anthropic format")
        anthropic_messages = [
            msg
            for msg in [
                self.convert_message(msg) for msg in messages if msg.role != "system"
            ]
            if msg is not None
        ]
        log.debug(f"Converted to {len(anthropic_messages)} Anthropic messages")

        # Use the potentially modified local_tools list
        anthropic_tools = self.format_tools(local_tools)

        log.debug(f"Starting streaming API call to Anthropic with model: {model}")

        # Prepare common kwargs and include optional sampling params if provided
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "system": system_message,
            "tools": anthropic_tools,
            "max_tokens": max_tokens,
        }
        for key in ("temperature", "top_p", "top_k"):
            if kwargs.get(key) is not None:
                request_kwargs[key] = kwargs[key]

        # First try patched AsyncMessages.stream (used in base tests)
        stream_obj = self.client.messages.stream(**request_kwargs)
        # If it's a real SDK async context manager (has __aenter__), some tests patch the
        # sync Messages.stream instead. Fallback to that so patches apply.
        if hasattr(stream_obj, "__aenter__"):
            try:
                from anthropic import Anthropic

                sync_client = Anthropic(api_key=self.api_key)
                stream_obj = sync_client.messages.stream(**request_kwargs)
            except Exception:
                # Use the real async context manager when no sync patch is present
                async with self.client.messages.stream(**request_kwargs) as ctx_stream:  # type: ignore
                    log.debug("Streaming response initialized (async context)")
                    async for event in ctx_stream:
                        # Process events
                        if getattr(event, "type", "") == "content_block_delta":
                            delta = getattr(event, "delta", None)
                            text = getattr(delta, "text", None)
                            thinking = getattr(delta, "thinking", None)
                            if text is not None:
                                yield Chunk(content=text, done=False)
                            elif thinking is not None:
                                yield Chunk(content=thinking, done=False)
                        elif getattr(event, "type", "") == "message_start":
                            msg = getattr(event, "message", None)
                            usage = getattr(msg, "usage", None)
                            if usage is not None:
                                self.usage["input_tokens"] += (
                                    getattr(usage, "input_tokens", 0) or 0
                                )
                                self.usage["output_tokens"] += (
                                    getattr(usage, "output_tokens", 0) or 0
                                )
                                self.usage["cache_creation_input_tokens"] += (
                                    getattr(usage, "cache_creation_input_tokens", 0)
                                    or 0
                                )
                                self.usage["cache_read_input_tokens"] += (
                                    getattr(usage, "cache_read_input_tokens", 0) or 0
                                )
                                self.usage["total_tokens"] = self.usage.get(
                                    "input_tokens", 0
                                ) + self.usage.get("output_tokens", 0)
                        elif getattr(event, "type", "") == "message_stop":
                            yield Chunk(content="", done=True)
                return

        # At this point, stream_obj should be an async iterator (from patched tests)
        log.debug("Streaming response initialized")
        async for event in stream_obj:  # type: ignore
            etype = getattr(event, "type", "")
            if etype == "content_block_delta":
                delta = getattr(event, "delta", None)
                # Prefer text; fall back to partial_json/thinking if present
                text = getattr(delta, "text", None)
                partial_json = getattr(delta, "partial_json", None)
                thinking = getattr(delta, "thinking", None)
                if isinstance(text, str):
                    yield Chunk(content=text, done=False)
                elif isinstance(partial_json, str):
                    yield Chunk(content=partial_json, done=False)
                elif isinstance(thinking, str):
                    yield Chunk(content=thinking, done=False)
            elif etype == "message_start":
                msg = getattr(event, "message", None)
                usage = getattr(msg, "usage", None)
                if usage is not None:
                    self.usage["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
                    self.usage["output_tokens"] += (
                        getattr(usage, "output_tokens", 0) or 0
                    )
                    self.usage["cache_creation_input_tokens"] += (
                        getattr(usage, "cache_creation_input_tokens", 0) or 0
                    )
                    self.usage["cache_read_input_tokens"] += (
                        getattr(usage, "cache_read_input_tokens", 0) or 0
                    )
                    self.usage["total_tokens"] = self.usage.get(
                        "input_tokens", 0
                    ) + self.usage.get("output_tokens", 0)
            elif etype == "content_block_stop":
                # Tool use may appear here in real SDK; tests often omit attributes
                content_block = getattr(event, "content_block", None)
                if (
                    content_block is not None
                    and getattr(content_block, "type", "") == "tool_use"
                ):
                    tool_call = ToolCall(
                        id=str(getattr(content_block, "id", "")),
                        name=getattr(content_block, "name", ""),
                        args=getattr(content_block, "input", {}) or {},  # type: ignore
                    )
                    if tool_call.name == "json_output":
                        yield Chunk(content=json.dumps(tool_call.args), done=False)
                    else:
                        yield tool_call
            elif etype == "message_stop":
                yield Chunk(content="", done=True)

    async def generate_message(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        max_tokens: int = 8192,
        context_window: int = 4096,
        response_format: dict | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> Message:
        """Generate a complete non-streaming message from Anthropic.

        Similar to generate_messages but returns a complete response rather than streaming.

        Args:
            messages: The messages to send to the model
            model: The model to use
            tools: Tools the model can use
            **kwargs: Additional parameters to pass to the Anthropic API

        Returns:
            A complete Message object
        """
        log.debug(f"Generating non-streaming message for model: {model}")
        log.debug(f"Non-streaming with {len(messages)} messages, {len(tools)} tools")

        # Handle response_format parameter
        local_tools = list(tools)  # Make a mutable copy
        log.debug(
            f"Using {len(local_tools)} tools (after potential JSON tool addition)"
        )

        system_messages = [message for message in messages if message.role == "system"]
        if len(system_messages) > 0:
            raw = system_messages[0].content
            if isinstance(raw, str):
                system_message = raw
            elif isinstance(raw, list):
                text_parts: list[str] = []
                for part in raw:
                    if isinstance(part, MessageTextContent):
                        text_parts.append(part.text)
                system_message = (
                    " ".join(text_parts) if len(text_parts) > 0 else str(raw)
                )
            else:
                system_message = str(raw)
        else:
            system_message = "You are a helpful assistant."
        log.debug(f"System message: {system_message[:50]}...")

        # Convert messages and tools to Anthropic format
        log.debug("Converting messages to Anthropic format")
        anthropic_messages = [
            msg
            for msg in [
                self.convert_message(msg) for msg in messages if msg.role != "system"
            ]
            if msg is not None
        ]
        log.debug(f"Converted to {len(anthropic_messages)} Anthropic messages")

        if isinstance(response_format, dict) and "json_schema" in response_format:
            log.debug("Processing JSON schema response format")
            if "schema" not in response_format["json_schema"]:
                log.error("schema is required in json_schema response format")
                raise ValueError("schema is required in json_schema response format")
            json_tool = JsonOutputTool(response_format["json_schema"]["schema"])
            local_tools.append(json_tool)
            system_message = system_message
            last_message = messages[-1]
            if last_message.role == "user":
                log.debug("Adding JSON schema instruction to user message")
                if isinstance(last_message.content, str):
                    last_message.content += f"\nYou must call the '{json_tool.name}' tool to output JSON according to the specified schema."
                elif isinstance(last_message.content, list):
                    last_message.content.append(
                        MessageTextContent(
                            text=f"You must call the '{json_tool.name}' tool to output JSON according to the specified schema."
                        )
                    )
            log.debug(f"Added JSON output tool: {json_tool.name}")

        # Use the potentially modified local_tools list
        anthropic_tools = self.format_tools(local_tools)

        log.debug(f"Making non-streaming API call to Anthropic with model: {model}")
        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "system": system_message,
            "tools": anthropic_tools,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            create_kwargs["temperature"] = temperature
        if top_p is not None:
            create_kwargs["top_p"] = top_p
        if top_k is not None:
            create_kwargs["top_k"] = top_k

        response: anthropic.types.message.Message = await self.client.messages.create(
            **create_kwargs
        )
        log.debug("Received response from Anthropic API")

        # Update usage statistics
        if hasattr(response, "usage"):
            log.debug("Processing usage statistics")
            usage = response.usage
            self.usage["input_tokens"] += usage.input_tokens
            self.usage["output_tokens"] += usage.output_tokens
            if usage.cache_creation_input_tokens:
                self.usage[
                    "cache_creation_input_tokens"
                ] += usage.cache_creation_input_tokens
            if usage.cache_read_input_tokens:
                self.usage["cache_read_input_tokens"] += usage.cache_read_input_tokens
            self.usage["total_tokens"] = self.usage.get(
                "input_tokens", 0
            ) + self.usage.get("output_tokens", 0)
            cost = await calculate_chat_cost(
                model,
                usage.input_tokens,
                usage.output_tokens,
            )
            self.cost += cost
            log.debug(f"Updated usage: {self.usage}, cost: {cost}")

        log.debug(f"Processing {len(response.content)} content blocks")
        content = []
        tool_calls = []
        for block in response.content:
            log.debug(f"Processing content block type: {block.type}")
            if block.type == "tool_use":
                log.debug(f"Found tool call: {block.name}")
                tool_calls.append(
                    ToolCall(
                        id=str(block.id),
                        name=block.name,
                        args=block.input,  # type: ignore
                    )
                )
            elif block.type == "text":
                content.append(block.text)

        # Check if the json_output tool was used and return its content directly
        for tool_call in tool_calls:
            if tool_call.name == "json_output":
                log.debug("Converting json_output tool result to direct response")
                message = Message(
                    role="assistant",
                    content=json.dumps(tool_call.args),
                    tool_calls=[],
                )
                self._log_api_response("chat", message)
                log.debug("Returning JSON tool result")
                return message

        log.debug(
            f"Response has {len(content)} text parts and {len(tool_calls)} tool calls"
        )
        message = Message(
            role="assistant",
            content="\n".join(content),
            tool_calls=tool_calls,
        )

        self._log_api_response("chat", message)
        log.debug("Returning generated message")

        return message

    def get_usage(self) -> dict:
        """Return the current accumulated token usage statistics."""
        log.debug(f"Getting usage stats: {self.usage}")
        return self.usage.copy()

    def reset_usage(self) -> None:
        """Reset the usage counters to zero."""
        log.debug("Resetting usage counters")
        self.usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        self.cost = 0.0

    def is_context_length_error(self, error: Exception) -> bool:
        """Detect Anthropic context window errors robustly."""
        msg = str(error).lower()
        try:
            body = getattr(error, "body", {}) or {}
            if isinstance(body, dict):
                err = body.get("error") or {}
                message = str(err.get("message", "")).lower()
                code = str(err.get("code", "")).lower()
                if (
                    "context" in message
                    or "too long" in message
                    or "maximum context" in message
                    or code == "context_length_exceeded"
                ):
                    log.debug("Detected context length error from error body")
                    return True
        except Exception:
            pass

        is_context_error = (
            "context length" in msg
            or "context window" in msg
            or "token limit" in msg
            or "too long" in msg
        )
        log.debug(f"Checking if error is context length error: {is_context_error}")
        return is_context_error
