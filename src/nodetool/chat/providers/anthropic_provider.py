"""
Anthropic provider implementation for chat completions.

This module implements the ChatProvider interface for Anthropic Claude models,
handling message conversion, streaming, and tool integration.
"""

import json
import base64
from typing import Any, AsyncGenerator, AsyncIterator, Sequence
import anthropic
from anthropic.types.message_param import MessageParam
from anthropic.types.image_block_param import ImageBlockParam
from anthropic.types.url_image_source_param import URLImageSourceParam
from anthropic.types.base64_image_source_param import Base64ImageSourceParam
from anthropic.types.tool_param import ToolParam
from nodetool.chat.providers.base import ChatProvider
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

    provider: Provider = Provider.Anthropic

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
                        # Handle image content - either data URI or raw base64
                        uri = part.image.uri
                        if uri.startswith("http"):
                            log.debug(f"Handling image URL: {uri[:50]}...")
                            # Handle image URL
                            media_type = "image/png"
                            data = uri
                            image_source = URLImageSourceParam(
                                type="url",
                                url=uri,
                            )
                        elif part.image.data:
                            log.debug("Handling raw image data")
                            # Handle raw image data
                            data = base64.b64encode(part.image.data).decode("utf-8")
                            media_type = "image/png"  # Default assumption
                            image_source = Base64ImageSourceParam(
                                type="base64",
                                media_type=media_type,  # type: ignore
                                data=data,
                            )
                        else:
                            log.error(f"Invalid image URI: {uri}")
                            raise ValueError(f"Invalid image URI: {uri}")

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
        formatted_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in tools
        ]
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
        system_message = (
            str(system_messages[0].content)
            if len(system_messages) > 0
            else "You are a helpful assistant."
        )
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
        async with self.client.messages.stream(
            model=model,
            messages=anthropic_messages,
            system=system_message,
            tools=anthropic_tools,
            max_tokens=max_tokens,
        ) as stream:
            log.debug("Streaming response initialized")
            async for event in stream:
                log.debug(f"Processing streaming event: {event.type}")

                if event.type == "content_block_delta":
                    log.debug(f"Content block delta type: {event.delta.type}")
                    if event.delta.type == "text_delta":
                        log.debug(f"Yielding text chunk: {event.delta.text[:50]}...")
                        yield Chunk(content=event.delta.text, done=False)
                    elif event.delta.type == "thinking_delta":
                        log.debug(
                            f"Yielding thinking chunk: {event.delta.thinking[:50]}..."
                        )
                        yield Chunk(content=event.delta.thinking, done=False)
                elif event.type == "content_block_start":
                    if (
                        hasattr(event, "content_block")
                        and event.content_block.type == "thinking"
                    ):
                        log.debug("Handling start of thinking block")
                        # Handle start of a thinking block if needed
                        pass
                elif event.type == "content_block_stop":
                    log.debug(f"Content block stop, type: {event.content_block.type}")
                    if event.content_block.type == "tool_use":
                        log.debug(f"Processing tool use: {event.content_block.name}")
                        tool_call = ToolCall(
                            id=str(event.content_block.id),
                            name=event.content_block.name,
                            args=event.content_block.input,  # type: ignore
                        )
                        # If this is the json_output tool, convert it to a normal text chunk
                        if tool_call.name == "json_output":
                            json_str = json.dumps(tool_call.args)
                            log.debug("Converting json_output tool to text chunk")
                            yield Chunk(content=json_str, done=False)
                        else:
                            log.debug(f"Yielding tool call: {tool_call.name}")
                            yield tool_call
                    elif event.content_block.type == "thinking":
                        log.debug("Handling complete thinking block")
                        # Handle complete thinking blocks if needed
                        pass
                elif event.type == "message_stop":
                    log.debug("Message stop event received")
                    # Update usage statistics when the message is complete
                    if hasattr(event, "message") and hasattr(event.message, "usage"):
                        log.debug("Processing usage statistics")
                        usage = event.message.usage
                        self.usage["input_tokens"] += usage.input_tokens
                        self.usage["output_tokens"] += usage.output_tokens
                        if usage.cache_creation_input_tokens:
                            self.usage[
                                "cache_creation_input_tokens"
                            ] += usage.cache_creation_input_tokens
                        if usage.cache_read_input_tokens:
                            self.usage[
                                "cache_read_input_tokens"
                            ] += usage.cache_read_input_tokens
                        cost = await calculate_chat_cost(
                            model,
                            usage.input_tokens,
                            usage.output_tokens,
                        )
                        self.cost += cost
                        log.debug(f"Updated usage: {self.usage}, cost: {cost}")

                    log.debug("Yielding final done chunk")
                    yield Chunk(content="", done=True)

    async def generate_message(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        max_tokens: int = 8192,
        context_window: int = 4096,
        response_format: dict | None = None,
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
        system_message = (
            str(system_messages[0].content)
            if len(system_messages) > 0
            else "You are a helpful assistant."
        )
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
        response: anthropic.types.message.Message = await self.client.messages.create(
            model=model,
            messages=anthropic_messages,
            system=system_message,
            tools=anthropic_tools,
            max_tokens=max_tokens,
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
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        self.cost = 0.0

    def is_context_length_error(self, error: Exception) -> bool:
        msg = str(error).lower()
        is_context_error = (
            "context length" in msg or "context window" in msg or "token limit" in msg
        )
        log.debug(f"Checking if error is context length error: {is_context_error}")
        return is_context_error
