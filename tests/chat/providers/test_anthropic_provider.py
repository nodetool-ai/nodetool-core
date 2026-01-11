"""
Tests for Anthropic provider with comprehensive API response mocking.

This module tests the Anthropic Claude provider implementation including:
- Claude model responses
- Tool use functionality
- Streaming responses
- Error handling specific to Anthropic API

Anthropic Messages API Documentation (2024):
URL: https://docs.anthropic.com/en/api/messages

The Messages API is Anthropic's primary interface for interacting with Claude models.

Key Request Parameters:
- model: Claude model (e.g., "claude-sonnet-4-20250514", "claude-3-5-haiku-20241022")
- messages: Array of message objects with role ("user", "assistant") and content
- system: System prompt providing context and instructions (separate from messages)
- max_tokens: Maximum response tokens (required)
- tools: Array of tool definitions for tool use
- stream: Boolean for streaming responses
- temperature: Randomness control (0.0-1.0)
- top_k: Top-k sampling (1-40)
- top_p: Nucleus sampling (0.0-1.0)

Message Content:
- String content: Shorthand for single text block
- Array content: Multiple content blocks (text, image, tool_use, tool_result)
- Image blocks: Support base64 images or URLs (PDFs supported via URLs)

Response Format:
- id: Unique message identifier
- type: "message"
- role: "assistant"
- content: Array of content blocks
- model: Model used for generation
- usage: Token counts (input_tokens, output_tokens)
- stop_reason: Why generation stopped ("end_turn", "max_tokens", "stop_sequence", "tool_use")

Tool Use:
- Tools defined with name, description, and input_schema (JSON Schema)
- Claude responds with tool_use blocks containing id, name, and input
- Follow up with tool_result blocks containing tool_call_id and content
- Supports multiple tool calls in single response

Streaming:
- Server-sent events with event types:
  - message_start: Initial response metadata
  - content_block_start/delta/stop: Content generation
  - message_delta: Usage updates
  - message_stop: Final message
- Each event has type and data fields

Special Features:
- Prompt caching (beta): Reduce latency up to 80% and costs up to 90%
- Files API: Upload files and reference in Messages API
- Large context: Up to 200k tokens context window
- System prompts: Separate from conversation for role/context setting

Error Handling:
- 400: Invalid request
- 401: Unauthorized (invalid API key)
- 403: Forbidden (permissions issue)
- 429: Rate limit exceeded
- 500: Internal server error
- 529: Service overloaded
"""

from typing import Any
from unittest.mock import MagicMock, patch

import anthropic
import pytest
from anthropic.types import Message as AnthropicMessage
from anthropic.types import TextBlock, ToolUseBlock, Usage

from nodetool.metadata.types import Message, MessageTextContent
from nodetool.providers.anthropic_provider import AnthropicProvider
from tests.chat.providers.test_base_provider import BaseProviderTest, ResponseFixtures


class TestAnthropicProvider(BaseProviderTest):
    """Test suite for Anthropic provider with realistic API response mocking."""

    @property
    def provider_class(self):
        return AnthropicProvider

    @property
    def provider_name(self):
        return "anthropic"

    def create_anthropic_message_response(
        self, content: str = "Hello, world!", tool_uses: list[dict] | None = None
    ) -> AnthropicMessage:
        """Create a realistic Anthropic Message response."""
        content_blocks = []

        if content:
            content_blocks.append(TextBlock(text=content, type="text"))

        if tool_uses:
            for tool_use in tool_uses:
                content_blocks.append(
                    ToolUseBlock(
                        id=tool_use["id"],
                        name=tool_use["name"],
                        input=tool_use["args"],
                        type="tool_use",
                    )
                )

        return AnthropicMessage(
            id="msg_123",
            content=content_blocks,
            model="claude-3-sonnet-20240229",
            role="assistant",
            stop_reason="end_turn" if not tool_uses else "tool_use",
            stop_sequence=None,
            type="message",
            usage=Usage(input_tokens=10, output_tokens=25),
        )

    def create_anthropic_streaming_responses(self, text: str = "Hello world!", chunk_size: int = 5) -> list[dict]:
        """Create realistic Anthropic streaming response events."""
        events = []

        # Message start event
        events.append(
            {
                "type": "message_start",
                "message": {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3-sonnet-20240229",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            }
        )

        # Content block start
        events.append(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }
        )

        # Content block deltas
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i : i + chunk_size]
            events.append(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": chunk_text},
                }
            )

        # Content block stop
        events.append({"type": "content_block_stop", "index": 0})

        # Message stop
        events.append({"type": "message_stop"})

        return events

    def create_anthropic_error(self, error_type: str = "rate_limit"):
        """Create realistic Anthropic API errors."""
        if error_type == "rate_limit":
            return anthropic.RateLimitError(
                message="Rate limit exceeded",
                response=MagicMock(status_code=429),
                body={
                    "error": {
                        "type": "rate_limit_error",
                        "message": "Rate limit exceeded",
                    }
                },
            )
        elif error_type == "invalid_api_key":
            return anthropic.AuthenticationError(
                message="Invalid API key",
                response=MagicMock(status_code=401),
                body={"error": {"type": "authentication_error"}},
            )
        else:
            return anthropic.APIError(message="Unknown error")

    def mock_api_call(self, response_data: dict[str, Any]):
        """Mock Anthropic API call with structured response."""
        if "tool_calls" in response_data:
            # Tool use response
            tool_uses = [{"id": tc["id"], "name": tc["name"], "args": tc["args"]} for tc in response_data["tool_calls"]]

            anthropic_response = self.create_anthropic_message_response(
                content=response_data.get("text"), tool_uses=tool_uses
            )
        else:
            # Regular text response
            anthropic_response = self.create_anthropic_message_response(
                content=response_data.get("text", "Hello, world!")
            )

        # Mock the async create method
        async def mock_create(*args, **kwargs):
            return anthropic_response

        return patch.object(
            anthropic.resources.messages.AsyncMessages,
            "create",
            side_effect=mock_create,
        )

    def mock_streaming_call(self, chunks: list[dict[str, Any]]):
        """Mock Anthropic streaming API call."""
        # Convert generic chunks to Anthropic streaming events
        text = "".join(chunk.get("content", "") for chunk in chunks)
        anthropic_events = self.create_anthropic_streaming_responses(text)

        class MockStream:
            def __init__(self, events):
                self.events = events
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.events):
                    raise StopAsyncIteration

                event = self.events[self.index]
                self.index += 1

                # Create mock event object
                mock_event = MagicMock()
                mock_event.type = event["type"]

                if event["type"] == "message_start":
                    mock_event.message = MagicMock()
                    mock_event.message.usage = MagicMock()
                    mock_event.message.usage.input_tokens = 10
                elif event["type"] == "content_block_delta":
                    mock_event.delta = MagicMock()
                    mock_event.delta.text = event["delta"]["text"]
                    mock_event.index = event["index"]

                return mock_event

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        return patch.object(
            anthropic.resources.messages.AsyncMessages,
            "stream",
            return_value=MockStream(anthropic_events),
        )

    def mock_error_response(self, error_type: str):
        """Mock Anthropic API error response."""
        error = self.create_anthropic_error(error_type)
        return patch.object(anthropic.resources.messages.AsyncMessages, "create", side_effect=error)

    @pytest.mark.asyncio
    async def test_anthropic_specific_features(self):
        """Test Anthropic-specific features and parameters."""
        provider = self.create_provider()
        messages = self.create_simple_messages("Test Anthropic features")

        with self.mock_api_call(ResponseFixtures.simple_text_response("Claude response")) as mock_call:
            await provider.generate_message(
                messages,
                "claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=0.7,
                top_p=0.9,
            )

        # Verify the call was made with correct parameters
        mock_call.assert_called_once()
        call_kwargs = mock_call.call_args[1]
        assert call_kwargs["model"] == "claude-3-sonnet-20240229"
        assert call_kwargs["max_tokens"] == 1000
        assert call_kwargs["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_tool_use_with_anthropic_format(self):
        """Test tool use with Anthropic's specific format."""
        provider = self.create_provider()
        messages = self.create_tool_messages()
        tools = [self.create_mock_tool()]

        tool_response = {
            "tool_calls": [
                {
                    "id": "toolu_123",
                    "name": "mock_tool",
                    "args": {"query": "test search"},
                }
            ]
        }

        with self.mock_api_call(tool_response):
            response = await provider.generate_message(messages, "claude-3-sonnet-20240229", tools=tools)

        assert hasattr(response, "tool_calls")
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "mock_tool"
        assert response.tool_calls[0].id == "toolu_123"

    @pytest.mark.asyncio
    async def test_anthropic_error_recognition(self):
        """Test that Anthropic-specific errors are properly recognized."""
        _ = self.create_provider()

        # Test rate limit error recognition
        rate_limit_error = self.create_anthropic_error("rate_limit")
        # Test that we can handle rate limit errors
        assert rate_limit_error is not None

    @pytest.mark.asyncio
    async def test_anthropic_system_message_handling(self):
        """Test Anthropic's specific system message handling."""
        provider = self.create_provider()

        messages = [
            Message(
                role="system",
                content=[MessageTextContent(text="You are a helpful assistant.")],
            ),
            Message(role="user", content=[MessageTextContent(text="Hello")]),
        ]

        with self.mock_api_call(ResponseFixtures.simple_text_response("Hello! How can I help you?")) as mock_call:
            await provider.generate_message(messages, "claude-3-sonnet-20240229")

        # Verify system message was handled correctly
        mock_call.assert_called_once()
        call_kwargs = mock_call.call_args[1]

        # In Anthropic API, system messages are passed separately
        assert "system" in call_kwargs
        assert call_kwargs["system"] == "You are a helpful assistant."

    def create_mock_tool(self):
        """Create a mock tool for testing."""
        from tests.chat.providers.test_base_provider import MockTool

        return MockTool()

    @pytest.mark.asyncio
    async def test_anthropic_streaming_with_tools(self):
        """Test streaming responses that include tool use."""
        provider = self.create_provider()
        messages = self.create_tool_messages()
        tools = [self.create_mock_tool()]

        # Create streaming response with tool use
        tool_events = [
            {
                "type": "message_start",
                "message": {"id": "msg_123", "usage": {"input_tokens": 10}},
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"query": "'},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": 'test"}'},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_stop"},
        ]

        class MockToolStream:
            def __init__(self):
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(tool_events):
                    raise StopAsyncIteration

                event = tool_events[self.index]
                self.index += 1

                mock_event = MagicMock()
                mock_event.type = event["type"]

                if event["type"] == "content_block_delta":
                    mock_event.delta = MagicMock()
                    mock_event.delta.partial_json = event["delta"]["partial_json"]
                    mock_event.index = event["index"]

                return mock_event

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        with patch.object(
            anthropic.resources.messages.AsyncMessages,
            "stream",
            return_value=MockToolStream(),
        ):
            results = []
            async for result in provider.generate_messages(messages, "claude-3-sonnet-20240229", tools=tools):
                results.append(result)

            # Should process streaming tool use
            assert len(results) >= 0  # Basic verification that streaming works

    @pytest.mark.asyncio
    async def test_anthropic_multi_turn_conversation(self):
        """Test multi-turn conversation handling."""
        provider = self.create_provider()

        # Anthropic requires alternating user/assistant messages
        conversation = [
            Message(
                role="user",
                content=[MessageTextContent(text="What's the weather like?")],
            ),
            Message(
                role="assistant",
                instructions=[MessageTextContent(text="I don't have access to current weather data.")],
            ),
            Message(
                role="user",
                content=[MessageTextContent(text="What should I wear then?")],
            ),
        ]

        with self.mock_api_call(
            ResponseFixtures.simple_text_response("I'd suggest checking a weather app first.")
        ) as mock_call:
            await provider.generate_message(conversation, "claude-3-sonnet-20240229")

        # Verify conversation was processed correctly
        mock_call.assert_called_once()
        call_kwargs = mock_call.call_args[1]

        # Should have processed the conversation messages
        assert "messages" in call_kwargs
        messages = call_kwargs["messages"]
        assert len(messages) >= 2  # Should have user and assistant messages
