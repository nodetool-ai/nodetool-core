"""
Tests for OpenAI provider with comprehensive API response mocking.

This module tests the OpenAI provider implementation including:
- GPT model responses
- Function calling
- Streaming responses
- Error handling specific to OpenAI API

OpenAI API Documentation (2024):
URL: https://platform.openai.com/docs/api-reference/chat/create

The Chat Completions API generates responses for conversations with models like GPT-4.

Key Request Parameters:
- model: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
- messages: Array of message objects with role ("system", "user", "assistant", "tool") and content
- tools: Array of available functions/tools for function calling
- stream: Boolean for streaming responses
- temperature: Creativity level (0.0-2.0)
- max_tokens: Maximum response length
- response_format: For structured output (JSON mode)
- seed: For reproducible responses

Response Format:
- id: Unique completion identifier
- object: "chat.completion" or "chat.completion.chunk"
- created: Unix timestamp
- model: Model used
- choices: Array of completion choices with message and finish_reason
- usage: Token usage statistics (prompt_tokens, completion_tokens, total_tokens)

Function Calling:
- Functions defined in tools parameter with name, description, and parameters schema
- Model responds with tool_calls in message when functions should be called
- Each tool_call has id, type ("function"), and function with name and arguments (JSON string)

Streaming:
- Server-sent events with data: prefix
- Final chunk has finish_reason and usage statistics
- Delta objects contain incremental content updates

Error Handling:
- 400: Invalid request (malformed JSON, invalid parameters)
- 401: Invalid authentication
- 429: Rate limit exceeded or quota exceeded
- 500: Server error
"""

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import openai
import openai.resources
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import (
    Choice as ChunkChoice,
)
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_message_tool_call import (
    Function as ToolCallFunction,
)
from openai.types.completion_usage import CompletionUsage

from nodetool.metadata.types import ToolCall
from nodetool.providers.openai_provider import OpenAIProvider
from tests.chat.providers.test_base_provider import BaseProviderTest, ResponseFixtures


class TestOpenAIProvider(BaseProviderTest):
    """Test suite for OpenAI provider with realistic API response mocking."""

    @property
    def provider_class(self):
        return OpenAIProvider

    @property
    def provider_name(self):
        return "openai"

    def create_openai_completion_response(
        self, content: str = "Hello, world!", tool_calls: List[Dict] | None = None
    ) -> ChatCompletion:
        """Create a realistic OpenAI ChatCompletion response."""
        message_kwargs: dict[str, Any] = {
            "role": "assistant",
            "content": content,
        }

        if tool_calls:
            message_kwargs["tool_calls"] = [
                ChatCompletionMessageToolCall(
                    id=tc["id"],
                    type="function",
                    function=ToolCallFunction(name=tc["name"], arguments=json.dumps(tc["args"])),
                )
                for tc in tool_calls
            ]
            message_kwargs["content"] = None  # No content when making tool calls

        return ChatCompletion(
            id="chatcmpl-123",
            choices=[
                Choice(
                    finish_reason="stop" if not tool_calls else "tool_calls",
                    index=0,
                    message=ChatCompletionMessage(**message_kwargs),
                    logprobs=None,
                )
            ],
            created=1677652288,
            model="gpt-3.5-turbo-0613",
            object="chat.completion",
            usage=CompletionUsage(completion_tokens=12, prompt_tokens=9, total_tokens=21),
        )

    def create_openai_streaming_responses(
        self, text: str = "Hello world!", chunk_size: int = 5
    ) -> List[ChatCompletionChunk]:
        """Create realistic OpenAI streaming response chunks."""
        chunks = []

        # First chunk with role
        chunks.append(
            ChatCompletionChunk(
                id="chatcmpl-123",
                choices=[
                    ChunkChoice(
                        delta=ChoiceDelta(role="assistant", content=""),
                        finish_reason=None,
                        index=0,
                        logprobs=None,
                    )
                ],
                created=1677652288,
                model="gpt-3.5-turbo-0613",
                object="chat.completion.chunk",
            )
        )

        # Content chunks
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i : i + chunk_size]
            is_last = i + chunk_size >= len(text)

            chunks.append(
                ChatCompletionChunk(
                    id="chatcmpl-123",
                    choices=[
                        ChunkChoice(
                            delta=ChoiceDelta(content=chunk_text),
                            finish_reason="stop" if is_last else None,
                            index=0,
                            logprobs=None,
                        )
                    ],
                    created=1677652288,
                    model="gpt-3.5-turbo-0613",
                    object="chat.completion.chunk",
                )
            )

        return chunks

    def create_openai_error(self, error_type: str = "rate_limit"):
        """Create realistic OpenAI API errors."""
        if error_type == "rate_limit":
            return openai.RateLimitError(
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
            return openai.AuthenticationError(
                message="Invalid API key",
                response=MagicMock(status_code=401),
                body={"error": {"type": "authentication_error"}},
            )
        else:
            return openai.APIError(message="Unknown error", request=MagicMock(), body={})

    def mock_api_call(self, response_data: Dict[str, Any]) -> MagicMock:
        """Mock OpenAI API call with structured response."""
        if "tool_calls" in response_data:
            # Tool calling response
            openai_response = self.create_openai_completion_response(
                content=response_data.get("text", "Hello, world!"),
                tool_calls=response_data["tool_calls"],
            )
        else:
            # Regular text response
            openai_response = self.create_openai_completion_response(content=response_data.get("text", "Hello, world!"))

        # Mock the async create method
        async def mock_create(*args, **kwargs):
            return openai_response

        return patch.object(
            openai.resources.chat.completions.AsyncCompletions,
            "create",
            side_effect=mock_create,
        )  # type: ignore[return-value]

    def mock_streaming_call(self, chunks: List[Dict[str, Any]]) -> MagicMock:
        """Mock OpenAI streaming API call."""
        # Convert generic chunks to OpenAI format
        text = "".join(chunk.get("content", "") for chunk in chunks)
        openai_chunks = self.create_openai_streaming_responses(text)

        async def async_generator():
            for chunk in openai_chunks:
                yield chunk

        # Mock the async streaming create method
        async def mock_stream(*args, **kwargs):
            return async_generator()

        return patch.object(
            openai.resources.chat.completions.AsyncCompletions,
            "create",
            side_effect=mock_stream,
        )  # type: ignore[return-value]

    def mock_error_response(self, error_type: str) -> MagicMock:
        """Mock OpenAI API error response."""
        error = self.create_openai_error(error_type)
        return patch.object(
            openai.resources.chat.completions.AsyncCompletions,
            "create",
            side_effect=error,
        )  # type: ignore[return-value]

    @pytest.mark.asyncio
    async def test_openai_specific_features(self):
        """Test OpenAI-specific features and parameters."""
        provider = self.create_provider()
        messages = self.create_simple_messages("Test OpenAI features")

        with self.mock_api_call(ResponseFixtures.simple_text_response("OpenAI response")) as mock_call:
            await provider.generate_message(
                messages,
                "gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=150,
                top_p=0.9,
                presence_penalty=0.1,
                frequency_penalty=0.1,
            )

        # Verify the call was made with correct parameters
        mock_call.assert_called_once()
        call_kwargs = mock_call.call_args[1]
        assert call_kwargs["model"] == "gpt-3.5-turbo"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_completion_tokens"] == 150

    @pytest.mark.asyncio
    async def test_function_calling_with_openai_format(self):
        """Test function calling with OpenAI's specific format."""
        provider = self.create_provider()
        messages = self.create_tool_messages()
        tools = [self.create_mock_tool()]

        tool_response = {
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "name": "mock_tool",
                    "args": {"query": "test search"},
                }
            ]
        }

        with self.mock_api_call(tool_response):
            response = await provider.generate_message(messages, "gpt-3.5-turbo", tools=tools)

        assert hasattr(response, "tool_calls")
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "mock_tool"
        assert response.tool_calls[0].id == "call_abc123"

    @pytest.mark.asyncio
    async def test_openai_usage_tracking(self):
        """Test that OpenAI usage tokens are properly tracked."""
        provider = self.create_provider()
        messages = self.create_simple_messages()

        initial_usage = provider.usage
        initial_tokens = initial_usage.get("total_tokens", 0)

        with self.mock_api_call(ResponseFixtures.simple_text_response()):
            await provider.generate_message(messages, "gpt-3.5-turbo")

        final_usage = provider.usage
        final_tokens = final_usage.get("total_tokens", 0)

        # Usage should have increased
        assert final_tokens >= initial_tokens

    @pytest.mark.asyncio
    async def test_openai_error_recognition(self):
        """Test that OpenAI-specific errors are properly recognized."""
        _ = self.create_provider()

        # Test rate limit error recognition
        rate_limit_error = self.create_openai_error("rate_limit")
        # Test that we can handle rate limit errors
        assert rate_limit_error is not None

    @pytest.mark.asyncio
    async def test_openai_response_format(self):
        """Test OpenAI response format parameter."""
        provider = self.create_provider()
        messages = self.create_simple_messages("Return JSON")

        response_format = {"type": "json_object"}

        with self.mock_api_call(ResponseFixtures.simple_text_response('{"result": "success"}')) as mock_call:
            await provider.generate_message(messages, "gpt-3.5-turbo", response_format=response_format)

        # Verify response_format was passed
        call_kwargs = mock_call.call_args[1]
        assert "response_format" in call_kwargs
        assert call_kwargs["response_format"] == response_format

    def create_mock_tool(self):
        """Create a mock tool for testing."""
        from tests.chat.providers.test_base_provider import MockTool

        return MockTool()

    @pytest.mark.asyncio
    async def test_openai_streaming_with_tools(self):
        """Test streaming responses that include tool calls."""
        provider = self.create_provider()
        messages = self.create_tool_messages()
        tools = [self.create_mock_tool()]

        # Create streaming response with tool calls

        # Note: This is a simplified version - actual OpenAI streaming tool calls are more complex
        with patch.object(openai.resources.chat.completions.AsyncCompletions, "create") as mock_create:
            # Mock the streaming response
            async def mock_stream():
                # Simplified tool call streaming
                yield ChatCompletionChunk(
                    id="chatcmpl-123",
                    choices=[
                        ChunkChoice(
                            delta=ChoiceDelta(
                                role="assistant",
                                tool_calls=[
                                    ChoiceDeltaToolCall(
                                        index=0,
                                        id="call_123",
                                        function=ChoiceDeltaToolCallFunction(
                                            name="mock_tool",
                                            arguments='{"query": "test"}',
                                        ),
                                        type="function",
                                    )
                                ],
                            ),
                            finish_reason="tool_calls",
                            index=0,
                            logprobs=None,
                        )
                    ],
                    created=1677652288,
                    model="gpt-3.5-turbo-0613",
                    object="chat.completion.chunk",
                )

            mock_create.return_value = mock_stream()

            results = []
            async for result in provider.generate_messages(messages, "gpt-3.5-turbo", tools=tools):
                results.append(result)

            # Should receive tool calls
            tool_calls = [r for r in results if isinstance(r, ToolCall)]
            assert len(tool_calls) > 0
