"""
Tests for Perplexity AI provider.

Perplexity provides access to specialized language models through an OpenAI-compatible API.
This test suite verifies that the Perplexity provider correctly:
- Initializes with the correct base URL
- Supports streaming and non-streaming completions
- Handles function calling
"""

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import openai
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import (
    Choice as ChunkChoice,
)
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_message_tool_call import (
    Function as ToolCallFunction,
)
from openai.types.completion_usage import CompletionUsage

from nodetool.providers.perplexity_provider import PerplexityProvider


class TestPerplexityProvider:
    """Test suite for Perplexity provider."""

    def test_initialization(self):
        """Test that Perplexity provider initializes with correct configuration."""
        provider = PerplexityProvider(secrets={"PERPLEXITY_API_KEY": "test-key"})
        assert provider.api_key == "test-key"
        assert provider.provider.value == "perplexity"

    def test_get_client_configuration(self):
        """Test that Perplexity client is configured with correct base URL."""
        provider = PerplexityProvider(secrets={"PERPLEXITY_API_KEY": "test-key"})

        import httpx

        with patch("nodetool.runtime.resources.require_scope") as mock_scope:
            # Create a real httpx.AsyncClient instead of a mock
            mock_http_client = httpx.AsyncClient()
            mock_scope.return_value.get_http_client.return_value = mock_http_client

            client = provider.get_client()

            # Verify base URL (may or may not have trailing slash)
            assert "api.perplexity.ai" in str(client.base_url)

    def test_tool_support(self):
        """Test tool/function calling support detection."""
        provider = PerplexityProvider(secrets={"PERPLEXITY_API_KEY": "test-key"})

        # Most models support tools
        assert provider.has_tool_support("sonar")
        assert provider.has_tool_support("sonar-pro")
        assert provider.has_tool_support("llama-3.1-sonar-large-128k-online")

    def test_required_secrets(self):
        """Test that Perplexity provider requires the correct API key."""
        required = PerplexityProvider.required_secrets()
        assert required == ["PERPLEXITY_API_KEY"]

    def test_container_env(self):
        """Test that container environment variables are correctly set."""
        provider = PerplexityProvider(secrets={"PERPLEXITY_API_KEY": "test-key"})

        mock_context = MagicMock()

        env = provider.get_container_env(mock_context)
        assert "PERPLEXITY_API_KEY" in env
        assert env["PERPLEXITY_API_KEY"] == "test-key"

    @pytest.mark.asyncio
    async def test_generate_message(self):
        """Test non-streaming message generation."""
        provider = PerplexityProvider(secrets={"PERPLEXITY_API_KEY": "test-key"})

        # Create mock response
        mock_response = ChatCompletion(
            id="chatcmpl-perplexity-123",
            created=1234567890,
            model="sonar",
            object="chat.completion",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        content="This is a test response from Perplexity AI.",
                        role="assistant",
                    ),
                )
            ],
            usage=CompletionUsage(
                completion_tokens=10,
                prompt_tokens=5,
                total_tokens=15,
            ),
        )

        with patch.object(provider, "get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create = MagicMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            from nodetool.metadata.types import Message

            messages = [Message(role="user", content="Hello")]

            result = await provider.generate_message(
                messages=messages, model="sonar", max_tokens=100
            )

            assert result.role == "assistant"
            assert result.content == "This is a test response from Perplexity AI."

    @pytest.mark.asyncio
    async def test_generate_message_with_tools(self):
        """Test non-streaming message generation with function calling.

        Note: Tool calling functionality is inherited from OpenAIProvider
        and tested there. This test verifies basic integration.
        """
        provider = PerplexityProvider(secrets={"PERPLEXITY_API_KEY": "test-key"})

        # Create mock response with tool call
        mock_response = ChatCompletion(
            id="chatcmpl-perplexity-tool-123",
            created=1234567890,
            model="sonar",
            object="chat.completion",
            choices=[
                Choice(
                    finish_reason="tool_calls",
                    index=0,
                    message=ChatCompletionMessage(
                        content=None,
                        role="assistant",
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                id="call_123",
                                function=ToolCallFunction(
                                    name="get_weather",
                                    arguments='{"location": "San Francisco"}',
                                ),
                                type="function",
                            )
                        ],
                    ),
                )
            ],
            usage=CompletionUsage(
                completion_tokens=10,
                prompt_tokens=5,
                total_tokens=15,
            ),
        )

        with patch.object(provider, "get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create = MagicMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            from nodetool.metadata.types import Message

            messages = [Message(role="user", content="What's the weather in SF?")]

            # Call without tools to test basic response processing
            result = await provider.generate_message(
                messages=messages, model="sonar", max_tokens=100
            )

            assert result.role == "assistant"
            assert result.tool_calls is not None
            assert len(result.tool_calls) == 1
            assert result.tool_calls[0].name == "get_weather"
            assert result.tool_calls[0].args["location"] == "San Francisco"

    @pytest.mark.asyncio
    async def test_streaming_generation(self):
        """Test streaming message generation."""
        provider = PerplexityProvider(secrets={"PERPLEXITY_API_KEY": "test-key"})

        # Create mock streaming response
        async def mock_stream():
            yield ChatCompletionChunk(
                id="chatcmpl-perplexity-stream-123",
                created=1234567890,
                model="sonar",
                object="chat.completion.chunk",
                choices=[
                    ChunkChoice(
                        delta=ChoiceDelta(content="Hello", role="assistant"),
                        finish_reason=None,
                        index=0,
                    )
                ],
            )
            yield ChatCompletionChunk(
                id="chatcmpl-perplexity-stream-123",
                created=1234567890,
                model="sonar",
                object="chat.completion.chunk",
                choices=[
                    ChunkChoice(
                        delta=ChoiceDelta(content=" world", role="assistant"),
                        finish_reason=None,
                        index=0,
                    )
                ],
            )
            yield ChatCompletionChunk(
                id="chatcmpl-perplexity-stream-123",
                created=1234567890,
                model="sonar",
                object="chat.completion.chunk",
                choices=[
                    ChunkChoice(
                        delta=ChoiceDelta(content="", role="assistant"),
                        finish_reason="stop",
                        index=0,
                    )
                ],
            )

        with patch.object(provider, "get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create = MagicMock(return_value=mock_stream())
            mock_get_client.return_value = mock_client

            from nodetool.metadata.types import Message
            from nodetool.workflows.types import Chunk

            messages = [Message(role="user", content="Say hello")]

            chunks = []
            async for item in provider.generate_messages(
                messages=messages, model="sonar", max_tokens=100
            ):
                if isinstance(item, Chunk):
                    chunks.append(item.content)

            # We should get the text chunks
            assert "Hello" in "".join(chunks)
            assert "world" in "".join(chunks)
