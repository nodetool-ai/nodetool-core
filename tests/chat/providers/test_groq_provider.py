"""
Tests for Groq provider.

Groq provides ultra-fast LLM inference through an OpenAI-compatible API.
This test suite verifies that the Groq provider correctly:
- Initializes with the correct base URL
- Supports streaming and non-streaming completions
- Handles function calling
"""

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

from nodetool.providers.groq_provider import GroqProvider


class TestGroqProvider:
    """Test suite for Groq provider."""

    def test_initialization(self):
        """Test that Groq provider initializes with correct configuration."""
        provider = GroqProvider(secrets={"GROQ_API_KEY": "test-key"})
        assert provider.api_key == "test-key"
        assert provider.provider.value == "groq"

    def test_get_client_configuration(self):
        """Test that Groq client is configured with correct base URL."""
        provider = GroqProvider(secrets={"GROQ_API_KEY": "test-key"})

        import httpx

        with patch("nodetool.runtime.resources.require_scope") as mock_scope:
            # Create a real httpx.AsyncClient instead of a mock
            mock_http_client = httpx.AsyncClient()
            mock_scope.return_value.get_http_client.return_value = mock_http_client

            client = provider.get_client()

            # Verify base URL
            assert str(client.base_url) == "https://api.groq.com/openai/v1/"

    def test_tool_support(self):
        """Test tool/function calling support detection."""
        provider = GroqProvider(secrets={"GROQ_API_KEY": "test-key"})

        # Groq models support tools
        assert provider.has_tool_support("llama-3.1-8b-instant")
        assert provider.has_tool_support("llama-3.3-70b-versatile")

    def test_required_secrets(self):
        """Test that Groq provider requires the correct API key."""
        required = GroqProvider.required_secrets()
        assert required == ["GROQ_API_KEY"]

    def test_container_env(self):
        """Test that container environment variables are correctly set."""
        provider = GroqProvider(secrets={"GROQ_API_KEY": "test-key"})

        mock_context = MagicMock()

        env = provider.get_container_env(mock_context)
        assert "GROQ_API_KEY" in env
        assert env["GROQ_API_KEY"] == "test-key"

    @pytest.mark.asyncio
    async def test_generate_message(self):
        """Test non-streaming message generation."""
        provider = GroqProvider(secrets={"GROQ_API_KEY": "test-key"})

        # Create mock response
        mock_response = ChatCompletion(
            id="chatcmpl-groq-123",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="Test response"),
                    logprobs=None,
                )
            ],
            created=1677652288,
            model="llama-3.1-8b-instant",
            object="chat.completion",
            usage=CompletionUsage(completion_tokens=12, prompt_tokens=9, total_tokens=21),
        )

        with patch.object(provider, "get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_client.chat.completions.create = MagicMock(return_value=mock_response)

            from nodetool.metadata.types import Message

            messages = [Message(role="user", content="Test message")]

            result = await provider.generate_message(
                messages=messages,
                model="llama-3.1-8b-instant",
                max_tokens=100,
            )

            assert result.role == "assistant"
            assert result.content == "Test response"

    @pytest.mark.asyncio
    async def test_get_available_language_models(self):
        """Test that Groq can fetch available language models."""
        provider = GroqProvider(secrets={"GROQ_API_KEY": "test-key"})

        # Mock the aiohttp response
        mock_models_data = {
            "data": [
                {
                    "id": "llama-3.1-8b-instant",
                    "name": "Llama 3.1 8B Instant",
                },
                {
                    "id": "llama-3.3-70b-versatile",
                    "name": "Llama 3.3 70B Versatile",
                },
            ]
        }

        # Create mock response object
        class MockResponse:
            status = 200

            async def json(self):
                return mock_models_data

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

        # Create mock session
        class MockSession:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            def get(self, *args, **kwargs):
                return MockResponse()

        with patch("aiohttp.ClientSession", MockSession):
            models = await provider.get_available_language_models()

            assert len(models) == 2
            model_ids = [m.id for m in models]
            assert "llama-3.1-8b-instant" in model_ids
            assert "llama-3.3-70b-versatile" in model_ids
            # Verify provider is set correctly
            for model in models:
                assert model.provider.value == "groq"
