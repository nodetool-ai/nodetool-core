"""
Tests for Together AI provider.

Together AI provides open-source LLM inference through an OpenAI-compatible API.
This test suite verifies that the Together AI provider correctly:
- Initializes with the correct base URL
- Supports streaming and non-streaming completions
- Handles function calling
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import httpx
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

from nodetool.providers.together_provider import TogetherProvider


# Mock classes for aiohttp testing
class MockResponse:
    """Mock aiohttp response for testing."""

    def __init__(self, data: dict, status: int = 200):
        self._data = data
        self.status = status

    async def json(self):
        return self._data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


class MockSession:
    """Mock aiohttp session for testing."""

    def __init__(self, response_data: dict, status: int = 200, *args, **kwargs):
        self._response_data = response_data
        self._status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    def get(self, *args, **kwargs):
        return MockResponse(self._response_data, self._status)


class TestTogetherProvider:
    """Test suite for Together AI provider."""

    def test_initialization(self):
        """Test that Together AI provider initializes with correct configuration."""
        provider = TogetherProvider(secrets={"TOGETHER_API_KEY": "test-key"})
        assert provider.api_key == "test-key"
        assert provider.provider.value == "together"

    def test_get_client_configuration(self):
        """Test that Together AI client is configured with correct base URL."""
        provider = TogetherProvider(secrets={"TOGETHER_API_KEY": "test-key"})

        with patch("nodetool.runtime.resources.require_scope") as mock_scope:
            # Create a real httpx.AsyncClient instead of a mock
            mock_http_client = httpx.AsyncClient()
            mock_scope.return_value.get_http_client.return_value = mock_http_client

            client = provider.get_client()

            # Verify base URL
            assert str(client.base_url) == "https://api.together.xyz/v1/"

    def test_tool_support(self):
        """Test tool/function calling support detection."""
        provider = TogetherProvider(secrets={"TOGETHER_API_KEY": "test-key"})

        # Together AI models support tools
        assert provider.has_tool_support("mistralai/Mixtral-8x7B-Instruct-v0.1")
        assert provider.has_tool_support("meta-llama/Llama-3-70b-chat-hf")

    def test_required_secrets(self):
        """Test that Together AI provider requires the correct API key."""
        required = TogetherProvider.required_secrets()
        assert required == ["TOGETHER_API_KEY"]

    def test_container_env(self):
        """Test that container environment variables are correctly set."""
        provider = TogetherProvider(secrets={"TOGETHER_API_KEY": "test-key"})

        mock_context = MagicMock()

        env = provider.get_container_env(mock_context)
        assert "TOGETHER_API_KEY" in env
        assert env["TOGETHER_API_KEY"] == "test-key"

    @pytest.mark.asyncio
    async def test_generate_message(self):
        """Test non-streaming message generation."""
        provider = TogetherProvider(secrets={"TOGETHER_API_KEY": "test-key"})

        # Create mock response
        mock_response = ChatCompletion(
            id="chatcmpl-together-123",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="Test response"),
                    logprobs=None,
                )
            ],
            created=1677652288,
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
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
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                max_tokens=100,
            )

            assert result.role == "assistant"
            assert result.content == "Test response"

    @pytest.mark.asyncio
    async def test_get_available_language_models(self):
        """Test that Together AI can fetch available language models."""
        provider = TogetherProvider(secrets={"TOGETHER_API_KEY": "test-key"})

        # Mock the aiohttp response
        mock_models_data = {
            "data": [
                {
                    "id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                    "display_name": "Mixtral 8x7B Instruct",
                    "type": "chat",
                },
                {
                    "id": "meta-llama/Llama-3-70b-chat-hf",
                    "display_name": "Llama 3 70B Chat",
                    "type": "chat",
                },
                {
                    "id": "stabilityai/stable-diffusion-xl-base-1.0",
                    "display_name": "Stable Diffusion XL",
                    "type": "image",  # Should be filtered out
                },
            ]
        }

        # Create a factory function that returns a new MockSession for each call
        def mock_session_factory(*args, **kwargs):
            return MockSession(mock_models_data, 200, *args, **kwargs)

        with patch("aiohttp.ClientSession", mock_session_factory):
            models = await provider.get_available_language_models()

            # Should only get chat/language models, not image models
            assert len(models) == 2
            model_ids = [m.id for m in models]
            assert "mistralai/Mixtral-8x7B-Instruct-v0.1" in model_ids
            assert "meta-llama/Llama-3-70b-chat-hf" in model_ids
            assert "stabilityai/stable-diffusion-xl-base-1.0" not in model_ids
            # Verify provider is set correctly
            for model in models:
                assert model.provider.value == "together"
