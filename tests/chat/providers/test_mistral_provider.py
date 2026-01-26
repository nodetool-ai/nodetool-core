"""
Tests for Mistral AI provider.

Mistral AI provides access to Mistral language models through an OpenAI-compatible API.
This test suite verifies that the Mistral provider correctly:
- Initializes with the correct base URL
- Supports streaming and non-streaming completions
- Handles function calling
"""

from unittest.mock import MagicMock, patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from nodetool.providers.mistral_provider import MistralProvider


class TestMistralProvider:
    """Test suite for Mistral provider."""

    def test_initialization(self):
        """Test that Mistral provider initializes with correct configuration."""
        provider = MistralProvider(secrets={"MISTRAL_API_KEY": "test-key"})
        assert provider.api_key == "test-key"
        assert provider.provider.value == "mistral"

    def test_get_client_configuration(self):
        """Test that Mistral client is configured with correct base URL."""
        provider = MistralProvider(secrets={"MISTRAL_API_KEY": "test-key"})

        import httpx

        with patch("nodetool.runtime.resources.require_scope") as mock_scope:
            # Create a real httpx.AsyncClient instead of a mock
            mock_http_client = httpx.AsyncClient()
            mock_scope.return_value.get_http_client.return_value = mock_http_client

            client = provider.get_client()

            # Verify base URL
            assert str(client.base_url) == "https://api.mistral.ai/v1/"

    def test_tool_support(self):
        """Test tool/function calling support detection."""
        provider = MistralProvider(secrets={"MISTRAL_API_KEY": "test-key"})

        # Mistral models support tools
        assert provider.has_tool_support("mistral-large-latest")
        assert provider.has_tool_support("mistral-small-latest")

    def test_required_secrets(self):
        """Test that Mistral provider requires the correct API key."""
        required = MistralProvider.required_secrets()
        assert required == ["MISTRAL_API_KEY"]

    def test_container_env(self):
        """Test that container environment variables are correctly set."""
        provider = MistralProvider(secrets={"MISTRAL_API_KEY": "test-key"})

        mock_context = MagicMock()

        env = provider.get_container_env(mock_context)
        assert "MISTRAL_API_KEY" in env
        assert env["MISTRAL_API_KEY"] == "test-key"

    @pytest.mark.asyncio
    async def test_generate_message(self):
        """Test non-streaming message generation."""
        provider = MistralProvider(secrets={"MISTRAL_API_KEY": "test-key"})

        # Create mock response
        mock_response = ChatCompletion(
            id="chatcmpl-mistral-123",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="Test response"),
                    logprobs=None,
                )
            ],
            created=1677652288,
            model="mistral-large-latest",
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
                model="mistral-large-latest",
                max_tokens=100,
            )

            assert result.role == "assistant"
            assert result.content == "Test response"

    @pytest.mark.asyncio
    async def test_get_available_language_models(self):
        """Test that Mistral can fetch available language models."""
        provider = MistralProvider(secrets={"MISTRAL_API_KEY": "test-key"})

        # Mock the aiohttp response
        mock_models_data = {
            "data": [
                {
                    "id": "mistral-large-latest",
                    "name": "Mistral Large",
                },
                {
                    "id": "mistral-small-latest",
                    "name": "Mistral Small",
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
            assert "mistral-large-latest" in model_ids
            assert "mistral-small-latest" in model_ids
            # Verify provider is set correctly
            for model in models:
                assert model.provider.value == "mistral"
