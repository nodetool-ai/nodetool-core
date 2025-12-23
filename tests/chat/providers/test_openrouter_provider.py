"""
Tests for OpenRouter provider.

OpenRouter provides access to multiple AI models through an OpenAI-compatible API.
This test suite verifies that the OpenRouter provider correctly:
- Initializes with the correct base URL and headers
- Supports streaming and non-streaming completions
- Handles function calling
- Correctly maps model context lengths
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

from nodetool.providers.openrouter_provider import OpenRouterProvider


class TestOpenRouterProvider:
    """Test suite for OpenRouter provider."""

    def test_initialization(self):
        """Test that OpenRouter provider initializes with correct configuration."""
        provider = OpenRouterProvider(secrets={"OPENROUTER_API_KEY": "test-key"})
        assert provider.api_key == "test-key"
        assert provider.provider.value == "openrouter"

    def test_get_client_configuration(self):
        """Test that OpenRouter client is configured with correct base URL and headers."""
        provider = OpenRouterProvider(secrets={"OPENROUTER_API_KEY": "test-key"})

        import httpx

        with patch("nodetool.runtime.resources.require_scope") as mock_scope:
            # Create a real httpx.AsyncClient instead of a mock
            mock_http_client = httpx.AsyncClient()
            mock_scope.return_value.get_http_client.return_value = mock_http_client

            client = provider.get_client()

            # Verify base URL
            assert str(client.base_url) == "https://openrouter.ai/api/v1/"

            # Verify OpenRouter-specific headers
            assert "HTTP-Referer" in client.default_headers
            assert "X-Title" in client.default_headers

    def test_context_length_openai_models(self):
        """Test context length detection for OpenAI models via OpenRouter."""
        provider = OpenRouterProvider(secrets={"OPENROUTER_API_KEY": "test-key"})

        # Test OpenAI models with provider prefix
        assert provider.get_context_length("openai/gpt-4o") == 128000
        assert provider.get_context_length("openai/gpt-4-turbo") == 128000
        assert provider.get_context_length("openai/gpt-4") == 8192
        assert provider.get_context_length("openai/gpt-4-32k") == 32768
        assert provider.get_context_length("openai/gpt-3.5-turbo") == 4096
        assert provider.get_context_length("openai/gpt-3.5-turbo-16k") == 16384

    def test_context_length_anthropic_models(self):
        """Test context length detection for Anthropic models via OpenRouter."""
        provider = OpenRouterProvider(secrets={"OPENROUTER_API_KEY": "test-key"})

        # Test Anthropic models
        assert provider.get_context_length("anthropic/claude-3-opus") == 200000
        assert provider.get_context_length("anthropic/claude-3-sonnet") == 200000
        assert provider.get_context_length("anthropic/claude-2") == 200000

    def test_context_length_other_models(self):
        """Test context length detection for other models via OpenRouter."""
        provider = OpenRouterProvider(secrets={"OPENROUTER_API_KEY": "test-key"})

        # Test Google Gemini
        assert provider.get_context_length("google/gemini-1.5-pro") == 1000000
        assert provider.get_context_length("google/gemini-pro") == 32768

        # Test Meta Llama
        assert provider.get_context_length("meta-llama/llama-3.1-70b") == 128000
        assert provider.get_context_length("meta-llama/llama-2-70b") == 8192

        # Test Mistral
        assert provider.get_context_length("mistralai/mistral-7b") == 32768
        assert provider.get_context_length("mistralai/mixtral-8x7b") == 32768

        # Test unknown model (fallback)
        assert provider.get_context_length("unknown/model") == 8192

    def test_tool_support(self):
        """Test tool/function calling support detection."""
        provider = OpenRouterProvider(secrets={"OPENROUTER_API_KEY": "test-key"})

        # Most models support tools
        assert provider.has_tool_support("openai/gpt-4")
        assert provider.has_tool_support("anthropic/claude-3-opus")
        assert provider.has_tool_support("google/gemini-pro")

        # O1/O3 models don't support tools
        assert not provider.has_tool_support("openai/o1-preview")
        assert not provider.has_tool_support("openai/o3-mini")

    def test_required_secrets(self):
        """Test that OpenRouter provider requires the correct API key."""
        required = OpenRouterProvider.required_secrets()
        assert required == ["OPENROUTER_API_KEY"]

    def test_container_env(self):
        """Test that container environment variables are correctly set."""
        provider = OpenRouterProvider(secrets={"OPENROUTER_API_KEY": "test-key"})

        mock_context = MagicMock()

        env = provider.get_container_env(mock_context)
        assert "OPENROUTER_API_KEY" in env
        assert env["OPENROUTER_API_KEY"] == "test-key"

    @pytest.mark.asyncio
    async def test_generate_message(self):
        """Test non-streaming message generation."""
        provider = OpenRouterProvider(secrets={"OPENROUTER_API_KEY": "test-key"})

        # Create mock response
        mock_response = ChatCompletion(
            id="chatcmpl-openrouter-123",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="Test response"),
                    logprobs=None,
                )
            ],
            created=1677652288,
            model="openai/gpt-4",
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
                model="openai/gpt-4",
                max_tokens=100,
            )

            assert result.role == "assistant"
            assert result.content == "Test response"

    @pytest.mark.asyncio
    async def test_cost_tracking(self):
        """Test that OpenRouter cost tracking works correctly."""
        provider = OpenRouterProvider(secrets={"OPENROUTER_API_KEY": "test-key"})

        # Create a CompletionUsage with additional cost attribute (via model_extra)
        # OpenRouter returns cost in the usage object
        mock_usage = CompletionUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        # Add cost attribute directly to the object (OpenRouter-specific)
        mock_usage.cost = 0.0025  # type: ignore

        # Create mock response with cost
        mock_response = ChatCompletion(
            id="chatcmpl-openrouter-123",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="Test response with cost tracking"),
                    logprobs=None,
                )
            ],
            created=1677652288,
            model="anthropic/claude-3-opus",
            object="chat.completion",
            usage=mock_usage,
        )

        with patch.object(provider, "get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_client.chat.completions.create = MagicMock(return_value=mock_response)

            from nodetool.metadata.types import Message

            messages = [Message(role="user", content="Test message")]

            # Reset provider cost tracking
            provider.cost = 0.0

            result = await provider.generate_message(
                messages=messages,
                model="anthropic/claude-3-opus",
                max_tokens=100,
            )

            # Verify response
            assert result.role == "assistant"
            assert result.content == "Test response with cost tracking"

            # Verify cost tracking
            assert provider.cost == 0.0025
            assert provider.usage["prompt_tokens"] == 100
            assert provider.usage["completion_tokens"] == 50
            assert provider.usage["total_tokens"] == 150

    @pytest.mark.asyncio
    async def test_text_to_image(self):
        """Test that OpenRouter text-to-image generation works correctly."""
        provider = OpenRouterProvider(secrets={"OPENROUTER_API_KEY": "test-key"})

        # Create a mock image response
        from openai.types import Image, ImagesResponse

        mock_image_data = b"fake_image_data"
        mock_b64 = "ZmFrZV9pbWFnZV9kYXRh"  # base64 of "fake_image_data"

        mock_image = Image(
            b64_json=mock_b64,
            url=None,
            revised_prompt=None,
        )

        mock_response = ImagesResponse(
            created=1677652288,
            data=[mock_image],
        )

        with patch.object(provider, "get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_client.images.generate = MagicMock(return_value=mock_response)

            from nodetool.metadata.types import ImageModel, Provider
            from nodetool.providers.types import TextToImageParams

            params = TextToImageParams(
                model=ImageModel(
                    id="openai/dall-e-3",
                    name="DALL-E 3",
                    provider=Provider.OpenRouter,
                ),
                prompt="A test image",
                width=1024,
                height=1024,
            )

            result = await provider.text_to_image(params, timeout_s=120)

            # Verify response
            assert result == mock_image_data
            assert len(result) > 0

            # Verify the client was called with correct parameters
            mock_client.images.generate.assert_called_once()
            call_args = mock_client.images.generate.call_args[1]
            assert call_args["model"] == "openai/dall-e-3"
            assert call_args["prompt"] == "A test image"
            assert call_args["n"] == 1

    @pytest.mark.asyncio
    async def test_get_available_image_models(self):
        """Test that OpenRouter can fetch available image models."""
        provider = OpenRouterProvider(secrets={"OPENROUTER_API_KEY": "test-key"})

        # Mock the aiohttp response
        mock_models_data = {
            "data": [
                {
                    "id": "openai/dall-e-3",
                    "name": "DALL-E 3",
                    "architecture": {"modality": "text->image"},
                },
                {
                    "id": "stability-ai/stable-diffusion-xl",
                    "name": "Stable Diffusion XL",
                    "architecture": {},
                },
                {
                    "id": "black-forest-labs/flux-pro",
                    "name": "Flux Pro",
                    "architecture": {},
                },
                {
                    "id": "openai/gpt-4",  # Non-image model
                    "name": "GPT-4",
                    "architecture": {"modality": "text"},
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
            models = await provider.get_available_image_models()

            # Verify we got image models (DALL-E, Stable Diffusion, Flux)
            assert len(models) >= 3
            model_ids = [m.id for m in models]
            assert "openai/dall-e-3" in model_ids
            assert "stability-ai/stable-diffusion-xl" in model_ids
            assert "black-forest-labs/flux-pro" in model_ids
            # GPT-4 should not be in the list
            assert "openai/gpt-4" not in model_ids
