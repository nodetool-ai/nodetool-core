"""
Tests for Mistral AI provider.

Mistral AI provides access to Mistral language models through an OpenAI-compatible API.
This test suite verifies that the Mistral provider correctly:
- Initializes with the correct base URL
- Supports streaming and non-streaming completions
- Handles function calling
- Supports embedding generation
- Supports vision/image-to-text through Pixtral models
"""

from unittest.mock import MagicMock, patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from nodetool.metadata.types import EmbeddingModel, Provider
from nodetool.providers.base import ProviderCapability
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

        async def mock_create(*args, **kwargs):
            return mock_response

        with patch.object(provider, "get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_client.chat.completions.create = mock_create

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

    @pytest.mark.asyncio
    async def test_get_available_embedding_models(self):
        """Test that Mistral returns available embedding models."""
        provider = MistralProvider(secrets={"MISTRAL_API_KEY": "test-key"})
        models = await provider.get_available_embedding_models()

        assert len(models) == 1
        assert all(isinstance(m, EmbeddingModel) for m in models)
        assert all(m.provider == Provider.Mistral for m in models)

        model_ids = [m.id for m in models]
        assert "mistral-embed" in model_ids

        # Check dimensions
        mistral_embed = next(m for m in models if m.id == "mistral-embed")
        assert mistral_embed.dimensions == 1024

    @pytest.mark.asyncio
    async def test_get_available_embedding_models_no_api_key(self):
        """Test that Mistral returns empty list when no API key is configured."""
        # Create provider with API key then set it to None for testing
        with patch.object(MistralProvider, "__init__", lambda self, secrets: None):
            provider = MistralProvider(secrets={})
            provider.api_key = None
            provider.client = None
            provider.cost = 0.0

            models = await provider.get_available_embedding_models()
            assert models == []

    @pytest.mark.asyncio
    async def test_generate_embedding_single_text(self):
        """Test generating embedding for a single text."""
        provider = MistralProvider(secrets={"MISTRAL_API_KEY": "test-key"})

        # Mock the embeddings response with exactly 1024 dimensions (matches mistral-embed)
        mock_embedding = [0.1] * 1024

        class MockData:
            def __init__(self, embedding):
                self.embedding = embedding

        class MockResponse:
            def __init__(self, embeddings):
                self.data = [MockData(emb) for emb in embeddings]

        async def mock_create(**kwargs):
            return MockResponse([mock_embedding])

        with patch.object(provider, "get_client") as mock_client:
            mock_client.return_value.embeddings.create = mock_create

            result = await provider.generate_embedding(
                text="Hello, world!",
                model="mistral-embed",
            )

            assert len(result) == 1
            assert result[0] == mock_embedding

    @pytest.mark.asyncio
    async def test_generate_embedding_multiple_texts(self):
        """Test generating embeddings for multiple texts."""
        provider = MistralProvider(secrets={"MISTRAL_API_KEY": "test-key"})

        mock_embeddings = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]

        class MockData:
            def __init__(self, embedding):
                self.embedding = embedding

        class MockResponse:
            def __init__(self, embeddings):
                self.data = [MockData(emb) for emb in embeddings]

        async def mock_create(**kwargs):
            return MockResponse(mock_embeddings)

        with patch.object(provider, "get_client") as mock_client:
            mock_client.return_value.embeddings.create = mock_create

            result = await provider.generate_embedding(
                text=["Text 1", "Text 2", "Text 3"],
                model="mistral-embed",
            )

            assert len(result) == 3
            assert result == mock_embeddings

    @pytest.mark.asyncio
    async def test_generate_embedding_empty_text_raises_error(self):
        """Test that empty text raises ValueError."""
        provider = MistralProvider(secrets={"MISTRAL_API_KEY": "test-key"})

        with pytest.raises(ValueError, match="text must not be empty"):
            await provider.generate_embedding(text="", model="mistral-embed")

    def test_embedding_capability_detected(self):
        """Test that the GENERATE_EMBEDDING capability is detected."""
        provider = MistralProvider(secrets={"MISTRAL_API_KEY": "test-key"})
        capabilities = provider.get_capabilities()

        assert ProviderCapability.GENERATE_EMBEDDING in capabilities


class TestMistralVisionSupport:
    """Test suite for Mistral vision/image-to-text support (Pixtral models)."""

    @pytest.mark.asyncio
    async def test_convert_message_with_image_content(self):
        """Test that image content is properly converted for Pixtral models."""
        from nodetool.metadata.types import (
            ImageRef,
            Message,
            MessageImageContent,
            MessageTextContent,
        )

        provider = MistralProvider(secrets={"MISTRAL_API_KEY": "test-key"})

        # Create a message with text and image content
        # Using a minimal valid base64 1x1 pixel PNG
        import base64

        # 1x1 transparent PNG pixel
        pixel_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )

        message = Message(
            role="user",
            content=[
                MessageTextContent(text="What is in this image?"),
                MessageImageContent(
                    image=ImageRef(data=pixel_png)
                ),
            ],
        )

        result = await provider.convert_message(message)

        # Verify the message structure
        assert result["role"] == "user"
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 2

        # First content part should be text
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "What is in this image?"

        # Second content part should be image
        assert result["content"][1]["type"] == "image_url"
        assert "image_url" in result["content"][1]
        assert "url" in result["content"][1]["image_url"]

    @pytest.mark.asyncio
    async def test_convert_message_with_image_uri(self):
        """Test that image URLs are properly converted for vision models."""
        from nodetool.metadata.types import (
            ImageRef,
            Message,
            MessageImageContent,
            MessageTextContent,
        )

        provider = MistralProvider(secrets={"MISTRAL_API_KEY": "test-key"})

        message = Message(
            role="user",
            content=[
                MessageTextContent(text="Describe this image"),
                MessageImageContent(
                    image=ImageRef(uri="https://example.com/image.jpg")
                ),
            ],
        )

        # Mock the uri_to_base64 to avoid actual fetch
        with patch.object(provider, "uri_to_base64") as mock_uri:
            mock_uri.return_value = "data:image/jpeg;base64,/9j/4AAQ..."

            result = await provider.convert_message(message)

            # Verify image URL was converted
            assert result["role"] == "user"
            assert len(result["content"]) == 2
            assert result["content"][1]["type"] == "image_url"
            mock_uri.assert_called_once_with("https://example.com/image.jpg")
