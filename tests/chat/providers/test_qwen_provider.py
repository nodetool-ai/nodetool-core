"""
Tests for Qwen (Alibaba Cloud) provider.

Qwen provides access to Qwen language models through an OpenAI-compatible API.
This test suite verifies that the Qwen provider correctly:
- Initializes with the correct base URL
- Supports streaming and non-streaming completions
- Handles function calling
- Supports vision/image-to-text through Qwen-VL models
"""

from unittest.mock import MagicMock, patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from nodetool.metadata.types import Provider
from nodetool.providers.base import ProviderCapability
from nodetool.providers.qwen_provider import QWEN_BASE_URL, QwenProvider


class TestQwenProvider:
    """Test suite for Qwen provider."""

    def test_initialization(self):
        """Test that Qwen provider initializes with correct configuration."""
        provider = QwenProvider(secrets={"DASHSCOPE_API_KEY": "test-key"})
        assert provider.api_key == "test-key"
        assert provider.provider.value == "qwen"

    def test_get_client_configuration(self):
        """Test that Qwen client is configured with correct base URL."""
        provider = QwenProvider(secrets={"DASHSCOPE_API_KEY": "test-key"})

        import httpx

        with patch("nodetool.runtime.resources.require_scope") as mock_scope:
            # Create a real httpx.AsyncClient instead of a mock
            mock_http_client = httpx.AsyncClient()
            mock_scope.return_value.get_http_client.return_value = mock_http_client

            client = provider.get_client()

            # Verify base URL
            assert str(client.base_url) == QWEN_BASE_URL + "/"

    def test_tool_support(self):
        """Test tool/function calling support detection."""
        provider = QwenProvider(secrets={"DASHSCOPE_API_KEY": "test-key"})

        # Qwen models support tools
        assert provider.has_tool_support("qwen-plus")
        assert provider.has_tool_support("qwen-max")
        assert provider.has_tool_support("qwen-turbo")

    def test_required_secrets(self):
        """Test that Qwen provider requires the correct API key."""
        required = QwenProvider.required_secrets()
        assert required == ["DASHSCOPE_API_KEY"]

    def test_container_env(self):
        """Test that container environment variables are correctly set."""
        provider = QwenProvider(secrets={"DASHSCOPE_API_KEY": "test-key"})

        mock_context = MagicMock()

        env = provider.get_container_env(mock_context)
        assert "DASHSCOPE_API_KEY" in env
        assert env["DASHSCOPE_API_KEY"] == "test-key"

    @pytest.mark.asyncio
    async def test_generate_message(self):
        """Test non-streaming message generation."""
        provider = QwenProvider(secrets={"DASHSCOPE_API_KEY": "test-key"})

        # Create mock response
        mock_response = ChatCompletion(
            id="chatcmpl-qwen-123",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="I am Qwen, developed by Alibaba Cloud."),
                    logprobs=None,
                )
            ],
            created=1677652288,
            model="qwen-plus",
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

            messages = [Message(role="user", content="Who are you?")]

            result = await provider.generate_message(
                messages=messages,
                model="qwen-plus",
                max_tokens=100,
            )

            assert result.role == "assistant"
            assert result.content == "I am Qwen, developed by Alibaba Cloud."

    @pytest.mark.asyncio
    async def test_get_available_language_models(self):
        """Test that Qwen returns available language models."""
        provider = QwenProvider(secrets={"DASHSCOPE_API_KEY": "test-key"})

        models = await provider.get_available_language_models()

        # Should return curated list of models
        assert len(models) > 0
        model_ids = [m.id for m in models]

        # Verify some key models are present
        assert "qwen-plus" in model_ids
        assert "qwen-max" in model_ids
        assert "qwen-turbo" in model_ids

        # Verify provider is set correctly
        for model in models:
            assert model.provider.value == "qwen"

    @pytest.mark.asyncio
    async def test_get_available_language_models_no_api_key(self):
        """Test that Qwen returns empty list when no API key is configured."""
        # Create provider with API key then set it to None for testing
        with patch.object(QwenProvider, "__init__", lambda self, secrets: None):
            provider = QwenProvider(secrets={})
            provider.api_key = None
            provider.client = None
            provider.cost = 0.0

            models = await provider.get_available_language_models()
            assert models == []

    def test_initialization_without_api_key_raises_error(self):
        """Test that initialization without API key raises an error."""
        with pytest.raises(AssertionError, match="DASHSCOPE_API_KEY is required"):
            QwenProvider(secrets={})


class TestQwenVisionSupport:
    """Test suite for Qwen vision/image-to-text support (Qwen-VL models)."""

    @pytest.mark.asyncio
    async def test_convert_message_with_image_content(self):
        """Test that image content is properly converted for Qwen-VL models."""
        from nodetool.metadata.types import (
            ImageRef,
            Message,
            MessageImageContent,
            MessageTextContent,
        )

        provider = QwenProvider(secrets={"DASHSCOPE_API_KEY": "test-key"})

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

        provider = QwenProvider(secrets={"DASHSCOPE_API_KEY": "test-key"})

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


class TestQwenCapabilities:
    """Test suite for Qwen provider capabilities."""

    def test_capabilities_detected(self):
        """Test that capabilities are correctly detected."""
        provider = QwenProvider(secrets={"DASHSCOPE_API_KEY": "test-key"})
        capabilities = provider.get_capabilities()

        # Should have message generation capability
        assert ProviderCapability.GENERATE_MESSAGE in capabilities
        assert ProviderCapability.GENERATE_MESSAGES in capabilities
