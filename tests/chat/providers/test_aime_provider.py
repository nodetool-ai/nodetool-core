"""
Tests for AIME provider.

AIME provides access to AI models through an OpenAI-compatible API.
This test suite verifies that the AIME provider correctly:
- Initializes with the correct credentials
- Uses OpenAI-compatible API for streaming and non-streaming completions
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import (
    Choice as ChunkChoice,
)
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta,
)
from openai.types.completion_usage import CompletionUsage

from nodetool.metadata.types import Message, MessageTextContent
from nodetool.providers.aime_provider import AIMEProvider


class TestAIMEProvider:
    """Test suite for AIME provider."""

    def test_initialization(self):
        """Test that AIME provider initializes with correct configuration."""
        provider = AIMEProvider(secrets={"AIME_API_KEY": "test-key"})
        assert provider.api_key == "test-key"
        assert provider.provider.value == "aime"
        assert provider._base_url == "https://api.aime.info/v1"

    def test_required_secrets(self):
        """Test that AIME provider requires the correct credentials."""
        required = AIMEProvider.required_secrets()
        assert "AIME_API_KEY" in required

    def test_container_env(self):
        """Test that container environment variables are correctly set."""
        provider = AIMEProvider(secrets={"AIME_API_KEY": "test-key"})

        mock_context = MagicMock()

        env = provider.get_container_env(mock_context)
        assert "AIME_API_KEY" in env
        assert env["AIME_API_KEY"] == "test-key"

    def test_tool_support(self):
        """Test that AIME supports tools via OpenAI-compatible API."""
        provider = AIMEProvider(secrets={"AIME_API_KEY": "test-key"})

        # AIME with OpenAI-compatible API supports tools
        assert provider.has_tool_support("gpt-oss:20b")

    def test_ensure_client(self):
        """Test that AIME client is configured with correct base URL."""
        provider = AIMEProvider(secrets={"AIME_API_KEY": "test-key"})

        client = provider._ensure_client()

        # Verify base URL
        assert str(client.base_url) == "https://api.aime.info/v1/"

    @pytest.mark.asyncio
    async def test_generate_message(self):
        """Test non-streaming message generation."""
        provider = AIMEProvider(secrets={"AIME_API_KEY": "test-key"})

        # Create mock response
        mock_response = ChatCompletion(
            id="chatcmpl-aime-123",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="Test response from AIME"),
                    logprobs=None,
                )
            ],
            created=1677652288,
            model="gpt-oss:20b",
            object="chat.completion",
            usage=CompletionUsage(completion_tokens=12, prompt_tokens=9, total_tokens=21),
        )

        with patch.object(provider, "_ensure_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            messages = [Message(role="user", content="Test message")]

            result = await provider.generate_message(
                messages=messages,
                model="gpt-oss:20b",
                max_tokens=100,
            )

            assert result.role == "assistant"
            assert result.content == "Test response from AIME"

    @pytest.mark.asyncio
    async def test_generate_messages_streaming(self):
        """Test streaming message generation."""
        provider = AIMEProvider(secrets={"AIME_API_KEY": "test-key"})

        # Create mock streaming chunks
        async def mock_stream():
            chunks = [
                ChatCompletionChunk(
                    id="chatcmpl-aime-123",
                    choices=[
                        ChunkChoice(
                            delta=ChoiceDelta(content="Hello", role="assistant"),
                            finish_reason=None,
                            index=0,
                        )
                    ],
                    created=1677652288,
                    model="gpt-oss:20b",
                    object="chat.completion.chunk",
                ),
                ChatCompletionChunk(
                    id="chatcmpl-aime-123",
                    choices=[
                        ChunkChoice(
                            delta=ChoiceDelta(content=" world"),
                            finish_reason=None,
                            index=0,
                        )
                    ],
                    created=1677652288,
                    model="gpt-oss:20b",
                    object="chat.completion.chunk",
                ),
                ChatCompletionChunk(
                    id="chatcmpl-aime-123",
                    choices=[
                        ChunkChoice(
                            delta=ChoiceDelta(content="!"),
                            finish_reason="stop",
                            index=0,
                        )
                    ],
                    created=1677652288,
                    model="gpt-oss:20b",
                    object="chat.completion.chunk",
                ),
            ]
            for chunk in chunks:
                yield chunk

        with patch.object(provider, "_ensure_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

            messages = [Message(role="user", content="Test")]

            chunks = []
            async for chunk in provider.generate_messages(
                messages=messages,
                model="gpt-oss:20b",
            ):
                chunks.append(chunk)

            # Should have received chunks
            assert len(chunks) == 3

            # Last chunk should be done
            assert chunks[-1].done is True

            # Reconstruct full text
            full_text = "".join(c.content for c in chunks)
            assert full_text == "Hello world!"

    @pytest.mark.asyncio
    async def test_get_available_language_models(self):
        """Test that AIME returns available language models."""
        provider = AIMEProvider(secrets={"AIME_API_KEY": "test-key"})

        # Mock the models list response
        class MockModel:
            def __init__(self, id):
                self.id = id

        mock_models_response = MagicMock()
        mock_models_response.data = [MockModel("gpt-oss:20b"), MockModel("llama-3.1-8b")]

        with patch.object(provider, "_ensure_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_client.models.list = AsyncMock(return_value=mock_models_response)

            models = await provider.get_available_language_models()

            assert len(models) == 2
            model_ids = [m.id for m in models]
            assert "gpt-oss:20b" in model_ids
            assert "llama-3.1-8b" in model_ids

            # Verify provider is set correctly
            for model in models:
                assert model.provider.value == "aime"

    @pytest.mark.asyncio
    async def test_get_available_language_models_fallback(self):
        """Test that AIME returns fallback models when API fails."""
        provider = AIMEProvider(secrets={"AIME_API_KEY": "test-key"})

        with patch.object(provider, "_ensure_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_client.models.list = AsyncMock(side_effect=Exception("API error"))

            models = await provider.get_available_language_models()

            # Should return fallback models
            assert len(models) > 0
            assert models[0].provider.value == "aime"

    def test_initialization_missing_key(self):
        """Test that initialization fails without AIME_API_KEY."""
        with pytest.raises(AssertionError, match="AIME_API_KEY is required"):
            AIMEProvider(secrets={})

    def test_usage_tracking(self):
        """Test that usage is tracked correctly."""
        provider = AIMEProvider(secrets={"AIME_API_KEY": "test-key"})

        # Initial usage should be zero
        usage = provider.get_usage()
        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0
        assert usage["total_tokens"] == 0

        # Test reset
        provider._usage["prompt_tokens"] = 100
        provider.reset_usage()
        usage = provider.get_usage()
        assert usage["prompt_tokens"] == 0
