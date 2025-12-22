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
            
            try:
                client = provider.get_client()
                
                # Verify base URL
                assert str(client.base_url) == "https://openrouter.ai/api/v1/"
                
                # Verify OpenRouter-specific headers
                assert "HTTP-Referer" in client.default_headers
                assert "X-Title" in client.default_headers
            finally:
                # Clean up the http client
                import asyncio
                asyncio.run(mock_http_client.aclose())

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
        
        from unittest.mock import MagicMock
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
                    message=ChatCompletionMessage(
                        role="assistant",
                        content="Test response"
                    ),
                    logprobs=None,
                )
            ],
            created=1677652288,
            model="openai/gpt-4",
            object="chat.completion",
            usage=CompletionUsage(
                completion_tokens=12, prompt_tokens=9, total_tokens=21
            ),
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

