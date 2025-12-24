"""
Tests for LM Studio provider with comprehensive API response mocking.

This module tests the LM Studio provider implementation including:
- Local model management and inference
- OpenAI-compatible API format
- Tool calling functionality
- Model listing
- Custom model support

LM Studio API Documentation:
URLs:
- https://lmstudio.ai/docs/developer/openai-compat

LM Studio provides an OpenAI-compatible REST API for running local large language models.

Core API Features:
- POST /v1/chat/completions: Generate next message in chat (streaming supported)
- GET /v1/models: List local models
- OpenAI-compatible API: Drop-in replacement for OpenAI Chat Completions API
- Compatible with existing OpenAI client libraries

Key Request Parameters:
- model: Model name (e.g., model identifier from LM Studio)
- messages: Array of message objects with role and content
- stream: Boolean for streaming responses
- tools: Optional array of tools/functions
- max_tokens: Maximum tokens to generate

Response Format:
- Standard OpenAI-compatible chat response with message, finish_reason, and metadata
- Streaming responses use server-sent events
- Usage statistics included in responses

Tool Support:
- Function calling with compatible models
- OpenAI-compatible tool calling API
- JSON schema-based tool definitions

Model Management:
- Models loaded via LM Studio GUI
- Support for various model formats (GGUF, etc.)
- Model names accessible via /v1/models endpoint

Local Advantages:
- Complete privacy and data control
- No API costs or external dependencies
- Custom model deployment
- Offline operation capability
"""

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest

from nodetool.metadata.types import Message
from nodetool.providers.lmstudio_provider import LMStudioProvider


@pytest.fixture
def provider():
    """Create a LMStudioProvider instance for testing."""
    with patch.dict(
        "os.environ",
        {
            "LMSTUDIO_API_URL": "http://localhost:1234",
        },
    ):
        return LMStudioProvider(secrets={})


@pytest.mark.asyncio
async def test_get_available_language_models_success(provider):
    """Test successful retrieval of available models."""

    class _Model:
        def __init__(self, id: str):
            self.id = id
            self.object = "model"
            self.created = 1677652288
            self.owned_by = "local"

    class _ModelsResponse:
        def __init__(self):
            self.data = [
                _Model("llama-2-7b-chat"),
                _Model("mistral-7b-instruct"),
            ]

    async def mock_list():
        return _ModelsResponse()

    with patch.object(openai.resources.models.AsyncModels, "list", side_effect=mock_list):
        models = await provider.get_available_language_models()

        assert len(models) == 2
        assert models[0].id == "llama-2-7b-chat"
        assert models[1].id == "mistral-7b-instruct"


@pytest.mark.asyncio
async def test_get_available_language_models_error(provider):
    """Test handling of errors when fetching models."""

    async def mock_list():
        raise httpx.ConnectError("Connection refused")

    with patch.object(openai.resources.models.AsyncModels, "list", side_effect=mock_list):
        models = await provider.get_available_language_models()
        assert len(models) == 0


@pytest.mark.asyncio
async def test_is_context_length_error(provider):
    """Test context length error detection."""
    # Test various error messages
    assert provider.is_context_length_error(Exception("context length exceeded"))
    assert provider.is_context_length_error(Exception("Context window too large"))
    assert provider.is_context_length_error(Exception("token limit reached"))
    assert provider.is_context_length_error(Exception("request too large"))
    assert provider.is_context_length_error(Exception("HTTP 413"))
    assert provider.is_context_length_error(Exception("maximum context length is 4096"))
    assert not provider.is_context_length_error(Exception("other error"))


@pytest.mark.asyncio
async def test_usage_tracking(provider):
    """Test usage statistics tracking."""
    # Initial usage should be zero
    usage = provider.get_usage()
    assert usage["prompt_tokens"] == 0
    assert usage["completion_tokens"] == 0
    assert usage["total_tokens"] == 0

    # Test reset
    provider._usage = {
        "prompt_tokens": 100,
        "completion_tokens": 200,
        "total_tokens": 300,
    }
    provider.reset_usage()
    usage = provider.get_usage()
    assert usage["prompt_tokens"] == 0
    assert usage["completion_tokens"] == 0
    assert usage["total_tokens"] == 0


@pytest.mark.asyncio
async def test_container_env(provider):
    """Test container environment variables."""
    from nodetool.workflows.processing_context import ProcessingContext

    context = ProcessingContext()
    env_vars = provider.get_container_env(context)
    assert "LMSTUDIO_API_URL" in env_vars
    assert env_vars["LMSTUDIO_API_URL"] == "http://localhost:1234"

