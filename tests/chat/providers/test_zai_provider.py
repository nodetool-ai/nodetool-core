"""
Tests for Z.AI provider.

Z.AI provides access to GLM models through an OpenAI-compatible API.
This test suite verifies that the Z.AI provider correctly:
- Initializes with the correct base URL (normal or coding plan)
- Supports streaming and non-streaming completions
- Handles function calling
- Respects the ZAI_USE_CODING_PLAN setting
"""

from unittest.mock import MagicMock, patch

import openai
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from nodetool.providers.zai_provider import ZAIProvider


class TestZAIProvider:
    """Test suite for Z.AI provider."""

    def test_initialization_normal_endpoint(self):
        """Test that Z.AI provider initializes with normal endpoint by default."""
        with patch("nodetool.config.environment.Environment.get_environment") as mock_env:
            mock_env.return_value.get.return_value = "false"
            provider = ZAIProvider(secrets={"ZHIPU_API_KEY": "test-key"})
            assert provider.api_key == "test-key"
            assert provider.provider.value == "zai"
            assert provider.base_url == "https://api.z.ai/api/paas/v4"

    def test_initialization_coding_plan_endpoint(self):
        """Test that Z.AI provider uses coding plan endpoint when configured."""
        with patch("nodetool.config.environment.Environment.get_environment") as mock_env:
            mock_env.return_value.get.return_value = "true"
            provider = ZAIProvider(secrets={"ZHIPU_API_KEY": "test-key"})
            assert provider.api_key == "test-key"
            assert provider.provider.value == "zai"
            assert provider.base_url == "https://api.z.ai/api/coding/paas/v4"

    def test_get_client_configuration_normal(self):
        """Test that Z.AI client is configured with normal base URL."""
        with patch("nodetool.config.environment.Environment.get_environment") as mock_env:
            mock_env.return_value.get.return_value = "false"
            provider = ZAIProvider(secrets={"ZHIPU_API_KEY": "test-key"})

            import httpx

            with patch("nodetool.runtime.resources.require_scope") as mock_scope:
                # Create a real httpx.AsyncClient instead of a mock
                mock_http_client = httpx.AsyncClient()
                mock_scope.return_value.get_http_client.return_value = mock_http_client

                client = provider.get_client()

                # Verify base URL (normal endpoint)
                assert str(client.base_url) == "https://api.z.ai/api/paas/v4/"

    def test_get_client_configuration_coding_plan(self):
        """Test that Z.AI client is configured with coding plan base URL."""
        with patch("nodetool.config.environment.Environment.get_environment") as mock_env:
            mock_env.return_value.get.return_value = "true"
            provider = ZAIProvider(secrets={"ZHIPU_API_KEY": "test-key"})

            import httpx

            with patch("nodetool.runtime.resources.require_scope") as mock_scope:
                # Create a real httpx.AsyncClient instead of a mock
                mock_http_client = httpx.AsyncClient()
                mock_scope.return_value.get_http_client.return_value = mock_http_client

                client = provider.get_client()

                # Verify base URL (coding plan endpoint)
                assert str(client.base_url) == "https://api.z.ai/api/coding/paas/v4/"

    def test_tool_support(self):
        """Test tool/function calling support detection."""
        with patch("nodetool.config.environment.Environment.get_environment") as mock_env:
            mock_env.return_value.get.return_value = "false"
            provider = ZAIProvider(secrets={"ZHIPU_API_KEY": "test-key"})

            # Z.AI GLM models support tools
            assert provider.has_tool_support("glm-4.5")
            assert provider.has_tool_support("glm-4.6")
            assert provider.has_tool_support("glm-4.7")

    def test_required_secrets(self):
        """Test that Z.AI provider requires the correct API key."""
        required = ZAIProvider.required_secrets()
        assert required == ["ZHIPU_API_KEY"]

    def test_container_env(self):
        """Test that container environment variables are correctly set."""
        with patch("nodetool.config.environment.Environment.get_environment") as mock_env:
            mock_env.return_value.get.return_value = "false"
            provider = ZAIProvider(secrets={"ZHIPU_API_KEY": "test-key"})

            mock_context = MagicMock()

            env = provider.get_container_env(mock_context)
            assert "ZHIPU_API_KEY" in env
            assert env["ZHIPU_API_KEY"] == "test-key"

    @pytest.mark.asyncio
    async def test_generate_message(self):
        """Test non-streaming message generation."""
        with patch("nodetool.config.environment.Environment.get_environment") as mock_env:
            mock_env.return_value.get.return_value = "false"
            provider = ZAIProvider(secrets={"ZHIPU_API_KEY": "test-key"})

            # Create mock response
            mock_response = ChatCompletion(
                id="chatcmpl-zai-123",
                choices=[
                    Choice(
                        finish_reason="stop",
                        index=0,
                        message=ChatCompletionMessage(role="assistant", content="Test response"),
                        logprobs=None,
                    )
                ],
                created=1677652288,
                model="glm-4.5",
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
                    model="glm-4.5",
                    max_tokens=100,
                )

                assert result.role == "assistant"
                assert result.content == "Test response"

    @pytest.mark.asyncio
    async def test_get_available_language_models(self):
        """Test that Z.AI can fetch available language models."""
        with patch("nodetool.config.environment.Environment.get_environment") as mock_env:
            mock_env.return_value.get.return_value = "false"
            provider = ZAIProvider(secrets={"ZHIPU_API_KEY": "test-key"})

            # Mock the aiohttp response
            mock_models_data = {
                "data": [
                    {
                        "id": "glm-4.5",
                        "name": "GLM-4.5",
                    },
                    {
                        "id": "glm-4.6",
                        "name": "GLM-4.6",
                    },
                    {
                        "id": "glm-4.7",
                        "name": "GLM-4.7",
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

                assert len(models) == 3
                model_ids = [m.id for m in models]
                assert "glm-4.5" in model_ids
                assert "glm-4.6" in model_ids
                assert "glm-4.7" in model_ids
                # Verify provider is set correctly
                for model in models:
                    assert model.provider.value == "zai"
