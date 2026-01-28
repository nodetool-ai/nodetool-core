"""Tests for ProcessingContext provider-based prediction methods."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.providers.base import BaseProvider, ProviderCapability
from nodetool.metadata.types import Provider, Message


class MockProvider(BaseProvider):
    """Mock provider for testing."""

    provider_name = "openai"  # Use openai for cost lookup tests

    def __init__(self, secrets=None):
        super().__init__(secrets or {})
        self._generate_message_mock = AsyncMock()
        self._generate_embedding_mock = AsyncMock()

    @classmethod
    def required_secrets(cls):
        return []

    def get_capabilities(self):
        return {
            ProviderCapability.GENERATE_MESSAGE,
            ProviderCapability.GENERATE_EMBEDDING,
        }

    async def generate_message(self, messages, model, tools=None, max_tokens=8192, **kwargs):
        result = await self._generate_message_mock(messages, model, tools, max_tokens, **kwargs)
        # Simulate cost tracking
        self.track_usage(model=model, input_tokens=100, output_tokens=50)
        return result or Message(role="assistant", content="test response")

    async def generate_embedding(self, text, model, **kwargs):
        result = await self._generate_embedding_mock(text, model, **kwargs)
        self.track_usage(model=model, input_tokens=50)
        return result or [[0.1, 0.2, 0.3]]


class TestGetProvider:
    """Tests for ProcessingContext.get_provider()."""

    @pytest.mark.asyncio
    async def test_get_provider_caching(self):
        """Provider instances should be cached."""
        ctx = ProcessingContext(user_id="test")

        with patch(
            "nodetool.providers.base.get_registered_provider",
            return_value=(MockProvider, {}),
        ):
            provider1 = await ctx.get_provider(Provider.OpenAI)
            provider2 = await ctx.get_provider(Provider.OpenAI)

            assert provider1 is provider2, "Provider should be cached"

    @pytest.mark.asyncio
    async def test_get_provider_different_providers(self):
        """Different providers should return different instances."""
        ctx = ProcessingContext(user_id="test")

        with patch(
            "nodetool.providers.base.get_registered_provider",
            return_value=(MockProvider, {}),
        ):
            provider1 = await ctx.get_provider(Provider.OpenAI)
            provider2 = await ctx.get_provider(Provider.Anthropic)

            assert provider1 is not provider2, "Different providers should be different instances"

    @pytest.mark.asyncio
    async def test_get_provider_passes_secrets(self):
        """Provider should receive secrets from environment."""
        ctx = ProcessingContext(
            user_id="test",
            environment={"TEST_API_KEY": "test-key-123"},
        )

        class MockProviderWithSecrets(MockProvider):
            @classmethod
            def required_secrets(cls):
                return ["TEST_API_KEY"]

        with patch(
            "nodetool.providers.base.get_registered_provider",
            return_value=(MockProviderWithSecrets, {}),
        ):
            provider = await ctx.get_provider(Provider.OpenAI)
            assert provider.secrets.get("TEST_API_KEY") == "test-key-123"


class TestRunProviderPrediction:
    """Tests for ProcessingContext.run_provider_prediction()."""

    @pytest.mark.asyncio
    async def test_run_provider_prediction_generate_message(self):
        """Test running a GENERATE_MESSAGE prediction."""
        ctx = ProcessingContext(user_id="test")

        mock_provider = MockProvider()
        mock_provider._generate_message_mock.return_value = Message(
            role="assistant", content="Hello, world!"
        )

        with patch.object(ctx, "get_provider", return_value=mock_provider):
            with patch("nodetool.models.prediction.Prediction.create", new_callable=AsyncMock):
                result = await ctx.run_provider_prediction(
                    node_id="test_node",
                    provider=Provider.OpenAI,
                    model="gpt-4o-mini",
                    capability=ProviderCapability.GENERATE_MESSAGE,
                    params={
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )

                assert result.role == "assistant"
                assert result.content == "Hello, world!"
                assert mock_provider.cost > 0

    @pytest.mark.asyncio
    async def test_run_provider_prediction_unsupported_capability(self):
        """Test that unsupported capability raises ValueError."""
        ctx = ProcessingContext(user_id="test")

        mock_provider = MockProvider()
        # Remove TEXT_TO_IMAGE from capabilities (not in default mock)

        with patch.object(ctx, "get_provider", return_value=mock_provider):
            with pytest.raises(ValueError, match="does not support capability"):
                await ctx.run_provider_prediction(
                    node_id="test_node",
                    provider=Provider.OpenAI,
                    model="dall-e-3",
                    capability=ProviderCapability.TEXT_TO_IMAGE,
                    params={"prompt": "A cat"},
                )

    @pytest.mark.asyncio
    async def test_run_provider_prediction_string_provider(self):
        """Test that string provider is converted to enum."""
        ctx = ProcessingContext(user_id="test")

        mock_provider = MockProvider()

        with patch.object(ctx, "get_provider", return_value=mock_provider) as mock_get:
            with patch("nodetool.models.prediction.Prediction.create", new_callable=AsyncMock):
                await ctx.run_provider_prediction(
                    node_id="test_node",
                    provider="openai",  # String provider
                    model="gpt-4o-mini",
                    capability=ProviderCapability.GENERATE_MESSAGE,
                    params={"messages": []},
                )

                # Verify get_provider was called with Provider enum
                mock_get.assert_called_once()
                call_args = mock_get.call_args[0]
                assert call_args[0] == Provider.OpenAI

    @pytest.mark.asyncio
    async def test_run_provider_prediction_logs_cost(self):
        """Test that prediction cost is logged."""
        ctx = ProcessingContext(user_id="test")

        mock_provider = MockProvider()

        with patch.object(ctx, "get_provider", return_value=mock_provider):
            with patch(
                "nodetool.models.prediction.Prediction.create", new_callable=AsyncMock
            ) as mock_create:
                await ctx.run_provider_prediction(
                    node_id="test_node",
                    provider=Provider.OpenAI,
                    model="gpt-4o-mini",
                    capability=ProviderCapability.GENERATE_MESSAGE,
                    params={"messages": []},
                )

                # Verify Prediction.create was called with cost
                mock_create.assert_called_once()
                call_kwargs = mock_create.call_args[1]
                assert call_kwargs["status"] == "completed"
                assert "cost" in call_kwargs
                assert call_kwargs["cost"] > 0

    @pytest.mark.asyncio
    async def test_run_provider_prediction_logs_failure(self):
        """Test that failed predictions are logged."""
        ctx = ProcessingContext(user_id="test")

        mock_provider = MockProvider()
        mock_provider._generate_message_mock.side_effect = RuntimeError("API error")

        with patch.object(ctx, "get_provider", return_value=mock_provider):
            with patch(
                "nodetool.models.prediction.Prediction.create", new_callable=AsyncMock
            ) as mock_create:
                with pytest.raises(RuntimeError, match="API error"):
                    await ctx.run_provider_prediction(
                        node_id="test_node",
                        provider=Provider.OpenAI,
                        model="gpt-4o-mini",
                        capability=ProviderCapability.GENERATE_MESSAGE,
                        params={"messages": []},
                    )

                # Verify failure was logged
                mock_create.assert_called_once()
                call_kwargs = mock_create.call_args[1]
                assert call_kwargs["status"] == "failed"
                assert "API error" in call_kwargs["error"]


class TestDispatchCapability:
    """Tests for ProcessingContext._dispatch_capability()."""

    @pytest.mark.asyncio
    async def test_dispatch_generate_message(self):
        """Test dispatching GENERATE_MESSAGE capability."""
        ctx = ProcessingContext(user_id="test")
        mock_provider = MockProvider()
        
        # Set up mock return value before the call
        expected_message = Message(role="assistant", content="Test response")
        mock_provider._generate_message_mock.return_value = expected_message

        result = await ctx._dispatch_capability(
            provider=mock_provider,
            capability=ProviderCapability.GENERATE_MESSAGE,
            model="gpt-4o-mini",
            params={"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1000},
        )

        mock_provider._generate_message_mock.assert_called_once()
        assert result.role == "assistant"
        assert result.content == "Test response"

    @pytest.mark.asyncio
    async def test_dispatch_generate_embedding(self):
        """Test dispatching GENERATE_EMBEDDING capability."""
        ctx = ProcessingContext(user_id="test")
        mock_provider = MockProvider()
        
        # Set up mock return value before the call
        expected_embedding = [[0.1, 0.2, 0.3]]
        mock_provider._generate_embedding_mock.return_value = expected_embedding

        result = await ctx._dispatch_capability(
            provider=mock_provider,
            capability=ProviderCapability.GENERATE_EMBEDDING,
            model="text-embedding-3-small",
            params={"text": "Hello world"},
        )

        mock_provider._generate_embedding_mock.assert_called_once()
        assert result == [[0.1, 0.2, 0.3]]

    @pytest.mark.asyncio
    async def test_dispatch_unsupported_capability(self):
        """Test that unsupported capability raises ValueError."""
        ctx = ProcessingContext(user_id="test")
        mock_provider = MockProvider()

        # Create a mock capability that doesn't exist
        with pytest.raises(ValueError, match="Unsupported capability"):
            await ctx._dispatch_capability(
                provider=mock_provider,
                capability="nonexistent_capability",  # type: ignore
                model="test-model",
                params={},
            )
