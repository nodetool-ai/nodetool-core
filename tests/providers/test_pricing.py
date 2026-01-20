"""Tests for provider pricing functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nodetool.metadata.types import ModelPricing, Provider


class TestModelPricing:
    """Tests for ModelPricing data model."""

    def test_model_pricing_defaults(self):
        """Test ModelPricing default values."""
        pricing = ModelPricing()
        assert pricing.endpoint_id == ""
        assert pricing.provider == Provider.Empty
        assert pricing.unit_price == 0.0
        assert pricing.unit == "request"
        assert pricing.currency == "USD"
        assert pricing.prompt_price is None
        assert pricing.completion_price is None
        assert pricing.request_price is None
        assert pricing.image_price is None

    def test_model_pricing_with_values(self):
        """Test ModelPricing with custom values."""
        pricing = ModelPricing(
            endpoint_id="gpt-4o-mini",
            provider=Provider.OpenRouter,
            unit_price=0.00015,
            unit="token",
            currency="USD",
            prompt_price=0.00015,
            completion_price=0.0006,
        )
        assert pricing.endpoint_id == "gpt-4o-mini"
        assert pricing.provider == Provider.OpenRouter
        assert pricing.unit_price == 0.00015
        assert pricing.unit == "token"
        assert pricing.prompt_price == 0.00015
        assert pricing.completion_price == 0.0006


class TestBaseProviderPricing:
    """Tests for BaseProvider.get_pricing method."""

    def test_base_provider_get_pricing_returns_empty_list(self):
        """Test that BaseProvider.get_pricing returns empty list by default."""
        from nodetool.providers.base import BaseProvider

        provider = BaseProvider()
        # Run async method synchronously for testing
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(provider.get_pricing())
        assert result == []


class TestKieProviderPricing:
    """Tests for KieProvider pricing functionality."""

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession")
    async def test_get_pricing_success(self, mock_session_class):
        """Test KieProvider.get_pricing returns pricing data."""
        from nodetool.providers.kie_provider import KieProvider

        # Setup mocks
        mock_session = MagicMock()
        mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_class.return_value.__aexit__ = AsyncMock()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "code": 200,
                "data": {
                    "records": [
                        {
                            "model": "flux-2/pro-text-to-image",
                            "price": 0.025,
                            "interfaceType": "text-to-image",
                        },
                        {
                            "model": "kling-2.6/text-to-video",
                            "price": 0.50,
                            "interfaceType": "text-to-video",
                        },
                    ],
                    "total": 2,
                },
            }
        )
        mock_session.post = MagicMock(
            return_value=MagicMock(__aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock())
        )

        provider = KieProvider(secrets={"KIE_API_KEY": "test_key"})
        pricing = await provider.get_pricing()

        assert len(pricing) == 2
        assert all(isinstance(p, ModelPricing) for p in pricing)
        assert pricing[0].endpoint_id == "flux-2/pro-text-to-image"
        assert pricing[0].unit_price == 0.025
        assert pricing[0].unit == "image"
        assert pricing[0].provider == Provider.KIE
        assert pricing[1].endpoint_id == "kling-2.6/text-to-video"
        assert pricing[1].unit_price == 0.50
        assert pricing[1].unit == "video"

    @pytest.mark.asyncio
    async def test_get_pricing_no_api_key(self):
        """Test KieProvider.get_pricing returns empty list without API key."""
        from nodetool.providers.kie_provider import KieProvider

        # We can't create a KieProvider without an API key (raises ValueError)
        # So we test the flow with a mocked empty key scenario
        with pytest.raises(ValueError, match="KIE_API_KEY is required"):
            KieProvider(secrets={})

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession")
    async def test_get_pricing_with_filter(self, mock_session_class):
        """Test KieProvider.get_pricing filters by endpoint_ids."""
        from nodetool.providers.kie_provider import KieProvider

        mock_session = MagicMock()
        mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_class.return_value.__aexit__ = AsyncMock()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "code": 200,
                "data": {
                    "records": [
                        {"model": "flux-2/pro-text-to-image", "price": 0.025, "interfaceType": "text-to-image"},
                        {"model": "other-model", "price": 0.10, "interfaceType": "text-to-image"},
                    ],
                    "total": 2,
                },
            }
        )
        mock_session.post = MagicMock(
            return_value=MagicMock(__aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock())
        )

        provider = KieProvider(secrets={"KIE_API_KEY": "test_key"})
        pricing = await provider.get_pricing(endpoint_ids=["flux-2/pro-text-to-image"])

        assert len(pricing) == 1
        assert pricing[0].endpoint_id == "flux-2/pro-text-to-image"


class TestOpenRouterProviderPricing:
    """Tests for OpenRouterProvider pricing functionality."""

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession")
    async def test_get_pricing_success(self, mock_session_class):
        """Test OpenRouterProvider.get_pricing returns pricing data."""
        from nodetool.providers.openrouter_provider import OpenRouterProvider

        mock_session = MagicMock()
        mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_class.return_value.__aexit__ = AsyncMock()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "data": [
                    {
                        "id": "openai/gpt-4o-mini",
                        "name": "GPT-4o Mini",
                        "pricing": {
                            "prompt": "0.00015",
                            "completion": "0.0006",
                            "request": "0",
                            "image": "0.007225",
                        },
                    },
                    {
                        "id": "anthropic/claude-3.5-sonnet",
                        "name": "Claude 3.5 Sonnet",
                        "pricing": {
                            "prompt": "0.003",
                            "completion": "0.015",
                        },
                    },
                ]
            }
        )
        mock_session.get = MagicMock(
            return_value=MagicMock(__aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock())
        )

        provider = OpenRouterProvider(secrets={"OPENROUTER_API_KEY": "test_key"})
        pricing = await provider.get_pricing()

        assert len(pricing) == 2
        assert all(isinstance(p, ModelPricing) for p in pricing)
        assert pricing[0].endpoint_id == "openai/gpt-4o-mini"
        assert pricing[0].unit == "token"
        assert pricing[0].provider == Provider.OpenRouter
        assert pricing[0].prompt_price == 0.00015
        assert pricing[0].completion_price == 0.0006
        assert pricing[0].image_price == 0.007225

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession")
    async def test_get_pricing_with_filter(self, mock_session_class):
        """Test OpenRouterProvider.get_pricing filters by endpoint_ids."""
        from nodetool.providers.openrouter_provider import OpenRouterProvider

        mock_session = MagicMock()
        mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_class.return_value.__aexit__ = AsyncMock()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "data": [
                    {"id": "openai/gpt-4o-mini", "pricing": {"prompt": "0.00015", "completion": "0.0006"}},
                    {"id": "anthropic/claude-3.5-sonnet", "pricing": {"prompt": "0.003", "completion": "0.015"}},
                ]
            }
        )
        mock_session.get = MagicMock(
            return_value=MagicMock(__aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock())
        )

        provider = OpenRouterProvider(secrets={"OPENROUTER_API_KEY": "test_key"})
        pricing = await provider.get_pricing(endpoint_ids=["openai/gpt-4o-mini"])

        assert len(pricing) == 1
        assert pricing[0].endpoint_id == "openai/gpt-4o-mini"


class TestFalPricing:
    """Tests for FAL pricing utility function."""

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession")
    async def test_fetch_fal_pricing_success(self, mock_session_class):
        """Test fetch_fal_pricing returns pricing data."""
        from nodetool.providers.fal_pricing import fetch_fal_pricing

        mock_session = MagicMock()
        mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_class.return_value.__aexit__ = AsyncMock()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "prices": [
                    {"endpoint_id": "fal-ai/flux/dev", "unit_price": 0.025, "unit": "image", "currency": "USD"}
                ],
                "has_more": False,
                "next_cursor": None,
            }
        )
        mock_session.get = MagicMock(
            return_value=MagicMock(__aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock())
        )

        pricing = await fetch_fal_pricing("test_api_key", endpoint_ids=["fal-ai/flux/dev"])

        assert len(pricing) == 1
        assert pricing[0].endpoint_id == "fal-ai/flux/dev"
        assert pricing[0].unit_price == 0.025
        assert pricing[0].unit == "image"
        assert pricing[0].provider == Provider.HuggingFaceFalAI

    @pytest.mark.asyncio
    async def test_fetch_fal_pricing_no_api_key(self):
        """Test fetch_fal_pricing returns empty list without API key."""
        from nodetool.providers.fal_pricing import fetch_fal_pricing

        pricing = await fetch_fal_pricing("", endpoint_ids=["fal-ai/flux/dev"])
        assert pricing == []

    @pytest.mark.asyncio
    async def test_fetch_fal_pricing_no_endpoint_ids(self):
        """Test fetch_fal_pricing returns empty list without endpoint_ids."""
        from nodetool.providers.fal_pricing import fetch_fal_pricing

        pricing = await fetch_fal_pricing("test_api_key", endpoint_ids=None)
        assert pricing == []
