"""Tests for the model pricing API endpoint."""

from unittest.mock import AsyncMock, patch

import pytest

from nodetool.metadata.types import ModelPricing, Provider


class TestPricingApiEndpoint:
    """Tests for GET /api/models/pricing/{provider} endpoint."""

    @pytest.mark.asyncio
    async def test_get_pricing_by_provider(self):
        """Test get_pricing_by_provider helper function."""
        from nodetool.api.model import get_pricing_by_provider

        # Create mock pricing data
        mock_pricing = [
            ModelPricing(
                endpoint_id="test-model-1",
                provider=Provider.KIE,
                unit_price=0.025,
                unit="image",
                currency="USD",
            ),
            ModelPricing(
                endpoint_id="test-model-2",
                provider=Provider.KIE,
                unit_price=0.50,
                unit="video",
                currency="USD",
            ),
        ]

        # Mock the provider
        mock_provider = AsyncMock()
        mock_provider.get_pricing = AsyncMock(return_value=mock_pricing)

        with patch("nodetool.api.model.get_provider", return_value=mock_provider):
            result = await get_pricing_by_provider(Provider.KIE, "test_user")

            assert len(result) == 2
            assert all(isinstance(p, ModelPricing) for p in result)
            assert result[0].endpoint_id == "test-model-1"
            assert result[1].endpoint_id == "test-model-2"

    @pytest.mark.asyncio
    async def test_get_pricing_with_endpoint_filter(self):
        """Test get_pricing_by_provider with endpoint filter."""
        from nodetool.api.model import get_pricing_by_provider

        mock_pricing = [
            ModelPricing(
                endpoint_id="test-model-1",
                provider=Provider.KIE,
                unit_price=0.025,
                unit="image",
                currency="USD",
            ),
        ]

        mock_provider = AsyncMock()
        mock_provider.get_pricing = AsyncMock(return_value=mock_pricing)

        with patch("nodetool.api.model.get_provider", return_value=mock_provider):
            result = await get_pricing_by_provider(
                Provider.KIE, "test_user", endpoint_ids=["test-model-1"]
            )

            assert len(result) == 1
            assert result[0].endpoint_id == "test-model-1"
            # Verify endpoint_ids was passed to provider
            mock_provider.get_pricing.assert_called_once_with(["test-model-1"])

    @pytest.mark.asyncio
    async def test_get_pricing_provider_not_available(self):
        """Test get_pricing_by_provider when provider not available."""
        from nodetool.api.model import get_pricing_by_provider

        with patch(
            "nodetool.api.model.get_provider",
            side_effect=ValueError("Provider not installed"),
        ):
            result = await get_pricing_by_provider(Provider.KIE, "test_user")
            assert result == []

    @pytest.mark.asyncio
    async def test_get_pricing_provider_error(self):
        """Test get_pricing_by_provider when provider raises error."""
        from nodetool.api.model import get_pricing_by_provider

        with patch(
            "nodetool.api.model.get_provider",
            side_effect=Exception("Unexpected error"),
        ):
            result = await get_pricing_by_provider(Provider.KIE, "test_user")
            assert result == []
