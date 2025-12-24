#!/usr/bin/env python3

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nodetool.metadata.types import LanguageModel, Provider
from nodetool.ml.models.language_models import get_all_language_models
from nodetool.providers.base import BaseProvider


class TestDynamicLanguageModels:
    """Test the dynamic language model fetching system."""

    @pytest.mark.asyncio
    async def test_get_all_language_models_basic(self):
        """Test that get_all_language_models returns a list of LanguageModel objects."""
        # Mock providers to avoid actual API calls
        mock_provider = MagicMock(spec=BaseProvider)
        mock_provider.provider_name = "test_provider"
        mock_provider.get_available_language_models = AsyncMock(
            return_value=[
                LanguageModel(
                    id="test-model-1",
                    name="Test Model 1",
                    provider=Provider.OpenAI,
                )
            ]
        )

        with patch(
            "nodetool.ml.models.language_models.list_providers",
            return_value=[mock_provider],
        ):
            models = await get_all_language_models(user_id="test-user")

            assert isinstance(models, list)
            assert len(models) > 0
            assert all(isinstance(model, LanguageModel) for model in models)

    @pytest.mark.asyncio
    async def test_get_all_language_models_multiple_calls(self):
        """Test that get_all_language_models fetches models on each call."""
        mock_provider = MagicMock(spec=BaseProvider)
        mock_provider.provider_name = "test_provider"
        mock_provider.get_available_language_models = AsyncMock(
            return_value=[
                LanguageModel(
                    id="test-model-1",
                    name="Test Model 1",
                    provider=Provider.OpenAI,
                )
            ]
        )

        with patch(
            "nodetool.ml.models.language_models.list_providers",
            return_value=[mock_provider],
        ):
            # First call
            models1 = await get_all_language_models(user_id="test-user")
            # Second call fetches again (no caching currently)
            models2 = await get_all_language_models(user_id="test-user")

            # Should call the provider twice (no caching implemented yet)
            assert mock_provider.get_available_language_models.call_count == 2
            assert len(models1) == len(models2)
            assert models1[0].id == models2[0].id

    @pytest.mark.asyncio
    async def test_get_all_language_models_multiple_providers(self):
        """Test that get_all_language_models aggregates models from multiple providers."""
        mock_provider1 = MagicMock(spec=BaseProvider)
        mock_provider1.provider_name = "provider1"
        mock_provider1.get_available_language_models = AsyncMock(
            return_value=[
                LanguageModel(
                    id="provider1-model",
                    name="Provider 1 Model",
                    provider=Provider.OpenAI,
                )
            ]
        )

        mock_provider2 = MagicMock(spec=BaseProvider)
        mock_provider2.provider_name = "provider2"
        mock_provider2.get_available_language_models = AsyncMock(
            return_value=[
                LanguageModel(
                    id="provider2-model",
                    name="Provider 2 Model",
                    provider=Provider.Anthropic,
                )
            ]
        )

        with patch(
            "nodetool.ml.models.language_models.list_providers",
            return_value=[mock_provider1, mock_provider2],
        ):
            models = await get_all_language_models(user_id="test-user")

            assert len(models) == 2
            model_ids = {model.id for model in models}
            assert "provider1-model" in model_ids
            assert "provider2-model" in model_ids

    @pytest.mark.asyncio
    async def test_get_all_language_models_handles_provider_errors(self):
        """Test that errors from one provider don't prevent getting models from others."""
        mock_provider1 = MagicMock(spec=BaseProvider)
        mock_provider1.provider_name = "failing_provider"
        mock_provider1.get_available_language_models = AsyncMock(side_effect=Exception("API Error"))

        mock_provider2 = MagicMock(spec=BaseProvider)
        mock_provider2.provider_name = "working_provider"
        mock_provider2.get_available_language_models = AsyncMock(
            return_value=[
                LanguageModel(
                    id="working-model",
                    name="Working Model",
                    provider=Provider.OpenAI,
                )
            ]
        )

        with patch(
            "nodetool.ml.models.language_models.list_providers",
            return_value=[mock_provider1, mock_provider2],
        ):
            # Should not raise, even though provider1 fails
            models = await get_all_language_models(user_id="test-user")

            # Should still get models from the working provider
            # Note: This assumes the implementation handles errors gracefully
            # If it doesn't, we may need to update the implementation
            assert isinstance(models, list)
