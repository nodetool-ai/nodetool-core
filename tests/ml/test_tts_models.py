"""
Tests for TTS model discovery and API endpoints.

This module tests the TTS model discovery functionality including:
- get_all_tts_models() function
- /api/models/tts endpoint
- Model type validation
"""

import pytest
from fastapi.testclient import TestClient

from nodetool.metadata.types import TTSModel
from nodetool.ml.models.tts_models import get_all_tts_models


@pytest.mark.asyncio
async def test_get_all_tts_models_returns_only_tts_models():
    """
    Test: Ensure get_all_tts_models() only returns TTSModel instances.

    This ensures TTS providers correctly return only TTS models.
    """
    models = await get_all_tts_models(user_id="test-user")

    # All returned models must be TTSModel instances
    for model in models:
        assert isinstance(model, TTSModel), f"Expected TTSModel, got {type(model).__name__}: {model}"
        assert model.type == "tts_model", f"Expected type='tts_model', got type='{model.type}'"


@pytest.mark.asyncio
async def test_tts_models_fastapi_serialization():
    """
    Test that TTS models can be serialized for FastAPI response validation.

    This simulates what FastAPI does internally when validating the response
    against the list[TTSModel] type annotation.
    """
    models = await get_all_tts_models(user_id="test-user")

    # Simulate FastAPI serialization
    for model in models:
        model_dict = model.model_dump()

        # These fields are required for FastAPI response validation
        assert "type" in model_dict
        assert model_dict["type"] == "tts_model"
        assert "id" in model_dict
        assert "name" in model_dict
        assert "provider" in model_dict
        assert "voices" in model_dict
        assert isinstance(model_dict["voices"], list)


@pytest.mark.asyncio
async def test_tts_models_have_required_fields():
    """
    Test that all TTS models have required fields populated.
    """
    models = await get_all_tts_models(user_id="test-user")

    for model in models:
        # ID and name should not be empty
        assert model.id, f"Model ID should not be empty: {model}"
        assert model.name, f"Model name should not be empty: {model}"

        # Provider should be set
        assert model.provider, f"Model provider should be set: {model}"

        # Voices should be a list (can be empty for some models)
        assert isinstance(model.voices, list), f"Model voices should be a list: {model}"


class TestTTSModelsAPI:
    """Tests for the /api/models/tts endpoint."""

    async def test_get_all_tts_models_endpoint(self, client: TestClient, headers: dict):
        """
        Test the /api/models/tts endpoint returns all TTS models from all providers.
        """
        response = client.get("/api/models/tts", headers=headers)
        assert response.status_code == 200
        models = response.json()
        assert isinstance(models, list)

        for model in models:
            assert model["type"] == "tts_model"
            assert model["id"] is not None
            assert model["name"] is not None
            assert model["provider"] is not None

    async def test_get_tts_models_endpoint_returns_only_tts_models(self, client: TestClient, headers: dict):
        """
        Test that the /api/models/tts endpoint only returns TTSModel instances.
        """
        response = client.get("/api/models/tts", headers=headers)
        assert response.status_code == 200
        models = response.json()

        for model in models:
            assert model["type"] == "tts_model", f"Expected type='tts_model', got type='{model.get('type')}'"

    async def test_get_tts_models_endpoint_response_format(self, client: TestClient, headers: dict):
        """
        Test that the /api/models/tts endpoint returns properly formatted response.
        """
        response = client.get("/api/models/tts", headers=headers)
        assert response.status_code == 200

        models = response.json()
        assert isinstance(models, list)

        if len(models) > 0:
            model = models[0]
            assert "type" in model
            assert "id" in model
            assert "name" in model
            assert "provider" in model
            assert "voices" in model
            assert isinstance(model["voices"], list)
