"""
Integration tests for model API endpoints.

This test ensures that the /api/models/llm and /api/models/image endpoints
correctly return only the expected model types, even when multi-modal providers
(like Gemini) are registered.
"""

import pytest
from nodetool.ml.models.language_models import get_all_language_models
from nodetool.ml.models.image_models import get_all_image_models
from nodetool.metadata.types import LanguageModel, ImageModel


@pytest.mark.asyncio
async def test_get_all_language_models_returns_only_language_models():
    """
    Regression test: Ensure get_all_language_models() only returns LanguageModel instances.

    This reproduces the bug where multi-modal providers (like Gemini) would return
    ImageModel instances mixed with LanguageModel instances, causing FastAPI
    ResponseValidationError.
    """
    models = await get_all_language_models()

    # All returned models must be LanguageModel instances
    for model in models:
        assert isinstance(
            model, LanguageModel
        ), f"Expected LanguageModel, got {type(model).__name__}: {model}"
        assert (
            model.type == "language_model"
        ), f"Expected type='language_model', got type='{model.type}'"


@pytest.mark.asyncio
async def test_get_all_image_models_returns_only_image_models():
    """
    Regression test: Ensure get_all_image_models() only returns ImageModel instances.

    This ensures image providers correctly return only image models.
    """
    models = await get_all_image_models()

    # All returned models must be ImageModel instances
    for model in models:
        assert isinstance(
            model, ImageModel
        ), f"Expected ImageModel, got {type(model).__name__}: {model}"
        assert (
            model.type == "image_model"
        ), f"Expected type='image_model', got type='{model.type}'"


@pytest.mark.asyncio
async def test_language_models_fastapi_serialization():
    """
    Test that language models can be serialized for FastAPI response validation.

    This simulates what FastAPI does internally when validating the response
    against the list[LanguageModel] type annotation.
    """
    models = await get_all_language_models()

    # Simulate FastAPI serialization
    for model in models:
        model_dict = model.model_dump()

        # These fields are required for FastAPI response validation
        assert "type" in model_dict
        assert model_dict["type"] == "language_model"
        assert "id" in model_dict
        assert "name" in model_dict
        assert "provider" in model_dict


@pytest.mark.asyncio
async def test_image_models_fastapi_serialization():
    """
    Test that image models can be serialized for FastAPI response validation.
    """
    models = await get_all_image_models()

    # Simulate FastAPI serialization
    for model in models:
        model_dict = model.model_dump()

        # These fields are required for FastAPI response validation
        assert "type" in model_dict
        assert model_dict["type"] == "image_model"
        assert "id" in model_dict
        assert "name" in model_dict
        assert "provider" in model_dict
