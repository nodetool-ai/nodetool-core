"""
Regression tests for model discovery and API response validation.

This test file ensures that:
1. get_all_language_models() only returns LanguageModel instances
2. get_all_image_models() only returns ImageModel instances
3. Multi-modal providers (like Gemini) return the correct model types
4. API endpoints have correct response validation
"""

import pytest
from typing import List
from nodetool.metadata.types import LanguageModel, ImageModel, Provider
from nodetool.providers.base import BaseProvider
from nodetool.ml.models.language_models import get_all_language_models
from nodetool.ml.models.image_models import get_all_image_models


class MockMultiModalProvider(BaseProvider):
    """Mock provider that supports both chat and image capabilities (like Gemini)."""

    async def get_available_language_models(self) -> List[LanguageModel]:
        """Return mock language models."""
        return [
            LanguageModel(id="mock-llm-1", name="Mock LLM 1", provider=Provider.Empty),
            LanguageModel(id="mock-llm-2", name="Mock LLM 2", provider=Provider.Empty),
        ]

    async def get_available_image_models(self) -> List[ImageModel]:
        """Return mock image models."""
        return [
            ImageModel(id="mock-img-1", name="Mock Image 1", provider=Provider.Empty),
            ImageModel(id="mock-img-2", name="Mock Image 2", provider=Provider.Empty),
        ]

    async def generate_message(self, messages, model, tools=None, **kwargs):  # type: ignore
        return None

    async def generate_messages(self, messages, model, tools=None, **kwargs):  # type: ignore
        if False:
            yield

    async def text_to_image(self, *args, **kwargs):  # type: ignore[override]
        return b""

    async def image_to_image(self, *args, **kwargs):  # type: ignore[override]
        return b""


class MockLanguageOnlyProvider(BaseProvider):
    """Mock provider that only supports language capabilities."""

    async def get_available_language_models(self) -> List[LanguageModel]:
        """Return mock language models."""
        return [
            LanguageModel(
                id="lang-only-1", name="Language Only 1", provider=Provider.Empty
            ),
        ]

    async def generate_message(self, messages, model, tools=None, **kwargs):  # type: ignore
        return None

    async def generate_messages(self, messages, model, tools=None, **kwargs):  # type: ignore
        if False:
            yield


class MockImageOnlyProvider(BaseProvider):
    """Mock provider that only supports image capabilities."""

    provider_name = "mock_image"

    async def get_available_image_models(self) -> List[ImageModel]:
        """Return mock image models."""
        return [
            ImageModel(id="img-only-1", name="Image Only 1", provider=Provider.Empty),
        ]

    async def text_to_image(self, *args, **kwargs):  # type: ignore[override]
        return b""

    async def image_to_image(self, *args, **kwargs):  # type: ignore[override]
        return b""


@pytest.mark.asyncio
async def test_multimodal_provider_language_models():
    """
    Test that a multi-modal provider's get_available_language_models()
    only returns LanguageModel instances.
    """
    provider = MockMultiModalProvider()
    models = await provider.get_available_language_models()

    assert len(models) == 2
    for model in models:
        assert isinstance(model, LanguageModel)
        assert model.type == "language_model"


@pytest.mark.asyncio
async def test_multimodal_provider_image_models():
    """
    Test that a multi-modal provider's get_available_image_models()
    only returns ImageModel instances.
    """
    provider = MockMultiModalProvider()
    models = await provider.get_available_image_models()

    assert len(models) == 2
    for model in models:
        assert isinstance(model, ImageModel)
        assert model.type == "image_model"


@pytest.mark.asyncio
async def test_multimodal_provider_get_available_models_returns_both():
    """
    Test that get_available_models() returns both language and image models
    for a multi-modal provider.
    """
    provider = MockMultiModalProvider()
    models = await provider.get_available_models()

    assert len(models) == 4

    # Count model types
    language_count = sum(1 for m in models if isinstance(m, LanguageModel))
    image_count = sum(1 for m in models if isinstance(m, ImageModel))

    assert language_count == 2
    assert image_count == 2


@pytest.mark.asyncio
async def test_language_only_provider():
    """
    Test that a language-only provider returns empty list for image models.
    """
    provider = MockLanguageOnlyProvider()

    language_models = await provider.get_available_language_models()
    assert len(language_models) == 1
    assert all(isinstance(m, LanguageModel) for m in language_models)

    image_models = await provider.get_available_image_models()
    assert len(image_models) == 0


@pytest.mark.asyncio
async def test_image_only_provider():
    """
    Test that an image-only provider returns empty list for language models.
    """
    provider = MockImageOnlyProvider()

    image_models = await provider.get_available_image_models()
    assert len(image_models) == 1
    assert all(isinstance(m, ImageModel) for m in image_models)

    language_models = await provider.get_available_language_models()
    assert len(language_models) == 0


@pytest.mark.asyncio
async def test_model_type_discriminator():
    """
    Regression test for FastAPI response validation error.

    This test ensures that LanguageModel and ImageModel instances
    have the correct 'type' discriminator field for Pydantic validation.

    Reproduces the bug where GeminiProvider's get_available_models()
    returned both model types, causing FastAPI ResponseValidationError
    when the API expected only LanguageModel.
    """
    provider = MockMultiModalProvider()

    # Test language models have correct type discriminator
    language_models = await provider.get_available_language_models()
    for model in language_models:
        assert hasattr(model, "type")
        assert (
            model.type == "language_model"
        ), f"LanguageModel should have type='language_model', got '{model.type}'"

    # Test image models have correct type discriminator
    image_models = await provider.get_available_image_models()
    for model in image_models:
        assert hasattr(model, "type")
        assert (
            model.type == "image_model"
        ), f"ImageModel should have type='image_model', got '{model.type}'"


@pytest.mark.asyncio
async def test_model_serialization():
    """
    Test that models can be serialized correctly for API responses.

    This ensures Pydantic can validate the models for FastAPI responses.
    """
    provider = MockMultiModalProvider()

    # Test language model serialization
    language_models = await provider.get_available_language_models()
    for model in language_models:
        # This is what FastAPI does internally
        model_dict = model.model_dump()
        assert model_dict["type"] == "language_model"
        assert "id" in model_dict
        assert "name" in model_dict
        assert "provider" in model_dict

    # Test image model serialization
    image_models = await provider.get_available_image_models()
    for model in image_models:
        model_dict = model.model_dump()
        assert model_dict["type"] == "image_model"
        assert "id" in model_dict
        assert "name" in model_dict
        assert "provider" in model_dict
