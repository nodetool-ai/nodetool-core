"""
Tests for the ImageProvider registry system.
"""

import pytest
from typing import List, Set
from nodetool.image.providers import (
    register_image_provider,
    get_image_provider,
    list_image_providers,
    ImageProvider,
)
from nodetool.chat.providers.base import ProviderCapability
from nodetool.image.types import TextToImageParams, ImageToImageParams, ImageBytes
from nodetool.metadata.types import (
    Provider,
    ImageModel,
)


class MockImageProvider(ImageProvider):
    """Mock image provider for testing."""

    provider_name = "mock"

    def get_capabilities(self) -> Set[ProviderCapability]:
        """Mock provider supports both image generation capabilities."""
        return {
            ProviderCapability.TEXT_TO_IMAGE,
            ProviderCapability.IMAGE_TO_IMAGE,
        }

    async def get_available_image_models(self) -> List[ImageModel]:
        """Return empty list for testing."""
        return []

    async def text_to_image(
        self, params: TextToImageParams, timeout_s: int | None = None, context=None
    ) -> ImageBytes:
        """Generate a mock image."""
        return b"mock_image_data"

    async def image_to_image(
        self,
        image: ImageBytes,
        params: ImageToImageParams,
        timeout_s: int | None = None,
        context=None,
    ) -> ImageBytes:
        """Transform a mock image."""
        return b"mock_transformed_image_data"


def test_register_provider():
    """Test provider registration."""
    register_image_provider("test_mock", lambda: MockImageProvider())
    assert "test_mock" in list_image_providers()


def test_get_provider():
    """Test provider retrieval."""
    register_image_provider("test_mock2", lambda: MockImageProvider())
    provider = get_image_provider("test_mock2")
    assert isinstance(provider, MockImageProvider)
    assert provider.provider_name == "mock"


def test_get_unknown_provider():
    """Test error when getting unknown provider."""
    with pytest.raises(ValueError) as exc_info:
        get_image_provider("unknown_provider")
    assert "not registered" in str(exc_info.value)


def test_list_providers():
    """Test listing registered providers."""
    register_image_provider("test_list1", lambda: MockImageProvider())
    register_image_provider("test_list2", lambda: MockImageProvider())

    providers = list_image_providers()
    assert "test_list1" in providers
    assert "test_list2" in providers


@pytest.mark.asyncio
async def test_mock_text_to_image():
    """Test mock text-to-image generation."""
    provider = MockImageProvider()

    params = TextToImageParams(
        model=ImageModel(provider=Provider.Empty, id="test-model", name="Test Model"),
        prompt="test prompt",
        width=512,
        height=512,
    )

    result = await provider.text_to_image(params)
    assert result == b"mock_image_data"


@pytest.mark.asyncio
async def test_mock_image_to_image():
    """Test mock image-to-image transformation."""
    provider = MockImageProvider()

    params = ImageToImageParams(
        model=ImageModel(provider=Provider.Empty, id="test-model", name="Test Model"),
        prompt="test transformation",
        strength=0.8,
    )

    result = await provider.image_to_image(b"input_image", params)
    assert result == b"mock_transformed_image_data"
