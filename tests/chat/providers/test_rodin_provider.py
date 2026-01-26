"""
Tests for the Rodin AI provider.

These tests verify:
- Provider registration and initialization
- Capability detection (TEXT_TO_3D, IMAGE_TO_3D)
- Model discovery
- Text-to-3D generation (mocked)
- Image-to-3D generation (mocked)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nodetool.metadata.types import Model3DModel, Provider
from nodetool.providers.base import ProviderCapability
from nodetool.providers.rodin_provider import RodinProvider, RODIN_3D_MODELS
from nodetool.providers.types import TextTo3DParams, ImageTo3DParams


class TestRodinProviderInitialization:
    """Tests for Rodin provider initialization and registration."""

    def test_required_secrets(self):
        """Test that required secrets are correctly defined."""
        secrets = RodinProvider.required_secrets()
        assert "RODIN_API_KEY" in secrets

    def test_initialization_with_api_key(self):
        """Test provider initialization with API key."""
        provider = RodinProvider(secrets={"RODIN_API_KEY": "test-key"})
        assert provider.api_key == "test-key"
        assert provider.provider_name == "rodin"

    def test_initialization_without_api_key(self):
        """Test provider initialization without API key."""
        provider = RodinProvider(secrets={})
        assert provider.api_key is None


class TestRodinProviderCapabilities:
    """Tests for Rodin provider capability detection."""

    def test_text_to_3d_capability(self):
        """Test that TEXT_TO_3D capability is detected."""
        provider = RodinProvider(secrets={"RODIN_API_KEY": "test-key"})
        capabilities = provider.get_capabilities()
        assert ProviderCapability.TEXT_TO_3D in capabilities

    def test_image_to_3d_capability(self):
        """Test that IMAGE_TO_3D capability is detected."""
        provider = RodinProvider(secrets={"RODIN_API_KEY": "test-key"})
        capabilities = provider.get_capabilities()
        assert ProviderCapability.IMAGE_TO_3D in capabilities


class TestRodinProviderModels:
    """Tests for Rodin provider model discovery."""

    @pytest.mark.asyncio
    async def test_get_available_3d_models_with_key(self):
        """Test model discovery with API key."""
        provider = RodinProvider(secrets={"RODIN_API_KEY": "test-key"})
        models = await provider.get_available_3d_models()
        assert len(models) == len(RODIN_3D_MODELS)
        for model in models:
            assert isinstance(model, Model3DModel)
            assert model.provider == Provider.Rodin

    @pytest.mark.asyncio
    async def test_get_available_3d_models_without_key(self):
        """Test model discovery without API key returns empty list."""
        provider = RodinProvider(secrets={})
        models = await provider.get_available_3d_models()
        assert len(models) == 0

    def test_model_tasks(self):
        """Test that models have correct supported tasks."""
        text_to_3d_models = [m for m in RODIN_3D_MODELS if "text_to_3d" in m.supported_tasks]
        image_to_3d_models = [m for m in RODIN_3D_MODELS if "image_to_3d" in m.supported_tasks]

        assert len(text_to_3d_models) >= 1
        assert len(image_to_3d_models) >= 1

    def test_model_output_formats(self):
        """Test that models have output formats defined."""
        for model in RODIN_3D_MODELS:
            assert len(model.output_formats) > 0
            assert "glb" in model.output_formats


class TestRodinTextTo3D:
    """Tests for Rodin text-to-3D generation."""

    @pytest.mark.asyncio
    async def test_text_to_3d_missing_api_key(self):
        """Test that text_to_3d fails without API key."""
        provider = RodinProvider(secrets={})
        text_models = [m for m in RODIN_3D_MODELS if "text_to_3d" in m.supported_tasks]
        params = TextTo3DParams(
            model=text_models[0],
            prompt="A cute robot",
        )
        with pytest.raises(ValueError, match="API key is not configured"):
            await provider.text_to_3d(params)

    @pytest.mark.asyncio
    async def test_text_to_3d_empty_prompt(self):
        """Test that text_to_3d fails with empty prompt."""
        provider = RodinProvider(secrets={"RODIN_API_KEY": "test-key"})
        text_models = [m for m in RODIN_3D_MODELS if "text_to_3d" in m.supported_tasks]
        params = TextTo3DParams(
            model=text_models[0],
            prompt="",
        )
        with pytest.raises(ValueError, match="prompt must not be empty"):
            await provider.text_to_3d(params)


class TestRodinImageTo3D:
    """Tests for Rodin image-to-3D generation."""

    @pytest.mark.asyncio
    async def test_image_to_3d_missing_api_key(self):
        """Test that image_to_3d fails without API key."""
        provider = RodinProvider(secrets={})
        image_models = [m for m in RODIN_3D_MODELS if "image_to_3d" in m.supported_tasks]
        params = ImageTo3DParams(
            model=image_models[0],
        )
        with pytest.raises(ValueError, match="API key is not configured"):
            await provider.image_to_3d(b"fake image", params)

    @pytest.mark.asyncio
    async def test_image_to_3d_empty_image(self):
        """Test that image_to_3d fails with empty image."""
        provider = RodinProvider(secrets={"RODIN_API_KEY": "test-key"})
        image_models = [m for m in RODIN_3D_MODELS if "image_to_3d" in m.supported_tasks]
        params = ImageTo3DParams(
            model=image_models[0],
        )
        with pytest.raises(ValueError, match="image must not be empty"):
            await provider.image_to_3d(b"", params)
