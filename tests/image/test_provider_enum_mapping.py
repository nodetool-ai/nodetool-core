"""Test Provider enum mapping to image providers."""

import pytest
from nodetool.metadata.types import Provider
from nodetool.image.providers.registry import get_image_provider


def test_huggingface_providers_map_correctly():
    """Test that HuggingFace Provider enums map to the correct provider."""
    # Test various HuggingFace providers (should all map to hf_inference except FalAI)
    hf_providers = [
        Provider.HuggingFaceBlackForestLabs,
        Provider.HuggingFaceCohere,
        Provider.HuggingFaceHFInference,
        Provider.HuggingFaceTogether,
    ]

    for provider_enum in hf_providers:
        try:
            provider = get_image_provider(provider_enum)
            assert provider is not None
            assert provider.provider_name == "hf_inference"
        except ValueError as e:
            # Provider might not be installed, but error should be about registration
            assert "not registered" in str(e) or "HF_TOKEN" in str(e)


def test_fal_ai_provider_maps_correctly():
    """Test that HuggingFaceFalAI maps to hf_inference provider (uses HuggingFace API with FAL backend)."""
    try:
        provider = get_image_provider(Provider.HuggingFaceFalAI)
        assert provider is not None
        # HuggingFaceFalAI uses HuggingFace's inference API with FAL as backend
        assert provider.provider_name == "hf_inference"
    except ValueError as e:
        # Provider might not be installed
        assert "not registered" in str(e) or "HF_TOKEN" in str(e)


def test_mlx_provider_maps_correctly():
    """Test that MLX provider maps correctly."""
    try:
        provider = get_image_provider(Provider.MLX)
        assert provider is not None
        assert provider.provider_name == "mlx"
    except ValueError as e:
        # Provider might not be installed
        assert "not registered" in str(e)


def test_unsupported_provider_raises_error():
    """Test that non-image providers raise appropriate errors."""
    with pytest.raises(ValueError) as exc_info:
        get_image_provider(Provider.OpenAI)

    assert "does not support image generation" in str(exc_info.value)


def test_string_provider_name_works():
    """Test that string provider names still work."""
    try:
        provider = get_image_provider("hf_inference")
        assert provider is not None
    except ValueError as e:
        # Provider might not be installed
        assert "not registered" in str(e) or "HF_TOKEN" in str(e)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
