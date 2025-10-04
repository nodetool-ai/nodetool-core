"""
Registry for ImageProvider implementations.

This module provides registration and lookup mechanisms for image providers.
Providers self-register using the register_image_provider function.
"""

from typing import Callable
from nodetool.image.providers.base import ImageProvider
from nodetool.metadata.types import Provider

_IMAGE_PROVIDER_REGISTRY: dict[str, Callable[[], ImageProvider]] = {}
_PROVIDER_ENUM_MAP: dict[Provider, str] = {}
_PROVIDERS_LOADED = False


def register_image_provider(
    provider_name: str, factory: Callable[[], ImageProvider]
) -> None:
    """Register an image provider factory.

    Args:
        provider_name: Unique name for the provider (e.g., "hf_inference", "fal_ai", "mlx")
        factory: Callable that returns a new ImageProvider instance
    """
    _IMAGE_PROVIDER_REGISTRY[provider_name] = factory


def _load_providers() -> None:
    """Import all image provider packages to ensure they're registered."""
    global _PROVIDERS_LOADED
    if _PROVIDERS_LOADED:
        return

    _PROVIDERS_LOADED = True

    # Import provider packages to trigger registration
    try:
        import nodetool.huggingface  # type: ignore
    except ImportError:
        pass

    try:
        import nodetool.fal  # type: ignore
    except ImportError:
        pass

    try:
        import nodetool.mlx  # type: ignore
    except ImportError:
        pass


def get_image_provider(provider: Provider | str) -> ImageProvider:
    """Get an image provider instance by Provider enum or name string.

    Args:
        provider: Provider enum value or provider name string

    Returns:
        A new ImageProvider instance

    Raises:
        ValueError: If the provider is not registered
    """
    # Ensure all providers are loaded
    _load_providers()

    # Map Provider enum to provider name
    if isinstance(provider, Provider):
        # All HuggingFace providers use the HuggingFace inference client
        if provider.value.startswith("huggingface_"):
            provider_name = "hf_inference"
        elif provider == Provider.MLX:
            provider_name = "mlx"
        elif provider == Provider.HuggingFace:
            provider_name = "huggingface"
        elif provider == Provider.FalAI:
            provider_name = "fal_ai"
        elif provider == Provider.Replicate:
            provider_name = "replicate"
        elif provider == Provider.Gemini:
            provider_name = "gemini"
        else:
            raise ValueError(
                f"Provider {provider.value} does not support image generation. "
                f"Supported providers: HuggingFace providers, MLX, Gemini. "
                f"Available registered providers: {', '.join(_IMAGE_PROVIDER_REGISTRY.keys())}"
            )
    else:
        provider_name = provider

    factory = _IMAGE_PROVIDER_REGISTRY.get(provider_name)
    if factory is None:
        raise ValueError(
            f"Image provider '{provider_name}' is not registered. "
            f"Available providers: {list(_IMAGE_PROVIDER_REGISTRY.keys())}"
        )
    return factory()


def list_image_providers() -> list[str]:
    """List all registered image provider names.

    Returns:
        List of registered provider names
    """
    _load_providers()
    return list(_IMAGE_PROVIDER_REGISTRY.keys())
