"""
Provider module for multi-modal AI services.

This module provides a unified interface for AI service providers including
language models (OpenAI, Anthropic, Ollama) and image generation services
(DALL-E, Gemini, FAL, etc.). Providers declare their capabilities and
implement the corresponding methods.
"""

import threading

# Base provider class and testing utilities
from nodetool.providers.base import (
    BaseProvider,
    MockProvider,
    ProviderCapability,
    get_registered_provider,
    register_provider,
    _PROVIDER_REGISTRY,
)
from nodetool.providers.fake_provider import (
    FakeProvider,
    create_fake_tool_call,
    create_simple_fake_provider,
    create_streaming_fake_provider,
    create_tool_calling_fake_provider,
)
from nodetool.metadata.types import Provider as ProviderEnum
from nodetool.workflows.types import Chunk


def import_providers():
    # import providers to ensure they are registered
    from nodetool.providers import anthropic_provider
    from nodetool.providers import gemini_provider
    from nodetool.providers import llama_provider
    from nodetool.providers import ollama_provider
    from nodetool.providers import openai_provider
    from nodetool.providers import fake_provider
    from nodetool.providers import huggingface_provider

    # TODO: implement better discovery of providers
    try:
        import nodetool.mlx.mlx_provider # type: ignore
    except ImportError:
        pass

    try:
        import nodetool.huggingface.huggingface_local_provider # type: ignore
    except ImportError:
        pass


# Provider instance cache
_provider_cache: dict[ProviderEnum, BaseProvider] = {}
_provider_cache_lock = threading.Lock()


def get_provider(provider_type: ProviderEnum, **kwargs) -> BaseProvider:
    """
    Get a chat provider instance based on the provider type.
    Providers are cached after first creation.

    Args:
        provider_type: The provider type enum

    Returns:
        A chat provider instance

    Raises:
        ValueError: If the provider type is not supported
    """
    with _provider_cache_lock:
        if provider_type in _provider_cache:
            return _provider_cache[provider_type]

        import_providers()

        provider_cls, kwargs = get_registered_provider(provider_type)
        if provider_cls is None:
            raise ValueError(
                f"Provider {provider_type.value} is not available. Install the corresponding package via nodetool's package manager."
            )

        provider: BaseProvider = provider_cls(**kwargs)

        _provider_cache[provider_type] = provider
        return provider


def list_providers() -> list["BaseProvider"]:
    """List all registered providers."""
    import_providers()

    # Get models from each registered chat provider
    provider_enums = list(_PROVIDER_REGISTRY.keys())
    providers = []
    for provider_enum in provider_enums:
        provider_cls, kwargs = get_registered_provider(provider_enum)
        provider = provider_cls(**kwargs)
        providers.append(provider)
    return providers

__all__ = [
    "BaseProvider",
    "MockProvider",
    "ProviderCapability",
    "register_provider",
    "FakeProvider",
    "create_fake_tool_call",
    "create_simple_fake_provider",
    "create_streaming_fake_provider",
    "create_tool_calling_fake_provider",
    "Chunk",
    "get_provider",
]
