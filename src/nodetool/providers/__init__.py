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
from nodetool.security.secret_helper import get_secret


def import_providers():
    # import providers to ensure they are registered
    from nodetool.providers import (  # noqa: F401
        anthropic_provider,
        gemini_provider,
        llama_provider,
        ollama_provider,
        openai_provider,
        fake_provider,
        huggingface_provider,
        vllm_provider,
    )

    # TODO: implement better discovery of providers
    try:
        import nodetool.mlx.mlx_provider  # type: ignore  # noqa: F401
    except ImportError:
        pass

    try:
        import nodetool.huggingface.huggingface_local_provider  # type: ignore  # noqa: F401
    except ImportError:
        pass


# Provider instance cache
_provider_cache: dict[ProviderEnum, BaseProvider] = {}
_provider_cache_lock = threading.Lock()


async def get_provider(provider_type: ProviderEnum, user_id: str = "1", **kwargs) -> BaseProvider:
    """
    Get a chat provider instance based on the provider type.
    Providers are cached after first creation.

    Args:
        provider_type: The provider type enum
        user_id: The user ID
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

        required_secrets = provider_cls.required_secrets()
        secrets = {}
        for secret in required_secrets:
            secrets[secret] = await get_secret(secret, user_id)

        _provider_cache[provider_type] = provider_cls(secrets=secrets, **kwargs)
        return _provider_cache[provider_type]


async def list_providers(user_id: str) -> list["BaseProvider"]:
    """List all registered providers for a given user."""
    import_providers()

    # Get models from each registered chat provider
    provider_enums = list(_PROVIDER_REGISTRY.keys())
    providers = []
    for provider_enum in provider_enums:
        provider_cls, kwargs = get_registered_provider(provider_enum)
        required_secrets = provider_cls.required_secrets()
        secrets = {}
        for secret in required_secrets:
            secret_value = await get_secret(secret, user_id)
            if not secret_value:
                continue
            secrets[secret] = secret_value
        if len(required_secrets) > 0 and len(secrets) == 0:
            continue
        provider = provider_cls(secrets=secrets, **kwargs)
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
