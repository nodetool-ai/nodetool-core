"""
Provider module for multi-modal AI services.

This module provides a unified interface for AI service providers including
language models (OpenAI, Anthropic, Ollama) and image generation services
(DALL-E, Gemini, FAL, etc.). Providers declare their capabilities and
implement the corresponding methods.
"""

import asyncio
import time
from typing import Optional

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
from nodetool.security.secret_helper import get_secret, get_secrets_batch


def import_providers():
    # import providers to ensure they are registered
    from nodetool.providers import (  # noqa: F401
        anthropic_provider,
        gemini_provider,
        llama_provider,
        comfy_provider,
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
_provider_cache_lock = asyncio.Lock()

# Provider list cache (per user_id) with timestamp
_provider_list_cache: dict[str, tuple[list["BaseProvider"], float]] = {}
_provider_list_cache_lock = asyncio.Lock()
_PROVIDER_LIST_CACHE_TTL = 60.0  # Cache for 60 seconds


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
    async with _provider_cache_lock:
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
    """
    List all registered providers for a given user.

    Results are cached for 60 seconds to avoid repeated database queries
    for secrets lookup.

    Args:
        user_id: The user ID to get providers for

    Returns:
        List of initialized provider instances for this user
    """
    import logging
    logger = logging.getLogger(__name__)

    # Check cache first
    async with _provider_list_cache_lock:
        if user_id in _provider_list_cache:
            cached_providers, cache_time = _provider_list_cache[user_id]
            if time.time() - cache_time < _PROVIDER_LIST_CACHE_TTL:
                logger.debug(f"Returning cached providers for user {user_id}")
                return cached_providers

    import_providers()

    # Get models from each registered chat provider
    provider_enums = list[ProviderEnum](_PROVIDER_REGISTRY.keys())

    # Collect all required secrets across all providers
    all_required_secrets = set()
    provider_secret_map = {}  # provider_enum -> list of required secrets
    for provider_enum in provider_enums:
        provider_cls, kwargs = get_registered_provider(provider_enum)
        required_secrets = provider_cls.required_secrets()
        provider_secret_map[provider_enum] = (provider_cls, kwargs, required_secrets)
        all_required_secrets.update(required_secrets)

    # Batch fetch all secrets in one query
    if all_required_secrets:
        secrets_dict = await get_secrets_batch(list(all_required_secrets), user_id)
    else:
        secrets_dict = {}

    # Initialize providers with their secrets
    providers = []
    for provider_enum, (provider_cls, kwargs, required_secrets) in provider_secret_map.items():
        # Collect this provider's secrets
        provider_secrets = {}
        for secret in required_secrets:
            secret_value = secrets_dict.get(secret)
            if secret_value:
                provider_secrets[secret] = secret_value

        # Skip provider if required secrets are missing
        if len(required_secrets) > 0 and len(provider_secrets) == 0:
            logger.debug(f"Skipping provider {provider_enum.value}: missing required secrets {required_secrets}")
            continue

        # Initialize and register provider
        provider = provider_cls(secrets=provider_secrets, **kwargs)
        providers.append(provider)

    # Cache the result
    async with _provider_list_cache_lock:
        _provider_list_cache[user_id] = (providers, time.time())

    logger.debug(f"Cached {len(providers)} providers for user {user_id}")
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
