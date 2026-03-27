"""
Provider module for AI services.

This module provides the provider infrastructure (base classes, registry,
caching) used by Python-only providers (HuggingFace Local, MLX).
Cloud/API providers (OpenAI, Anthropic, Gemini, etc.) are implemented
in the TypeScript server.
"""
import asyncio
import sys
import threading
import traceback
from typing import Optional

from nodetool.config.env_guard import RUNNING_PYTEST
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import Provider as ProviderEnum

# Base provider class and testing utilities
from nodetool.providers.base import (
    _PROVIDER_REGISTRY,
    BaseProvider,
    MockProvider,
    ProviderCapability,
    get_registered_provider,
    register_provider,
)
from nodetool.providers.fake_provider import (
    FakeProvider,
    create_fake_tool_call,
    create_simple_fake_provider,
    create_streaming_fake_provider,
    create_tool_calling_fake_provider,
)
from nodetool.security.secret_helper import get_secret, get_secrets_batch
from nodetool.workflows.types import Chunk

log = get_logger(__name__)

_providers_imported = False


def import_providers():
    global _providers_imported
    if _providers_imported:
        return

    _providers_imported = True

    # Import Python-only providers (local compute)
    # Cloud/API providers are handled by the TypeScript server.

    if RUNNING_PYTEST:
        log.debug("Skipping MLX/FAL provider import under pytest")
    else:
        try:
            import nodetool.fal.fal_provider  # type: ignore

            log.debug("FAL provider imported successfully")
        except ImportError as e:
            log.debug(f"FAL provider not available: {e}")
        except Exception as e:
            traceback.print_exc()
            log.warning(f"Unexpected error importing FAL provider: {e}")

        try:
            import nodetool.mlx.mlx_provider  # type: ignore

            log.debug("MLX provider imported successfully")
        except ImportError as e:
            log.debug(f"MLX provider not available: {e}")
        except Exception as e:
            log.warning(f"Unexpected error importing MLX provider: {e}")

    try:
        import nodetool.huggingface.huggingface_local_provider  # type: ignore

        log.debug("HuggingFace local provider imported successfully")
    except ImportError as e:
        log.debug(f"HuggingFace local provider not available: {e}")
    except Exception as e:
        log.warning(f"Unexpected error importing HuggingFace local provider: {e}")


# Provider instance cache
_provider_cache: dict[ProviderEnum, BaseProvider] = {}
_provider_cache_lock: asyncio.Lock | None = None
_provider_cache_locks: dict[int, asyncio.Lock] = {}
_provider_cache_locks_lock = threading.Lock()


def clear_provider_cache() -> int:
    global _provider_cache
    count = len(_provider_cache)
    _provider_cache.clear()
    if count > 0:
        log.info(f"Cleared {count} providers from cache")
    return count


def _get_provider_cache_lock() -> asyncio.Lock:
    """Get or create the provider cache lock lazily."""
    global _provider_cache_lock
    if _provider_cache_lock is None:
        _provider_cache_lock = asyncio.Lock()

    loop_id = id(asyncio.get_running_loop())

    with _provider_cache_locks_lock:
        if loop_id not in _provider_cache_locks:
            _provider_cache_locks[loop_id] = asyncio.Lock()
        return _provider_cache_locks[loop_id]


async def get_provider(provider_type: ProviderEnum, user_id: str = "1", **kwargs) -> BaseProvider:
    """
    Get a provider instance based on the provider type.
    Providers are cached after first creation.
    """
    async with _get_provider_cache_lock():
        if provider_type in _provider_cache:
            return _provider_cache[provider_type]

        import_providers()

        provider_cls, kwargs = get_registered_provider(provider_type)
        if provider_cls is None:
            raise ValueError(
                f"Provider {provider_type.value} is not available. "
                "Cloud providers are handled by the TS server; "
                "local providers require the corresponding Python package."
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
    """
    import logging

    logger = logging.getLogger(__name__)

    import_providers()

    provider_enums = list[ProviderEnum](_PROVIDER_REGISTRY.keys())

    all_required_secrets = set()
    provider_secret_map = {}
    for provider_enum in provider_enums:
        provider_cls, kwargs = get_registered_provider(provider_enum)
        required_secrets = provider_cls.required_secrets()
        provider_secret_map[provider_enum] = (provider_cls, kwargs, required_secrets)
        all_required_secrets.update(required_secrets)

    if all_required_secrets:
        secrets_dict = await get_secrets_batch(list(all_required_secrets), user_id)
    else:
        secrets_dict = {}

    providers = []
    for provider_enum, (provider_cls, kwargs, required_secrets) in provider_secret_map.items():
        provider_secrets = {}
        for secret in required_secrets:
            secret_value = secrets_dict.get(secret)
            if secret_value:
                provider_secrets[secret] = secret_value

        if len(required_secrets) > 0 and len(provider_secrets) == 0:
            logger.debug(f"Skipping provider {provider_enum.value}: missing required secrets {required_secrets}")
            continue

        provider = provider_cls(secrets=provider_secrets, **kwargs)
        providers.append(provider)

    return providers


__all__ = [
    "BaseProvider",
    "Chunk",
    "FakeProvider",
    "MockProvider",
    "ProviderCapability",
    "clear_provider_cache",
    "create_fake_tool_call",
    "create_simple_fake_provider",
    "create_streaming_fake_provider",
    "create_tool_calling_fake_provider",
    "get_provider",
    "register_provider",
]
