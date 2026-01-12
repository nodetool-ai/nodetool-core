"""
Provider module for multi-modal AI services.

This module provides a unified interface for AI service providers including
language models (OpenAI, Anthropic, Ollama) and image generation services
(DALL-E, Gemini, FAL, etc.). Providers declare their capabilities and
implement the corresponding methods.
"""

import asyncio
import os
import shutil
import subprocess
import sys
import threading
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

_SAFE_IMPORT_CACHE: dict[str, bool] = {}


def _safe_import_check(module_name: str) -> bool:
    """Return True if importing `module_name` appears safe in this process.

    Some optional providers can hard-crash the interpreter on import when native
    dependencies are missing or misconfigured. Probe importability in a
    subprocess to avoid taking down the current process.
    """
    cached = _SAFE_IMPORT_CACHE.get(module_name)
    if cached is not None:
        return cached

    try:
        result = subprocess.run(
            [sys.executable, "-c", f"import {module_name}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        ok = result.returncode == 0
    except Exception:
        ok = False

    _SAFE_IMPORT_CACHE[module_name] = ok
    return ok


def _is_llama_server_available() -> bool:
    """Check if llama-server binary is available in PATH or via environment variable.

    Returns:
        True if llama-server binary can be found, False otherwise.
    """
    # Check environment variable first (allows custom path)
    binary_name = os.environ.get("LLAMA_SERVER_BINARY", "llama-server")

    # If it's an absolute path, check if file exists
    if os.path.isabs(binary_name) or os.path.sep in binary_name:
        return os.path.isfile(binary_name) and os.access(binary_name, os.X_OK)

    # Otherwise, check if it's in PATH
    return shutil.which(binary_name) is not None


def import_providers():
    # import providers to ensure they are registered
    from nodetool.providers import (
        anthropic_provider,
        cerebras_provider,
        comfy_local_provider,
        comfy_runpod_provider,
        fake_provider,
        gemini_provider,
        huggingface_provider,
        kie_provider,
        lmstudio_provider,
        minimax_provider,
        ollama_provider,
        openai_provider,
        openrouter_provider,
        vllm_provider,
    )

    # Conditionally import llama_provider only if llama-server binary is available
    if _is_llama_server_available():
        try:
            from nodetool.providers import llama_provider  # type: ignore

            log.debug("Llama provider imported successfully (llama-server binary found)")
        except ImportError as e:
            log.warning(
                f"Llama provider could not be imported despite binary being available: {e}. "
                "Some llama.cpp features may be unavailable."
            )
        except Exception as e:
            log.warning(f"Unexpected error importing Llama provider: {e}. Some llama.cpp features may be unavailable.")
    else:
        log.debug(
            "Llama provider skipped: llama-server binary not found in PATH or LLAMA_SERVER_BINARY. "
            "Install llama.cpp to enable local LLM inference."
        )

    # Optional providers that may have missing dependencies
    # These are imported with better error handling and logging
    mlx_module = "nodetool.mlx.mlx_provider"
    if RUNNING_PYTEST:
        log.debug("Skipping MLX provider import under pytest")
    elif _safe_import_check(mlx_module):
        try:
            import nodetool.mlx.mlx_provider  # type: ignore

            log.debug("MLX provider imported successfully")
        except ImportError as e:
            log.warning(
                f"MLX provider could not be imported (some features may be unavailable): {e}. "
                "This is expected if optional MLX dependencies (e.g., mflux) are not installed. "
                "MLX language models can still be discovered from the HuggingFace cache."
            )
        except Exception as e:
            log.warning(
                f"Unexpected error importing MLX provider: {e}. "
                "MLX language models can still be discovered from the HuggingFace cache."
            )
    else:
        log.warning("MLX provider import skipped: module failed a safe-import probe (likely missing native deps).")

    try:
        import nodetool.huggingface.huggingface_local_provider  # type: ignore

        log.debug("HuggingFace local provider imported successfully")
    except ImportError as e:
        log.debug(f"HuggingFace local provider could not be imported: {e}")
    except Exception as e:
        log.warning(f"Unexpected error importing HuggingFace local provider: {e}")


# Provider instance cache
_provider_cache: dict[ProviderEnum, BaseProvider] = {}
_provider_cache_lock: asyncio.Lock | None = None
_provider_cache_locks: dict[int, asyncio.Lock] = {}
_provider_cache_locks_lock = threading.Lock()


def _get_provider_cache_lock() -> asyncio.Lock:
    """Get or create the provider cache lock lazily.

    This ensures the lock is created in the correct event loop.
    Multiple event loops (threads) each get their own lock to avoid
    the "locked by different event loop" error.
    """
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
    async with _get_provider_cache_lock():
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

    import_providers()

    # Get providers from the registry
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

    return providers


__all__ = [
    "BaseProvider",
    "Chunk",
    "FakeProvider",
    "MockProvider",
    "ProviderCapability",
    "create_fake_tool_call",
    "create_simple_fake_provider",
    "create_streaming_fake_provider",
    "create_tool_calling_fake_provider",
    "get_provider",
    "register_provider",
]
