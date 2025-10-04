#!/usr/bin/env python

import asyncio
from nodetool.config.logging_config import get_logger
from typing import List, Any
import aiohttp
from nodetool.metadata.types import ImageModel, Provider
import time

log = get_logger(__name__)

# Cache TTL: 60 minutes = 3600 seconds
CACHE_TTL = 3600


class ImageModelCache:
    """
    A class to manage caching of image models.
    """

    def __init__(self):
        self.cache = {}

    def get(self, key: str) -> Any:
        if key in self.cache:
            value, expiry_time = self.cache[key]
            if expiry_time is None or time.time() < expiry_time:
                return value
            else:
                del self.cache[key]  # Remove expired entry
        return None

    def set(self, key: str, value: Any, ttl: int = 3600):
        expiry_time = time.time() + ttl
        self.cache[key] = (value, expiry_time)

    def clear(self):
        self.cache.clear()


# Dedicated in-memory cache for image models
image_model_cache = ImageModelCache()


async def get_all_image_models() -> List[ImageModel]:
    """
    Get all image models from all registered providers.

    This function discovers models by calling each registered provider's
    get_available_models() method. Each provider is responsible for
    checking API keys and returning appropriate models.

    Returns:
        List of all available ImageModel instances from all providers
    """
    from nodetool.image.providers.registry import (
        list_image_providers,
        get_image_provider,
    )

    models = []

    # Load all providers to ensure they're registered
    try:
        import nodetool.image.providers  # noqa: F401
    except ImportError:
        pass

    # Get models from each registered provider
    provider_names = list_image_providers()
    log.debug(
        f"Discovering models from {len(provider_names)} providers: {provider_names}"
    )

    for provider_name in provider_names:
        try:
            provider = get_image_provider(provider_name)
            provider_models = await provider.get_available_image_models()
            models.extend(provider_models)
            log.debug(
                f"Provider '{provider_name}' returned {len(provider_models)} models"
            )
        except Exception as e:
            # Don't fail if one provider fails - just log and continue
            log.warning(f"Failed to get models from provider '{provider_name}': {e}")
            continue

    log.info(
        f"Discovered {len(models)} total image models from {len(provider_names)} providers"
    )
    return models


def clear_image_model_cache() -> None:
    """
    Clear the in-memory image model cache.

    This will force the next call to get_cached_hf_image_models() to fetch fresh data from the API.
    """
    image_model_cache.clear()
    log.info("Image model cache cleared")
