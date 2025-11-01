#!/usr/bin/env python

import asyncio
from nodetool.config.logging_config import get_logger
from typing import List
from nodetool.metadata.types import VideoModel

from nodetool.providers import list_providers
from nodetool.ml.models.model_cache import _model_cache

log = get_logger(__name__)


async def get_all_video_models(user_id: str) -> List[VideoModel]:
    """
    Get all video models from all registered providers.
    Results are cached for 6 hours to reduce API calls.

    This function discovers models by calling each registered provider's
    get_available_video_models() method. Each provider is responsible for
    checking API keys and returning appropriate models.

    Args:
        user_id: The user ID to get the video models for

    Returns:
        List of all available VideoModel instances from all providers
    """
    # Check cache first
    cache_key = f"video_models:all:{user_id}"
    cached_models = _model_cache.get(cache_key)
    if cached_models is not None:
        log.info(f"Returning {len(cached_models)} cached video models")
        return cached_models

    models = []

    # Get models from each registered provider
    providers = await list_providers(user_id)
    log.debug(f"Discovering video models from {len(providers)} providers: {providers}")

    for provider in providers:
        print(f"Getting video models from provider: {provider.provider_name}")
        provider_models = await provider.get_available_video_models()
        models.extend(provider_models)
        log.debug(
            f"Provider '{provider.provider_name}' returned {len(provider_models)} video models"
        )

    log.info(
        f"Discovered {len(models)} total video models from {len(providers)} providers"
    )

    # Cache the results
    _model_cache.set(cache_key, models)

    return models
