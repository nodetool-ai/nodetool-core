#!/usr/bin/env python

import asyncio
from nodetool.config.logging_config import get_logger
from typing import List
from nodetool.metadata.types import ImageModel

from nodetool.providers import list_providers
from nodetool.ml.models.model_cache import _model_cache

log = get_logger(__name__)


async def get_all_image_models() -> List[ImageModel]:
    """
    Get all image models from all registered providers.
    Results are cached for 6 hours to reduce API calls.

    This function discovers models by calling each registered provider's
    get_available_models() method. Each provider is responsible for
    checking API keys and returning appropriate models.

    Returns:
        List of all available ImageModel instances from all providers
    """
    # Check cache first
    cache_key = "image_models:all"
    cached_models = _model_cache.get(cache_key)
    if cached_models is not None:
        log.info(f"Returning {len(cached_models)} cached image models")
        return cached_models

    models = []

    # Get models from each registered provider
    providers = list_providers()
    log.debug(f"Discovering models from {len(providers)} providers: {providers}")

    for provider in providers:
        provider_models = await provider.get_available_image_models()
        models.extend(provider_models)
        log.debug(
            f"Provider '{provider.provider_name}' returned {len(provider_models)} models"
        )

    log.info(
        f"Discovered {len(models)} total image models from {len(providers)} providers"
    )

    # Cache the results
    _model_cache.set(cache_key, models)

    return models


if __name__ == "__main__":
    import asyncio

    models = asyncio.run(get_all_image_models())
    print(models)
