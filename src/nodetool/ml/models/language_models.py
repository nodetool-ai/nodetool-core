#!/usr/bin/env python

import asyncio
from nodetool.config.logging_config import get_logger
from typing import List
from nodetool.metadata.types import LanguageModel
from nodetool.providers import list_providers
from nodetool.ml.models.model_cache import _model_cache

log = get_logger(__name__)


async def get_all_language_models() -> List[LanguageModel]:
    """
    Get all language models from all registered chat providers.
    Results are cached for 6 hours to reduce API calls.

    This function discovers models by calling each registered chat provider's
    get_available_models() method. Each provider is responsible for
    checking API keys and returning appropriate models.

    Returns:
        List of all available LanguageModel instances from all providers
    """
    # Check cache first
    cache_key = "language_models:all"
    cached_models = _model_cache.get(cache_key)
    if cached_models is not None:
        log.info(f"Returning {len(cached_models)} cached language models")
        return cached_models

    models = []

    for provider in list_providers():
        print(provider)
        provider_models = await provider.get_available_language_models()
        models.extend(provider_models)
        log.debug(
            f"Provider '{provider.provider_name}' returned {len(provider_models)} models"
        )

    log.info(
        f"Discovered {len(models)} total language models from {len(list_providers())} providers"
    )

    # Cache the results
    _model_cache.set(cache_key, models)

    return models



if __name__ == "__main__":
    print(asyncio.run(get_all_language_models()))
