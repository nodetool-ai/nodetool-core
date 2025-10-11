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

    Provider failures are handled gracefully - we cache whatever models
    we successfully retrieve, even if some providers fail.

    Returns:
        List of all available LanguageModel instances from all providers
    """
    log.info("🔍 TRACE: get_all_language_models() CALLED")

    # Check cache first
    cache_key = "language_models:all"
    cached_models = _model_cache.get(cache_key)
    if cached_models is not None:
        log.info(f"Returning {len(cached_models)} cached language models")
        return cached_models

    models = []
    successful_providers = 0
    failed_providers = []

    for provider in list_providers():
        try:
            provider_models = await provider.get_available_language_models()
            models.extend(provider_models)
            successful_providers += 1
            log.debug(
                f"✓ Provider '{provider.provider_name}' returned {len(provider_models)} models"
            )
        except Exception as e:
            failed_providers.append(provider.provider_name)
            log.warning(
                f"✗ Failed to get models from provider '{provider.provider_name}': {e}"
            )

    log.info(
        f"Discovered {len(models)} total language models from {successful_providers}/{len(list_providers())} providers"
    )

    if failed_providers:
        log.warning(f"Failed providers: {', '.join(failed_providers)}")

    # Cache the results even if some providers failed
    # This prevents repeated failures from slowing down requests
    if models:  # Only cache if we got at least some models
        log.debug(f"About to cache {len(models)} language models with key: {cache_key}")
        try:
            _model_cache.set(cache_key, models)
            log.info(f"✓ Successfully cached {len(models)} language models from {successful_providers} providers")
        except Exception as e:
            log.error(f"✗ Failed to cache language models: {e}", exc_info=True)
    else:
        log.warning("No models retrieved from any provider - skipping cache to allow retry")

    return models



if __name__ == "__main__":
    print(asyncio.run(get_all_language_models()))
