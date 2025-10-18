#!/usr/bin/env python

"""
Text-to-Speech model discovery and management.

This module provides functionality to discover and list all available TTS models
across all registered providers that support the TEXT_TO_SPEECH capability.
"""

from nodetool.config.logging_config import get_logger
from typing import List
from nodetool.metadata.types import TTSModel
from nodetool.ml.models.model_cache import _model_cache

log = get_logger(__name__)


async def get_all_tts_models() -> List[TTSModel]:
    """
    Get all TTS models from all registered providers.
    Results are cached for 6 hours to reduce API calls.

    This function discovers models by calling each registered provider's
    get_available_tts_models() method. Each provider is responsible for
    checking API keys and returning appropriate models.

    Returns:
        List of all available TTSModel instances from all providers
    """
    # Check cache first
    cache_key = "tts_models:all"
    cached_models = _model_cache.get(cache_key)
    if cached_models is not None:
        log.info(f"Returning {len(cached_models)} cached TTS models")
        return cached_models

    from nodetool.providers import list_providers

    models = []
    for provider in list_providers():
        provider_models = await provider.get_available_tts_models()
        models.extend(provider_models)
        log.debug(
            f"Provider '{provider.provider_name}' returned {len(provider_models)} TTS models"
        )

    log.info(f"Discovered {len(models)} total TTS models")

    # Cache the results
    _model_cache.set(cache_key, models)

    return models


if __name__ == "__main__":
    import asyncio

    print(asyncio.run(get_all_tts_models()))
