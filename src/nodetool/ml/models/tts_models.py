#!/usr/bin/env python

"""
Text-to-Speech model discovery and management.

This module provides functionality to discover and list all available TTS models
across all registered providers that support the TEXT_TO_SPEECH capability.
"""

import asyncio
import time
from typing import List

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import TTSModel

log = get_logger(__name__)

# Cache for TTS models (per user_id) with timestamp
_tts_models_cache: dict[str, tuple[list[TTSModel], float]] = {}
_tts_models_cache_lock = asyncio.Lock()
_TTS_MODELS_CACHE_TTL = 6 * 3600  # 6 hours in seconds


async def get_all_tts_models(user_id: str) -> list[TTSModel]:
    """
    Get all TTS models from all registered providers.
    Results are cached for 6 hours to reduce API calls.

    This function discovers models by calling each registered provider's
    get_available_tts_models() method. Each provider is responsible for
    checking API keys and returning appropriate models.

    Args:
        user_id: The user ID to get the TTS models for

    Returns:
        List of all available TTSModel instances from all providers
    """
    # Check cache first
    async with _tts_models_cache_lock:
        if user_id in _tts_models_cache:
            cached_models, cache_time = _tts_models_cache[user_id]
            if time.time() - cache_time < _TTS_MODELS_CACHE_TTL:
                log.debug(f"Returning cached TTS models for user {user_id}")
                return cached_models

    from nodetool.providers import list_providers

    models = []
    for provider in await list_providers(user_id):
        provider_models = await provider.get_available_tts_models()
        models.extend(provider_models)
        log.debug(f"Provider '{provider.provider_name}' returned {len(provider_models)} TTS models")

    log.info(f"Discovered {len(models)} total TTS models")

    # Cache the result
    async with _tts_models_cache_lock:
        _tts_models_cache[user_id] = (models, time.time())

    return models
