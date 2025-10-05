#!/usr/bin/env python

"""
Text-to-Speech model discovery and management.

This module provides functionality to discover and list all available TTS models
across all registered providers that support the TEXT_TO_SPEECH capability.
"""

from nodetool.config.logging_config import get_logger
from typing import List
from nodetool.metadata.types import TTSModel, Provider
import aiohttp

log = get_logger(__name__)



async def get_all_tts_models() -> List[TTSModel]:
    """
    Get all TTS models from all registered providers.

    This function discovers models by calling each registered provider's
    get_available_tts_models() method. Each provider is responsible for
    checking API keys and returning appropriate models.

    Returns:
        List of all available TTSModel instances from all providers
    """
    from nodetool.providers.base import (
        ProviderCapability,
    )
    from nodetool.providers import list_providers
    models = []
    for provider in list_providers():
        # Check if provider supports TTS
        capabilities = provider.get_capabilities()
        if ProviderCapability.TEXT_TO_SPEECH not in capabilities:
            log.debug(
                f"Provider '{provider.provider_name}' does not support TEXT_TO_SPEECH, skipping"
            )
            continue

        provider_models = await provider.get_available_tts_models()
        models.extend(provider_models)
        log.debug(
            f"Provider '{provider.provider_name}' returned {len(provider_models)} TTS models"
        )

    log.info(f"Discovered {len(models)} total TTS models")
    return models


if __name__ == "__main__":
    import asyncio

    print(asyncio.run(get_all_tts_models()))
