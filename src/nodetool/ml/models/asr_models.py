#!/usr/bin/env python

"""
Automatic Speech Recognition (ASR) model discovery and management.

This module provides functionality to discover and list all available ASR models
across all registered providers that support the AUTOMATIC_SPEECH_RECOGNITION capability.
"""

from typing import List

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import ASRModel

log = get_logger(__name__)


async def get_all_asr_models(user_id: str) -> list[ASRModel]:
    """
    Get all ASR models from all registered providers.
    Results are cached for 6 hours to reduce API calls.

    This function discovers models by calling each registered provider's
    get_available_asr_models() method. Each provider is responsible for
    checking API keys and returning appropriate models.

    Args:
        user_id: The user ID to get the ASR models for

    Returns:
        List of all available ASRModel instances from all providers
    """
    from nodetool.providers import list_providers

    models = []
    for provider in await list_providers(user_id):
        provider_models = await provider.get_available_asr_models()
        models.extend(provider_models)
        log.debug(f"Provider '{provider.provider_name}' returned {len(provider_models)} ASR models")

    log.info(f"Discovered {len(models)} total ASR models")

    return models
