#!/usr/bin/env python

"""
Automatic Speech Recognition (ASR) model discovery and management.

This module provides functionality to discover and list all available ASR models
across all registered providers that support the AUTOMATIC_SPEECH_RECOGNITION capability.
"""

from nodetool.config.logging_config import get_logger
from typing import List
from nodetool.metadata.types import ASRModel, Provider

log = get_logger(__name__)


async def get_all_asr_models() -> List[ASRModel]:
    """
    Get all ASR models from all registered providers.

    This function discovers models by calling each registered provider's
    get_available_asr_models() method. Each provider is responsible for
    checking API keys and returning appropriate models.

    Returns:
        List of all available ASRModel instances from all providers
    """
    from nodetool.providers.base import (
        ProviderCapability,
    )
    from nodetool.providers import list_providers

    models = []
    for provider in list_providers():
        # Check if provider supports ASR
        capabilities = provider.get_capabilities()
        if ProviderCapability.AUTOMATIC_SPEECH_RECOGNITION not in capabilities:
            log.debug(
                f"Provider '{provider.provider_name}' does not support AUTOMATIC_SPEECH_RECOGNITION, skipping"
            )
            continue

        provider_models = await provider.get_available_asr_models()
        models.extend(provider_models)
        log.debug(
            f"Provider '{provider.provider_name}' returned {len(provider_models)} ASR models"
        )

    log.info(f"Discovered {len(models)} total ASR models")
    return models


if __name__ == "__main__":
    import asyncio

    print(asyncio.run(get_all_asr_models()))
