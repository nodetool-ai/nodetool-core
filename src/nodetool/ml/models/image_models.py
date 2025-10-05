#!/usr/bin/env python

import asyncio
from nodetool.config.logging_config import get_logger
from typing import List, Any
import aiohttp
from nodetool.metadata.types import ImageModel, Provider
import time

from nodetool.providers import list_providers

log = get_logger(__name__)



async def get_all_image_models() -> List[ImageModel]:
    """
    Get all image models from all registered providers.

    This function discovers models by calling each registered provider's
    get_available_models() method. Each provider is responsible for
    checking API keys and returning appropriate models.

    Returns:
        List of all available ImageModel instances from all providers
    """
    models = []

    # Get models from each registered provider
    providers = list_providers()
    log.debug(
        f"Discovering models from {len(providers)} providers: {providers}"
    )

    for provider in providers:
        provider_models = await provider.get_available_image_models()
        models.extend(provider_models)
        log.debug(
            f"Provider '{provider.provider_name}' returned {len(provider_models)} models"
        )

    log.info(
        f"Discovered {len(models)} total image models from {len(providers)} providers"
    )
    return models