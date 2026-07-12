#!/usr/bin/env python


import asyncio

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import VideoModel
from nodetool.providers import list_providers

log = get_logger(__name__)


async def get_all_video_models(user_id: str) -> list[VideoModel]:
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
    models = []

    # Get models from each registered provider
    providers = await list_providers(user_id)
    log.debug(f"Discovering video models from {len(providers)} providers: {providers}")

    if not providers:
        return models

    for provider in providers:
        print(f"Getting video models from provider: {provider.provider_name}")

    tasks = [provider.get_available_video_models() for provider in providers]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for provider, result in zip(providers, results, strict=False):
        if isinstance(result, Exception):
            log.error(f"Error getting video models from {provider.provider_name}: {result}")
        else:
            models.extend(result)
            log.debug(f"Provider '{provider.provider_name}' returned {len(result)} video models")

    log.info(f"Discovered {len(models)} total video models from {len(providers)} providers")

    return models
