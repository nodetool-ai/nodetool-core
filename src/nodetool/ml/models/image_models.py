#!/usr/bin/env python

import asyncio

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import ImageModel
from nodetool.providers import list_providers

log = get_logger(__name__)


async def get_all_image_models(user_id: str) -> list[ImageModel]:
    """
    Get all image models from all registered providers.
    Results are cached for 6 hours to reduce API calls.

    This function discovers models by calling each registered provider's
    get_available_models() method. Each provider is responsible for
    checking API keys and returning appropriate models.

    Args:
        user_id: The user ID to get the image models for

    Returns:
        List of all available ImageModel instances from all providers
    """
    models = []

    # Get models from each registered provider
    providers = await list_providers(user_id)
    log.debug(f"Discovering models from {len(providers)} providers: {providers}")

    for provider in providers:
        provider_models = await provider.get_available_image_models()
        models.extend(provider_models)
        log.debug(f"Provider '{provider.provider_name}' returned {len(provider_models)} models")

    log.info(f"Discovered {len(models)} total image models from {len(providers)} providers")

    return models


if __name__ == "__main__":
    import asyncio

    models = asyncio.run(get_all_image_models(user_id="1"))
    print(models)
