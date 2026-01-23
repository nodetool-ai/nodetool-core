#!/usr/bin/env python

import asyncio

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import LanguageModel
from nodetool.providers import list_providers

log = get_logger(__name__)


async def get_all_language_models(user_id: str) -> list[LanguageModel]:
    """
    Get all language models from all registered chat providers.
    Results are cached for 6 hours to reduce API calls.

    This function discovers models by calling each registered chat provider's
    get_available_models() method concurrently using asyncio.gather().
    Each provider is responsible for checking API keys and returning appropriate models.

    Provider failures are handled gracefully - we use return_exceptions=True
    to ensure one failing provider doesn't cancel others.

    Args:
        user_id: The user ID to get the language models for

    Returns:
        List of all available LanguageModel instances from all providers
    """
    providers = await list_providers(user_id)

    if not providers:
        log.warning("No providers available")
        return []

    # Fetch models from all providers concurrently
    # return_exceptions=True ensures one failure doesn't cancel others
    results = await asyncio.gather(
        *(provider.get_available_language_models() for provider in providers),
        return_exceptions=True,
    )

    models: list[LanguageModel] = []
    successful_providers = 0
    failed_providers: list[str] = []

    for provider, result in zip(providers, results, strict=True):
        if isinstance(result, Exception):
            failed_providers.append(provider.provider_name)
            log.warning(f"Failed to get models from provider '{provider.provider_name}': {result}")
        else:
            models.extend(result)
            successful_providers += 1
            log.debug(f"Provider '{provider.provider_name}' returned {len(result)} models")

    log.info(f"Discovered {len(models)} total language models from {successful_providers}/{len(providers)} providers")

    if failed_providers:
        log.warning(f"Failed providers: {', '.join(failed_providers)}")

    return models
