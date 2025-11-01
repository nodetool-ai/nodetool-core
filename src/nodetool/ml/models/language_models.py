#!/usr/bin/env python

import asyncio
from nodetool.config.logging_config import get_logger
from typing import List
from nodetool.metadata.types import LanguageModel
from nodetool.providers import list_providers

log = get_logger(__name__)


async def get_all_language_models(user_id: str) -> List[LanguageModel]:
    """
    Get all language models from all registered chat providers.
    Results are cached for 6 hours to reduce API calls.

    This function discovers models by calling each registered chat provider's
    get_available_models() method. Each provider is responsible for
    checking API keys and returning appropriate models.

    Provider failures are handled gracefully - we cache whatever models
    we successfully retrieve, even if some providers fail.

    Args:
        user_id: The user ID to get the language models for

    Returns:
        List of all available LanguageModel instances from all providers
    """
    models = []
    successful_providers = 0
    failed_providers = []
    providers = await list_providers(user_id)

    for provider in providers:
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
        f"Discovered {len(models)} total language models from {successful_providers}/{len(providers)} providers"
    )

    if failed_providers:
        log.warning(f"Failed providers: {', '.join(failed_providers)}")

    return models