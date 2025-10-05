#!/usr/bin/env python

import asyncio
from nodetool.config.logging_config import get_logger
from typing import List
from nodetool.metadata.types import LanguageModel
from nodetool.providers import list_providers

log = get_logger(__name__)


async def get_all_language_models() -> List[LanguageModel]:
    """
    Get all language models from all registered chat providers.

    This function discovers models by calling each registered chat provider's
    get_available_models() method. Each provider is responsible for
    checking API keys and returning appropriate models.

    Returns:
        List of all available LanguageModel instances from all providers
    """
    models = []

    for provider in list_providers():
        print(provider)
        provider_models = await provider.get_available_language_models()
        models.extend(provider_models)
        log.debug(
            f"Provider '{provider.provider_name}' returned {len(provider_models)} models"
        )

    log.info(
        f"Discovered {len(models)} total language models from {len(list_providers())} providers"
    )
    return models



if __name__ == "__main__":
    print(asyncio.run(get_all_language_models()))
