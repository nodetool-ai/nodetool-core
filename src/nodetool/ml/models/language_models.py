#!/usr/bin/env python

import asyncio
from nodetool.config.logging_config import get_logger
from typing import List, Dict, Any
import aiohttp
from nodetool.metadata.types import LanguageModel, Provider
from nodetool.integrations.huggingface.huggingface_models import (
    get_mlx_language_models_from_hf_cache,
)

log = get_logger(__name__)

# Cache TTL: 60 minutes = 3600 seconds
CACHE_TTL = 3600

from typing import Any
import time


class LanguageModelCache:
    """
    A class to manage caching of language models.
    """

    def __init__(self):
        self.cache = {}

    def get(self, key: str) -> Any:
        if key in self.cache:
            value, expiry_time = self.cache[key]
            if expiry_time is None or time.time() < expiry_time:
                return value
            else:
                del self.cache[key]  # Remove expired entry
        return None

    def set(self, key: str, value: Any, ttl: int = 3600):
        expiry_time = time.time() + ttl
        self.cache[key] = (value, expiry_time)

    def clear(self):
        self.cache.clear()


# Dedicated in-memory cache for language models
_language_model_cache = LanguageModelCache()

# Provider mapping for HuggingFace Hub API
HF_PROVIDER_MAPPING = {
    "black-forest-labs": Provider.HuggingFaceBlackForestLabs,
    "cerebras": Provider.HuggingFaceCerebras,
    "cohere": Provider.HuggingFaceCohere,
    "fal-ai": Provider.HuggingFaceFalAI,
    "featherless-ai": Provider.HuggingFaceFeatherlessAI,
    "fireworks-ai": Provider.HuggingFaceFireworksAI,
    "groq": Provider.HuggingFaceGroq,
    "hf-inference": Provider.HuggingFaceHFInference,
    "hyperbolic": Provider.HuggingFaceHyperbolic,
    "nebius": Provider.HuggingFaceNebius,
    "novita": Provider.HuggingFaceNovita,
    "nscale": Provider.HuggingFaceNscale,
    "openai": Provider.HuggingFaceOpenAI,
    "replicate": Provider.HuggingFaceReplicate,
    "sambanova": Provider.HuggingFaceSambanova,
    "together": Provider.HuggingFaceTogether,
}


async def fetch_models_from_hf_provider(provider: str) -> List[LanguageModel]:
    """
    Fetch models from HuggingFace Hub API for a specific provider.

    Args:
        provider: The provider value (e.g., "groq", "cerebras", etc.)

    Returns:
        List of LanguageModel instances
    """
    try:
        url = f"https://huggingface.co/api/models?inference_provider={provider}&pipeline_tag=text-generation&limit=1000"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    models_data = await response.json()

                    models = []
                    for model_data in models_data:
                        model_id = model_data.get("id", "")
                        if model_id:
                            # Use the model name from the API if available, otherwise use the ID
                            model_name = (
                                model_data.get("name") or model_id.split("/")[-1]
                                if "/" in model_id
                                else model_id
                            )

                            # Get the appropriate provider enum value
                            provider_enum = HF_PROVIDER_MAPPING.get(provider)
                            if provider_enum is None:
                                log.warning(
                                    f"Unknown provider: {provider}, skipping model: {model_id}"
                                )
                                continue

                            models.append(
                                LanguageModel(
                                    id=model_id,
                                    name=model_name,
                                    provider=provider_enum,
                                )
                            )

                    # Preserve API order to match test expectations
                    log.debug(
                        f"Fetched {len(models)} models from HF provider: {provider}"
                    )
                    return models
                else:
                    log.warning(
                        f"Failed to fetch models for provider {provider}: HTTP {response.status}"
                    )
                    return []

    except Exception as e:
        log.error(f"Error fetching models for provider {provider}: {e}")
        return []


async def get_all_language_models() -> List[LanguageModel]:
    """
    Get all language models from all registered chat providers.

    This function discovers models by calling each registered chat provider's
    get_available_models() method. Each provider is responsible for
    checking API keys and returning appropriate models.

    Returns:
        List of all available LanguageModel instances from all providers
    """
    from nodetool.chat.providers.base import (
        _CHAT_PROVIDER_REGISTRY,
        get_registered_provider,
    )

    models = []

    # Load all chat providers to ensure they're registered
    # This triggers the @register_chat_provider decorators
    from nodetool.chat.providers import import_providers

    import_providers()

    # Get models from each registered chat provider
    provider_enums = list(_CHAT_PROVIDER_REGISTRY.keys())
    log.debug(
        f"Discovering models from {len(provider_enums)} chat providers: {provider_enums}"
    )

    for provider_enum in provider_enums:
        try:
            provider_cls, kwargs = get_registered_provider(provider_enum)
            provider = provider_cls(**kwargs)
            provider_models = await provider.get_available_language_models()
            models.extend(provider_models)
            log.debug(
                f"Provider '{provider_enum}' returned {len(provider_models)} models"
            )
        except Exception as e:
            # Don't fail if one provider fails - just log and continue
            log.warning(f"Failed to get models from provider '{provider_enum}': {e}")
            continue

    log.info(
        f"Discovered {len(models)} total language models from {len(provider_enums)} providers"
    )
    return models


def clear_language_model_cache() -> None:
    """
    Clear the in-memory language model cache.

    This will force the next call to get_cached_hf_models() to fetch fresh data from the API.
    """
    _language_model_cache.clear()
    log.info("Language model cache cleared")


if __name__ == "__main__":
    print(asyncio.run(get_all_language_models()))
