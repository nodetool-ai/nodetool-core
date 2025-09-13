#!/usr/bin/env python

import asyncio
from nodetool.config.logging_config import get_logger
from typing import List, Dict, Any
import aiohttp
from nodetool.metadata.types import LanguageModel, Provider
from nodetool.config.environment import Environment
from nodetool.storage.memory_node_cache import MemoryNodeCache

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

# Hardcoded models for providers that don't support HF Hub API
anthropic_models = [
    LanguageModel(
        id="claude-3-5-haiku-latest",
        name="Claude 3.5 Haiku",
        provider=Provider.Anthropic,
    ),
    LanguageModel(
        id="claude-3-5-sonnet-latest",
        name="Claude 3.5 Sonnet",
        provider=Provider.Anthropic,
    ),
    LanguageModel(
        id="claude-3-7-sonnet-latest",
        name="Claude 3.7 Sonnet",
        provider=Provider.Anthropic,
    ),
    LanguageModel(
        id="claude-sonnet-4-20250514",
        name="Claude Sonnet 4",
        provider=Provider.Anthropic,
    ),
    LanguageModel(
        id="claude-opus-4-20250514",
        name="Claude Opus 4",
        provider=Provider.Anthropic,
    ),
]

gemini_models = [
    LanguageModel(
        id="gemini-2.5-pro-exp-03-25",
        name="Gemini 2.5 Pro Experimental",
        provider=Provider.Gemini,
    ),
    LanguageModel(
        id="gemini-2.5-flash-preview-04-17",
        name="Gemini 2.5 Flash",
        provider=Provider.Gemini,
    ),
    LanguageModel(
        id="gemini-2.0-flash",
        name="Gemini 2.0 Flash",
        provider=Provider.Gemini,
    ),
    LanguageModel(
        id="gemini-2.0-flash-lite",
        name="Gemini 2.0 Flash Lite",
        provider=Provider.Gemini,
    ),
    LanguageModel(
        id="gemini-2.0-flash-exp-image-generation",
        name="Gemini 2.0 Flash Exp Image Generation",
        provider=Provider.Gemini,
    ),
]

# Keep a small static OpenAI list for compatibility (tests import this symbol),
# but runtime aggregation uses dynamic fetching when OPENAI_API_KEY is set.
openai_models = [
    LanguageModel(
        id="gpt-4o",
        name="GPT-4o",
        provider=Provider.OpenAI,
    ),
    LanguageModel(
        id="gpt-4o-mini",
        name="GPT-4o Mini",
        provider=Provider.OpenAI,
    ),
]

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
                            provider_enum = HF_PROVIDER_MAPPING.get(
                                provider, Provider.HuggingFace
                            )

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


async def get_cached_hf_models() -> List[LanguageModel]:
    """
    Get HuggingFace models from in-memory cache or fetch them dynamically.

    Returns:
        List of LanguageModel instances from HuggingFace providers
    """
    cache_key = "hf_language_models"

    # Try to get from cache first
    cached_models = _language_model_cache.get(cache_key)
    if cached_models is not None:
        log.debug("Using cached HuggingFace models")
        return cached_models

    log.debug("Fetching HuggingFace models from API")

    # List of providers to fetch from
    providers = [
        "black-forest-labs",
        "cerebras",
        "cohere",
        "fal-ai",
        "featherless-ai",
        "fireworks-ai",
        "groq",
        "hf-inference",
        "hyperbolic",
        "nebius",
        "novita",
        "nscale",
        "replicate",
        "sambanova",
        "together",
    ]

    # Fetch models from all providers concurrently
    tasks = [fetch_models_from_hf_provider(provider) for provider in providers]
    provider_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Combine all models
    all_models = []
    for i, result in enumerate(provider_results):
        if isinstance(result, Exception):
            log.error(f"Error fetching models for provider {providers[i]}: {result}")
        elif isinstance(result, list):
            all_models.extend(result)

    # Cache the results in memory
    _language_model_cache.set(cache_key, all_models, ttl=CACHE_TTL)
    log.info(f"Cached {len(all_models)} HuggingFace models in memory")

    return all_models


async def fetch_openai_language_models() -> List[LanguageModel]:
    """
    Fetch available OpenAI models using the OpenAI REST API.

    Returns:
        A list of LanguageModel entries with provider set to Provider.OpenAI.

    Notes:
        - Only runs if OPENAI_API_KEY is available in the environment.
        - Uses a short timeout to avoid blocking if the network is unavailable.
    """
    env = Environment.get_environment()
    api_key = env.get("OPENAI_API_KEY")
    if not api_key:
        return []

    try:
        timeout = aiohttp.ClientTimeout(total=3)
        headers = {
            "Authorization": f"Bearer {api_key}",
        }
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.get("https://api.openai.com/v1/models") as response:
                if response.status != 200:
                    log.warning(
                        f"Failed to fetch OpenAI models: HTTP {response.status}"
                    )
                    return []
                payload: Dict[str, Any] = await response.json()
                data = payload.get("data", [])

                models: List[LanguageModel] = []
                for item in data:
                    model_id = item.get("id")
                    if not model_id:
                        continue
                    models.append(
                        LanguageModel(
                            id=model_id,
                            name=model_id,
                            provider=Provider.OpenAI,
                        )
                    )
                log.debug(f"Fetched {len(models)} OpenAI models")
                return models
    except Exception as e:
        log.error(f"Error fetching OpenAI models: {e}")
        return []


async def get_cached_openai_models() -> List[LanguageModel]:
    """
    Get OpenAI models from in-memory cache or fetch them dynamically.

    Returns:
        List of LanguageModel instances for OpenAI
    """
    cache_key = "openai_language_models"

    cached_models = _language_model_cache.get(cache_key)
    if cached_models is not None:
        log.debug("Using cached OpenAI models")
        return cached_models

    log.debug("Fetching OpenAI models from API")
    models = await fetch_openai_language_models()
    # Only cache non-empty results to allow retries if API was temporarily unavailable
    if models:
        _language_model_cache.set(cache_key, models, ttl=CACHE_TTL)
        log.info(f"Cached {len(models)} OpenAI models in memory")

    return models


async def fetch_gemini_language_models() -> List[LanguageModel]:
    """
    Fetch available Gemini models using the Google Generative Language REST API.

    Returns:
        A list of LanguageModel entries with provider set to Provider.Gemini.

    Notes:
        - Only runs if GEMINI_API_KEY is available in the environment.
        - Uses a short timeout to avoid blocking if the network is unavailable.
    """
    env = Environment.get_environment()
    api_key = env.get("GEMINI_API_KEY")
    if not api_key:
        return []

    try:
        timeout = aiohttp.ClientTimeout(total=3)
        # API permits key either as header or query parameter; use query to avoid header nuances
        url = f"https://generativelanguage.googleapis.com/v1/models?key={api_key}"
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    log.warning(
                        f"Failed to fetch Gemini models: HTTP {response.status}"
                    )
                    return []
                payload: Dict[str, Any] = await response.json()
                items = payload.get("models") or payload.get("data") or []

                models: List[LanguageModel] = []
                for item in items:
                    # Typical id format is name: "models/gemini-1.5-flash"; strip prefix
                    raw_name: str | None = item.get("name")
                    if not raw_name:
                        continue
                    model_id = raw_name.split("/")[-1]
                    display_name = item.get("displayName") or model_id
                    models.append(
                        LanguageModel(
                            id=model_id,
                            name=display_name,
                            provider=Provider.Gemini,
                        )
                    )
                log.debug(f"Fetched {len(models)} Gemini models")
                return models
    except Exception as e:
        log.error(f"Error fetching Gemini models: {e}")
        return []


async def get_cached_gemini_models() -> List[LanguageModel]:
    """
    Get Gemini models from in-memory cache or fetch them dynamically.

    Returns:
        List of LanguageModel instances for Gemini
    """
    cache_key = "gemini_language_models"

    cached_models = _language_model_cache.get(cache_key)
    if cached_models is not None:
        log.debug("Using cached Gemini models")
        return cached_models

    log.debug("Fetching Gemini models from API")
    models = await fetch_gemini_language_models()
    if models:
        _language_model_cache.set(cache_key, models, ttl=CACHE_TTL)
        log.info(f"Cached {len(models)} Gemini models in memory")

    return models


async def fetch_anthropic_language_models() -> List[LanguageModel]:
    """
    Fetch available Anthropic models using the Anthropic REST API.

    Returns:
        A list of LanguageModel entries with provider set to Provider.Anthropic.

    Notes:
        - Only runs if ANTHROPIC_API_KEY is available in the environment.
        - Uses a short timeout to avoid blocking if the network is unavailable.
        - Anthropic requires the 'anthropic-version' header.
    """
    env = Environment.get_environment()
    api_key = env.get("ANTHROPIC_API_KEY")
    if not api_key:
        return []

    try:
        timeout = aiohttp.ClientTimeout(total=3)
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.get("https://api.anthropic.com/v1/models") as response:
                if response.status != 200:
                    log.warning(
                        f"Failed to fetch Anthropic models: HTTP {response.status}"
                    )
                    return []
                payload: Dict[str, Any] = await response.json()
                data = payload.get("data", [])

                models: List[LanguageModel] = []
                for item in data:
                    model_id = item.get("id") or item.get("name")
                    if not model_id:
                        continue
                    models.append(
                        LanguageModel(
                            id=model_id,
                            name=model_id,
                            provider=Provider.Anthropic,
                        )
                    )
                log.debug(f"Fetched {len(models)} Anthropic models")
                return models
    except Exception as e:
        log.error(f"Error fetching Anthropic models: {e}")
        return []


async def get_cached_anthropic_models() -> List[LanguageModel]:
    """
    Get Anthropic models from in-memory cache or fetch them dynamically.

    Returns:
        List of LanguageModel instances for Anthropic
    """
    cache_key = "anthropic_language_models"

    cached_models = _language_model_cache.get(cache_key)
    if cached_models is not None:
        log.debug("Using cached Anthropic models")
        return cached_models

    log.debug("Fetching Anthropic models from API")
    models = await fetch_anthropic_language_models()
    if models:
        _language_model_cache.set(cache_key, models, ttl=CACHE_TTL)
        log.info(f"Cached {len(models)} Anthropic models in memory")

    return models


async def get_all_language_models() -> List[LanguageModel]:
    """
    Get all language models from all providers, including dynamically fetched HF models.

    Returns:
        List of all available LanguageModel instances
    """
    env = Environment.get_environment()
    models = []

    # Add static models based on API keys
    if "ANTHROPIC_API_KEY" in env:
        models.extend(await get_cached_anthropic_models())
    if "GEMINI_API_KEY" in env:
        models.extend(await get_cached_gemini_models())
    if "OPENAI_API_KEY" in env:
        models.extend(await get_cached_openai_models())
    if "HF_TOKEN" in env or "HUGGINGFACE_API_KEY" in env:
        models.extend(await get_cached_hf_models())

    return models


def clear_language_model_cache() -> None:
    """
    Clear the in-memory language model cache.

    This will force the next call to get_cached_hf_models() to fetch fresh data from the API.
    """
    _language_model_cache.clear()
    log.info("Language model cache cleared")
