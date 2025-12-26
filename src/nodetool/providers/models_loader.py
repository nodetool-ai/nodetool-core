"""
Models loader utility for reading models.json and providing model information.

This module loads the models.json file which contains detailed information about
models from various AI providers including pricing, limits, and capabilities.
"""

import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import LanguageModel, Provider

log = get_logger(__name__)


@lru_cache(maxsize=1)
def load_models_data() -> Dict[str, Any]:
    """
    Load and cache the models.json file.
    
    Returns:
        Dictionary containing all provider and model data from models.json
    """
    # Get the path to models.json (in the package_metadata directory)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    package_metadata_dir = os.path.abspath(os.path.join(current_dir, "..", "package_metadata"))
    models_json_path = os.path.join(package_metadata_dir, "models.json")
    
    try:
        with open(models_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        log.info(f"Loaded models.json with {len(data)} providers")
        return data
    except Exception as e:
        log.error(f"Failed to load models.json from {models_json_path}: {e}")
        return {}


def get_provider_models(provider_id: str) -> Dict[str, Any]:
    """
    Get all models for a specific provider from models.json.
    
    Args:
        provider_id: Provider identifier (e.g., "openai", "anthropic", "google")
    
    Returns:
        Dictionary of models for the provider, empty dict if not found
    """
    data = load_models_data()
    provider_data = data.get(provider_id, {})
    return provider_data.get("models", {})


def get_model_cost_info(provider_id: str, model_id: str) -> Optional[Dict[str, float]]:
    """
    Get cost information for a specific model.
    
    Args:
        provider_id: Provider identifier (e.g., "openai", "anthropic")
        model_id: Model identifier (e.g., "gpt-4.1-nano")
    
    Returns:
        Dictionary with cost info (input, output, cache_read, cache_write per 1M tokens)
        or None if not found
    """
    models = get_provider_models(provider_id)
    model_data = models.get(model_id)
    if model_data:
        return model_data.get("cost")
    return None


def get_openai_models() -> List[LanguageModel]:
    """
    Get list of OpenAI models from models.json.
    
    Returns:
        List of LanguageModel instances for OpenAI
    """
    models = get_provider_models("openai")
    result = []
    for model_id, model_data in models.items():
        result.append(
            LanguageModel(
                id=model_id,
                name=model_data.get("name", model_id),
                provider=Provider.OpenAI,
            )
        )
    log.debug(f"Loaded {len(result)} OpenAI models from models.json")
    return result


def get_anthropic_models() -> List[LanguageModel]:
    """
    Get list of Anthropic models from models.json.
    
    Returns:
        List of LanguageModel instances for Anthropic
    """
    models = get_provider_models("anthropic")
    result = []
    for model_id, model_data in models.items():
        result.append(
            LanguageModel(
                id=model_id,
                name=model_data.get("name", model_id),
                provider=Provider.Anthropic,
            )
        )
    log.debug(f"Loaded {len(result)} Anthropic models from models.json")
    return result


def get_gemini_models() -> List[LanguageModel]:
    """
    Get list of Gemini models from models.json.
    
    Returns:
        List of LanguageModel instances for Gemini (Google)
    """
    models = get_provider_models("google")
    result = []
    for model_id, model_data in models.items():
        # Filter for models that have text output (language models)
        modalities = model_data.get("modalities", {})
        output_modalities = modalities.get("output", [])
        if "text" in output_modalities:
            result.append(
                LanguageModel(
                    id=model_id,
                    name=model_data.get("name", model_id),
                    provider=Provider.Gemini,
                )
            )
    log.debug(f"Loaded {len(result)} Gemini models from models.json")
    return result


def calculate_cost_from_models_json(
    provider_id: str,
    model_id: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_tokens: int = 0,
) -> float:
    """
    Calculate cost in credits using models.json pricing data.
    
    Args:
        provider_id: Provider identifier (e.g., "openai", "anthropic", "google")
        model_id: Model identifier
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cached_tokens: Number of cached input tokens (if applicable)
    
    Returns:
        Cost in credits (1 credit = $0.01 USD)
    """
    cost_info = get_model_cost_info(provider_id, model_id)
    if not cost_info:
        log.warning(f"No cost info found for {provider_id}/{model_id}, returning 0")
        return 0.0
    
    # Cost in models.json is per 1M tokens (e.g., $0.1 = 0.1)
    # Convert to credits: 1 credit = $0.01 USD
    # So $0.1 = 10 credits
    
    # Calculate input cost
    input_cost_per_1m = cost_info.get("input", 0)
    cached_cost_per_1m = cost_info.get("cache_read", input_cost_per_1m)
    output_cost_per_1m = cost_info.get("output", 0)
    
    # Separate regular and cached input tokens
    regular_input_tokens = max(0, input_tokens - cached_tokens)
    
    # Calculate cost in USD
    input_cost_usd = (regular_input_tokens / 1_000_000) * input_cost_per_1m
    cached_cost_usd = (cached_tokens / 1_000_000) * cached_cost_per_1m
    output_cost_usd = (output_tokens / 1_000_000) * output_cost_per_1m
    
    total_cost_usd = input_cost_usd + cached_cost_usd + output_cost_usd
    
    # Convert to credits (1 credit = $0.01)
    total_cost_credits = total_cost_usd * 100
    
    log.debug(
        f"Cost calculation for {provider_id}/{model_id}: "
        f"{input_tokens} input + {cached_tokens} cached + {output_tokens} output "
        f"= ${total_cost_usd:.6f} = {total_cost_credits:.4f} credits"
    )
    
    return total_cost_credits
