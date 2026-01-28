"""
Centralized cost calculation for all AI providers.

This module provides a unified interface for calculating API costs in credits.
1 credit = $0.01 USD (i.e., 1000 credits = $10 USD)
All rates include a 50% premium over provider base costs.

Usage:
    from nodetool.providers.cost_calculator import CostCalculator, UsageInfo

    # Calculate cost for a chat completion
    usage = UsageInfo(input_tokens=1000, output_tokens=500)
    cost = CostCalculator.calculate("gpt-4o-mini", usage)

    # Or use backward-compatible helper functions
    cost = await calculate_chat_cost("gpt-4o-mini", input_tokens=1000, output_tokens=500)
"""

from enum import Enum
from dataclasses import dataclass

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


class CostType(str, Enum):
    """Types of cost calculation methods."""

    TOKEN_BASED = "token_based"  # Chat models (input/output tokens)
    EMBEDDING = "embedding"  # Embedding models (input tokens only)
    CHARACTER_BASED = "character_based"  # TTS models (input characters)
    DURATION_BASED = "duration_based"  # ASR models (audio duration)
    IMAGE_BASED = "image_based"  # Image generation (per image)
    VIDEO_BASED = "video_based"  # Video generation (per second/frame)


@dataclass
class PricingTier:
    """Pricing configuration for a model tier."""

    cost_type: CostType
    # Token-based pricing
    input_per_1k_tokens: float = 0.0
    output_per_1k_tokens: float = 0.0
    cached_per_1k_tokens: float = 0.0
    # Character/duration/unit pricing
    per_1k_chars: float = 0.0
    per_minute: float = 0.0
    per_image: float = 0.0
    per_second_video: float = 0.0


# Pricing tiers by tier name - derived from openai_prediction.py CREDIT_PRICING_TIERS
PRICING_TIERS: dict[str, PricingTier] = {
    # GPT-5 Series (newest flagship models)
    "gpt5_tier": PricingTier(
        cost_type=CostType.TOKEN_BASED,
        input_per_1k_tokens=0.002625,
        output_per_1k_tokens=0.021,
        cached_per_1k_tokens=0.0002625,
    ),
    "gpt5_pro_tier": PricingTier(
        cost_type=CostType.TOKEN_BASED,
        input_per_1k_tokens=0.0315,
        output_per_1k_tokens=0.252,
    ),
    "gpt5_mini_tier": PricingTier(
        cost_type=CostType.TOKEN_BASED,
        input_per_1k_tokens=0.000375,
        output_per_1k_tokens=0.003,
        cached_per_1k_tokens=0.0000375,
    ),
    # GPT-4.1 family
    "gpt4_1_tier": PricingTier(
        cost_type=CostType.TOKEN_BASED,
        input_per_1k_tokens=0.0045,
        output_per_1k_tokens=0.018,
        cached_per_1k_tokens=0.001125,
    ),
    "gpt4_1_mini_tier": PricingTier(
        cost_type=CostType.TOKEN_BASED,
        input_per_1k_tokens=0.0012,
        output_per_1k_tokens=0.0048,
        cached_per_1k_tokens=0.0003,
    ),
    "gpt4_1_nano_tier": PricingTier(
        cost_type=CostType.TOKEN_BASED,
        input_per_1k_tokens=0.0003,
        output_per_1k_tokens=0.0012,
        cached_per_1k_tokens=0.000075,
    ),
    # O4-mini (reasoning)
    "o4_mini_tier": PricingTier(
        cost_type=CostType.TOKEN_BASED,
        input_per_1k_tokens=0.006,
        output_per_1k_tokens=0.024,
        cached_per_1k_tokens=0.0015,
    ),
    # O1 Series (existing reasoning models)
    "o1_tier": PricingTier(
        cost_type=CostType.TOKEN_BASED,
        input_per_1k_tokens=2.25,
        output_per_1k_tokens=9.0,
    ),
    "o1_mini_tier": PricingTier(
        cost_type=CostType.TOKEN_BASED,
        input_per_1k_tokens=0.45,
        output_per_1k_tokens=1.8,
    ),
    # GPT-4o Series
    "top_tier_chat": PricingTier(
        cost_type=CostType.TOKEN_BASED,
        input_per_1k_tokens=0.375,
        output_per_1k_tokens=1.5,
    ),
    "low_tier_chat": PricingTier(
        cost_type=CostType.TOKEN_BASED,
        input_per_1k_tokens=0.0225,
        output_per_1k_tokens=0.09,
    ),
    # GPT-4 Turbo
    "gpt4_turbo": PricingTier(
        cost_type=CostType.TOKEN_BASED,
        input_per_1k_tokens=1.5,
        output_per_1k_tokens=6.0,
    ),
    # Image generation - gpt-image-1
    "image_gpt_low": PricingTier(
        cost_type=CostType.IMAGE_BASED,
        per_image=1.5,
    ),
    "image_gpt_medium": PricingTier(
        cost_type=CostType.IMAGE_BASED,
        per_image=6.0,
    ),
    "image_gpt_high": PricingTier(
        cost_type=CostType.IMAGE_BASED,
        per_image=25.0,
    ),
    # Image generation GPT-image-1.5
    "image_gpt_1_5": PricingTier(
        cost_type=CostType.TOKEN_BASED,
        input_per_1k_tokens=0.0075,
        output_per_1k_tokens=0.015,
    ),
    # Whisper / Speech-to-Text
    "whisper_standard": PricingTier(
        cost_type=CostType.DURATION_BASED,
        per_minute=0.9,
    ),
    "whisper_low_cost": PricingTier(
        cost_type=CostType.DURATION_BASED,
        per_minute=0.45,
    ),
    # TTS / Text-to-Speech
    "tts_standard": PricingTier(
        cost_type=CostType.CHARACTER_BASED,
        per_1k_chars=0.09,
    ),
    "tts_hd": PricingTier(
        cost_type=CostType.CHARACTER_BASED,
        per_1k_chars=2.25,
    ),
    "tts_ultra_hd": PricingTier(
        cost_type=CostType.CHARACTER_BASED,
        per_1k_chars=4.5,
    ),
    # Embeddings
    "embedding_small": PricingTier(
        cost_type=CostType.EMBEDDING,
        input_per_1k_tokens=0.003,
    ),
    "embedding_large": PricingTier(
        cost_type=CostType.EMBEDDING,
        input_per_1k_tokens=0.0195,
    ),
    # Anthropic Claude 4 family (2025)
    "claude_opus_4": PricingTier(
        cost_type=CostType.TOKEN_BASED,
        input_per_1k_tokens=0.045,
        output_per_1k_tokens=0.15,
    ),
    "claude_sonnet_4": PricingTier(
        cost_type=CostType.TOKEN_BASED,
        input_per_1k_tokens=0.0075,
        output_per_1k_tokens=0.0375,
    ),
    "claude_haiku_4": PricingTier(
        cost_type=CostType.TOKEN_BASED,
        input_per_1k_tokens=0.00225,
        output_per_1k_tokens=0.0075,
    ),
    # Claude 3.7 family
    "claude_3_7_sonnet": PricingTier(
        cost_type=CostType.TOKEN_BASED,
        input_per_1k_tokens=0.009,
        output_per_1k_tokens=0.027,
    ),
    # Claude 3.5 family
    "claude_3_5_sonnet": PricingTier(
        cost_type=CostType.TOKEN_BASED,
        input_per_1k_tokens=0.0075,
        output_per_1k_tokens=0.0225,
    ),
    "claude_3_5_haiku": PricingTier(
        cost_type=CostType.TOKEN_BASED,
        input_per_1k_tokens=0.0015,
        output_per_1k_tokens=0.006,
    ),
    # Claude 3 Opus
    "claude_3_opus": PricingTier(
        cost_type=CostType.TOKEN_BASED,
        input_per_1k_tokens=0.045,
        output_per_1k_tokens=0.15,
    ),
    # Claude 3 Sonnet
    "claude_3_sonnet": PricingTier(
        cost_type=CostType.TOKEN_BASED,
        input_per_1k_tokens=0.0075,
        output_per_1k_tokens=0.0225,
    ),
    # Claude 3 Haiku
    "claude_3_haiku": PricingTier(
        cost_type=CostType.TOKEN_BASED,
        input_per_1k_tokens=0.000375,
        output_per_1k_tokens=0.0015,
    ),
}

# Model ID to tier mapping - derived from openai_prediction.py MODEL_TO_TIER_MAP
# Keys can be either:
#   - str: model ID (provider-agnostic, used as fallback)
#   - tuple[str, str]: (provider, model_id) for provider-specific pricing
MODEL_TO_TIER: dict[str | tuple[str, str], str] = {
    # GPT-5 Series (newest models) - OpenAI specific
    ("openai", "gpt-5.2"): "gpt5_tier",
    ("openai", "gpt-5.2-pro"): "gpt5_pro_tier",
    ("openai", "gpt-5-mini"): "gpt5_mini_tier",
    # GPT-4.1 Family - OpenAI specific
    ("openai", "gpt-4.1"): "gpt4_1_tier",
    ("openai", "gpt-4.1-mini"): "gpt4_1_mini_tier",
    ("openai", "gpt-4.1-nano"): "gpt4_1_nano_tier",
    # O4 Series (reasoning models) - OpenAI specific
    ("openai", "o4-mini"): "o4_mini_tier",
    # O1 Series (existing reasoning models) - OpenAI specific
    ("openai", "o1"): "o1_tier",
    ("openai", "o1-preview"): "o1_tier",
    ("openai", "o1-mini"): "o1_mini_tier",
    # O3 Series (future models) - OpenAI specific
    ("openai", "o3"): "o1_tier",
    ("openai", "o3-mini"): "o1_mini_tier",
    # GPT-4o Series - OpenAI specific
    ("openai", "gpt-4o"): "top_tier_chat",
    ("openai", "gpt-4o-2024-11-20"): "top_tier_chat",
    ("openai", "gpt-4o-2024-08-06"): "top_tier_chat",
    ("openai", "gpt-4o-2024-05-13"): "top_tier_chat",
    ("openai", "gpt-4o-search-preview"): "top_tier_chat",
    ("openai", "gpt-4o-mini"): "low_tier_chat",
    ("openai", "gpt-4o-mini-2024-07-18"): "low_tier_chat",
    ("openai", "gpt-4o-mini-search-preview"): "low_tier_chat",
    # GPT-4 Turbo Series - OpenAI specific
    ("openai", "gpt-4-turbo"): "gpt4_turbo",
    ("openai", "gpt-4-turbo-2024-04-09"): "gpt4_turbo",
    ("openai", "gpt-4-turbo-preview"): "gpt4_turbo",
    ("openai", "gpt-4-0125-preview"): "gpt4_turbo",
    ("openai", "gpt-4-1106-preview"): "gpt4_turbo",
    ("openai", "computer-use-preview"): "top_tier_chat",
    # Image models - OpenAI specific
    ("openai", "gpt-image-1.5"): "image_gpt_1_5",
    # Whisper / Speech-to-Text - OpenAI specific
    ("openai", "whisper-1"): "whisper_standard",
    ("openai", "gpt-4o-transcribe"): "whisper_standard",
    ("openai", "gpt-4o-mini-transcribe"): "whisper_low_cost",
    # TTS / Text-to-Speech - OpenAI specific
    ("openai", "gpt-4o-mini-tts"): "tts_standard",
    ("openai", "tts-1"): "tts_hd",
    ("openai", "tts-1-hd"): "tts_ultra_hd",
    # Embeddings - OpenAI specific
    ("openai", "text-embedding-3-small"): "embedding_small",
    ("openai", "text-embedding-3-large"): "embedding_large",
    # Anthropic Models - Anthropic specific
    ("anthropic", "claude-opus-4-20250514"): "claude_opus_4",
    ("anthropic", "claude-opus-4-20250501"): "claude_opus_4",
    ("anthropic", "claude-sonnet-4-20250514"): "claude_sonnet_4",
    ("anthropic", "claude-sonnet-4-20250501"): "claude_sonnet_4",
    ("anthropic", "claude-haiku-4-20250514"): "claude_haiku_4",
    ("anthropic", "claude-haiku-4-20250501"): "claude_haiku_4",
    ("anthropic", "claude-3-7-sonnet-20250511"): "claude_3_7_sonnet",
    ("anthropic", "claude-3-7-sonnet-20250219"): "claude_3_7_sonnet",
    ("anthropic", "claude-3-5-sonnet-20241022"): "claude_3_5_sonnet",
    ("anthropic", "claude-3-5-sonnet-20240620"): "claude_3_5_sonnet",
    ("anthropic", "claude-3-5-sonnet-latest"): "claude_3_5_sonnet",
    ("anthropic", "claude-3-5-haiku-20241022"): "claude_3_5_haiku",
    ("anthropic", "claude-3-5-haiku-latest"): "claude_3_5_haiku",
    ("anthropic", "claude-3-opus-20240229"): "claude_3_opus",
    ("anthropic", "claude-3-opus-latest"): "claude_3_opus",
    ("anthropic", "claude-3-sonnet-20240229"): "claude_3_sonnet",
    ("anthropic", "claude-3-sonnet-latest"): "claude_3_sonnet",
    ("anthropic", "claude-3-haiku-20240307"): "claude_3_haiku",
    ("anthropic", "claude-3-haiku-latest"): "claude_3_haiku",
}


@dataclass
class UsageInfo:
    """Standardized usage information from API responses."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    input_characters: int = 0
    duration_seconds: float = 0.0
    image_count: int = 0
    video_seconds: float = 0.0


class CostCalculator:
    """Centralized cost calculator for all providers."""

    @staticmethod
    def get_tier(model_id: str, provider: str) -> str | None:
        """Get the pricing tier for a model ID scoped by provider.

        Provider is required for unambiguous pricing lookup.

        Args:
            model_id: The model identifier
            provider: Provider name (e.g., "openai", "anthropic")

        Returns:
            The pricing tier name, or None if not found
        """
        model_lower = model_id.lower()
        provider_lower = provider.lower()

        # Direct lookup with (provider, model) key
        provider_key = (provider_lower, model_lower)
        if provider_key in MODEL_TO_TIER:
            return MODEL_TO_TIER[provider_key]

        # Try prefix match with provider key - sort by model length descending
        # Only look at tuple keys that match the provider
        provider_prefixes = [
            key for key in MODEL_TO_TIER.keys()
            if isinstance(key, tuple) and len(key) == 2 and key[0] == provider_lower
        ]
        sorted_provider_prefixes = sorted(
            provider_prefixes, key=lambda x: len(x[1]), reverse=True
        )
        for prefix_key in sorted_provider_prefixes:
            if model_lower.startswith(prefix_key[1]):
                return MODEL_TO_TIER[prefix_key]

        return None

    @staticmethod
    def calculate(
        model_id: str,
        usage: UsageInfo,
        provider: str,
    ) -> float:
        """
        Calculate cost in credits for an API call.

        Args:
            model_id: The model identifier
            usage: Usage information from the API response
            provider: Provider name for pricing lookup (required)

        Returns:
            Cost in credits (1 credit = $0.01 USD)
        """
        tier_name = CostCalculator.get_tier(model_id, provider)
        if tier_name is None:
            log.warning(
                f"No pricing tier found for model: {model_id} (provider: {provider})"
            )
            return 0.0

        tier = PRICING_TIERS.get(tier_name)
        if tier is None:
            log.warning(f"Pricing tier '{tier_name}' not defined")
            return 0.0

        return CostCalculator._calculate_for_tier(tier, usage)

    @staticmethod
    def _calculate_for_tier(tier: PricingTier, usage: UsageInfo) -> float:
        """Calculate cost based on tier type."""
        if tier.cost_type == CostType.TOKEN_BASED:
            # Handle cached tokens properly
            if usage.cached_tokens > 0 and tier.cached_per_1k_tokens > 0:
                non_cached_input = max(0, usage.input_tokens - usage.cached_tokens)
                input_cost = (non_cached_input / 1000) * tier.input_per_1k_tokens
                cached_cost = (usage.cached_tokens / 1000) * tier.cached_per_1k_tokens
            else:
                input_cost = (usage.input_tokens / 1000) * tier.input_per_1k_tokens
                cached_cost = 0.0

            output_cost = (usage.output_tokens / 1000) * tier.output_per_1k_tokens
            return input_cost + output_cost + cached_cost

        elif tier.cost_type == CostType.EMBEDDING:
            return (usage.input_tokens / 1000) * tier.input_per_1k_tokens

        elif tier.cost_type == CostType.CHARACTER_BASED:
            return (usage.input_characters / 1000) * tier.per_1k_chars

        elif tier.cost_type == CostType.DURATION_BASED:
            duration_minutes = usage.duration_seconds / 60.0
            return duration_minutes * tier.per_minute

        elif tier.cost_type == CostType.IMAGE_BASED:
            return usage.image_count * tier.per_image

        elif tier.cost_type == CostType.VIDEO_BASED:
            return usage.video_seconds * tier.per_second_video

        return 0.0


# Convenience functions for backward compatibility with openai_prediction.py


async def calculate_chat_cost(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    provider: str = "openai",
) -> float:
    """Calculate chat completion cost. Backward-compatible function."""
    usage = UsageInfo(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
    )
    return CostCalculator.calculate(model_id, usage, provider)


async def calculate_embedding_cost(
    model_id: str,
    input_tokens: int,
    provider: str = "openai",
) -> float:
    """Calculate embedding cost. Backward-compatible function."""
    usage = UsageInfo(input_tokens=input_tokens)
    return CostCalculator.calculate(model_id, usage, provider)


async def calculate_speech_cost(
    model_id: str,
    input_chars: int,
    provider: str = "openai",
) -> float:
    """Calculate TTS cost. Backward-compatible function."""
    usage = UsageInfo(input_characters=input_chars)
    return CostCalculator.calculate(model_id, usage, provider)


async def calculate_whisper_cost(
    model_id: str,
    duration_seconds: float,
    provider: str = "openai",
) -> float:
    """Calculate ASR/Whisper cost. Backward-compatible function."""
    usage = UsageInfo(duration_seconds=duration_seconds)
    return CostCalculator.calculate(model_id, usage, provider)


async def calculate_image_cost(
    model_id: str,
    image_count: int = 1,
    quality: str = "medium",
    provider: str = "openai",
) -> float:
    """Calculate image generation cost. Backward-compatible function."""
    # Adjust tier based on quality for gpt-image models
    tier_override = None
    if "gpt-image" in model_id.lower() and "1.5" not in model_id:
        quality_map = {
            "low": "image_gpt_low",
            "medium": "image_gpt_medium",
            "high": "image_gpt_high",
        }
        tier_override = quality_map.get(quality.lower(), "image_gpt_medium")

    usage = UsageInfo(image_count=image_count)

    if tier_override:
        tier = PRICING_TIERS.get(tier_override)
        if tier:
            return CostCalculator._calculate_for_tier(tier, usage)

    return CostCalculator.calculate(model_id, usage, provider)
