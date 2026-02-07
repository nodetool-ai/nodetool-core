"""Tests for the centralized cost calculator module."""

import pytest

from nodetool.providers.cost_calculator import (
    MODEL_TO_TIER,
    PRICING_TIERS,
    CostCalculator,
    CostType,
    PricingTier,
    UsageInfo,
    calculate_chat_cost,
    calculate_embedding_cost,
    calculate_image_cost,
    calculate_speech_cost,
    calculate_whisper_cost,
)


class TestCostType:
    """Tests for CostType enum."""

    def test_cost_type_values(self):
        assert CostType.TOKEN_BASED == "token_based"
        assert CostType.EMBEDDING == "embedding"
        assert CostType.CHARACTER_BASED == "character_based"
        assert CostType.DURATION_BASED == "duration_based"
        assert CostType.IMAGE_BASED == "image_based"
        assert CostType.VIDEO_BASED == "video_based"


class TestPricingTier:
    """Tests for PricingTier dataclass."""

    def test_pricing_tier_defaults(self):
        tier = PricingTier(cost_type=CostType.TOKEN_BASED)
        assert tier.cost_type == CostType.TOKEN_BASED
        assert tier.input_per_1k_tokens == 0.0
        assert tier.output_per_1k_tokens == 0.0
        assert tier.cached_per_1k_tokens == 0.0
        assert tier.per_1k_chars == 0.0
        assert tier.per_minute == 0.0
        assert tier.per_image == 0.0
        assert tier.per_second_video == 0.0

    def test_pricing_tier_with_values(self):
        tier = PricingTier(
            cost_type=CostType.TOKEN_BASED,
            input_per_1k_tokens=0.01,
            output_per_1k_tokens=0.02,
            cached_per_1k_tokens=0.005,
        )
        assert tier.input_per_1k_tokens == 0.01
        assert tier.output_per_1k_tokens == 0.02
        assert tier.cached_per_1k_tokens == 0.005


class TestUsageInfo:
    """Tests for UsageInfo dataclass."""

    def test_usage_info_defaults(self):
        usage = UsageInfo()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.cached_tokens == 0
        assert usage.reasoning_tokens == 0
        assert usage.input_characters == 0
        assert usage.duration_seconds == 0.0
        assert usage.image_count == 0
        assert usage.video_seconds == 0.0

    def test_usage_info_with_values(self):
        usage = UsageInfo(
            input_tokens=1000,
            output_tokens=500,
            cached_tokens=200,
        )
        assert usage.input_tokens == 1000
        assert usage.output_tokens == 500
        assert usage.cached_tokens == 200


class TestCostCalculator:
    """Tests for CostCalculator class."""

    def test_get_tier_direct_match(self):
        """Test direct model lookup with provider."""
        assert CostCalculator.get_tier("gpt-4o-mini", "openai") == "low_tier_chat"
        assert CostCalculator.get_tier("gpt-4o", "openai") == "top_tier_chat"
        assert CostCalculator.get_tier("text-embedding-3-small", "openai") == "embedding_small"
        assert CostCalculator.get_tier("whisper-1", "openai") == "whisper_standard"

    def test_get_tier_case_insensitive(self):
        """Test that model and provider matching is case-insensitive."""
        assert CostCalculator.get_tier("GPT-4O-MINI", "openai") == "low_tier_chat"
        assert CostCalculator.get_tier("GPT-4O", "OpenAI") == "top_tier_chat"
        assert CostCalculator.get_tier("gpt-4o-mini", "OPENAI") == "low_tier_chat"

    def test_get_tier_prefix_match(self):
        """Models with version suffixes should match their prefix."""
        assert CostCalculator.get_tier("gpt-4o-2024-11-20", "openai") == "top_tier_chat"
        assert CostCalculator.get_tier("gpt-4o-mini-2024-07-18", "openai") == "low_tier_chat"

    def test_get_tier_unknown_model(self):
        """Test that unknown models return None."""
        assert CostCalculator.get_tier("unknown-model", "openai") is None
        assert CostCalculator.get_tier("my-custom-model", "anthropic") is None

    def test_get_tier_unknown_provider(self):
        """Test that unknown provider returns None."""
        assert CostCalculator.get_tier("gpt-4o-mini", "unknown-provider") is None

    def test_get_tier_with_openai(self):
        """Test OpenAI-specific tier lookup."""
        assert CostCalculator.get_tier("gpt-4o-mini", "openai") == "low_tier_chat"
        assert CostCalculator.get_tier("gpt-4o", "openai") == "top_tier_chat"
        assert CostCalculator.get_tier("gpt-5.2", "openai") == "gpt5_tier"

    def test_get_tier_with_anthropic(self):
        """Test Anthropic-specific tier lookup."""
        assert CostCalculator.get_tier("claude-3-5-sonnet-latest", "anthropic") == "claude_3_5_sonnet"
        assert CostCalculator.get_tier("claude-3-haiku-20240307", "anthropic") == "claude_3_haiku"

    def test_calculate_with_openai(self):
        """Test cost calculation with OpenAI provider."""
        usage = UsageInfo(input_tokens=1000, output_tokens=500)
        cost = CostCalculator.calculate("gpt-4o-mini", usage, "openai")
        # Expected: (1000/1000 * 0.0225) + (500/1000 * 0.09) = 0.0225 + 0.045 = 0.0675
        assert cost == pytest.approx(0.0675, rel=1e-6)

    def test_calculate_with_anthropic(self):
        """Test cost calculation with Anthropic provider."""
        usage = UsageInfo(input_tokens=1000, output_tokens=500)
        cost = CostCalculator.calculate("claude-3-5-sonnet-latest", usage, "anthropic")
        # Expected: (1000/1000 * 0.0075) + (500/1000 * 0.0225) = 0.0075 + 0.01125 = 0.01875
        assert cost == pytest.approx(0.01875, rel=1e-6)

    def test_calculate_token_based_cost(self):
        # GPT-4o-mini: input=0.0225/1k, output=0.09/1k
        usage = UsageInfo(input_tokens=1000, output_tokens=500)
        cost = CostCalculator.calculate("gpt-4o-mini", usage, "openai")
        # Expected: (1000/1000 * 0.0225) + (500/1000 * 0.09) = 0.0225 + 0.045 = 0.0675
        assert cost == pytest.approx(0.0675, rel=1e-6)

    def test_calculate_token_based_with_cached_tokens(self):
        # gpt-5.2: input=0.002625, output=0.021, cached=0.0002625
        usage = UsageInfo(input_tokens=1000, output_tokens=500, cached_tokens=200)
        cost = CostCalculator.calculate("gpt-5.2", usage, "openai")
        # Expected:
        # non_cached_input = 1000 - 200 = 800
        # input_cost = (800/1000 * 0.002625) = 0.0021
        # cached_cost = (200/1000 * 0.0002625) = 0.0000525
        # output_cost = (500/1000 * 0.021) = 0.0105
        # total = 0.0021 + 0.0000525 + 0.0105 = 0.0126525
        assert cost == pytest.approx(0.0126525, rel=1e-6)

    def test_calculate_embedding_cost(self):
        # text-embedding-3-small: 0.003/1k tokens
        usage = UsageInfo(input_tokens=1000)
        cost = CostCalculator.calculate("text-embedding-3-small", usage, "openai")
        # Expected: 1000/1000 * 0.003 = 0.003
        assert cost == pytest.approx(0.003, rel=1e-6)

    def test_calculate_character_based_cost(self):
        # tts-1: 2.25/1k chars
        usage = UsageInfo(input_characters=500)
        cost = CostCalculator.calculate("tts-1", usage, "openai")
        # Expected: 500/1000 * 2.25 = 1.125
        assert cost == pytest.approx(1.125, rel=1e-6)

    def test_calculate_duration_based_cost(self):
        # whisper-1: 0.9/minute
        usage = UsageInfo(duration_seconds=120)  # 2 minutes
        cost = CostCalculator.calculate("whisper-1", usage, "openai")
        # Expected: (120/60) * 0.9 = 2 * 0.9 = 1.8
        assert cost == pytest.approx(1.8, rel=1e-6)

    def test_calculate_unknown_model_returns_zero(self):
        usage = UsageInfo(input_tokens=1000, output_tokens=500)
        cost = CostCalculator.calculate("unknown-model", usage, "openai")
        assert cost == 0.0

    def test_calculate_unknown_provider_returns_zero(self):
        usage = UsageInfo(input_tokens=1000, output_tokens=500)
        cost = CostCalculator.calculate("gpt-4o-mini", usage, "unknown-provider")
        assert cost == 0.0


class TestBackwardCompatibleFunctions:
    """Tests for backward-compatible helper functions."""

    @pytest.mark.asyncio
    async def test_calculate_chat_cost(self):
        cost = await calculate_chat_cost("gpt-4o-mini", input_tokens=1000, output_tokens=500)
        assert cost == pytest.approx(0.0675, rel=1e-6)

    @pytest.mark.asyncio
    async def test_calculate_chat_cost_with_cached_tokens(self):
        cost = await calculate_chat_cost(
            "gpt-5.2", input_tokens=1000, output_tokens=500, cached_tokens=200
        )
        assert cost == pytest.approx(0.0126525, rel=1e-6)

    @pytest.mark.asyncio
    async def test_calculate_embedding_cost(self):
        cost = await calculate_embedding_cost("text-embedding-3-small", input_tokens=1000)
        assert cost == pytest.approx(0.003, rel=1e-6)

    @pytest.mark.asyncio
    async def test_calculate_speech_cost(self):
        cost = await calculate_speech_cost("tts-1", input_chars=500)
        assert cost == pytest.approx(1.125, rel=1e-6)

    @pytest.mark.asyncio
    async def test_calculate_whisper_cost(self):
        cost = await calculate_whisper_cost("whisper-1", duration_seconds=120)
        assert cost == pytest.approx(1.8, rel=1e-6)

    @pytest.mark.asyncio
    async def test_calculate_image_cost_low_quality(self):
        cost = await calculate_image_cost("gpt-image-1", image_count=1, quality="low")
        # image_gpt_low: 1.5 per image
        assert cost == pytest.approx(1.5, rel=1e-6)

    @pytest.mark.asyncio
    async def test_calculate_image_cost_medium_quality(self):
        cost = await calculate_image_cost("gpt-image-1", image_count=1, quality="medium")
        # image_gpt_medium: 6.0 per image
        assert cost == pytest.approx(6.0, rel=1e-6)

    @pytest.mark.asyncio
    async def test_calculate_image_cost_high_quality(self):
        cost = await calculate_image_cost("gpt-image-1", image_count=1, quality="high")
        # image_gpt_high: 25.0 per image
        assert cost == pytest.approx(25.0, rel=1e-6)

    @pytest.mark.asyncio
    async def test_calculate_image_cost_multiple_images(self):
        cost = await calculate_image_cost("gpt-image-1", image_count=3, quality="medium")
        # 3 * 6.0 = 18.0
        assert cost == pytest.approx(18.0, rel=1e-6)


class TestPricingTiersCompleteness:
    """Tests to verify pricing tiers are properly defined."""

    def test_all_model_tiers_exist(self):
        """Verify all model mappings point to existing tiers."""
        for key, tier_name in MODEL_TO_TIER.items():
            assert (
                tier_name in PRICING_TIERS
            ), f"Model {key} maps to undefined tier {tier_name}"

    def test_all_entries_are_provider_keyed(self):
        """Verify all entries in MODEL_TO_TIER are (provider, model) tuples."""
        for key in MODEL_TO_TIER.keys():
            assert isinstance(key, tuple), f"Key {key} should be a tuple"
            assert len(key) == 2, f"Key {key} should have 2 elements"
            assert isinstance(key[0], str), f"Provider in {key} should be a string"
            assert isinstance(key[1], str), f"Model in {key} should be a string"

    def test_token_based_tiers_have_required_fields(self):
        """Verify token-based tiers have input and output pricing."""
        for tier_name, tier in PRICING_TIERS.items():
            if tier.cost_type == CostType.TOKEN_BASED:
                assert (
                    tier.input_per_1k_tokens > 0
                ), f"Tier {tier_name} missing input_per_1k_tokens"
                assert (
                    tier.output_per_1k_tokens > 0
                ), f"Tier {tier_name} missing output_per_1k_tokens"

    def test_embedding_tiers_have_required_fields(self):
        """Verify embedding tiers have input pricing."""
        for tier_name, tier in PRICING_TIERS.items():
            if tier.cost_type == CostType.EMBEDDING:
                assert (
                    tier.input_per_1k_tokens > 0
                ), f"Tier {tier_name} missing input_per_1k_tokens"

    def test_character_based_tiers_have_required_fields(self):
        """Verify character-based tiers have per_1k_chars pricing."""
        for tier_name, tier in PRICING_TIERS.items():
            if tier.cost_type == CostType.CHARACTER_BASED:
                assert tier.per_1k_chars > 0, f"Tier {tier_name} missing per_1k_chars"

    def test_duration_based_tiers_have_required_fields(self):
        """Verify duration-based tiers have per_minute pricing."""
        for tier_name, tier in PRICING_TIERS.items():
            if tier.cost_type == CostType.DURATION_BASED:
                assert tier.per_minute > 0, f"Tier {tier_name} missing per_minute"

    def test_image_based_tiers_have_required_fields(self):
        """Verify image-based tiers have per_image pricing."""
        for tier_name, tier in PRICING_TIERS.items():
            if tier.cost_type == CostType.IMAGE_BASED:
                assert tier.per_image > 0, f"Tier {tier_name} missing per_image"
