"""Tests for ProcessingContext cost tracking functionality."""

import pytest

from nodetool.providers.cost_calculator import UsageInfo
from nodetool.workflows.processing_context import ProcessingContext


class TestCostTrackingInitialization:
    """Tests for cost tracking initialization in ProcessingContext."""

    def test_context_initializes_with_zero_cost(self):
        """Verify that a new context starts with zero cost."""
        ctx = ProcessingContext(user_id="test")
        assert ctx.get_total_cost() == 0.0
        assert ctx.get_operation_costs() == []

    def test_context_cost_variables_are_private(self):
        """Verify that cost tracking uses private variables."""
        ctx = ProcessingContext(user_id="test")
        assert hasattr(ctx, "_total_cost")
        assert hasattr(ctx, "_operation_costs")
        assert ctx._total_cost == 0.0
        assert ctx._operation_costs == []


class TestTrackOperationCost:
    """Tests for track_operation_cost method."""

    def test_track_simple_token_operation(self):
        """Test tracking a simple token-based operation."""
        ctx = ProcessingContext(user_id="test")

        usage = UsageInfo(input_tokens=1000, output_tokens=500)
        cost = ctx.track_operation_cost(
            model="gpt-4o-mini",
            provider="openai",
            usage_info=usage,
            node_id="node_1",
        )

        # Expected cost: (1000/1000 * 0.0225) + (500/1000 * 0.09) = 0.0675
        assert cost == pytest.approx(0.0675, rel=1e-6)
        assert ctx.get_total_cost() == pytest.approx(0.0675, rel=1e-6)

    def test_track_operation_with_cached_tokens(self):
        """Test tracking operation with cached tokens."""
        ctx = ProcessingContext(user_id="test")

        usage = UsageInfo(
            input_tokens=1000,
            output_tokens=500,
            cached_tokens=200,
        )
        cost = ctx.track_operation_cost(
            model="gpt-5.2",
            provider="openai",
            usage_info=usage,
            node_id="node_1",
        )

        # Note: gpt-5.2 is a real model in the CostCalculator with the following pricing:
        # gpt-5.2: input=0.002625, output=0.021, cached=0.0002625
        # non_cached_input = 1000 - 200 = 800
        # input_cost = (800/1000 * 0.002625) = 0.0021
        # cached_cost = (200/1000 * 0.0002625) = 0.0000525
        # output_cost = (500/1000 * 0.021) = 0.0105
        # total = 0.0126525
        assert cost == pytest.approx(0.0126525, rel=1e-6)
        assert ctx.get_total_cost() == pytest.approx(0.0126525, rel=1e-6)

    def test_track_embedding_operation(self):
        """Test tracking an embedding operation."""
        ctx = ProcessingContext(user_id="test")

        usage = UsageInfo(input_tokens=1000)
        cost = ctx.track_operation_cost(
            model="text-embedding-3-small",
            provider="openai",
            usage_info=usage,
            node_id="node_1",
            operation_type="embedding",
        )

        # Expected: 1000/1000 * 0.003 = 0.003
        assert cost == pytest.approx(0.003, rel=1e-6)
        assert ctx.get_total_cost() == pytest.approx(0.003, rel=1e-6)

    def test_track_character_based_operation(self):
        """Test tracking a character-based operation (TTS)."""
        ctx = ProcessingContext(user_id="test")

        usage = UsageInfo(input_characters=500)
        cost = ctx.track_operation_cost(
            model="tts-1",
            provider="openai",
            usage_info=usage,
            node_id="node_1",
            operation_type="tts",
        )

        # Expected: 500/1000 * 2.25 = 1.125
        assert cost == pytest.approx(1.125, rel=1e-6)
        assert ctx.get_total_cost() == pytest.approx(1.125, rel=1e-6)

    def test_track_duration_based_operation(self):
        """Test tracking a duration-based operation (ASR)."""
        ctx = ProcessingContext(user_id="test")

        usage = UsageInfo(duration_seconds=120)  # 2 minutes
        cost = ctx.track_operation_cost(
            model="whisper-1",
            provider="openai",
            usage_info=usage,
            node_id="node_1",
            operation_type="asr",
        )

        # Expected: (120/60) * 0.9 = 1.8
        assert cost == pytest.approx(1.8, rel=1e-6)
        assert ctx.get_total_cost() == pytest.approx(1.8, rel=1e-6)

    def test_track_multiple_operations_accumulate(self):
        """Test that multiple operations accumulate correctly."""
        ctx = ProcessingContext(user_id="test")

        # First operation
        usage1 = UsageInfo(input_tokens=1000, output_tokens=500)
        cost1 = ctx.track_operation_cost(
            model="gpt-4o-mini",
            provider="openai",
            usage_info=usage1,
            node_id="node_1",
        )

        # Second operation
        usage2 = UsageInfo(input_tokens=2000, output_tokens=1000)
        cost2 = ctx.track_operation_cost(
            model="gpt-4o-mini",
            provider="openai",
            usage_info=usage2,
            node_id="node_2",
        )

        # Verify costs
        assert cost1 == pytest.approx(0.0675, rel=1e-6)
        assert cost2 == pytest.approx(0.135, rel=1e-6)
        assert ctx.get_total_cost() == pytest.approx(0.2025, rel=1e-6)

    def test_track_operation_stores_details(self):
        """Test that operation details are stored correctly."""
        ctx = ProcessingContext(user_id="test")

        usage = UsageInfo(
            input_tokens=1000,
            output_tokens=500,
            cached_tokens=100,
            reasoning_tokens=50,
        )
        cost = ctx.track_operation_cost(
            model="gpt-4o-mini",
            provider="openai",
            usage_info=usage,
            node_id="node_1",
            operation_type="prediction",
        )

        operations = ctx.get_operation_costs()
        assert len(operations) == 1

        op = operations[0]
        assert op["node_id"] == "node_1"
        assert op["model"] == "gpt-4o-mini"
        assert op["provider"] == "openai"
        assert op["operation_type"] == "prediction"
        assert op["cost"] == cost
        assert "timestamp" in op
        assert op["usage"]["input_tokens"] == 1000
        assert op["usage"]["output_tokens"] == 500
        assert op["usage"]["cached_tokens"] == 100
        assert op["usage"]["reasoning_tokens"] == 50

    def test_track_anthropic_operation(self):
        """Test tracking an Anthropic model operation."""
        ctx = ProcessingContext(user_id="test")

        usage = UsageInfo(input_tokens=1000, output_tokens=500)
        cost = ctx.track_operation_cost(
            model="claude-3-5-sonnet-latest",
            provider="anthropic",
            usage_info=usage,
            node_id="node_1",
        )

        # Expected: (1000/1000 * 0.0075) + (500/1000 * 0.0225) = 0.01875
        assert cost == pytest.approx(0.01875, rel=1e-6)
        assert ctx.get_total_cost() == pytest.approx(0.01875, rel=1e-6)

    def test_track_unknown_model_returns_zero(self):
        """Test that unknown models return zero cost."""
        ctx = ProcessingContext(user_id="test")

        usage = UsageInfo(input_tokens=1000, output_tokens=500)
        cost = ctx.track_operation_cost(
            model="unknown-model",
            provider="openai",
            usage_info=usage,
            node_id="node_1",
        )

        assert cost == 0.0
        assert ctx.get_total_cost() == 0.0


class TestAddToTotalCost:
    """Tests for add_to_total_cost method."""

    def test_add_cost_to_empty_context(self):
        """Test adding cost to a new context."""
        ctx = ProcessingContext(user_id="test")
        ctx.add_to_total_cost(1.5)

        assert ctx.get_total_cost() == pytest.approx(1.5, rel=1e-6)

    def test_add_multiple_costs(self):
        """Test adding multiple costs accumulates correctly."""
        ctx = ProcessingContext(user_id="test")

        ctx.add_to_total_cost(1.0)
        ctx.add_to_total_cost(2.5)
        ctx.add_to_total_cost(0.75)

        assert ctx.get_total_cost() == pytest.approx(4.25, rel=1e-6)

    def test_add_cost_with_existing_tracked_operations(self):
        """Test that add_to_total_cost works alongside track_operation_cost."""
        ctx = ProcessingContext(user_id="test")

        # Track an operation
        usage = UsageInfo(input_tokens=1000, output_tokens=500)
        cost1 = ctx.track_operation_cost(
            model="gpt-4o-mini",
            provider="openai",
            usage_info=usage,
        )

        # Add manual cost
        ctx.add_to_total_cost(1.0)

        # Total should be sum of both
        assert ctx.get_total_cost() == pytest.approx(cost1 + 1.0, rel=1e-6)

    def test_add_zero_cost(self):
        """Test adding zero cost doesn't break anything."""
        ctx = ProcessingContext(user_id="test")
        ctx.add_to_total_cost(0.0)

        assert ctx.get_total_cost() == 0.0

    def test_add_negative_cost(self):
        """Test that negative costs can be added (for refunds/adjustments)."""
        ctx = ProcessingContext(user_id="test")

        ctx.add_to_total_cost(5.0)
        ctx.add_to_total_cost(-1.0)

        assert ctx.get_total_cost() == pytest.approx(4.0, rel=1e-6)


class TestResetTotalCost:
    """Tests for reset_total_cost method."""

    def test_reset_empty_context(self):
        """Test resetting an empty context doesn't break anything."""
        ctx = ProcessingContext(user_id="test")
        ctx.reset_total_cost()

        assert ctx.get_total_cost() == 0.0
        assert ctx.get_operation_costs() == []

    def test_reset_context_with_cost(self):
        """Test resetting a context with accumulated cost."""
        ctx = ProcessingContext(user_id="test")

        usage = UsageInfo(input_tokens=1000, output_tokens=500)
        ctx.track_operation_cost(
            model="gpt-4o-mini",
            provider="openai",
            usage_info=usage,
        )

        assert ctx.get_total_cost() > 0
        assert len(ctx.get_operation_costs()) > 0

        ctx.reset_total_cost()

        assert ctx.get_total_cost() == 0.0
        assert ctx.get_operation_costs() == []

    def test_reset_and_accumulate_again(self):
        """Test that cost can be tracked again after reset."""
        ctx = ProcessingContext(user_id="test")

        # First accumulation
        usage1 = UsageInfo(input_tokens=1000, output_tokens=500)
        ctx.track_operation_cost(
            model="gpt-4o-mini",
            provider="openai",
            usage_info=usage1,
        )
        first_cost = ctx.get_total_cost()

        # Reset
        ctx.reset_total_cost()
        assert ctx.get_total_cost() == 0.0

        # Second accumulation
        usage2 = UsageInfo(input_tokens=500, output_tokens=250)
        ctx.track_operation_cost(
            model="gpt-4o-mini",
            provider="openai",
            usage_info=usage2,
        )
        second_cost = ctx.get_total_cost()

        # Verify second cost is independent of first
        assert second_cost < first_cost
        assert second_cost == pytest.approx(0.03375, rel=1e-6)


class TestGetTotalCost:
    """Tests for get_total_cost method."""

    def test_get_total_cost_empty_context(self):
        """Test getting cost from empty context."""
        ctx = ProcessingContext(user_id="test")
        assert ctx.get_total_cost() == 0.0

    def test_get_total_cost_with_operations(self):
        """Test getting cost after operations."""
        ctx = ProcessingContext(user_id="test")

        usage = UsageInfo(input_tokens=1000, output_tokens=500)
        expected_cost = ctx.track_operation_cost(
            model="gpt-4o-mini",
            provider="openai",
            usage_info=usage,
        )

        assert ctx.get_total_cost() == pytest.approx(expected_cost, rel=1e-6)

    def test_get_total_cost_is_read_only(self):
        """Test that get_total_cost returns the value, not a reference."""
        ctx = ProcessingContext(user_id="test")

        ctx.add_to_total_cost(5.0)
        cost = ctx.get_total_cost()

        # Verify returned value is as expected
        assert cost == pytest.approx(5.0, rel=1e-6)

        # Modifying the local variable shouldn't affect the context
        # (not that it would with a float, but this tests the concept)
        assert ctx.get_total_cost() == pytest.approx(5.0, rel=1e-6)


class TestGetOperationCosts:
    """Tests for get_operation_costs method."""

    def test_get_operation_costs_empty(self):
        """Test getting operation costs from empty context."""
        ctx = ProcessingContext(user_id="test")
        operations = ctx.get_operation_costs()

        assert operations == []

    def test_get_operation_costs_with_operations(self):
        """Test getting operation costs after tracking operations."""
        ctx = ProcessingContext(user_id="test")

        usage1 = UsageInfo(input_tokens=1000, output_tokens=500)
        ctx.track_operation_cost(
            model="gpt-4o-mini",
            provider="openai",
            usage_info=usage1,
            node_id="node_1",
        )

        usage2 = UsageInfo(input_tokens=500, output_tokens=250)
        ctx.track_operation_cost(
            model="gpt-4o",
            provider="openai",
            usage_info=usage2,
            node_id="node_2",
        )

        operations = ctx.get_operation_costs()
        assert len(operations) == 2
        assert operations[0]["node_id"] == "node_1"
        assert operations[0]["model"] == "gpt-4o-mini"
        assert operations[1]["node_id"] == "node_2"
        assert operations[1]["model"] == "gpt-4o"

    def test_get_operation_costs_returns_copy(self):
        """Test that get_operation_costs returns a copy, not the original list."""
        ctx = ProcessingContext(user_id="test")

        usage = UsageInfo(input_tokens=1000, output_tokens=500)
        ctx.track_operation_cost(
            model="gpt-4o-mini",
            provider="openai",
            usage_info=usage,
        )

        operations = ctx.get_operation_costs()
        operations.append({"fake": "data"})

        # Original should be unchanged
        assert len(ctx.get_operation_costs()) == 1


class TestCostTrackingEdgeCases:
    """Tests for edge cases in cost tracking."""

    def test_multiple_contexts_independent(self):
        """Test that multiple contexts track costs independently."""
        ctx1 = ProcessingContext(user_id="user1")
        ctx2 = ProcessingContext(user_id="user2")

        usage = UsageInfo(input_tokens=1000, output_tokens=500)

        ctx1.track_operation_cost(
            model="gpt-4o-mini",
            provider="openai",
            usage_info=usage,
        )

        ctx2.track_operation_cost(
            model="gpt-4o",
            provider="openai",
            usage_info=usage,
        )

        # Costs should be different
        assert ctx1.get_total_cost() != ctx2.get_total_cost()
        assert len(ctx1.get_operation_costs()) == 1
        assert len(ctx2.get_operation_costs()) == 1

    def test_context_with_workflow_id_tracks_cost(self):
        """Test that context with workflow_id tracks cost correctly."""
        ctx = ProcessingContext(user_id="test", workflow_id="workflow_123")

        usage = UsageInfo(input_tokens=1000, output_tokens=500)
        cost = ctx.track_operation_cost(
            model="gpt-4o-mini",
            provider="openai",
            usage_info=usage,
        )

        assert ctx.get_total_cost() == cost
        assert ctx.workflow_id == "workflow_123"

    def test_all_usage_fields_stored(self):
        """Test that all usage fields are stored in operation records."""
        ctx = ProcessingContext(user_id="test")

        usage = UsageInfo(
            input_tokens=100,
            output_tokens=50,
            cached_tokens=20,
            reasoning_tokens=10,
            input_characters=500,
            duration_seconds=1.5,
            image_count=2,
            video_seconds=5.0,
        )

        ctx.track_operation_cost(
            model="gpt-4o-mini",
            provider="openai",
            usage_info=usage,
        )

        operations = ctx.get_operation_costs()
        assert len(operations) == 1

        usage_data = operations[0]["usage"]
        assert usage_data["input_tokens"] == 100
        assert usage_data["output_tokens"] == 50
        assert usage_data["cached_tokens"] == 20
        assert usage_data["reasoning_tokens"] == 10
        assert usage_data["input_characters"] == 500
        assert usage_data["duration_seconds"] == 1.5
        assert usage_data["image_count"] == 2
        assert usage_data["video_seconds"] == 5.0


class TestCostTrackingIntegrationWithPredictions:
    """Tests for cost tracking integration with prediction methods."""

    @pytest.mark.asyncio
    async def test_run_provider_prediction_tracks_cost(self):
        """Test that run_provider_prediction tracks cost in context."""
        from unittest.mock import AsyncMock, patch

        from nodetool.metadata.types import Message, Provider
        from nodetool.providers.base import BaseProvider, ProviderCapability

        class MockProvider(BaseProvider):
            provider_name = "openai"

            @classmethod
            def required_secrets(cls):
                return []

            def get_capabilities(self):
                return {ProviderCapability.GENERATE_MESSAGE}

            async def generate_message(
                self, messages, model, tools=None, max_tokens=8192, **kwargs
            ):
                # Simulate tracking usage
                self.track_usage(model=model, input_tokens=100, output_tokens=50)
                return Message(role="assistant", content="test response")

        ctx = ProcessingContext(user_id="test")

        mock_provider = MockProvider()

        with patch.object(ctx, "get_provider", return_value=mock_provider), patch(
            "nodetool.models.prediction.Prediction.create", new_callable=AsyncMock
        ):
            # Initial cost should be zero
            assert ctx.get_total_cost() == 0.0

            # Run prediction
            result = await ctx.run_provider_prediction(
                node_id="test_node",
                provider=Provider.OpenAI,
                model="gpt-4o-mini",
                capability=ProviderCapability.GENERATE_MESSAGE,
                params={"messages": [{"role": "user", "content": "Hello"}]},
            )

            # Cost should be tracked
            assert ctx.get_total_cost() > 0.0
            assert result.content == "test response"

            # Operation should be recorded
            operations = ctx.get_operation_costs()
            assert len(operations) == 1
            assert operations[0]["node_id"] == "test_node"
            assert operations[0]["model"] == "gpt-4o-mini"
            assert operations[0]["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_multiple_predictions_accumulate_cost(self):
        """Test that multiple predictions accumulate cost correctly."""
        from unittest.mock import AsyncMock, patch

        from nodetool.metadata.types import Message, Provider
        from nodetool.providers.base import BaseProvider, ProviderCapability

        class MockProvider(BaseProvider):
            provider_name = "openai"

            @classmethod
            def required_secrets(cls):
                return []

            def get_capabilities(self):
                return {ProviderCapability.GENERATE_MESSAGE}

            async def generate_message(
                self, messages, model, tools=None, max_tokens=8192, **kwargs
            ):
                self.track_usage(model=model, input_tokens=100, output_tokens=50)
                return Message(role="assistant", content="response")

        ctx = ProcessingContext(user_id="test")
        mock_provider = MockProvider()

        with patch.object(ctx, "get_provider", return_value=mock_provider), patch(
            "nodetool.models.prediction.Prediction.create", new_callable=AsyncMock
        ):
            # First prediction
            await ctx.run_provider_prediction(
                node_id="node_1",
                provider=Provider.OpenAI,
                model="gpt-4o-mini",
                capability=ProviderCapability.GENERATE_MESSAGE,
                params={"messages": []},
            )
            cost_after_first = ctx.get_total_cost()

            # Second prediction
            await ctx.run_provider_prediction(
                node_id="node_2",
                provider=Provider.OpenAI,
                model="gpt-4o-mini",
                capability=ProviderCapability.GENERATE_MESSAGE,
                params={"messages": []},
            )
            cost_after_second = ctx.get_total_cost()

            # Cost should accumulate
            assert cost_after_second > cost_after_first
            assert cost_after_second == pytest.approx(cost_after_first * 2, rel=1e-6)

            # Should have two operations recorded
            operations = ctx.get_operation_costs()
            assert len(operations) == 2

    @pytest.mark.asyncio
    async def test_failed_prediction_does_not_track_cost(self):
        """Test that failed predictions don't track cost in context."""
        from unittest.mock import AsyncMock, patch

        from nodetool.metadata.types import Provider
        from nodetool.providers.base import BaseProvider, ProviderCapability

        class MockProvider(BaseProvider):
            provider_name = "openai"

            @classmethod
            def required_secrets(cls):
                return []

            def get_capabilities(self):
                return {ProviderCapability.GENERATE_MESSAGE}

            async def generate_message(
                self, messages, model, tools=None, max_tokens=8192, **kwargs
            ):
                raise RuntimeError("API error")

        ctx = ProcessingContext(user_id="test")
        mock_provider = MockProvider()

        with patch.object(ctx, "get_provider", return_value=mock_provider), patch(
            "nodetool.models.prediction.Prediction.create", new_callable=AsyncMock
        ):
            with pytest.raises(RuntimeError, match="API error"):
                await ctx.run_provider_prediction(
                    node_id="test_node",
                    provider=Provider.OpenAI,
                    model="gpt-4o-mini",
                    capability=ProviderCapability.GENERATE_MESSAGE,
                    params={"messages": []},
                )

            # Cost should remain zero for failed operations
            assert ctx.get_total_cost() == 0.0
            assert len(ctx.get_operation_costs()) == 0

    @pytest.mark.asyncio
    async def test_cost_tracking_with_different_providers(self):
        """Test cost tracking with different providers."""
        from unittest.mock import AsyncMock, patch

        from nodetool.metadata.types import Message, Provider
        from nodetool.providers.base import BaseProvider, ProviderCapability

        class OpenAIProvider(BaseProvider):
            provider_name = "openai"

            @classmethod
            def required_secrets(cls):
                return []

            def get_capabilities(self):
                return {ProviderCapability.GENERATE_MESSAGE}

            async def generate_message(
                self, messages, model, tools=None, max_tokens=8192, **kwargs
            ):
                self.track_usage(model=model, input_tokens=1000, output_tokens=500)
                return Message(role="assistant", content="OpenAI response")

        class AnthropicProvider(BaseProvider):
            provider_name = "anthropic"

            @classmethod
            def required_secrets(cls):
                return []

            def get_capabilities(self):
                return {ProviderCapability.GENERATE_MESSAGE}

            async def generate_message(
                self, messages, model, tools=None, max_tokens=8192, **kwargs
            ):
                self.track_usage(model=model, input_tokens=1000, output_tokens=500)
                return Message(role="assistant", content="Anthropic response")

        ctx = ProcessingContext(user_id="test")

        async def mock_get_provider(provider_enum):
            if provider_enum == Provider.OpenAI:
                return OpenAIProvider()
            elif provider_enum == Provider.Anthropic:
                return AnthropicProvider()
            raise ValueError("Unknown provider")

        with patch.object(ctx, "get_provider", side_effect=mock_get_provider), patch(
            "nodetool.models.prediction.Prediction.create", new_callable=AsyncMock
        ):
            # OpenAI prediction
            await ctx.run_provider_prediction(
                node_id="node_1",
                provider=Provider.OpenAI,
                model="gpt-4o-mini",
                capability=ProviderCapability.GENERATE_MESSAGE,
                params={"messages": []},
            )

            # Anthropic prediction
            await ctx.run_provider_prediction(
                node_id="node_2",
                provider=Provider.Anthropic,
                model="claude-3-5-sonnet-latest",
                capability=ProviderCapability.GENERATE_MESSAGE,
                params={"messages": []},
            )

            # Both operations should be tracked
            operations = ctx.get_operation_costs()
            assert len(operations) == 2
            assert operations[0]["provider"] == "openai"
            assert operations[1]["provider"] == "anthropic"

            # Total cost should be sum of both
            total_cost = ctx.get_total_cost()
            assert total_cost > 0.0

