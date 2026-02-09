"""Tests for AsyncGeneratorOrchestrator class."""

import asyncio

import pytest

from nodetool.concurrency.async_generator_orchestrator import (
    AsyncGeneratorOrchestrator,
)


class TestAsyncGeneratorOrchestrator:
    """Tests for AsyncGeneratorOrchestrator class."""

    def test_init_with_generators(self):
        """Test initialization with generators."""
        async def gen1():
            yield 1

        async def gen2():
            yield 2

        orchestrator = AsyncGeneratorOrchestrator(gen1(), gen2())
        assert len(orchestrator._generators) == 2

    def test_init_without_generators_raises_error(self):
        """Test that initialization without generators raises ValueError."""
        with pytest.raises(ValueError, match="At least one generator must be provided"):
            AsyncGeneratorOrchestrator()

    @pytest.mark.asyncio
    async def test_round_robin_basic(self):
        """Test basic round-robin functionality."""

        async def gen1():
            for i in range(3):
                yield f"gen1-{i}"

        async def gen2():
            for i in range(3):
                yield f"gen2-{i}"

        orchestrator = AsyncGeneratorOrchestrator(gen1(), gen2())
        result = [item async for item in orchestrator.round_robin()]

        # Should alternate between generators
        assert result == ["gen1-0", "gen2-0", "gen1-1", "gen2-1", "gen1-2", "gen2-2"]

    @pytest.mark.asyncio
    async def test_round_robin_uneven_lengths(self):
        """Test round-robin with generators of different lengths."""

        async def gen_long():
            for i in range(5):
                yield f"long-{i}"

        async def gen_short():
            for i in range(2):
                yield f"short-{i}"

        orchestrator = AsyncGeneratorOrchestrator(gen_long(), gen_short())
        result = [item async for item in orchestrator.round_robin()]

        # Should interleave until short is exhausted, then continue with long
        assert result == [
            "long-0",
            "short-0",
            "long-1",
            "short-1",
            "long-2",
            "long-3",
            "long-4",
        ]

    @pytest.mark.asyncio
    async def test_round_robin_single_generator(self):
        """Test round-robin with a single generator."""

        async def gen():
            for i in range(3):
                yield i

        orchestrator = AsyncGeneratorOrchestrator(gen())
        result = [item async for item in orchestrator.round_robin()]

        assert result == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_round_robin_three_generators(self):
        """Test round-robin with three generators."""

        async def gen1():
            for i in range(2):
                yield f"a{i}"

        async def gen2():
            for i in range(2):
                yield f"b{i}"

        async def gen3():
            for i in range(2):
                yield f"c{i}"

        orchestrator = AsyncGeneratorOrchestrator(gen1(), gen2(), gen3())
        result = [item async for item in orchestrator.round_robin()]

        assert result == ["a0", "b0", "c0", "a1", "b1", "c1"]

    @pytest.mark.asyncio
    async def test_round_robin_empty_generators(self):
        """Test round-robin with empty generators."""

        async def gen_empty():
            return
            yield  # type: ignore[unreachable]

        async def gen_items():
            for i in range(2):
                yield i

        orchestrator = AsyncGeneratorOrchestrator(gen_empty(), gen_items())
        result = [item async for item in orchestrator.round_robin()]

        assert result == [0, 1]

    @pytest.mark.asyncio
    async def test_priority_round_robin_basic(self):
        """Test basic priority round-robin functionality."""

        async def high():
            for i in range(10):
                yield f"high-{i}"

        async def low():
            for i in range(5):
                yield f"low-{i}"

        orchestrator = AsyncGeneratorOrchestrator(high(), low())
        result = [item async for item in orchestrator.priority_round_robin([2, 1])]

        # Should yield 2 from high, 1 from low, repeat
        assert result[:6] == ["high-0", "high-1", "low-0", "high-2", "high-3", "low-1"]

    @pytest.mark.asyncio
    async def test_priority_round_robin_equal_priorities(self):
        """Test priority round-robin with equal priorities (should be like round-robin)."""

        async def gen1():
            for i in range(3):
                yield f"a{i}"

        async def gen2():
            for i in range(3):
                yield f"b{i}"

        orchestrator = AsyncGeneratorOrchestrator(gen1(), gen2())
        result = [item async for item in orchestrator.priority_round_robin([1, 1])]

        assert result == ["a0", "b0", "a1", "b1", "a2", "b2"]

    @pytest.mark.asyncio
    async def test_priority_round_robin_wrong_length_raises_error(self):
        """Test that wrong priorities length raises ValueError."""

        async def gen1():
            yield 1

        async def gen2():
            yield 2

        orchestrator = AsyncGeneratorOrchestrator(gen1(), gen2())

        with pytest.raises(ValueError, match=r"Priorities length .* must match generators count"):
            async for _ in orchestrator.priority_round_robin([1]):
                pass

    @pytest.mark.asyncio
    async def test_priority_round_robin_zero_priority_raises_error(self):
        """Test that zero priority raises ValueError."""

        async def gen1():
            yield 1

        async def gen2():
            yield 2

        orchestrator = AsyncGeneratorOrchestrator(gen1(), gen2())

        with pytest.raises(ValueError, match="All priorities must be at least 1"):
            async for _ in orchestrator.priority_round_robin([0, 1]):
                pass

    @pytest.mark.asyncio
    async def test_priority_round_robin_three_generators(self):
        """Test priority round-robin with three generators."""

        async def high():
            for i in range(9):
                yield f"h{i}"

        async def medium():
            for i in range(6):
                yield f"m{i}"

        async def low():
            for i in range(3):
                yield f"l{i}"

        orchestrator = AsyncGeneratorOrchestrator(high(), medium(), low())
        result = [item async for item in orchestrator.priority_round_robin([3, 2, 1])]

        # Should yield 3 from high, 2 from medium, 1 from low
        assert result[:6] == ["h0", "h1", "h2", "m0", "m1", "l0"]

    @pytest.mark.asyncio
    async def test_selective_consume_with_condition(self):
        """Test selective_consume with a filtering condition."""

        async def errors():
            yield "error1"
            yield "warning"
            yield "error2"

        async def info():
            yield "info1"
            yield "info2"

        orchestrator = AsyncGeneratorOrchestrator(errors(), info())
        result = [
            item async for item in orchestrator.selective_consume(
                condition=lambda x, i: "error" in x
            )
        ]

        assert result == ["error1", "error2"]

    @pytest.mark.asyncio
    async def test_selective_consume_with_generator_index(self):
        """Test selective_consume with generator index included."""

        async def gen1():
            yield "a"
            yield "b"

        async def gen2():
            yield "c"
            yield "d"

        orchestrator = AsyncGeneratorOrchestrator(gen1(), gen2())
        result = [
            item async for item in orchestrator.selective_consume(
                condition=lambda x, i: True, include_generator_index=True
            )
        ]

        assert result == [("a", 0), ("b", 0), ("c", 1), ("d", 1)]

    @pytest.mark.asyncio
    async def test_selective_consume_no_condition(self):
        """Test selective_consume without condition (should merge all)."""

        async def gen1():
            yield 1
            yield 2

        async def gen2():
            yield 3
            yield 4

        orchestrator = AsyncGeneratorOrchestrator(gen1(), gen2())
        result = [item async for item in orchestrator.selective_consume()]

        # Should consume all items in order (generator 0 first, then generator 1)
        assert result == [1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_selective_consume_filters_by_index(self):
        """Test selective_consume filtering by generator index."""

        async def gen1():
            yield "a1"
            yield "a2"

        async def gen2():
            yield "b1"
            yield "b2"

        async def gen3():
            yield "c1"
            yield "c2"

        orchestrator = AsyncGeneratorOrchestrator(gen1(), gen2(), gen3())
        result = [
            item async for item in orchestrator.selective_consume(
                condition=lambda x, i: i in [0, 2]  # Only from gen1 and gen3
            )
        ]

        assert result == ["a1", "a2", "c1", "c2"]

    @pytest.mark.asyncio
    async def test_fair_merge_basic(self):
        """Test basic fair_merge functionality."""

        async def gen1():
            for i in range(5):
                yield i

        async def gen2():
            for i in range(5):
                yield f"{i}"

        orchestrator = AsyncGeneratorOrchestrator(gen1(), gen2())
        result = [item async for item in orchestrator.fair_merge()]

        # Should interleave fairly
        assert len(result) == 10
        # Check that generators are interleaved
        assert result[0] == 0
        assert result[1] == "0"

    @pytest.mark.asyncio
    async def test_fair_merge_with_slow_generator(self):
        """Test fair_merge with one slow generator."""

        async def fast():
            for i in range(10):
                yield f"fast-{i}"

        async def slow():
            for i in range(3):
                await asyncio.sleep(0.01)
                yield f"slow-{i}"

        orchestrator = AsyncGeneratorOrchestrator(fast(), slow())
        result = [item async for item in orchestrator.fair_merge(timeout=0.05)]

        # Should still include items from slow generator
        assert any("slow" in str(item) for item in result)
        assert len(result) == 13  # 10 from fast, 3 from slow

    @pytest.mark.asyncio
    async def test_fair_merge_custom_timeout(self):
        """Test fair_merge with custom timeout."""

        async def gen1():
            for i in range(3):
                yield i

        async def gen2():
            for i in range(3):
                yield f"{i}"

        orchestrator = AsyncGeneratorOrchestrator(gen1(), gen2())
        result = [item async for item in orchestrator.fair_merge(timeout=1.0)]

        assert len(result) == 6

    @pytest.mark.asyncio
    async def test_race_basic(self):
        """Test basic race functionality."""

        async def delayed():
            await asyncio.sleep(0.05)
            yield "delayed"

        async def immediate():
            yield "immediate"

        orchestrator = AsyncGeneratorOrchestrator(delayed(), immediate())
        result = [item async for item in orchestrator.race()]

        # Immediate should come first
        assert result[0] == "immediate"
        assert result[1] == "delayed"

    @pytest.mark.asyncio
    async def test_race_all_available(self):
        """Test race with all generators having items available."""

        async def gen1():
            yield "a"

        async def gen2():
            yield "b"

        async def gen3():
            yield "c"

        orchestrator = AsyncGeneratorOrchestrator(gen1(), gen2(), gen3())
        result = [item async for item in orchestrator.race()]

        # All items should be yielded
        assert len(result) == 3
        assert set(result) == {"a", "b", "c"}

    @pytest.mark.asyncio
    async def test_race_multiple_items_per_generator(self):
        """Test race with generators yielding multiple items."""

        async def gen1():
            yield "a1"
            await asyncio.sleep(0.01)
            yield "a2"

        async def gen2():
            await asyncio.sleep(0.005)
            yield "b1"
            yield "b2"

        orchestrator = AsyncGeneratorOrchestrator(gen1(), gen2())
        result = [item async for item in orchestrator.race()]

        # All items should be yielded
        assert len(result) == 4
        assert set(result) == {"a1", "a2", "b1", "b2"}

    @pytest.mark.asyncio
    async def test_race_one_generator_exhausted_early(self):
        """Test race when one generator finishes early."""

        async def short():
            yield "s1"

        async def long():
            yield "l1"
            yield "l2"
            yield "l3"

        orchestrator = AsyncGeneratorOrchestrator(short(), long())
        result = [item async for item in orchestrator.race()]

        # All items should be yielded
        assert len(result) == 4
        assert set(result) == {"s1", "l1", "l2", "l3"}
