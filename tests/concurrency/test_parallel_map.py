import asyncio

import pytest

from nodetool.concurrency.parallel_map import parallel_map


class TestParallelMap:
    """Tests for parallel_map function."""

    @pytest.mark.asyncio
    async def test_empty_list(self):
        """Test that empty list returns empty list."""

        async def mapper(x):
            return x

        result = await parallel_map(
            items=[],
            mapper=mapper,
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_single_item(self):
        """Test that single item is processed correctly."""

        async def mapper(x):
            return x * 2

        result = await parallel_map(
            items=[5],
            mapper=mapper,
        )
        assert result == [10]

    @pytest.mark.asyncio
    async def test_results_preserved_in_order(self):
        """Test that results are returned in the same order as input items."""

        async def mapper(x):
            return x * 2

        result = await parallel_map(
            items=[1, 2, 3, 4, 5],
            mapper=mapper,
        )
        assert result == [2, 4, 6, 8, 10]

    @pytest.mark.asyncio
    async def test_concurrent_processing(self):
        """Test that items are processed concurrently."""
        start_times = []
        lock = asyncio.Lock()

        async def slow_mapper(x):
            async with lock:
                start_times.append((x, asyncio.get_event_loop().time()))
            await asyncio.sleep(0.05)
            return x * 2

        result = await parallel_map(
            items=[1, 2, 3, 4],
            mapper=slow_mapper,
            max_concurrent=2,
        )
        assert result == [2, 4, 6, 8]
        assert len(start_times) == 4
        times = [t for _, t in start_times]
        assert times[1] - times[0] < 0.05  # Second item started before first finished
        assert times[2] - times[1] >= 0.05  # Third item waited for semaphore

    @pytest.mark.asyncio
    async def test_max_concurrent_controls_parallelism(self):
        """Test that max_concurrent limits concurrent operations."""
        active_count = 0
        max_active = 0
        lock = asyncio.Lock()

        async def increment_and_wait(x):
            nonlocal active_count, max_active
            async with lock:
                active_count += 1
                max_active = max(max_active, active_count)
            await asyncio.sleep(0.1)
            async with lock:
                active_count -= 1
            return x

        await parallel_map(
            items=[1, 2, 3, 4, 5, 6],
            mapper=increment_and_wait,
            max_concurrent=3,
        )
        assert max_active == 3

    @pytest.mark.asyncio
    async def test_max_concurrent_one_is_sequential(self):
        """Test that max_concurrent=1 processes items sequentially."""
        execution_order = []

        async def record_order(x):
            execution_order.append(x)
            await asyncio.sleep(0.01)
            return x

        result = await parallel_map(
            items=[1, 2, 3],
            mapper=record_order,
            max_concurrent=1,
        )
        assert result == [1, 2, 3]
        assert execution_order == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_invalid_max_concurrent_zero(self):
        """Test that max_concurrent=0 raises ValueError."""

        async def mapper(x):
            return x

        with pytest.raises(ValueError, match="max_concurrent must be a positive integer"):
            await parallel_map(
                items=[1, 2, 3],
                mapper=mapper,
                max_concurrent=0,
            )

    @pytest.mark.asyncio
    async def test_invalid_max_concurrent_negative(self):
        """Test that negative max_concurrent raises ValueError."""

        async def mapper(x):
            return x

        with pytest.raises(ValueError, match="max_concurrent must be a positive integer"):
            await parallel_map(
                items=[1, 2, 3],
                mapper=mapper,
                max_concurrent=-1,
            )

    @pytest.mark.asyncio
    async def test_exception_propagates(self):
        """Test that mapper exceptions are propagated."""

        async def failing_mapper(x):
            raise ValueError(f"failed on {x}")

        with pytest.raises(ValueError, match="failed on [123]"):
            await parallel_map(
                items=[1, 2, 3],
                mapper=failing_mapper,
            )

    @pytest.mark.asyncio
    async def test_async_mapper(self):
        """Test that async mapper functions work correctly."""

        async def async_mapper(x):
            await asyncio.sleep(0.01)
            return x + 1

        result = await parallel_map(
            items=[1, 2, 3],
            mapper=async_mapper,
        )
        assert result == [2, 3, 4]

    @pytest.mark.asyncio
    async def test_default_max_concurrent(self):
        """Test that default max_concurrent is 10."""
        start_times = []
        lock = asyncio.Lock()

        async def quick_mapper(x):
            async with lock:
                start_times.append((x, asyncio.get_event_loop().time()))
            await asyncio.sleep(0.05)
            return x

        await parallel_map(
            items=list(range(15)),
            mapper=quick_mapper,
        )
        assert len(start_times) == 15
        times = [t for _, t in start_times]
        assert times[9] - times[0] < 0.05  # First 10 started quickly
        assert times[10] - times[9] >= 0.05  # 11th waited for semaphore
