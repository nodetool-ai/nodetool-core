import asyncio

import pytest

from nodetool.concurrency.batching import batched_async_iterable, process_in_batches


class TestBatchedAsyncIterable:
    """Tests for batched_async_iterable function."""

    def test_empty_list(self):
        """Test that empty list yields nothing."""
        result = []

        async def collect():
            async for batch in batched_async_iterable([], 10):
                result.append(batch)

        asyncio.run(collect())
        assert result == []

    def test_single_batch(self):
        """Test that items fitting in one batch yield single list."""
        result = []

        async def collect():
            async for batch in batched_async_iterable([1, 2, 3], 10):
                result.append(batch)

        asyncio.run(collect())
        assert result == [[1, 2, 3]]

    def test_multiple_batches(self):
        """Test that items are split into correct batch sizes."""
        result = []

        async def collect():
            async for batch in batched_async_iterable([1, 2, 3, 4, 5], 2):
                result.append(batch)

        asyncio.run(collect())
        assert result == [[1, 2], [3, 4], [5]]

    def test_exact_batch_size(self):
        """Test that items exactly matching batch size work correctly."""
        result = []

        async def collect():
            async for batch in batched_async_iterable([1, 2, 3, 4], 2):
                result.append(batch)

        asyncio.run(collect())
        assert result == [[1, 2], [3, 4]]

    def test_invalid_batch_size_zero(self):
        """Test that batch_size=0 raises ValueError."""

        async def iterate():
            async for _batch in batched_async_iterable([1, 2, 3], 0):
                pass

        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            asyncio.run(iterate())

    def test_invalid_batch_size_negative(self):
        """Test that negative batch_size raises ValueError."""

        async def iterate():
            async for _batch in batched_async_iterable([1, 2, 3], -1):
                pass

        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            asyncio.run(iterate())

    def test_batch_size_one(self):
        """Test that batch_size=1 yields individual items."""
        result = []

        async def collect():
            async for batch in batched_async_iterable([1, 2, 3], 1):
                result.append(batch)

        asyncio.run(collect())
        assert result == [[1], [2], [3]]


class TestProcessInBatches:
    """Tests for process_in_batches function."""

    @pytest.mark.asyncio
    async def test_empty_list(self):
        """Test that empty list returns empty list."""

        async def processor(x):
            return x

        result = await process_in_batches(
            items=[],
            processor=processor,
            batch_size=10,
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_single_batch(self):
        """Test that single batch is processed correctly."""
        results = []

        async def processor(batch):
            results.append(batch)
            return len(batch)

        result = await process_in_batches(
            items=[1, 2, 3],
            processor=processor,
            batch_size=10,
        )
        assert result == [3]
        assert results == [[1, 2, 3]]

    @pytest.mark.asyncio
    async def test_multiple_batches_sequential(self):
        """Test that multiple batches are processed sequentially."""
        batch_log = []

        async def processor(batch):
            batch_log.append(batch)
            return f"processed_{len(batch)}"

        result = await process_in_batches(
            items=[1, 2, 3, 4, 5],
            processor=processor,
            batch_size=2,
            max_concurrent=1,
        )
        assert result == ["processed_2", "processed_2", "processed_1"]
        assert batch_log == [[1, 2], [3, 4], [5]]

    @pytest.mark.asyncio
    async def test_concurrent_batch_processing(self):
        """Test that batches are processed concurrently with max_concurrent."""
        start_times = []
        lock = asyncio.Lock()

        async def slow_processor(batch):
            async with lock:
                start_times.append((batch, asyncio.get_event_loop().time()))
            await asyncio.sleep(0.1)
            return len(batch)

        result = await process_in_batches(
            items=[1, 2, 3, 4, 5, 6],
            processor=slow_processor,
            batch_size=2,
            max_concurrent=2,
        )
        assert result == [2, 2, 2]
        assert len(start_times) == 3
        times = [t for _, t in start_times]
        assert times[1] - times[0] < 0.05  # Second batch started before first finished

    @pytest.mark.asyncio
    async def test_invalid_batch_size(self):
        """Test that invalid batch_size raises ValueError."""

        async def processor(x):
            return x

        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            await process_in_batches(
                items=[1, 2, 3],
                processor=processor,
                batch_size=0,
            )

    @pytest.mark.asyncio
    async def test_invalid_max_concurrent(self):
        """Test that invalid max_concurrent raises ValueError."""

        async def processor(x):
            return x

        with pytest.raises(ValueError, match="max_concurrent must be a positive integer"):
            await process_in_batches(
                items=[1, 2, 3],
                processor=processor,
                batch_size=2,
                max_concurrent=0,
            )

    @pytest.mark.asyncio
    async def test_processor_exception_propagates(self):
        """Test that processor exceptions are propagated."""

        async def failing_processor(batch):
            raise ValueError("batch processing failed")

        with pytest.raises(ValueError, match="batch processing failed"):
            await process_in_batches(
                items=[1, 2, 3],
                processor=failing_processor,
                batch_size=2,
            )

    @pytest.mark.asyncio
    async def test_results_preserved_in_order(self):
        """Test that results are returned in the order of input batches."""
        call_order = []

        async def processor(batch):
            call_order.append(tuple(batch))
            return f"batch_{tuple(batch)}"

        result = await process_in_batches(
            items=[1, 2, 3, 4, 5, 6],
            processor=processor,
            batch_size=2,
            max_concurrent=3,
        )
        assert result == [
            "batch_(1, 2)",
            "batch_(3, 4)",
            "batch_(5, 6)",
        ]
        assert call_order == [(1, 2), (3, 4), (5, 6)]

    @pytest.mark.asyncio
    async def test_max_concurrent_one_is_sequential(self):
        """Test that max_concurrent=1 processes batches sequentially."""
        execution_order = []

        async def processor(batch):
            execution_order.append(tuple(batch))
            await asyncio.sleep(0.05)
            return tuple(batch)

        result = await process_in_batches(
            items=[1, 2, 3, 4],
            processor=processor,
            batch_size=2,
            max_concurrent=1,
        )
        assert result == [(1, 2), (3, 4)]
        assert execution_order == [(1, 2), (3, 4)]

    @pytest.mark.asyncio
    async def test_async_processor(self):
        """Test that async processor functions work correctly."""
        call_count = 0

        async def async_processor(batch):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return sum(batch)

        result = await process_in_batches(
            items=[1, 2, 3, 4],
            processor=async_processor,
            batch_size=2,
        )
        assert result == [3, 7]
        assert call_count == 2
