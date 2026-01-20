import asyncio

import pytest

from nodetool.concurrency.async_priority_queue import AsyncPriorityQueue


class TestAsyncPriorityQueueInit:
    """Tests for AsyncPriorityQueue initialization."""

    def test_init_unlimited(self):
        """Test that queue can be initialized with no max size."""
        queue = AsyncPriorityQueue()
        assert queue.max_size is None
        assert queue.qsize == 0
        assert queue.empty

    def test_init_with_max_size(self):
        """Test that queue can be initialized with a max size."""
        queue = AsyncPriorityQueue(max_size=10)
        assert queue.max_size == 10
        assert queue.qsize == 0
        assert queue.empty

    def test_init_invalid_max_size_zero(self):
        """Test that initializing with max_size=0 raises ValueError."""
        with pytest.raises(ValueError):
            AsyncPriorityQueue(max_size=0)

    def test_init_invalid_max_size_negative(self):
        """Test that initializing with negative max_size raises ValueError."""
        with pytest.raises(ValueError):
            AsyncPriorityQueue(max_size=-1)


class TestAsyncPriorityQueueProperties:
    """Tests for AsyncPriorityQueue properties."""

    def test_qsize_empty(self):
        """Test that qsize is 0 for empty queue."""
        queue = AsyncPriorityQueue()
        assert queue.qsize == 0

    def test_qsize_with_items(self):
        """Test that qsize reflects number of items."""
        queue = AsyncPriorityQueue()
        asyncio.run(queue.put(0, "item1"))
        asyncio.run(queue.put(1, "item2"))
        assert queue.qsize == 2

    def test_empty_true(self):
        """Test that empty returns True for empty queue."""
        queue = AsyncPriorityQueue()
        assert queue.empty

    def test_empty_false(self):
        """Test that empty returns False when items are present."""
        queue = AsyncPriorityQueue()
        asyncio.run(queue.put(0, "item"))
        assert not queue.empty

    def test_full_false_unlimited(self):
        """Test that full returns False for unlimited queue."""
        queue = AsyncPriorityQueue()
        assert not queue.full

    def test_full_false_with_capacity(self):
        """Test that full returns False when under capacity."""
        queue = AsyncPriorityQueue(max_size=5)
        for i in range(3):
            asyncio.run(queue.put(i, f"item{i}"))
        assert not queue.full

    def test_full_true_at_capacity(self):
        """Test that full returns True when at capacity."""
        queue = AsyncPriorityQueue(max_size=3)
        for i in range(3):
            asyncio.run(queue.put(i, f"item{i}"))
        assert queue.full


class TestAsyncPriorityQueuePut:
    """Tests for AsyncPriorityQueue.put()."""

    def test_put_single_item(self):
        """Test putting a single item."""
        queue = AsyncPriorityQueue()
        asyncio.run(queue.put(0, "item"))
        assert queue.qsize == 1

    def test_put_multiple_items(self):
        """Test putting multiple items."""
        queue = AsyncPriorityQueue()
        asyncio.run(queue.put(2, "low"))
        asyncio.run(queue.put(0, "high"))
        asyncio.run(queue.put(1, "medium"))
        assert queue.qsize == 3

    def test_put_full_queue_raises(self):
        """Test that putting to full queue raises QueueFull."""
        queue = AsyncPriorityQueue(max_size=2)
        asyncio.run(queue.put(0, "item1"))
        asyncio.run(queue.put(1, "item2"))
        with pytest.raises(asyncio.QueueFull):
            asyncio.run(queue.put(2, "item3"))


class TestAsyncPriorityQueueGet:
    """Tests for AsyncPriorityQueue.get()."""

    @pytest.mark.asyncio
    async def test_get_returns_priority_order(self):
        """Test that get returns items in priority order."""
        queue: AsyncPriorityQueue[str] = AsyncPriorityQueue()
        await queue.put(2, "low")
        await queue.put(0, "high")
        await queue.put(1, "medium")

        assert await queue.get() == "high"
        assert await queue.get() == "medium"
        assert await queue.get() == "low"

    @pytest.mark.asyncio
    async def test_get_empty_queue_raises(self):
        """Test that getting from empty queue raises QueueEmpty."""
        queue: AsyncPriorityQueue[str] = AsyncPriorityQueue()
        with pytest.raises(asyncio.QueueEmpty):
            await queue.get()

    @pytest.mark.asyncio
    async def test_get_with_timeout_success(self):
        """Test that get with timeout succeeds when item is available."""
        queue: AsyncPriorityQueue[str] = AsyncPriorityQueue()
        await queue.put(0, "item")

        result = await queue.get(timeout=1.0)
        assert result == "item"

    @pytest.mark.asyncio
    async def test_get_with_timeout_expires(self):
        """Test that get with timeout raises TimeoutError when empty."""
        queue: AsyncPriorityQueue[str] = AsyncPriorityQueue()
        import time

        start = time.time()
        with pytest.raises(TimeoutError):
            await queue.get(timeout=0.1)
        elapsed = time.time() - start
        assert 0.05 <= elapsed < 0.3

    @pytest.mark.asyncio
    async def test_get_zero_timeout_empty(self):
        """Test that get with zero timeout raises immediately on empty queue."""
        queue: AsyncPriorityQueue[str] = AsyncPriorityQueue()
        with pytest.raises(asyncio.QueueEmpty):
            await queue.get(timeout=0)


class TestAsyncPriorityQueueGetNowait:
    """Tests for AsyncPriorityQueue.get_nowait()."""

    def test_get_nowait_success(self):
        """Test get_nowait returns item immediately."""
        queue: AsyncPriorityQueue[str] = AsyncPriorityQueue()
        asyncio.run(queue.put(0, "item"))
        result = queue.get_nowait()
        assert result == "item"
        assert queue.empty

    def test_get_nowait_empty_raises(self):
        """Test get_nowait raises on empty queue."""
        queue: AsyncPriorityQueue[str] = AsyncPriorityQueue()
        with pytest.raises(asyncio.QueueEmpty):
            queue.get_nowait()


class TestAsyncPriorityQueuePutNowait:
    """Tests for AsyncPriorityQueue.put_nowait()."""

    def test_put_nowait_success(self):
        """Test put_nowait adds item without waiting."""
        queue: AsyncPriorityQueue[str] = AsyncPriorityQueue()
        queue.put_nowait(0, "item")
        assert queue.qsize == 1

    def test_put_nowait_full_raises(self):
        """Test put_nowait raises on full queue."""
        queue: AsyncPriorityQueue[str] = AsyncPriorityQueue(max_size=2)
        queue.put_nowait(0, "item1")
        queue.put_nowait(1, "item2")
        with pytest.raises(asyncio.QueueFull):
            queue.put_nowait(2, "item3")


class TestAsyncPriorityQueuePeek:
    """Tests for AsyncPriorityQueue.peek()."""

    def test_peek_empty_returns_none(self):
        """Test peek returns None for empty queue."""
        queue: AsyncPriorityQueue[str] = AsyncPriorityQueue()
        assert queue.peek() is None

    def test_peek_returns_highest_priority(self):
        """Test peek returns highest priority item without removing."""
        queue: AsyncPriorityQueue[str] = AsyncPriorityQueue()
        asyncio.run(queue.put(2, "low"))
        asyncio.run(queue.put(0, "high"))
        asyncio.run(queue.put(1, "medium"))

        assert queue.peek() == "high"
        assert queue.qsize == 3


class TestAsyncPriorityQueueClear:
    """Tests for AsyncPriorityQueue.clear()."""

    def test_clear_returns_all_items(self):
        """Test clear returns all items."""
        queue: AsyncPriorityQueue[str] = AsyncPriorityQueue()
        asyncio.run(queue.put(0, "item1"))
        asyncio.run(queue.put(1, "item2"))

        items = queue.clear()
        assert len(items) == 2
        assert queue.empty

    def test_clear_empty_queue(self):
        """Test clear on empty queue returns empty list."""
        queue: AsyncPriorityQueue[str] = AsyncPriorityQueue()
        items = queue.clear()
        assert items == []


class TestAsyncPriorityQueueFIFO:
    """Tests for FIFO ordering when priorities are equal."""

    @pytest.mark.asyncio
    async def test_same_priority_fifo(self):
        """Test that items with same priority are returned in FIFO order."""
        queue: AsyncPriorityQueue[str] = AsyncPriorityQueue()
        await queue.put(1, "first")
        await queue.put(1, "second")
        await queue.put(1, "third")

        assert await queue.get() == "first"
        assert await queue.get() == "second"
        assert await queue.get() == "third"


class TestAsyncPriorityQueueAsyncOperations:
    """Tests for async operations on the queue."""

    @pytest.mark.asyncio
    async def test_concurrent_get(self):
        """Test that multiple tasks can wait for items."""
        queue: AsyncPriorityQueue[str] = AsyncPriorityQueue()

        async def put_after_delay():
            await asyncio.sleep(0.05)
            await queue.put(0, "item")

        put_task = asyncio.create_task(put_after_delay())
        result = await queue.get(timeout=1.0)
        await put_task

        assert result == "item"

    @pytest.mark.asyncio
    async def test_concurrent_put_and_get(self):
        """Test concurrent put and get operations."""
        queue: AsyncPriorityQueue[str] = AsyncPriorityQueue()

        results: list[str] = []

        async def producer():
            for i in range(5):
                await queue.put(i % 2, f"item{i}")
                await asyncio.sleep(0.01)

        async def consumer():
            for _ in range(5):
                item = await queue.get(timeout=1.0)
                results.append(item)

        await asyncio.gather(producer(), consumer())

        assert len(results) == 5
        assert "item0" in results
        assert "item2" in results
        assert "item4" in results


class TestAsyncPriorityQueueIterator:
    """Tests for async iteration."""

    @pytest.mark.asyncio
    async def test_async_iteration(self):
        """Test that the queue is async iterable."""
        queue: AsyncPriorityQueue[str] = AsyncPriorityQueue()
        await queue.put(1, "item2")
        await queue.put(0, "item1")

        items = []
        async for item in queue:
            items.append(item)
            if len(items) >= 2:
                break

        assert len(items) == 2
        assert items[0] == "item1"
        assert items[1] == "item2"
