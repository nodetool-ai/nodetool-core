import asyncio

import pytest

from nodetool.concurrency import AsyncQueue, QueueShutdownError


class TestAsyncQueueInitialization:
    """Tests for AsyncQueue initialization."""

    def test_unbounded_queue(self):
        """Test that unbounded queue has None max_size."""
        queue: AsyncQueue[int] = AsyncQueue()
        assert queue.max_size is None
        assert queue.empty()
        assert queue.qsize() == 0

    def test_bounded_queue(self):
        """Test that bounded queue has correct max_size."""
        queue: AsyncQueue[int] = AsyncQueue(max_size=100)
        assert queue.max_size == 100
        assert queue.empty()
        assert queue.qsize() == 0

    def test_invalid_max_size_zero(self):
        """Test that max_size=0 raises ValueError."""
        with pytest.raises(ValueError, match="max_size must be a positive integer"):
            AsyncQueue(max_size=0)

    def test_invalid_max_size_negative(self):
        """Test that negative max_size raises ValueError."""
        with pytest.raises(ValueError, match="max_size must be a positive integer"):
            AsyncQueue(max_size=-1)


class TestAsyncQueueBasicOperations:
    """Tests for basic queue operations."""

    @pytest.mark.asyncio
    async def test_put_and_get(self):
        """Test basic put and get operations."""
        queue: AsyncQueue[int] = AsyncQueue()

        await queue.put(42)
        assert queue.qsize() == 1

        result = await queue.get()
        assert result == 42
        assert queue.empty()

    @pytest.mark.asyncio
    async def test_put_nowait_and_get_nowait(self):
        """Test non-blocking put_nowait and get_nowait."""
        queue: AsyncQueue[int] = AsyncQueue()

        result = queue.put_nowait(42)
        assert result is True
        assert queue.qsize() == 1

        result = queue.get_nowait()
        assert result == 42
        assert queue.empty()

    @pytest.mark.asyncio
    async def test_put_nowait_on_full_queue(self):
        """Test that put_nowait returns False on full queue."""
        queue: AsyncQueue[int] = AsyncQueue(max_size=2)

        queue.put_nowait(1)
        queue.put_nowait(2)
        assert queue.full()

        result = queue.put_nowait(3)
        assert result is False

    @pytest.mark.asyncio
    async def test_get_nowait_on_empty_queue(self):
        """Test that get_nowait returns None on empty queue."""
        queue: AsyncQueue[int] = AsyncQueue()

        result = queue.get_nowait()
        assert result is None

    @pytest.mark.asyncio
    async def test_multiple_puts_and_gets(self):
        """Test multiple items in order."""
        queue: AsyncQueue[int] = AsyncQueue()

        for i in range(10):
            await queue.put(i)

        for i in range(10):
            result = await queue.get()
            assert result == i


class TestAsyncQueueTimeout:
    """Tests for timeout functionality."""

    @pytest.mark.asyncio
    async def test_get_timeout(self):
        """Test that get times out correctly."""
        queue: AsyncQueue[int] = AsyncQueue()

        result = await queue.get(timeout=0.1)
        assert result is None

    @pytest.mark.asyncio
    async def test_put_timeout_on_full_queue(self):
        """Test that put times out on full queue."""
        queue: AsyncQueue[int] = AsyncQueue(max_size=1)

        await queue.put(1)
        result = await queue.put(2, timeout=0.1)
        assert result is False


class TestAsyncQueueShutdown:
    """Tests for queue shutdown."""

    @pytest.mark.asyncio
    async def test_put_raises_after_shutdown(self):
        """Test that put raises QueueShutdownError after shutdown."""
        queue: AsyncQueue[int] = AsyncQueue()
        queue.shutdown()

        with pytest.raises(QueueShutdownError):
            await queue.put(42)

    @pytest.mark.asyncio
    async def test_put_nowait_raises_after_shutdown(self):
        """Test that put_nowait raises QueueShutdownError after shutdown."""
        queue: AsyncQueue[int] = AsyncQueue()
        queue.shutdown()

        with pytest.raises(QueueShutdownError):
            queue.put_nowait(42)

    @pytest.mark.asyncio
    async def test_get_raises_on_empty_shutdown_queue(self):
        """Test that get raises QueueShutdownError on empty shutdown queue."""
        queue: AsyncQueue[int] = AsyncQueue()
        queue.shutdown()

        with pytest.raises(QueueShutdownError):
            await queue.get()

    @pytest.mark.asyncio
    async def test_get_nowait_raises_on_empty_shutdown_queue(self):
        """Test that get_nowait raises QueueShutdownError on empty shutdown queue."""
        queue: AsyncQueue[int] = AsyncQueue()
        queue.shutdown()

        with pytest.raises(QueueShutdownError):
            queue.get_nowait()

    @pytest.mark.asyncio
    async def test_get_returns_items_before_shutdown(self):
        """Test that get returns items added before shutdown."""
        queue: AsyncQueue[int] = AsyncQueue()
        await queue.put(42)
        await queue.put(43)
        queue.shutdown()

        item = await queue.get()
        assert item == 42
        item = await queue.get()
        assert item == 43


class TestAsyncQueueStats:
    """Tests for queue statistics."""

    @pytest.mark.asyncio
    async def test_stats_initial(self):
        """Test initial statistics are zero."""
        queue: AsyncQueue[int] = AsyncQueue(max_size=10)
        stats = queue.stats

        assert stats.current_size == 0
        assert stats.max_size == 10
        assert stats.put_waiters == 0
        assert stats.get_waiters == 0
        assert stats.total_puts == 0
        assert stats.total_gets == 0

    @pytest.mark.asyncio
    async def test_stats_update(self):
        """Test statistics update correctly."""
        queue: AsyncQueue[int] = AsyncQueue(max_size=10)

        await queue.put(1)
        await queue.put(2)
        await queue.get()

        stats = queue.stats
        assert stats.current_size == 1
        assert stats.total_puts == 2
        assert stats.total_gets == 1


class TestAsyncQueueRepr:
    """Tests for queue string representation."""

    def test_repr_unbounded_active(self):
        """Test repr for unbounded active queue."""
        queue: AsyncQueue[int] = AsyncQueue()
        assert "AsyncQueue" in repr(queue)
        assert "unbounded" in repr(queue) or "None" in repr(queue)

    def test_repr_bounded_active(self):
        """Test repr for bounded active queue."""
        queue: AsyncQueue[int] = AsyncQueue(max_size=10)
        queue.put_nowait(1)
        assert "AsyncQueue" in repr(queue)
        assert "1/10" in repr(queue)
        assert "active" in repr(queue)

    def test_repr_shutdown(self):
        """Test repr for shutdown queue."""
        queue: AsyncQueue[int] = AsyncQueue()
        queue.shutdown()
        assert "shutdown" in repr(queue)


class TestAsyncQueueEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_empty_queue_full_false(self):
        """Test that empty bounded queue reports full=False."""
        queue: AsyncQueue[int] = AsyncQueue(max_size=10)
        assert not queue.full()

    @pytest.mark.asyncio
    async def test_unbounded_queue_never_full(self):
        """Test that unbounded queue never reports full."""
        queue: AsyncQueue[int] = AsyncQueue()
        for i in range(1000):
            await queue.put(i)
            assert not queue.full()
