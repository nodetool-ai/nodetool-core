import asyncio
import time

import pytest

from nodetool.concurrency.async_utils import AsyncSemaphore, gather_with_concurrency


class TestAsyncSemaphore:
    """Tests for AsyncSemaphore class."""

    def test_init_with_valid_max_tasks(self):
        """Test initialization with positive max_tasks."""
        sem = AsyncSemaphore(5)
        assert sem.max_tasks == 5
        assert sem.available == 5

    def test_init_with_invalid_max_tasks(self):
        """Test that invalid max_tasks raises ValueError."""
        with pytest.raises(ValueError, match="max_tasks must be a positive integer"):
            AsyncSemaphore(0)

        with pytest.raises(ValueError, match="max_tasks must be a positive integer"):
            AsyncSemaphore(-1)

    def test_acquire_and_release(self):
        """Test basic acquire and release functionality."""
        sem = AsyncSemaphore(2)

        async def run():
            assert sem.available == 2

            await sem.acquire()
            assert sem.available == 1

            await sem.acquire()
            assert sem.available == 0

            sem.release()
            assert sem.available == 1

            sem.release()
            assert sem.available == 2

        asyncio.run(run())

    @pytest.mark.asyncio
    async def test_acquire_no_timeout(self):
        """Test acquire without timeout waits indefinitely."""
        sem = AsyncSemaphore(1)

        await sem.acquire()
        assert sem.locked()

        acquire_task = asyncio.create_task(sem.acquire())
        await asyncio.sleep(0.1)
        assert not acquire_task.done()

        sem.release()
        await acquire_task
        assert sem.available == 0

    @pytest.mark.asyncio
    async def test_acquire_with_timeout_success(self):
        """Test acquire with timeout succeeds when slot available."""
        sem = AsyncSemaphore(1)

        result = await sem.acquire(timeout=1.0)
        assert result is True
        assert sem.available == 0

    @pytest.mark.asyncio
    async def test_acquire_with_timeout_expires(self):
        """Test acquire with timeout expires when no slots available."""
        sem = AsyncSemaphore(1)

        await sem.acquire()
        assert sem.locked()

        start = time.time()
        result = await sem.acquire(timeout=0.2)
        elapsed = time.time() - start

        assert result is False
        assert 0.15 <= elapsed < 0.5

    @pytest.mark.asyncio
    async def test_acquire_with_zero_timeout(self):
        """Test acquire with zero timeout returns immediately if available."""
        sem = AsyncSemaphore(1)

        result = await sem.acquire(timeout=0)
        assert result is True
        assert sem.available == 0

        result = await sem.acquire(timeout=0)
        assert result is False

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager usage."""
        sem = AsyncSemaphore(1)

        async with sem:
            assert sem.available == 0
            assert sem.locked()

        assert sem.available == 1
        assert not sem.locked()

    @pytest.mark.asyncio
    async def test_context_manager_exception(self):
        """Test context manager releases on exception."""
        sem = AsyncSemaphore(1)

        with pytest.raises(ValueError):
            async with sem:
                assert sem.available == 0
                raise ValueError("test error")

        assert sem.available == 1

    @pytest.mark.asyncio
    async def test_multiple_concurrent_acquires(self):
        """Test multiple tasks acquiring the semaphore concurrently."""
        sem = AsyncSemaphore(2)
        acquired_count = 0
        lock = asyncio.Lock()

        async def acquire_and_hold():
            nonlocal acquired_count
            async with sem:
                async with lock:
                    acquired_count += 1
                await asyncio.sleep(0.1)

        tasks = [acquire_and_hold() for _ in range(4)]
        await asyncio.gather(*tasks)

        assert acquired_count == 4


class TestGatherWithConcurrency:
    """Tests for gather_with_concurrency function."""

    @pytest.mark.asyncio
    async def test_empty_list(self):
        """Test with empty list returns empty list."""
        result = await gather_with_concurrency([], max_concurrent=5)
        assert result == []

    @pytest.mark.asyncio
    async def test_invalid_max_concurrent(self):
        """Test that invalid max_concurrent raises ValueError."""
        with pytest.raises(ValueError, match="max_concurrent must be a positive integer"):
            await gather_with_concurrency([asyncio.sleep(0)], max_concurrent=0)

        with pytest.raises(ValueError, match="max_concurrent must be a positive integer"):
            await gather_with_concurrency([asyncio.sleep(0)], max_concurrent=-1)

    @pytest.mark.asyncio
    async def test_sequential_execution(self):
        """Test that max_concurrent=1 runs tasks sequentially."""
        execution_order = []
        lock = asyncio.Lock()

        async def task(n):
            async with lock:
                execution_order.append(n)
            await asyncio.sleep(0.05)
            return n * 2

        tasks = [task(i) for i in range(3)]
        results = await gather_with_concurrency(tasks, max_concurrent=1)

        assert results == [0, 2, 4]
        assert execution_order == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Test that max_concurrent > 1 allows concurrent execution."""
        start_times = []
        lock = asyncio.Lock()

        async def task(n):
            async with lock:
                start_times.append(n)
            await asyncio.sleep(0.1)
            return n * 2

        tasks = [task(i) for i in range(3)]
        results = await gather_with_concurrency(tasks, max_concurrent=3)

        assert results == [0, 2, 4]
        assert len(start_times) == 3
        assert start_times[0] < start_times[2] < start_times[1] or start_times[0] < start_times[1] < start_times[2]

    @pytest.mark.asyncio
    async def test_respects_limit(self):
        """Test that execution respects max_concurrent limit."""
        max_concurrent = 2
        active_count = 0
        max_active_seen = 0
        lock = asyncio.Lock()

        async def task(n):
            nonlocal active_count, max_active_seen

            async with lock:
                active_count += 1
                max_active_seen = max(max_active_seen, active_count)

            await asyncio.sleep(0.1)

            async with lock:
                active_count -= 1

            return n

        tasks = [task(i) for i in range(6)]
        await gather_with_concurrency(tasks, max_concurrent=max_concurrent)

        assert max_active_seen == max_concurrent

    @pytest.mark.asyncio
    async def test_preserves_order(self):
        """Test that results are returned in the same order as input."""

        async def task(n):
            await asyncio.sleep(0.1 - n * 0.02)
            return n

        tasks = [task(i) for i in range(5)]
        output = await gather_with_concurrency(tasks, max_concurrent=3)

        assert output == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_exception_propagation(self):
        """Test that exceptions are propagated correctly."""

        async def failing_task():
            await asyncio.sleep(0.01)
            raise ValueError("test error")

        tasks = [asyncio.sleep(0.01), failing_task()]

        with pytest.raises(ValueError, match="test error"):
            await gather_with_concurrency(tasks, max_concurrent=2)
