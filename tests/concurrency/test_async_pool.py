import asyncio

import pytest

from nodetool.concurrency.async_pool import (
    AsyncPool,
    AsyncPoolClosedError,
    AsyncPoolFullError,
    PoolConfig,
    PoolStats,
)


class TestAsyncPool:
    """Tests for AsyncPool class."""

    def test_init_with_valid_params(self):
        """Test initialization with valid parameters."""
        pool = AsyncPool(max_workers=4)
        assert pool.config.max_workers == 4
        assert pool.config.max_queue_size == 0
        assert pool.config.timeout is None
        assert pool.is_closed is False
        assert pool.active_workers == 0
        assert pool.queued_tasks == 0

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        pool = AsyncPool(max_workers=8, max_queue_size=100, timeout=30.0)
        assert pool.config.max_workers == 8
        assert pool.config.max_queue_size == 100
        assert pool.config.timeout == 30.0

    def test_init_with_invalid_max_workers(self):
        """Test that invalid max_workers raises ValueError."""
        with pytest.raises(ValueError, match="max_workers must be at least 1"):
            AsyncPool(max_workers=0)

        with pytest.raises(ValueError, match="max_workers must be at least 1"):
            AsyncPool(max_workers=-1)

    def test_init_with_invalid_queue_size(self):
        """Test that invalid max_queue_size raises ValueError."""
        with pytest.raises(ValueError, match="max_queue_size must be non-negative"):
            AsyncPool(max_queue_size=-1)

    def test_stats_property(self):
        """Test that stats property returns correct initial values."""
        pool = AsyncPool(max_workers=2)
        stats = pool.stats

        assert isinstance(stats, PoolStats)
        assert stats.total_tasks == 0
        assert stats.completed_tasks == 0
        assert stats.failed_tasks == 0
        assert stats.cancelled_tasks == 0
        assert stats.queued_tasks == 0
        assert stats.active_workers == 0

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_submit_before_start_raises(self):
        """Test that submitting before start raises AsyncPoolClosedError."""
        pool = AsyncPool(max_workers=2)

        async def task():
            return "result"

        with pytest.raises(AsyncPoolClosedError):
            pool.submit(task)

    @pytest.mark.asyncio
    async def test_context_manager_starts_pool(self):
        """Test that context manager starts the pool."""
        pool = AsyncPool(max_workers=2)

        async with pool:
            assert pool.active_workers == 2
            assert pool.is_closed is False

        assert pool.is_closed is True

    @pytest.mark.asyncio
    async def test_basic_task_submission(self):
        """Test basic task submission and result retrieval."""

        async def double(x: int) -> int:
            return x * 2

        async with AsyncPool(max_workers=2) as pool:
            future = pool.submit(double, 5)
            result = await future

        assert result == 10

    @pytest.mark.asyncio
    async def test_multiple_task_submission(self):
        """Test submitting multiple tasks."""

        async def double(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        async with AsyncPool(max_workers=4) as pool:
            futures = [pool.submit(double, i) for i in range(10)]
            results = await pool.gather_results(futures)

        assert results == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

    @pytest.mark.asyncio
    async def test_task_exception_handling(self):
        """Test that exceptions in tasks are captured in the future."""

        async def fail_task():
            raise ValueError("Task failed")

        async with AsyncPool(max_workers=2) as pool:
            future = pool.submit(fail_task)

            with pytest.raises(ValueError, match="Task failed"):
                await future

    @pytest.mark.asyncio
    async def test_gather_results_raise_on_error(self):
        """Test gather_results with raise_on_error=True."""

        async def success_task(x: int) -> int:
            return x

        async def fail_task():
            raise RuntimeError("Error")

        async with AsyncPool(max_workers=2) as pool:
            futures = [
                pool.submit(success_task, 1),
                pool.submit(fail_task),
                pool.submit(success_task, 3),
            ]

            with pytest.raises(RuntimeError):
                await pool.gather_results(futures, raise_on_error=True)

    @pytest.mark.asyncio
    async def test_gather_results_no_raise_on_error(self):
        """Test gather_results with raise_on_error=False."""

        async def success_task(x: int) -> int:
            return x

        async def fail_task():
            raise RuntimeError("Error")

        async with AsyncPool(max_workers=2) as pool:
            futures = [
                pool.submit(success_task, 1),
                pool.submit(fail_task),
                pool.submit(success_task, 3),
            ]

            results = await pool.gather_results(futures, raise_on_error=False)

        assert results[0] == 1
        assert isinstance(results[1], RuntimeError)
        assert results[2] == 3

    @pytest.mark.asyncio
    async def test_map_function(self):
        """Test the map method."""

        async def double(x: int) -> int:
            return x * 2

        async with AsyncPool(max_workers=4) as pool:
            results = await pool.map(double, [1, 2, 3, 4, 5])

        assert results == [2, 4, 6, 8, 10]

    @pytest.mark.asyncio
    async def test_map_with_timeout(self):
        """Test the map method with timeout via gather_results timeout."""

        async def slow_task(x: int) -> int:
            await asyncio.sleep(0.2)
            return x * 2

        async with AsyncPool(max_workers=2) as pool:
            futures = [pool.submit(slow_task, i) for i in range(3)]

            with pytest.raises(asyncio.CancelledError):
                await pool.gather_results(futures, timeout=0.1)

    @pytest.mark.asyncio
    async def test_drain_waits_for_queued_tasks(self):
        """Test that drain waits for all queued tasks."""

        async def slow_task(x: int) -> int:
            await asyncio.sleep(0.1)
            return x

        async with AsyncPool(max_workers=2) as pool:
            for i in range(5):
                pool.submit(slow_task, i)

            await pool.drain()

            stats = pool.stats
            assert stats.completed_tasks == 5

    @pytest.mark.asyncio
    async def test_submit_with_timeout(self):
        """Test submit_with_timeout method."""

        async def slow_task():
            await asyncio.sleep(0.1)
            return "done"

        async with AsyncPool(max_workers=2) as pool:
            result = await pool.submit_with_timeout(slow_task, timeout=1.0)

        assert result == "done"

    @pytest.mark.asyncio
    async def test_submit_with_timeout_expires(self):
        """Test submit_with_timeout with expired timeout."""

        async def slow_task():
            await asyncio.sleep(0.5)
            return "done"

        async with AsyncPool(max_workers=2) as pool:
            with pytest.raises(asyncio.TimeoutError):
                await pool.submit_with_timeout(slow_task, timeout=0.05)

    @pytest.mark.asyncio
    async def test_max_workers_limit(self):
        """Test that max_workers limits concurrent execution."""
        execution_order: list[int] = []
        start_times: list[float] = []

        async def track_execution(task_id: int) -> int:
            start_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.1)
            execution_order.append(task_id)
            return task_id

        async with AsyncPool(max_workers=2) as pool:
            futures = [pool.submit(track_execution, i) for i in range(4)]
            await pool.gather_results(futures)

        assert pool.active_workers <= 2

    @pytest.mark.asyncio
    async def test_close_prevents_new_submissions(self):
        """Test that close prevents new submissions."""

        async def task():
            return "result"

        async with AsyncPool(max_workers=2) as pool:
            pool.submit(task)
            await pool.drain()

        with pytest.raises(AsyncPoolClosedError):
            pool.submit(task)

    @pytest.mark.asyncio
    async def test_close_waits_for_pending_tasks(self):
        """Test that close waits for pending tasks."""
        results: list[int] = []

        async def slow_task(x: int) -> int:
            await asyncio.sleep(0.2)
            results.append(x)
            return x

        pool = AsyncPool(max_workers=2)
        await pool.start()

        pool.submit(slow_task, 1)
        pool.submit(slow_task, 2)

        await pool.close()

        assert len(results) == 2
        assert pool.active_workers == 0

    @pytest.mark.asyncio
    async def test_queue_full_raises_error(self):
        """Test that full queue raises AsyncPoolFullError."""

        async def task():
            return "result"

        pool = AsyncPool(max_workers=1, max_queue_size=2)
        await pool.start()

        pool.submit(task)
        pool.submit(task)

        with pytest.raises(AsyncPoolFullError):
            pool.submit(task)

        await pool.close()

    @pytest.mark.asyncio
    async def test_empty_map(self):
        """Test map with empty list."""

        async def double(x: int) -> int:
            return x * 2

        async with AsyncPool(max_workers=2) as pool:
            results = await pool.map(double, [])

        assert results == []

    @pytest.mark.asyncio
    async def test_empty_gather(self):
        """Test gather_results with empty list."""
        async with AsyncPool(max_workers=2) as pool:
            results = await pool.gather_results([])

        assert results == []

    @pytest.mark.asyncio
    async def test_stats_update_on_completion(self):
        """Test that stats are updated correctly."""

        async def success_task(x: int) -> int:
            return x

        async def fail_task():
            raise ValueError("Error")

        async with AsyncPool(max_workers=2) as pool:
            futures = [
                pool.submit(success_task, 1),
                pool.submit(success_task, 2),
                pool.submit(fail_task),
            ]
            await pool.gather_results(futures, raise_on_error=False)

        stats = pool.stats
        assert stats.total_tasks == 3
        assert stats.completed_tasks == 2
        assert stats.failed_tasks == 1

    @pytest.mark.asyncio
    async def test_config_property(self):
        """Test that config property returns correct configuration."""
        pool = AsyncPool(max_workers=4, max_queue_size=10, timeout=30.0)
        config = pool.config

        assert isinstance(config, PoolConfig)
        assert config.max_workers == 4
        assert config.max_queue_size == 10
        assert config.timeout == 30.0

    @pytest.mark.asyncio
    async def test_double_close_is_safe(self):
        """Test that closing twice is safe."""
        pool = AsyncPool(max_workers=2)
        await pool.start()

        await pool.close()
        await pool.close()

        assert pool.is_closed is True

    @pytest.mark.asyncio
    async def test_context_manager_exception_during_execution(self):
        """Test that failed tasks set their exception on the future."""

        async def fail_task():
            await asyncio.sleep(0.05)
            raise RuntimeError("Error")

        async with AsyncPool(max_workers=2) as pool:
            future = pool.submit(fail_task)

            with pytest.raises(RuntimeError):
                await future

        assert pool.is_closed is True

    @pytest.mark.asyncio
    async def test_gather_results_with_timeout(self):
        """Test gather_results with a timeout covering all futures."""

        async def slow_task(x: int) -> int:
            await asyncio.sleep(0.3)
            return x

        async with AsyncPool(max_workers=2) as pool:
            futures = [pool.submit(slow_task, i) for i in range(3)]

            with pytest.raises(asyncio.CancelledError):
                await pool.gather_results(futures, timeout=0.1)
