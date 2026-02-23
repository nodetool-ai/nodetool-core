"""Tests for AsyncCountDownLatch class."""

import asyncio

import pytest

from nodetool.concurrency.async_countdown_latch import AsyncCountDownLatch


class TestAsyncCountDownLatch:
    """Tests for AsyncCountDownLatch class."""

    def test_init_positive_count(self):
        """Test initialization with a positive count."""
        latch = AsyncCountDownLatch(5)
        assert latch.count == 5
        assert not latch.is_done()

    def test_init_zero_count(self):
        """Test initialization with zero count."""
        latch = AsyncCountDownLatch(0)
        assert latch.count == 0
        assert latch.is_done()

    def test_init_negative_count_raises(self):
        """Test that negative count raises ValueError."""
        with pytest.raises(ValueError, match="count must be non-negative"):
            AsyncCountDownLatch(-1)

    @pytest.mark.asyncio
    async def test_count_down_by_one(self):
        """Test counting down by one."""
        latch = AsyncCountDownLatch(3)
        assert latch.count == 3

        latch.count_down()
        assert latch.count == 2
        assert not latch.is_done()

        latch.count_down()
        assert latch.count == 1
        assert not latch.is_done()

        latch.count_down()
        assert latch.count == 0
        assert latch.is_done()

    @pytest.mark.asyncio
    async def test_count_down_by_multiple(self):
        """Test counting down by more than one."""
        latch = AsyncCountDownLatch(10)
        latch.count_down(3)
        assert latch.count == 7

        latch.count_down(4)
        assert latch.count == 3

        latch.count_down(3)
        assert latch.count == 0
        assert latch.is_done()

    @pytest.mark.asyncio
    async def test_count_down_zero_decrement_raises(self):
        """Test that decrementing by zero raises ValueError."""
        latch = AsyncCountDownLatch(5)
        with pytest.raises(ValueError, match="decrement amount must be positive"):
            latch.count_down(0)

    @pytest.mark.asyncio
    async def test_count_down_negative_decrement_raises(self):
        """Test that negative decrement raises ValueError."""
        latch = AsyncCountDownLatch(5)
        with pytest.raises(ValueError, match="decrement amount must be positive"):
            latch.count_down(-1)

    @pytest.mark.asyncio
    async def test_count_down_past_zero_raises(self):
        """Test that decrementing past zero raises ValueError."""
        latch = AsyncCountDownLatch(3)
        with pytest.raises(ValueError, match="cannot decrement by"):
            latch.count_down(5)

    @pytest.mark.asyncio
    async def test_count_down_when_zero_raises(self):
        """Test that counting down when already at zero raises RuntimeError."""
        latch = AsyncCountDownLatch(1)
        latch.count_down()
        assert latch.is_done()

        with pytest.raises(RuntimeError, match="countdown latch already at zero"):
            latch.count_down()

    @pytest.mark.asyncio
    async def test_wait_returns_immediately_if_zero(self):
        """Test that wait returns immediately if count is zero."""
        latch = AsyncCountDownLatch(0)
        await latch.wait()  # Should not block
        assert latch.is_done()

    @pytest.mark.asyncio
    async def test_wait_blocks_until_zero(self):
        """Test that wait blocks until count reaches zero."""
        latch = AsyncCountDownLatch(3)
        results = []

        async def waiter():
            results.append("waiting")
            await latch.wait()
            results.append("done")

        async def counter():
            await asyncio.sleep(0.1)
            latch.count_down()
            await asyncio.sleep(0.1)
            latch.count_down()
            await asyncio.sleep(0.1)
            latch.count_down()

        # Start both tasks
        wait_task = asyncio.create_task(waiter())
        counter_task = asyncio.create_task(counter())

        # Wait a bit - waiter should be blocked
        await asyncio.sleep(0.05)
        assert results == ["waiting"]
        assert latch.count == 3

        # Wait for completion
        await wait_task
        await counter_task

        assert results == ["waiting", "done"]
        assert latch.is_done()

    @pytest.mark.asyncio
    async def test_multiple_waiters(self):
        """Test multiple tasks waiting on the same latch."""
        latch = AsyncCountDownLatch(3)
        results = []

        async def waiter(task_id):
            results.append(f"{task_id}_waiting")
            await latch.wait()
            results.append(f"{task_id}_done")

        # Start multiple waiters
        waiters = [asyncio.create_task(waiter(i)) for i in range(5)]

        # Give them time to start waiting
        await asyncio.sleep(0.05)

        # All should be waiting
        assert len([r for r in results if "_waiting" in r]) == 5
        assert len([r for r in results if "_done" in r]) == 0

        # Count down to zero
        latch.count_down(3)

        # All waiters should complete
        await asyncio.gather(*waiters)

        assert len([r for r in results if "_done" in r]) == 5

    @pytest.mark.asyncio
    async def test_wait_timeout_success(self):
        """Test wait_timeout returns True when count reaches zero before timeout."""
        latch = AsyncCountDownLatch(2)
        results = []

        async def countdown_later():
            await asyncio.sleep(0.1)
            latch.count_down(2)

        async def waiter():
            results.append(await latch.wait_timeout(1.0))

        # Start both tasks
        await asyncio.gather(countdown_later(), waiter())

        assert results == [True]
        assert latch.is_done()

    @pytest.mark.asyncio
    async def test_wait_timeout_expires(self):
        """Test wait_timeout raises TimeoutError when timeout expires."""
        latch = AsyncCountDownLatch(5)

        with pytest.raises(TimeoutError):
            await latch.wait_timeout(0.1)

        # Count should not have changed
        assert latch.count == 5

    @pytest.mark.asyncio
    async def test_wait_timeout_invalid_timeout(self):
        """Test wait_timeout raises ValueError for invalid timeout."""
        latch = AsyncCountDownLatch(5)

        with pytest.raises(ValueError, match="timeout must be positive"):
            await latch.wait_timeout(0)

        with pytest.raises(ValueError, match="timeout must be positive"):
            await latch.wait_timeout(-1)

    @pytest.mark.asyncio
    async def test_is_done_property(self):
        """Test is_done property reflects the latch state."""
        latch = AsyncCountDownLatch(2)

        assert not latch.is_done()
        assert latch.count == 2

        latch.count_down()
        assert not latch.is_done()
        assert latch.count == 1

        latch.count_down()
        assert latch.is_done()
        assert latch.count == 0

    @pytest.mark.asyncio
    async def test_count_property(self):
        """Test count property returns current count."""
        latch = AsyncCountDownLatch(5)

        assert latch.count == 5
        latch.count_down(2)
        assert latch.count == 3
        latch.count_down(3)
        assert latch.count == 0

    @pytest.mark.asyncio
    async def test_integration_with_concurrent_workers(self):
        """Test coordinating multiple workers with the latch."""
        latch = AsyncCountDownLatch(5)
        completed_workers = []

        async def worker(worker_id):
            """Simulate a worker that does work and counts down."""
            await asyncio.sleep(0.01 * worker_id)  # Variable work time
            completed_workers.append(worker_id)
            latch.count_down()

        # Launch workers
        workers = [asyncio.create_task(worker(i)) for i in range(5)]

        # Wait for all workers to complete
        await latch.wait()

        # Verify all workers completed
        await asyncio.gather(*workers)
        assert len(completed_workers) == 5
        assert set(completed_workers) == {0, 1, 2, 3, 4}
        assert latch.is_done()

    @pytest.mark.asyncio
    async def test_coordinated_start(self):
        """Test using latch for coordinated start of multiple tasks."""
        start_latch = AsyncCountDownLatch(3)  # All workers must be ready
        results = []

        async def worker(worker_id):
            # Worker is ready
            results.append(f"worker_{worker_id}_ready")
            start_latch.count_down()

            # Wait for all workers to be ready
            await start_latch.wait()
            results.append(f"worker_{worker_id}_started")

        # Launch workers
        workers = [asyncio.create_task(worker(i)) for i in range(3)]

        # Wait for all to complete
        await asyncio.gather(*workers)

        # All workers should be ready before any start
        ready_count = sum(1 for r in results if "_ready" in r)
        started_count = sum(1 for r in results if "_started" in r)

        assert ready_count == 3
        assert started_count == 3

        # Check that all ready events came before start events
        ready_indices = [i for i, r in enumerate(results) if "_ready" in r]
        start_indices = [i for i, r in enumerate(results) if "_started" in r]
        assert max(ready_indices) < min(start_indices)

    @pytest.mark.asyncio
    async def test_wait_already_done_returns_immediately(self):
        """Test that wait returns immediately when latch is already done."""
        latch = AsyncCountDownLatch(1)
        latch.count_down()

        # Should return immediately
        await latch.wait()

        # Multiple waits should all work
        await latch.wait()
        await latch.wait()

    @pytest.mark.asyncio
    async def test_zero_count_latch_allows_immediate_waits(self):
        """Test that a zero-count latch allows immediate waits."""
        latch = AsyncCountDownLatch(0)

        # Multiple concurrent waits should all succeed
        tasks = [asyncio.create_task(latch.wait()) for _ in range(5)]
        await asyncio.gather(*tasks)

        assert latch.is_done()
