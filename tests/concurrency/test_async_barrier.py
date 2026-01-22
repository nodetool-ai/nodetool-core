import asyncio

import pytest

from nodetool.concurrency import AsyncBarrier, AsyncTaskGroup, BrokenBarrierError


class TestAsyncBarrier:
    """Tests for AsyncBarrier class."""

    def test_init_valid_parties(self):
        """Test initialization with valid parties count."""
        barrier = AsyncBarrier(3)
        assert barrier.parties == 3
        assert barrier.n_waiting == 0
        assert not barrier.broken

    def test_init_invalid_parties(self):
        """Test that invalid parties count raises ValueError."""
        with pytest.raises(ValueError, match="parties must be at least 1"):
            AsyncBarrier(0)

        with pytest.raises(ValueError, match="parties must be at least 1"):
            AsyncBarrier(-1)

    def test_init_with_timeout(self):
        """Test initialization with timeout."""
        barrier = AsyncBarrier(3, timeout=5.0)
        assert barrier._timeout == 5.0

    def test_repr(self):
        """Test string representation."""
        barrier = AsyncBarrier(3)
        assert "parties=3" in repr(barrier)
        assert "waiting=0" in repr(barrier)
        assert "intact" in repr(barrier)

    @pytest.mark.asyncio
    async def test_wait_all_arrive_together(self):
        """Test that all parties proceed together when they arrive together."""
        barrier = AsyncBarrier(3)
        results = []

        async def participant(idx):
            result = await barrier.wait()
            results.append((idx, result))

        tasks = [asyncio.create_task(participant(i)) for i in range(3)]
        await asyncio.gather(*tasks)

        assert len(results) == 3
        indices = {r[0] for r in results}
        assert indices == {0, 1, 2}

    @pytest.mark.asyncio
    async def test_wait_staggered_arrivals(self):
        """Test barrier with staggered task arrivals."""
        barrier = AsyncBarrier(3)
        results = []
        order = []

        async def participant(idx, delay):
            order.append(f"start_{idx}")
            await asyncio.sleep(delay)
            order.append(f"wait_{idx}")
            result = await barrier.wait()
            order.append(f"done_{idx}")
            results.append((idx, result))

        tasks = [
            asyncio.create_task(participant(0, 0.1)),
            asyncio.create_task(participant(1, 0.05)),
            asyncio.create_task(participant(2, 0.0)),
        ]
        await asyncio.gather(*tasks)

        assert len(results) == 3
        assert len([o for o in order if o.startswith("done_")]) == 3

    @pytest.mark.asyncio
    async def test_wait_returns_arrival_index(self):
        """Test that wait returns arrival index for leader election."""
        barrier = AsyncBarrier(3)
        indices = []

        async def participant(idx):
            await barrier.wait()
            indices.append(idx)

        tasks = [asyncio.create_task(participant(i)) for i in range(3)]
        await asyncio.gather(*tasks)

        assert len(indices) == 3
        assert set(indices) == {0, 1, 2}

    @pytest.mark.asyncio
    async def test_barrier_reuse(self):
        """Test that barrier can be reused after first wait."""
        barrier = AsyncBarrier(2)
        phase1_results = []
        phase2_results = []

        async def participant(idx):
            await barrier.wait()
            phase1_results.append(idx)
            await barrier.wait()
            phase2_results.append(idx)

        tasks = [asyncio.create_task(participant(i)) for i in range(2)]
        await asyncio.gather(*tasks)

        assert set(phase1_results) == {0, 1}
        assert set(phase2_results) == {0, 1}

    @pytest.mark.asyncio
    async def test_n_waiting_property(self):
        """Test that n_waiting returns correct count."""
        barrier = AsyncBarrier(3)
        assert barrier.n_waiting == 0

        async def waiter(idx):
            await barrier.wait()

        task1 = asyncio.create_task(waiter(1))
        await asyncio.sleep(0.01)
        assert barrier.n_waiting == 1

        task2 = asyncio.create_task(waiter(2))
        await asyncio.sleep(0.01)
        assert barrier.n_waiting == 2

        task3 = asyncio.create_task(waiter(3))
        await asyncio.sleep(0.01)

        await asyncio.gather(task1, task2, task3)

    @pytest.mark.asyncio
    async def test_reset(self):
        """Test manual reset of barrier."""
        barrier = AsyncBarrier(2)

        async def waiter(idx):
            await barrier.wait()

        task1 = asyncio.create_task(waiter(1))
        await asyncio.sleep(0.01)

        barrier.reset()

        assert not barrier.broken
        await asyncio.sleep(0.05)
        assert barrier.n_waiting == 0

        task2 = asyncio.create_task(waiter(2))
        await asyncio.sleep(0.01)
        assert barrier.n_waiting == 1

        await barrier.wait()

        await asyncio.gather(task1, task2)

    @pytest.mark.asyncio
    async def test_abort(self):
        """Test aborting the barrier."""
        barrier = AsyncBarrier(3)

        async def waiter(idx):
            await barrier.wait()

        task1 = asyncio.create_task(waiter(1))
        await asyncio.sleep(0.01)

        task2 = asyncio.create_task(waiter(2))
        await asyncio.sleep(0.01)

        barrier.abort()

        assert barrier.broken

        with pytest.raises(BrokenBarrierError):
            await task1

        with pytest.raises(BrokenBarrierError):
            await task2

    @pytest.mark.asyncio
    async def test_timeout(self):
        """Test that timeout raises BrokenBarrierError."""
        barrier = AsyncBarrier(3, timeout=0.05)

        async def slow_waiter():
            await barrier.wait()

        task = asyncio.create_task(slow_waiter())
        await asyncio.sleep(0.01)

        with pytest.raises(BrokenBarrierError, match="timed out"):
            await task

        assert barrier.broken

    @pytest.mark.asyncio
    async def test_timeout_partial_group(self):
        """Test timeout with only some parties participating."""
        barrier = AsyncBarrier(3, timeout=0.05)

        async def quick_waiter():
            await barrier.wait()

        task = asyncio.create_task(quick_waiter())

        with pytest.raises(BrokenBarrierError, match="timed out"):
            await task

    @pytest.mark.asyncio
    async def test_broken_after_timeout(self):
        """Test that barrier is broken after timeout."""
        barrier = AsyncBarrier(3, timeout=0.05)

        async def waiter():
            await barrier.wait()

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.01)

        try:
            await task
        except BrokenBarrierError:
            pass

        assert barrier.broken

    @pytest.mark.asyncio
    async def test_wait_after_broken_raises_error(self):
        """Test that wait after barrier is broken raises BrokenBarrierError."""
        barrier = AsyncBarrier(3, timeout=0.01)

        async def waiter():
            await barrier.wait()

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.01)

        try:
            await task
        except BrokenBarrierError:
            pass

        with pytest.raises(BrokenBarrierError, match="was broken"):
            await barrier.wait()

    @pytest.mark.asyncio
    async def test_integration_with_task_group(self):
        """Test AsyncBarrier works well with AsyncTaskGroup."""
        barrier = AsyncBarrier(3)
        phase1_complete = []
        phase2_complete = []

        async def worker(worker_id):
            await barrier.wait()
            phase1_complete.append(worker_id)

            await barrier.wait()
            phase2_complete.append(worker_id)

        group = AsyncTaskGroup()
        group.spawn("w1", worker(1))
        group.spawn("w2", worker(2))
        group.spawn("w3", worker(3))

        await group.run()

        assert set(phase1_complete) == {1, 2, 3}
        assert set(phase2_complete) == {1, 2, 3}

    @pytest.mark.asyncio
    async def test_single_party(self):
        """Test barrier with single party."""
        barrier = AsyncBarrier(1)

        async def single():
            result = await barrier.wait()
            return result

        result = await single()
        assert result == 0

    @pytest.mark.asyncio
    async def test_large_party_count(self):
        """Test barrier with many parties."""
        barrier = AsyncBarrier(10)
        results = []

        async def participant(idx):
            await barrier.wait()
            results.append(idx)

        tasks = [asyncio.create_task(participant(i)) for i in range(10)]
        await asyncio.gather(*tasks)

        assert len(results) == 10
        assert set(results) == set(range(10))

    @pytest.mark.asyncio
    async def test_cancellation_during_wait(self):
        """Test cancellation of task during wait."""
        barrier = AsyncBarrier(3)

        async def cancellable_waiter():
            await barrier.wait()

        task = asyncio.create_task(cancellable_waiter())
        await asyncio.sleep(0.01)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert barrier.n_waiting == 0

    @pytest.mark.asyncio
    async def test_concurrent_phased_workflow(self):
        """Test real-world scenario: concurrent phased workflow."""
        barrier1 = AsyncBarrier(3)
        barrier2 = AsyncBarrier(3)
        phase1_results = []
        phase2_results = []

        async def phase_worker(worker_id):
            await asyncio.sleep(worker_id * 0.05)
            phase1_results.append(f"phase1_{worker_id}")

            await barrier1.wait()

            await asyncio.sleep(worker_id * 0.03)
            phase2_results.append(f"phase2_{worker_id}")

            await barrier2.wait()

        tasks = [asyncio.create_task(phase_worker(i)) for i in range(3)]
        await asyncio.gather(*tasks)

        assert len(phase1_results) == 3
        assert len(phase2_results) == 3
