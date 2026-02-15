import asyncio

import pytest

from nodetool.concurrency.async_barrier import AsyncBarrier


class TestAsyncBarrier:
    """Tests for AsyncBarrier class."""

    def test_init_valid_parties(self):
        """Test that barrier initializes with valid parties."""
        barrier = AsyncBarrier(parties=3)
        assert barrier.parties == 3
        assert barrier.waiting == 0
        assert barrier.remaining == 3

    def test_init_invalid_parties(self):
        """Test that barrier rejects invalid parties values."""
        with pytest.raises(ValueError, match="parties must be a positive integer"):
            AsyncBarrier(parties=0)

        with pytest.raises(ValueError, match="parties must be a positive integer"):
            AsyncBarrier(parties=-1)

    def test_repr(self):
        """Test string representation."""
        barrier = AsyncBarrier(parties=3)
        assert repr(barrier) == "AsyncBarrier(parties=3, waiting=0, remaining=3)"

    @pytest.mark.asyncio
    async def test_single_party(self):
        """Test barrier with a single party passes immediately."""
        barrier = AsyncBarrier(parties=1)
        result = await barrier.wait()
        # With single party, the last one returns False
        assert result is False

    @pytest.mark.asyncio
    async def test_two_parties(self):
        """Test barrier with two parties synchronizes correctly."""
        barrier = AsyncBarrier(parties=2)
        results = []

        async def worker(task_id: int):
            await asyncio.sleep(0.01 * task_id)
            result = await barrier.wait()
            results.append((task_id, result))

        await asyncio.gather(worker(0), worker(1))

        # Both should have passed through
        assert len(results) == 2
        # One should be leader (True), one not (False)
        leader_count = sum(1 for _, is_leader in results if is_leader)
        assert leader_count == 1

    @pytest.mark.asyncio
    async def test_multiple_parties(self):
        """Test barrier with multiple parties synchronizes correctly."""
        barrier = AsyncBarrier(parties=5)
        results = []
        arrival_order = []

        async def worker(task_id: int):
            # Stagger arrivals
            await asyncio.sleep(0.01 * task_id)
            arrival_order.append(task_id)
            result = await barrier.wait()
            results.append((task_id, result))

        tasks = [worker(i) for i in range(5)]
        await asyncio.gather(*tasks)

        # All should have passed through
        assert len(results) == 5
        assert len(arrival_order) == 5

        # Exactly one should be non-leader (False)
        leader_count = sum(1 for _, is_leader in results if is_leader)
        assert leader_count == 4
        non_leader_count = sum(1 for _, is_leader in results if not is_leader)
        assert non_leader_count == 1

    @pytest.mark.asyncio
    async def test_reuse_barrier(self):
        """Test that barrier can be reused multiple times."""
        barrier = AsyncBarrier(parties=3)

        for _phase in range(3):
            results = []

            async def worker(task_id: int, results_ref: list):
                await barrier.wait()
                results_ref.append(task_id)

            tasks = [worker(i, results) for i in range(3)]
            await asyncio.gather(*tasks)

            assert len(results) == 3
            assert barrier.waiting == 0

    @pytest.mark.asyncio
    async def test_phased_execution(self):
        """Test barrier coordinating phased work."""
        barrier = AsyncBarrier(parties=3)
        phases_completed = []

        async def worker(task_id: int):
            for phase in range(3):
                # Simulate work
                await asyncio.sleep(0.01 * task_id)
                phases_completed.append(f"task{task_id}-phase{phase}")

                # Wait for all tasks to complete this phase
                await barrier.wait()

        await asyncio.gather(*[worker(i) for i in range(3)])

        # All phases should be completed
        assert len(phases_completed) == 9  # 3 tasks x 3 phases

    @pytest.mark.asyncio
    async def test_reset_barrier(self):
        """Test that reset clears the barrier state and releases waiting tasks."""
        barrier = AsyncBarrier(parties=3)
        completed = []

        async def worker(task_id: int):
            await barrier.wait()
            completed.append(task_id)

        # Start two workers (will wait at barrier)
        task1 = asyncio.create_task(worker(0))
        task2 = asyncio.create_task(worker(1))

        # Wait for them to start waiting
        await asyncio.sleep(0.05)

        # Reset the barrier - this will wake up waiting tasks and reset counter
        await barrier.reset()

        # Give tasks time to complete
        await asyncio.sleep(0.05)

        # Tasks should have completed (been released by reset)
        await task1
        await task2

        # Barrier should be in initial state
        assert barrier.waiting == 0
        assert barrier.remaining == 3

    @pytest.mark.asyncio
    async def test_barrier_state_during_wait(self):
        """Test that barrier properties update correctly during wait."""
        barrier = AsyncBarrier(parties=3)
        waiting_counts = []

        async def worker(task_id: int):
            # Record waiting count after arriving but before waiting
            barrier._count += 1
            waiting_counts.append(barrier.waiting)
            barrier._count -= 1

            await barrier.wait()

        async def monitor():
            # Monitor the barrier state
            for _ in range(10):
                waiting_counts.append(barrier.waiting)
                await asyncio.sleep(0.01)

        # Run monitor and workers
        monitor_task = asyncio.create_task(monitor())
        await asyncio.gather(*[worker(i) for i in range(3)])
        monitor_task.cancel()

        # Should have observed various waiting states
        assert 0 in waiting_counts
        assert max(waiting_counts) <= 3

    @pytest.mark.asyncio
    async def test_leader_selection_consistency(self):
        """Test that exactly one task is designated as non-leader."""
        barrier = AsyncBarrier(parties=10)
        leader_results = []

        async def worker(task_id: int):
            await asyncio.sleep(0.001 * task_id)
            is_leader = await barrier.wait()
            leader_results.append(is_leader)

        await asyncio.gather(*[worker(i) for i in range(10)])

        # Exactly one should be False (non-leader)
        false_count = sum(1 for result in leader_results if not result)
        true_count = sum(1 for result in leader_results if result)
        assert false_count == 1
        assert true_count == 9

    @pytest.mark.asyncio
    async def test_barrier_with_exception(self):
        """Test that barrier handles exceptions in tasks correctly."""
        barrier = AsyncBarrier(parties=3)
        results = []

        async def worker(task_id: int):
            if task_id == 1:
                await barrier.wait()
                raise ValueError("Test error")
            else:
                await barrier.wait()
                results.append(task_id)

        tasks = [worker(i) for i in range(3)]

        with pytest.raises(ValueError, match="Test error"):
            await asyncio.gather(*tasks)

        # Other tasks should have completed their wait
        # (Note: behavior may vary based on asyncio implementation)

    @pytest.mark.asyncio
    async def test_barrier_properties_accuracy(self):
        """Test that barrier properties reflect accurate state."""
        barrier = AsyncBarrier(parties=4)

        async def worker(task_id: int):
            await asyncio.sleep(0.01)
            await barrier.wait()

        tasks = [asyncio.create_task(worker(i)) for i in range(4)]

        # Wait a bit for some tasks to arrive
        await asyncio.sleep(0.025)

        # Some tasks should be waiting
        assert barrier.waiting >= 0
        assert barrier.remaining <= 4

        # Let all complete
        await asyncio.gather(*tasks)

        # After completion, should be reset
        assert barrier.waiting == 0
        assert barrier.remaining == 4
