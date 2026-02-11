"""Tests for AsyncCancellationScope."""

import asyncio

import pytest

from nodetool.concurrency import AsyncCancellationScope, CancellationError


class TestAsyncCancellationScope:
    """Test suite for AsyncCancellationScope."""

    def test_initial_state(self) -> None:
        """Test that a new scope is not cancelled."""
        scope = AsyncCancellationScope()
        assert not scope.is_cancelled()

    def test_cancel_sets_flag(self) -> None:
        """Test that cancel() sets the cancelled flag."""
        scope = AsyncCancellationScope()
        assert not scope.is_cancelled()
        scope.cancel()
        assert scope.is_cancelled()

    def test_raise_if_cancelled_when_not_cancelled(self) -> None:
        """Test raise_if_cancelled does nothing when not cancelled."""
        scope = AsyncCancellationScope()
        scope.raise_if_cancelled()  # Should not raise

    def test_raise_if_cancelled_when_cancelled(self) -> None:
        """Test raise_if_cancelled raises when cancelled."""
        scope = AsyncCancellationScope()
        scope.cancel()
        with pytest.raises(CancellationError):
            scope.raise_if_cancelled()

    @pytest.mark.asyncio
    async def test_wait_for_cancelled_blocks_until_cancelled(self) -> None:
        """Test wait_for_cancelled blocks until scope is cancelled."""
        scope = AsyncCancellationScope()
        cancelled = False

        async def waiter() -> None:
            nonlocal cancelled
            await scope.wait_for_cancelled()
            cancelled = True

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.01)  # Give waiter time to start
        assert not cancelled

        scope.cancel()
        await asyncio.sleep(0.01)  # Let waiter process
        assert cancelled
        await task

    @pytest.mark.asyncio
    async def test_wait_for_cancelled_returns_immediately_if_cancelled(self) -> None:
        """Test wait_for_cancelled returns immediately if already cancelled."""
        scope = AsyncCancellationScope()
        scope.cancel()
        await scope.wait_for_cancelled()  # Should return immediately

    @pytest.mark.asyncio
    async def test_cleanup_callbacks_on_exit(self) -> None:
        """Test cleanup callbacks are called on scope exit."""
        cleanup_called = []

        def cleanup1() -> None:
            cleanup_called.append(1)

        def cleanup2() -> None:
            cleanup_called.append(2)

        async with AsyncCancellationScope() as scope:
            scope.add_cleanup_callback(cleanup1)
            scope.add_cleanup_callback(cleanup2)

        assert cleanup_called == [1, 2]

    @pytest.mark.asyncio
    async def test_cleanup_callback_removal(self) -> None:
        """Test that cleanup callbacks can be removed."""
        cleanup_called = []

        def cleanup() -> None:
            cleanup_called.append(1)

        async with AsyncCancellationScope() as scope:
            remove = scope.add_cleanup_callback(cleanup)
            remove()  # Remove the callback

        assert cleanup_called == []

    @pytest.mark.asyncio
    async def test_async_cleanup_callbacks(self) -> None:
        """Test that async cleanup callbacks are scheduled."""
        cleanup_called = []

        async def async_cleanup() -> None:
            await asyncio.sleep(0.01)
            cleanup_called.append(1)

        async with AsyncCancellationScope() as scope:
            scope.add_cleanup_callback(async_cleanup)

        # Give async cleanup time to run
        await asyncio.sleep(0.05)
        assert cleanup_called == [1]

    @pytest.mark.asyncio
    async def test_cleanup_callbacks_dont_fail_scope(self) -> None:
        """Test that failing cleanup callbacks don't break scope exit."""
        def failing_cleanup() -> None:
            raise RuntimeError("Cleanup failed")

        def good_cleanup() -> None:
            pass

        # Should not raise despite failing cleanup
        async with AsyncCancellationScope() as scope:
            scope.add_cleanup_callback(failing_cleanup)
            scope.add_cleanup_callback(good_cleanup)

    @pytest.mark.asyncio
    async def test_cooperative_cancellation_pattern(self) -> None:
        """Test the cooperative cancellation pattern."""
        iterations = []

        async def worker(scope: AsyncCancellationScope) -> None:
            for i in range(100):
                scope.raise_if_cancelled()
                iterations.append(i)
                await asyncio.sleep(0.01)

        scope = AsyncCancellationScope()
        task = asyncio.create_task(worker(scope))

        # Cancel after a few iterations
        await asyncio.sleep(0.05)
        scope.cancel()
        await asyncio.sleep(0.01)

        with pytest.raises(CancellationError):
            await task

        assert len(iterations) < 100  # Should have stopped early

    @pytest.mark.asyncio
    async def test_multiple_workers_with_single_scope(self) -> None:
        """Test coordinating cancellation across multiple workers."""
        results = []

        async def worker(scope: AsyncCancellationScope, worker_id: int) -> None:
            try:
                while not scope.is_cancelled():
                    results.append(worker_id)
                    await asyncio.sleep(0.01)
            except CancellationError:
                results.append(f"{worker_id}_cleaned")

        scope = AsyncCancellationScope()
        tasks = [
            asyncio.create_task(worker(scope, i))
            for i in range(3)
        ]

        # Let them run a bit then cancel
        await asyncio.sleep(0.05)
        scope.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        # Each worker should have done some work
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_sync_context_manager(self) -> None:
        """Test using scope as a sync context manager."""
        cleanup_called = []

        def cleanup() -> None:
            cleanup_called.append(1)

        with AsyncCancellationScope() as scope:
            scope.add_cleanup_callback(cleanup)

        assert cleanup_called == [1]

    @pytest.mark.asyncio
    async def test_nested_scopes(self) -> None:
        """Test using nested cancellation scopes."""
        outer_cancelled = False
        inner_cancelled = False

        async with AsyncCancellationScope() as outer:
            async with AsyncCancellationScope() as inner:
                outer.cancel()
                outer_cancelled = outer.is_cancelled()
                inner_cancelled = inner.is_cancelled()

        # Scopes are independent
        assert outer_cancelled
        assert not inner_cancelled

    @pytest.mark.asyncio
    async def test_cancel_from_outside_task(self) -> None:
        """Test cancelling a scope from outside the task using it."""
        task_started = False
        task_stopped = False

        async def worker(scope: AsyncCancellationScope) -> None:
            nonlocal task_started, task_stopped
            task_started = True
            try:
                while not scope.is_cancelled():
                    await asyncio.sleep(0.01)
            except CancellationError:
                pass
            task_stopped = True

        scope = AsyncCancellationScope()
        task = asyncio.create_task(worker(scope))

        # Wait for task to start
        await asyncio.sleep(0.05)
        assert task_started
        assert not task_stopped

        # Cancel from outside
        scope.cancel()
        await asyncio.sleep(0.05)
        assert task_stopped
        await task
