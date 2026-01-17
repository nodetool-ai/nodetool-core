import asyncio
import time

import pytest

from nodetool.concurrency.async_lock import AsyncLock


class TestAsyncLock:
    """Tests for AsyncLock class."""

    def test_init_unlocked(self):
        """Test that lock starts in unlocked state."""
        lock = AsyncLock()
        assert not lock.locked

    def test_repr(self):
        """Test string representation."""
        lock = AsyncLock()
        assert repr(lock) == "AsyncLock(unlocked)"

        async def hold_lock():
            async with lock:
                assert repr(lock) == "AsyncLock(locked)"

        asyncio.run(hold_lock())

    def test_acquire_and_release(self):
        """Test basic acquire and release functionality."""
        lock = AsyncLock()

        async def run():
            assert not lock.locked

            await lock.acquire()
            assert lock.locked

            lock.release()
            assert not lock.locked

        asyncio.run(run())

    @pytest.mark.asyncio
    async def test_acquire_no_timeout(self):
        """Test acquire without timeout waits indefinitely."""
        lock = AsyncLock()

        await lock.acquire()
        assert lock.locked

        acquire_task = asyncio.create_task(lock.acquire())
        await asyncio.sleep(0.1)
        assert not acquire_task.done()

        lock.release()
        await acquire_task
        assert lock.locked

        lock.release()
        assert not lock.locked

    @pytest.mark.asyncio
    async def test_acquire_with_timeout_success(self):
        """Test acquire with timeout succeeds when lock is available."""
        lock = AsyncLock()

        result = await lock.acquire(timeout=1.0)
        assert result is True
        assert lock.locked

    @pytest.mark.asyncio
    async def test_acquire_with_timeout_expires(self):
        """Test acquire with timeout expires when lock is held."""
        lock = AsyncLock()

        await lock.acquire()
        assert lock.locked

        start = time.time()
        result = await lock.acquire(timeout=0.2)
        elapsed = time.time() - start

        assert result is False
        assert 0.15 <= elapsed < 0.5

    @pytest.mark.asyncio
    async def test_acquire_with_zero_timeout_available(self):
        """Test acquire with zero timeout succeeds when lock is available."""
        lock = AsyncLock()

        result = await lock.acquire(timeout=0)
        assert result is True
        assert lock.locked

    @pytest.mark.asyncio
    async def test_acquire_with_zero_timeout_unavailable(self):
        """Test acquire with zero timeout fails when lock is held."""
        lock = AsyncLock()

        await lock.acquire()

        result = await lock.acquire(timeout=0)
        assert result is False

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager usage."""
        lock = AsyncLock()

        async with lock:
            assert lock.locked

        assert not lock.locked

    @pytest.mark.asyncio
    async def test_context_manager_exception(self):
        """Test context manager releases on exception."""
        lock = AsyncLock()

        with pytest.raises(ValueError):
            async with lock:
                assert lock.locked
                raise ValueError("test error")

        assert not lock.locked

    @pytest.mark.asyncio
    async def test_exclusive_access(self):
        """Test that lock provides exclusive access."""
        lock = AsyncLock()
        access_count = 0
        max_concurrent = 0
        lock_inner = asyncio.Lock()

        async def critical_section():
            nonlocal access_count, max_concurrent

            async with lock:
                async with lock_inner:
                    access_count += 1
                    current = access_count
                await asyncio.sleep(0.05)
                async with lock_inner:
                    after_sleep = access_count

                assert current == after_sleep

        tasks = [critical_section() for _ in range(5)]
        await asyncio.gather(*tasks)
        assert access_count == 5

    @pytest.mark.asyncio
    async def test_nested_acquire_and_release(self):
        """Test that acquire and release can be nested correctly."""
        lock = AsyncLock()

        await lock.acquire()
        assert lock.locked

        lock.release()
        assert not lock.locked

        await lock.acquire()
        assert lock.locked

        lock.release()
        assert not lock.locked

    @pytest.mark.asyncio
    async def test_multiple_waiters(self):
        """Test multiple tasks waiting for the lock are served sequentially."""
        lock = AsyncLock()
        acquired = []
        lock_inner = asyncio.Lock()

        async def try_acquire(num):
            result = await lock.acquire(timeout=5.0)
            if result:
                async with lock_inner:
                    acquired.append(num)
                await asyncio.sleep(0.01)
                lock.release()

        await lock.acquire()
        tasks = [try_acquire(i) for i in range(3)]
        lock.release()
        await asyncio.sleep(0.01)
        await asyncio.gather(*tasks)

        assert len(acquired) == 3
        assert sorted(acquired) == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_reentrant_behavior(self):
        """Test that reentrant acquire fails (like asyncio.Lock)."""
        lock = AsyncLock()

        await lock.acquire()

        result = await lock.acquire(timeout=0)
        assert result is False

        lock.release()
        assert not lock.locked
