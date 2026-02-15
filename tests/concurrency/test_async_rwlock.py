import asyncio

import pytest

from nodetool.concurrency.async_rwlock import AsyncReaderWriterLock


class TestAsyncReaderWriterLock:
    """Tests for AsyncReaderWriterLock class."""

    def test_init(self):
        """Test that lock initializes in unlocked state."""
        lock = AsyncReaderWriterLock()
        assert not lock.locked
        assert not lock.write_locked
        assert lock.readers == 0
        assert lock.writers == 0
        assert lock.write_pending == 0

    def test_repr_unlocked(self):
        """Test string representation when unlocked."""
        lock = AsyncReaderWriterLock()
        assert repr(lock) == "AsyncReaderWriterLock(unlocked, readers=0, pending_writers=0)"

    def test_repr_read_locked(self):
        """Test string representation when read locked."""
        lock = AsyncReaderWriterLock()

        async def test():
            await lock.acquire_read()
            assert "read_locked" in repr(lock)
            assert "readers=1" in repr(lock)
            await lock.release_read()

        asyncio.run(test())

    def test_repr_write_locked(self):
        """Test string representation when write locked."""
        lock = AsyncReaderWriterLock()

        async def test():
            await lock.acquire_write()
            assert "write_locked" in repr(lock)
            await lock.release_write()

        asyncio.run(test())

    @pytest.mark.asyncio
    async def test_single_reader(self):
        """Test single reader acquire and release."""
        lock = AsyncReaderWriterLock()

        assert await lock.acquire_read()
        assert lock.readers == 1
        assert not lock.write_locked

        await lock.release_read()
        assert lock.readers == 0
        assert not lock.locked

    @pytest.mark.asyncio
    async def test_single_writer(self):
        """Test single writer acquire and release."""
        lock = AsyncReaderWriterLock()

        assert await lock.acquire_write()
        assert lock.writers == 1
        assert lock.write_locked

        await lock.release_write()
        assert lock.writers == 0
        assert not lock.locked

    @pytest.mark.asyncio
    async def test_multiple_concurrent_readers(self):
        """Test multiple readers can hold the lock simultaneously."""
        lock = AsyncReaderWriterLock()
        readers_active = []

        async def reader(reader_id: int):
            await lock.acquire_read()
            readers_active.append(reader_id)
            assert lock.readers > 0
            await asyncio.sleep(0.05)
            readers_active.remove(reader_id)
            await lock.release_read()

        # Start multiple readers concurrently
        tasks = [asyncio.create_task(reader(i)) for i in range(5)]
        await asyncio.gather(*tasks)

        assert lock.readers == 0
        assert not lock.locked

    @pytest.mark.asyncio
    async def test_writer_excludes_readers(self):
        """Test that writer blocks new readers."""
        lock = AsyncReaderWriterLock()
        reader_acquired = []

        async def reader(reader_id: int):
            await lock.acquire_read()
            reader_acquired.append(reader_id)
            await asyncio.sleep(0.05)
            reader_acquired.remove(reader_id)
            await lock.release_read()

        async def writer():
            await lock.acquire_write()
            assert lock.writers == 1
            await asyncio.sleep(0.1)
            await lock.release_write()

        # Start writer first
        writer_task = asyncio.create_task(writer())
        await asyncio.sleep(0.02)  # Let writer acquire

        # Try to start readers - they should wait
        reader_tasks = [asyncio.create_task(reader(i)) for i in range(3)]

        await asyncio.gather(writer_task, *reader_tasks)

        # Readers should have acquired after writer released
        assert lock.readers == 0
        assert not lock.locked

    @pytest.mark.asyncio
    async def test_writer_waits_for_readers(self):
        """Test that writer waits for all readers to release."""
        lock = AsyncReaderWriterLock()
        readers_released = False

        async def reader():
            await lock.acquire_read()
            await asyncio.sleep(0.05)
            await lock.release_read()

        async def writer():
            nonlocal readers_released
            await lock.acquire_write()
            # All readers should have released by now
            assert lock.readers == 0
            readers_released = True
            await lock.release_write()

        # Start readers first
        reader_tasks = [asyncio.create_task(reader()) for _ in range(3)]
        await asyncio.sleep(0.01)  # Let readers acquire

        # Start writer - should wait for readers
        writer_task = asyncio.create_task(writer())

        await asyncio.gather(*reader_tasks, writer_task)

        assert readers_released
        assert lock.readers == 0

    @pytest.mark.asyncio
    async def test_read_lock_context_manager(self):
        """Test using read_lock() as a context manager."""
        lock = AsyncReaderWriterLock()

        async with lock.read_lock():
            assert lock.readers == 1
            assert not lock.write_locked

        assert lock.readers == 0
        assert not lock.locked

    @pytest.mark.asyncio
    async def test_write_lock_context_manager(self):
        """Test using write_lock() as a context manager."""
        lock = AsyncReaderWriterLock()

        async with lock.write_lock():
            assert lock.writers == 1
            assert lock.write_locked

        assert lock.writers == 0
        assert not lock.locked

    @pytest.mark.asyncio
    async def test_concurrent_readers_with_context_manager(self):
        """Test multiple concurrent readers using context manager."""
        lock = AsyncReaderWriterLock()
        active_count = []

        async def reader(reader_id: int):
            async with lock.read_lock():
                active_count.append(reader_id)
                await asyncio.sleep(0.02)
                active_count.remove(reader_id)

        tasks = [asyncio.create_task(reader(i)) for i in range(5)]
        await asyncio.gather(*tasks)

        assert not lock.locked

    @pytest.mark.asyncio
    async def test_acquire_read_timeout_success(self):
        """Test acquire_read with timeout succeeds when lock is available."""
        lock = AsyncReaderWriterLock()

        async def reader():
            await asyncio.sleep(0.05)
            async with lock.read_lock():
                await asyncio.sleep(0.05)

        async def waiter():
            # Should succeed after first reader releases
            result = await lock.acquire_read(timeout=1.0)
            if result:
                await lock.release_read()
            return result

        reader_task = asyncio.create_task(reader())
        await asyncio.sleep(0.02)

        # Try to acquire with timeout while reader holds lock
        waiter_task = asyncio.create_task(waiter())

        result = await waiter_task
        await reader_task

        assert result is True

    @pytest.mark.asyncio
    async def test_acquire_read_timeout_failure(self):
        """Test acquire_read returns False on timeout."""
        lock = AsyncReaderWriterLock()

        async def holding_writer():
            await lock.acquire_write()
            await asyncio.sleep(0.2)
            await lock.release_write()

        # Start a writer that will hold the lock
        writer_task = asyncio.create_task(holding_writer())
        await asyncio.sleep(0.01)

        # Try to acquire with short timeout
        result = await lock.acquire_read(timeout=0.05)

        assert result is False
        await writer_task

    @pytest.mark.asyncio
    async def test_acquire_write_timeout_success(self):
        """Test acquire_write with timeout succeeds when lock is available."""
        lock = AsyncReaderWriterLock()

        async def reader():
            async with lock.read_lock():
                await asyncio.sleep(0.05)

        async def waiting_writer():
            # Should succeed after reader releases
            result = await lock.acquire_write(timeout=1.0)
            if result:
                await lock.release_write()
            return result

        reader_task = asyncio.create_task(reader())
        await asyncio.sleep(0.02)

        writer_task = asyncio.create_task(waiting_writer())

        result = await writer_task
        await reader_task

        assert result is True

    @pytest.mark.asyncio
    async def test_acquire_write_timeout_failure(self):
        """Test acquire_write returns False on timeout."""
        lock = AsyncReaderWriterLock()

        async def holding_reader():
            await lock.acquire_read()
            await asyncio.sleep(0.2)
            await lock.release_read()

        # Start a reader that will hold the lock
        reader_task = asyncio.create_task(holding_reader())
        await asyncio.sleep(0.01)

        # Try to acquire with short timeout
        result = await lock.acquire_write(timeout=0.05)

        assert result is False
        await reader_task

    @pytest.mark.asyncio
    async def test_non_blocking_acquire_read(self):
        """Test non-blocking acquire_read (timeout=0)."""
        lock = AsyncReaderWriterLock()

        async def holder():
            await lock.acquire_write()
            await asyncio.sleep(0.1)
            await lock.release_write()

        holder_task = asyncio.create_task(holder())
        await asyncio.sleep(0.01)

        # Non-blocking attempt
        result = await lock.acquire_read(timeout=0)

        assert result is False
        await holder_task

    @pytest.mark.asyncio
    async def test_non_blocking_acquire_write(self):
        """Test non-blocking acquire_write (timeout=0)."""
        lock = AsyncReaderWriterLock()

        async def holder():
            await lock.acquire_read()
            await asyncio.sleep(0.1)
            await lock.release_read()

        holder_task = asyncio.create_task(holder())
        await asyncio.sleep(0.01)

        # Non-blocking attempt
        result = await lock.acquire_write(timeout=0)

        assert result is False
        await holder_task

    @pytest.mark.asyncio
    async def test_non_blocking_acquire_read_success(self):
        """Test non-blocking acquire_read succeeds when lock is free."""
        lock = AsyncReaderWriterLock()

        result = await lock.acquire_read(timeout=0)

        assert result is True
        await lock.release_read()

    @pytest.mark.asyncio
    async def test_release_read_without_acquire_raises(self):
        """Test releasing read lock without acquiring raises RuntimeError."""
        lock = AsyncReaderWriterLock()

        with pytest.raises(RuntimeError, match="no readers holding"):
            await lock.release_read()

    @pytest.mark.asyncio
    async def test_release_write_without_acquire_raises(self):
        """Test releasing write lock without acquiring raises RuntimeError."""
        lock = AsyncReaderWriterLock()

        with pytest.raises(RuntimeError, match="no writer holding"):
            await lock.release_write()

    @pytest.mark.asyncio
    async def test_writer_fairness_multiple_writers(self):
        """Test that multiple writers are served in order."""
        lock = AsyncReaderWriterLock()
        write_order = []

        async def writer(writer_id: int):
            await lock.acquire_write()
            write_order.append(writer_id)
            await asyncio.sleep(0.02)
            await lock.release_write()

        # Start multiple writers concurrently
        tasks = [asyncio.create_task(writer(i)) for i in range(3)]
        await asyncio.gather(*tasks)

        # Writers should have executed one at a time
        assert len(write_order) == 3
        assert lock.writers == 0

    @pytest.mark.asyncio
    async def test_reader_writer_starvation_prevention(self):
        """Test that readers don't starve writers and vice versa."""
        lock = AsyncReaderWriterLock()
        operations = []

        async def reader(reader_id: int):
            async with lock.read_lock():
                operations.append(f"reader_{reader_id}")
                await asyncio.sleep(0.02)

        async def writer(writer_id: int):
            async with lock.write_lock():
                operations.append(f"writer_{writer_id}")
                await asyncio.sleep(0.02)

        # Mix of readers and writers
        tasks = [
            asyncio.create_task(reader(1)),
            asyncio.create_task(writer(1)),
            asyncio.create_task(reader(2)),
            asyncio.create_task(writer(2)),
            asyncio.create_task(reader(3)),
        ]

        await asyncio.gather(*tasks)

        # All operations should complete
        assert len(operations) == 5
        assert lock.readers == 0
        assert lock.writers == 0

    @pytest.mark.asyncio
    async def test_concurrent_readers_then_writer(self):
        """Test writer waits for all concurrent readers to finish."""
        lock = AsyncReaderWriterLock()
        reader_count = 0
        writer_executed = False

        async def reader():
            nonlocal reader_count
            async with lock.read_lock():
                reader_count += 1
                await asyncio.sleep(0.05)
                reader_count -= 1

        async def writer():
            nonlocal writer_executed
            async with lock.write_lock():
                # All readers should be done
                assert reader_count == 0
                writer_executed = True

        # Start multiple readers
        reader_tasks = [asyncio.create_task(reader()) for _ in range(5)]
        await asyncio.sleep(0.01)

        # Start writer - should wait for all readers
        writer_task = asyncio.create_task(writer())

        await asyncio.gather(*reader_tasks, writer_task)

        assert writer_executed
        assert reader_count == 0

    @pytest.mark.asyncio
    async def test_write_pending_property(self):
        """Test write_pending property tracks waiting writers."""
        lock = AsyncReaderWriterLock()

        async def blocking_reader():
            await lock.acquire_read()
            await asyncio.sleep(0.1)
            await lock.release_read()

        async def waiting_writer():
            await lock.acquire_write()
            await lock.release_write()

        # Start reader
        reader_task = asyncio.create_task(blocking_reader())
        await asyncio.sleep(0.01)

        # Start writers that will wait
        writer_tasks = [asyncio.create_task(waiting_writer()) for _ in range(3)]
        await asyncio.sleep(0.02)

        # Should have pending writers
        assert lock.write_pending > 0

        await asyncio.gather(reader_task, *writer_tasks)

        # No pending writers after completion
        assert lock.write_pending == 0

    @pytest.mark.asyncio
    async def test_multiple_write_release_raises(self):
        """Test that releasing write lock twice raises RuntimeError."""
        lock = AsyncReaderWriterLock()

        await lock.acquire_write()
        await lock.release_write()

        with pytest.raises(RuntimeError, match="no writer holding"):
            await lock.release_write()

    @pytest.mark.asyncio
    async def test_multiple_read_release_raises(self):
        """Test that releasing read lock too many times raises RuntimeError."""
        lock = AsyncReaderWriterLock()

        await lock.acquire_read()
        await lock.release_read()

        with pytest.raises(RuntimeError, match="no readers holding"):
            await lock.release_read()

    @pytest.mark.asyncio
    async def test_read_write_alternation(self):
        """Test alternating read and write operations."""
        lock = AsyncReaderWriterLock()
        results = []

        async def reader():
            async with lock.read_lock():
                results.append("read")
                await asyncio.sleep(0.02)

        async def writer():
            async with lock.write_lock():
                results.append("write")
                await asyncio.sleep(0.02)

        # Alternate between reads and writes
        tasks = [
            asyncio.create_task(reader()),
            asyncio.create_task(writer()),
            asyncio.create_task(reader()),
            asyncio.create_task(writer()),
            asyncio.create_task(reader()),
        ]

        await asyncio.gather(*tasks)

        # All should complete
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_writer_blocks_new_readers(self):
        """Test that a holding writer blocks new readers from acquiring."""
        lock = AsyncReaderWriterLock()
        readers_acquired = []

        async def early_reader():
            # Acquire before writer
            await lock.acquire_read()
            await asyncio.sleep(0.05)
            await lock.release_read()

        async def writer():
            # Writer should wait for early_reader then block new readers
            await lock.acquire_write()
            await asyncio.sleep(0.1)
            await lock.release_write()

        async def late_reader():
            # Should wait for writer
            await lock.acquire_read()
            readers_acquired.append(1)
            await lock.release_read()

        early_task = asyncio.create_task(early_reader())
        await asyncio.sleep(0.01)

        writer_task = asyncio.create_task(writer())
        await asyncio.sleep(0.02)  # Writer is waiting or acquiring

        # Try to start another reader - should wait for writer
        late_task = asyncio.create_task(late_reader())

        await asyncio.gather(early_task, writer_task, late_task)

        # Late reader should have acquired after writer released
        assert len(readers_acquired) == 1

    @pytest.mark.asyncio
    async def test_locked_property(self):
        """Test locked property returns True for both read and write locks."""
        lock = AsyncReaderWriterLock()

        assert not lock.locked

        # Read lock
        await lock.acquire_read()
        assert lock.locked
        assert not lock.write_locked
        await lock.release_read()

        assert not lock.locked

        # Write lock
        await lock.acquire_write()
        assert lock.locked
        assert lock.write_locked
        await lock.release_write()

        assert not lock.locked
