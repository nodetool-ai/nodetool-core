"""Tests for AsyncBoundedBuffer."""
import asyncio
import pytest

from nodetool.concurrency import (
    AsyncBoundedBuffer,
    BufferFullError,
    BufferStatistics,
    OverflowStrategy,
)


@pytest.mark.asyncio
async def test_basic_put_get() -> None:
    """Test basic put and get operations."""
    buffer = AsyncBoundedBuffer[int](capacity=5)

    await buffer.put(1)
    await buffer.put(2)
    await buffer.put(3)

    assert buffer.size == 3
    assert await buffer.get() == 1
    assert await buffer.get() == 2
    assert buffer.size == 1


@pytest.mark.asyncio
async def test_fifo_order() -> None:
    """Test that items are retrieved in FIFO order."""
    buffer = AsyncBoundedBuffer[str](capacity=10)

    items = ["first", "second", "third", "fourth"]
    for item in items:
        await buffer.put(item)

    retrieved = []
    for _ in items:
        retrieved.append(await buffer.get())

    assert retrieved == items


@pytest.mark.asyncio
async def test_blocking_when_full() -> None:
    """Test BLOCK strategy blocks producer when full."""
    buffer = AsyncBoundedBuffer[int](capacity=2, overflow_strategy=OverflowStrategy.BLOCK)

    # Fill the buffer
    await buffer.put(1)
    await buffer.put(2)
    assert buffer.full

    # This should block until we make space
    put_task = asyncio.create_task(buffer.put(3))

    # Give it a moment to verify it's blocked
    await asyncio.sleep(0.01)
    assert not put_task.done()

    # Consume one item to make space
    assert await buffer.get() == 1

    # Now the put should complete
    await asyncio.wait_for(put_task, timeout=0.1)
    assert buffer.size == 2


@pytest.mark.asyncio
async def test_drop_oldest_strategy() -> None:
    """Test DROP_OLDEST strategy drops oldest items when full."""
    buffer = AsyncBoundedBuffer[int](
        capacity=3, overflow_strategy=OverflowStrategy.DROP_OLDEST
    )

    await buffer.put(1)
    await buffer.put(2)
    await buffer.put(3)
    # Buffer is full: [1, 2, 3]

    await buffer.put(4)  # Should drop 1
    # Buffer now: [2, 3, 4]

    assert buffer.size == 3
    assert await buffer.get() == 2
    assert await buffer.get() == 3
    assert await buffer.get() == 4


@pytest.mark.asyncio
async def test_drop_newest_strategy() -> None:
    """Test DROP_NEWEST strategy returns False when full."""
    buffer = AsyncBoundedBuffer[int](
        capacity=3, overflow_strategy=OverflowStrategy.DROP_NEWEST
    )

    await buffer.put(1)
    await buffer.put(2)
    await buffer.put(3)

    # These should be dropped
    assert not await buffer.put(4)
    assert not await buffer.put(5)

    assert buffer.size == 3
    assert await buffer.get() == 1
    assert await buffer.get() == 2
    assert await buffer.get() == 3


@pytest.mark.asyncio
async def test_raise_strategy() -> None:
    """Test RAISE strategy raises exception when full."""
    buffer = AsyncBoundedBuffer[int](
        capacity=2, overflow_strategy=OverflowStrategy.RAISE
    )

    await buffer.put(1)
    await buffer.put(2)

    with pytest.raises(BufferFullError):
        await buffer.put(3)


@pytest.mark.asyncio
async def test_put_nowait_block_strategy() -> None:
    """Test put_nowait with BLOCK strategy returns False when full."""
    buffer = AsyncBoundedBuffer[int](capacity=2, overflow_strategy=OverflowStrategy.BLOCK)

    assert await buffer.put(1)
    assert await buffer.put(2)

    # put_nowait should return False when full
    assert not buffer.put_nowait(3)

    # But items should still be in buffer
    assert buffer.size == 2
    assert await buffer.get() == 1
    assert await buffer.get() == 2


@pytest.mark.asyncio
async def test_get_nowait() -> None:
    """Test get_nowait returns None when empty."""
    buffer = AsyncBoundedBuffer[int](capacity=5)

    assert buffer.get_nowait() is None

    await buffer.put(1)
    assert buffer.get_nowait() == 1
    assert buffer.get_nowait() is None


@pytest.mark.asyncio
async def test_get_or_wait_with_timeout() -> None:
    """Test get_or_wait with timeout returns None on timeout."""
    buffer = AsyncBoundedBuffer[int](capacity=5)

    # Should timeout and return None
    result = await buffer.get_or_wait(timeout=0.1)
    assert result is None


@pytest.mark.asyncio
async def test_close_buffer() -> None:
    """Test closing the buffer."""
    buffer = AsyncBoundedBuffer[int](capacity=5)

    await buffer.put(1)
    await buffer.put(2)
    buffer.close()

    assert buffer.closed

    # Can still get remaining items
    assert await buffer.get() == 1
    assert await buffer.get() == 2

    # But get() now raises since buffer is closed and empty
    with pytest.raises(RuntimeError, match="closed and empty"):
        await buffer.get()

    # put() should also fail
    with pytest.raises(RuntimeError, match="closed"):
        await buffer.put(3)


@pytest.mark.asyncio
async def test_async_iteration() -> None:
    """Test async iteration over buffer."""
    buffer = AsyncBoundedBuffer[int](capacity=10)

    items = [1, 2, 3, 4, 5]
    for item in items:
        await buffer.put(item)

    buffer.close()

    result = []
    async for item in buffer:
        result.append(item)

    assert result == items


@pytest.mark.asyncio
async def test_statistics() -> None:
    """Test buffer statistics tracking."""
    buffer = AsyncBoundedBuffer[int](
        capacity=3, overflow_strategy=OverflowStrategy.DROP_OLDEST
    )

    await buffer.put(1)
    await buffer.put(2)
    await buffer.put(3)
    await buffer.put(4)  # Drops 1

    stats = buffer.statistics
    assert stats.puts == 4
    assert stats.gets == 0
    assert stats.drops == 1
    assert stats.overflows == 1

    await buffer.get()
    await buffer.get()

    assert stats.gets == 2

    # Reset statistics
    stats.reset()
    assert stats.puts == 0
    assert stats.gets == 0
    assert stats.drops == 0
    assert stats.overflows == 0


@pytest.mark.asyncio
async def test_properties() -> None:
    """Test buffer properties."""
    buffer = AsyncBoundedBuffer[int](capacity=5)

    assert buffer.capacity == 5
    assert buffer.size == 0
    assert buffer.empty
    assert not buffer.full
    assert buffer.available == 5

    await buffer.put(1)
    await buffer.put(2)

    assert buffer.size == 2
    assert not buffer.empty
    assert not buffer.full
    assert buffer.available == 3

    await buffer.put(3)
    await buffer.put(4)
    await buffer.put(5)

    assert buffer.size == 5
    assert not buffer.empty
    assert buffer.full
    assert buffer.available == 0


@pytest.mark.asyncio
async def test_clear() -> None:
    """Test clearing the buffer."""
    buffer = AsyncBoundedBuffer[int](capacity=5)

    await buffer.put(1)
    await buffer.put(2)
    await buffer.put(3)

    assert buffer.size == 3

    buffer.clear()

    assert buffer.size == 0
    assert buffer.empty


@pytest.mark.asyncio
async def test_len_and_repr() -> None:
    """Test __len__ and __repr__."""
    buffer = AsyncBoundedBuffer[int](capacity=5)

    assert len(buffer) == 0

    await buffer.put(1)
    await buffer.put(2)

    assert len(buffer) == 2

    repr_str = repr(buffer)
    assert "AsyncBoundedBuffer" in repr_str
    assert "capacity=5" in repr_str
    assert "size=2" in repr_str


@pytest.mark.asyncio
async def test_producer_consumer_pattern() -> None:
    """Test realistic producer-consumer scenario."""
    buffer = AsyncBoundedBuffer[int](capacity=5)

    produced = []
    consumed = []

    async def producer() -> None:
        for i in range(10):
            await buffer.put(i)
            produced.append(i)
            await asyncio.sleep(0.01)  # Simulate work

        buffer.close()

    async def consumer() -> None:
        async for item in buffer:
            consumed.append(item)
            await asyncio.sleep(0.02)  # Simulate slower processing

    # Run producer and consumer concurrently
    await asyncio.gather(producer(), consumer())

    assert produced == list(range(10))
    assert consumed == list(range(10))


@pytest.mark.asyncio
async def test_multiple_producers() -> None:
    """Test multiple producers writing to the same buffer."""
    buffer = AsyncBoundedBuffer[int](capacity=10)

    async def producer(id: int) -> None:
        for i in range(5):
            await buffer.put(id * 10 + i)

    # Run multiple producers
    producers = [producer(i) for i in range(3)]
    await asyncio.gather(*producers)
    buffer.close()

    # All items should be in the buffer
    items = []
    async for item in buffer:
        items.append(item)

    assert len(items) == 15


@pytest.mark.asyncio
async def test_invalid_capacity() -> None:
    """Test that invalid capacity raises ValueError."""
    with pytest.raises(ValueError, match="capacity must be at least 1"):
        AsyncBoundedBuffer[int](capacity=0)


@pytest.mark.asyncio
async def test_overflow_constants() -> None:
    """Test OverflowStrategy string constants."""
    assert OverflowStrategy.BLOCK == "block"
    assert OverflowStrategy.DROP_OLDEST == "drop_oldest"
    assert OverflowStrategy.DROP_NEWEST == "drop_newest"
    assert OverflowStrategy.RAISE == "raise"


@pytest.mark.asyncio
async def test_buffer_statistics_repr() -> None:
    """Test BufferStatistics __repr__."""
    stats = BufferStatistics()
    stats.puts = 10
    stats.gets = 8

    repr_str = repr(stats)
    assert "BufferStatistics" in repr_str
    assert "puts=10" in repr_str
    assert "gets=8" in repr_str
