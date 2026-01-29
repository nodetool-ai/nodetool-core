import asyncio

import pytest

from nodetool.concurrency.async_buffer import (
    AsyncBuffer,
    AsyncBufferFullError,
    BufferClosedError,
)


class TestAsyncBufferInit:
    """Tests for AsyncBuffer initialization."""

    def test_valid_max_size(self):
        """Test that valid max_size is accepted."""
        buffer = AsyncBuffer(max_size=10)
        assert buffer.max_size == 10
        assert buffer.size == 0
        assert buffer.is_empty
        assert not buffer.is_full

    def test_max_size_one(self):
        """Test that max_size=1 is valid."""
        buffer = AsyncBuffer(max_size=1)
        assert buffer.max_size == 1

    def test_invalid_max_size_zero(self):
        """Test that max_size=0 raises ValueError."""
        with pytest.raises(ValueError, match="max_size must be a positive integer"):
            AsyncBuffer(max_size=0)

    def test_invalid_max_size_negative(self):
        """Test that negative max_size raises ValueError."""
        with pytest.raises(ValueError, match="max_size must be a positive integer"):
            AsyncBuffer(max_size=-1)

    def test_invalid_max_size_float(self):
        """Test that float max_size raises ValueError."""
        with pytest.raises(ValueError, match="max_size must be a positive integer"):
            AsyncBuffer(max_size=10.5)

    def test_block_on_full_default_true(self):
        """Test that block_on_full defaults to True."""
        buffer = AsyncBuffer(max_size=10)
        assert buffer._block_on_full is True

    def test_block_on_full_false(self):
        """Test that block_on_full can be set to False."""
        buffer = AsyncBuffer(max_size=10, block_on_full=False)
        assert buffer._block_on_full is False

    def test_timeout_default_none(self):
        """Test that timeout defaults to None."""
        buffer = AsyncBuffer(max_size=10)
        assert buffer._timeout is None

    def test_timeout_custom(self):
        """Test that custom timeout is accepted."""
        buffer = AsyncBuffer(max_size=10, timeout=5.0)
        assert buffer._timeout == 5.0


class TestAsyncBufferProperties:
    """Tests for AsyncBuffer properties."""

    def test_available_property(self):
        """Test that available property returns correct value."""
        buffer = AsyncBuffer(max_size=10)
        assert buffer.available == 10
        buffer._buffer.append(1)
        assert buffer.available == 9

    def test_is_empty_true(self):
        """Test is_empty when buffer is empty."""
        buffer = AsyncBuffer(max_size=10)
        assert buffer.is_empty is True

    def test_is_empty_false(self):
        """Test is_empty when buffer has items."""
        buffer = AsyncBuffer(max_size=10)
        buffer._buffer.append(1)
        assert buffer.is_empty is False

    def test_is_full_true(self):
        """Test is_full when buffer is at capacity."""
        buffer = AsyncBuffer(max_size=2)
        buffer._buffer.append(1)
        buffer._buffer.append(2)
        assert buffer.is_full is True

    def test_is_full_false(self):
        """Test is_full when buffer has space."""
        buffer = AsyncBuffer(max_size=10)
        buffer._buffer.append(1)
        assert buffer.is_full is False

    def test_is_closed_false_initially(self):
        """Test that buffer is not closed on init."""
        buffer = AsyncBuffer(max_size=10)
        assert buffer.is_closed is False


class TestAsyncBufferPut:
    """Tests for AsyncBuffer.put() method."""

    @pytest.mark.asyncio
    async def test_put_single_item(self):
        """Test putting a single item."""
        buffer = AsyncBuffer(max_size=10)
        result = await buffer.put("item")
        assert result is True
        assert buffer.size == 1
        assert buffer._buffer[0] == "item"

    @pytest.mark.asyncio
    async def test_put_multiple_items(self):
        """Test putting multiple items."""
        buffer = AsyncBuffer(max_size=10)
        await buffer.put(1)
        await buffer.put(2)
        await buffer.put(3)
        assert buffer.size == 3
        assert list(buffer._buffer) == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_put_non_blocking_when_full(self):
        """Test that put() raises when full and block_on_full=False."""
        buffer = AsyncBuffer(max_size=2, block_on_full=False)
        await buffer.put(1)
        await buffer.put(2)
        with pytest.raises(AsyncBufferFullError):
            await buffer.put(3)

    @pytest.mark.asyncio
    async def test_put_raises_on_closed_buffer(self):
        """Test that put() raises on closed buffer."""
        buffer = AsyncBuffer(max_size=10)
        await buffer.aclose()
        with pytest.raises(BufferClosedError):
            await buffer.put("item")


class TestAsyncBufferGet:
    """Tests for AsyncBuffer.get() method."""

    @pytest.mark.asyncio
    async def test_get_single_item(self):
        """Test getting a single item."""
        buffer = AsyncBuffer(max_size=10)
        await buffer.put("item")
        item = await buffer.get()
        assert item == "item"
        assert buffer.is_empty

    @pytest.mark.asyncio
    async def test_get_fifo_order(self):
        """Test that items are retrieved in FIFO order."""
        buffer = AsyncBuffer(max_size=10)
        await buffer.put(1)
        await buffer.put(2)
        await buffer.put(3)
        assert await buffer.get() == 1
        assert await buffer.get() == 2
        assert await buffer.get() == 3

    @pytest.mark.asyncio
    async def test_get_blocks_on_empty(self):
        """Test that get() blocks when buffer is empty."""
        buffer = AsyncBuffer(max_size=10)
        result = None

        async def get_with_timeout():
            nonlocal result
            result = await buffer.get()

        task = asyncio.create_task(get_with_timeout())
        await asyncio.sleep(0.05)
        assert result is None
        await buffer.put("item")
        await asyncio.sleep(0.05)
        assert result == "item"
        task.cancel()

    @pytest.mark.asyncio
    async def test_get_raises_on_closed_empty_buffer(self):
        """Test that get() raises on closed empty buffer."""
        buffer = AsyncBuffer(max_size=10)
        await buffer.aclose()
        with pytest.raises(BufferClosedError):
            await buffer.get()


class TestAsyncBufferGetBatch:
    """Tests for AsyncBuffer.get_batch() method."""

    @pytest.mark.asyncio
    async def test_get_batch_single_item(self):
        """Test getting a batch with one item."""
        buffer = AsyncBuffer(max_size=10)
        await buffer.put(1)
        batch = await buffer.get_batch(5)
        assert batch == [1]

    @pytest.mark.asyncio
    async def test_get_batch_multiple_items(self):
        """Test getting a batch with multiple items."""
        buffer = AsyncBuffer(max_size=10)
        await buffer.put(1)
        await buffer.put(2)
        await buffer.put(3)
        batch = await buffer.get_batch(2)
        assert batch == [1, 2]

    @pytest.mark.asyncio
    async def test_get_batch_exceeds_available(self):
        """Test getting batch larger than available items."""
        buffer = AsyncBuffer(max_size=10)
        await buffer.put(1)
        await buffer.put(2)
        batch = await buffer.get_batch(10)
        assert batch == [1, 2]

    @pytest.mark.asyncio
    async def test_get_batch_empty_buffer(self):
        """Test getting batch from empty buffer returns empty list."""
        buffer = AsyncBuffer(max_size=10)
        batch = await buffer.get_batch(5)
        assert batch == []

    @pytest.mark.asyncio
    async def test_get_batch_invalid_size(self):
        """Test that invalid batch_size raises ValueError."""
        buffer = AsyncBuffer(max_size=10)
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            await buffer.get_batch(0)

    @pytest.mark.asyncio
    async def test_get_batch_negative_size(self):
        """Test that negative batch_size raises ValueError."""
        buffer = AsyncBuffer(max_size=10)
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            await buffer.get_batch(-1)


class TestAsyncBufferFlush:
    """Tests for AsyncBuffer.flush() method."""

    @pytest.mark.asyncio
    async def test_flush_returns_all_items(self):
        """Test that flush() returns all items."""
        buffer = AsyncBuffer(max_size=10)
        await buffer.put(1)
        await buffer.put(2)
        await buffer.put(3)
        items = await buffer.flush()
        assert items == [1, 2, 3]
        assert buffer.is_empty

    @pytest.mark.asyncio
    async def test_flush_empty_buffer(self):
        """Test that flush() on empty buffer returns empty list."""
        buffer = AsyncBuffer(max_size=10)
        items = await buffer.flush()
        assert items == []

    @pytest.mark.asyncio
    async def test_flush_raises_on_closed_buffer(self):
        """Test that flush() raises on closed buffer."""
        buffer = AsyncBuffer(max_size=10)
        await buffer.aclose()
        with pytest.raises(BufferClosedError):
            await buffer.flush()


class TestAsyncBufferClose:
    """Tests for AsyncBuffer.close() and aclose() methods."""

    @pytest.mark.asyncio
    async def test_close_sets_closed_flag(self):
        """Test that close() sets the closed flag."""
        buffer = AsyncBuffer(max_size=10)
        await buffer.aclose()
        assert buffer.is_closed

    @pytest.mark.asyncio
    async def test_close_wakes_waiting_getters(self):
        """Test that close() wakes waiting get() calls."""
        buffer = AsyncBuffer(max_size=10)
        error_raised = False

        async def try_get():
            nonlocal error_raised
            try:
                await buffer.get()
            except BufferClosedError:
                error_raised = True

        task = asyncio.create_task(try_get())
        await asyncio.sleep(0.05)
        await buffer.aclose()
        await asyncio.sleep(0.05)
        assert error_raised
        task.cancel()


class TestAsyncBufferContextManager:
    """Tests for AsyncBuffer as async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test using buffer as async context manager."""
        async with AsyncBuffer(max_size=10) as buffer:
            await buffer.put(1)
            item = await buffer.get()
            assert item == 1

    @pytest.mark.asyncio
    async def test_context_manager_closes_on_exit(self):
        """Test that context manager closes buffer on exit."""
        buffer = None
        async with AsyncBuffer(max_size=10) as buf:
            buffer = buf
        assert buffer.is_closed


class TestAsyncBufferAsyncIterator:
    """Tests for AsyncBuffer async iteration."""

    @pytest.mark.asyncio
    async def test_async_iterator(self):
        """Test async iteration over buffer items."""
        buffer = AsyncBuffer(max_size=10)
        await buffer.put(1)
        await buffer.put(2)
        await buffer.put(3)
        await buffer.aclose()

        items = [item async for item in buffer]
        assert items == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_async_iterator_empty_buffer(self):
        """Test async iteration over empty buffer."""
        buffer = AsyncBuffer(max_size=10)
        await buffer.aclose()

        items = [item async for item in buffer]
        assert items == []

    @pytest.mark.asyncio
    async def test_async_iterator_cancelled(self):
        """Test that async iteration handles cancellation."""
        buffer = AsyncBuffer(max_size=10)
        await buffer.put(1)
        await buffer.put(2)
        await buffer.put(3)

        items = []
        async for item in buffer:
            items.append(item)
            if item == 2:
                break

        assert items == [1, 2]
