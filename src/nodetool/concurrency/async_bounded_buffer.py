"""
Async Bounded Buffer for producer-consumer patterns with overflow strategies.

Provides a fixed-size buffer with configurable overflow/drop policies when full.
Ideal for scenarios where you want to limit memory usage while still maintaining
flow control between producers and consumers.
"""
import asyncio
from typing import Generic, TypeVar

T = TypeVar("T")


class OverflowStrategy:
    """Strategy for handling items when buffer is full."""

    BLOCK = "block"  # Block producer until space is available (default)
    DROP_OLDEST = "drop_oldest"  # Drop oldest item to make room
    DROP_NEWEST = "drop_newest"  # Drop the new item
    RAISE = "raise"  # Raise BufferFullError


class BufferFullError(Exception):
    """Raised when trying to add to a full buffer with OverflowStrategy.RAISE."""

    pass


class BufferStatistics:
    """Statistics for buffer operations."""

    def __init__(self) -> None:
        self.puts: int = 0
        self.gets: int = 0
        self.drops: int = 0
        self.overflows: int = 0

    def reset(self) -> None:
        """Reset all statistics."""
        self.puts = 0
        self.gets = 0
        self.drops = 0
        self.overflows = 0

    def __repr__(self) -> str:
        return (
            f"BufferStatistics(puts={self.puts}, gets={self.gets}, "
            f"drops={self.drops}, overflows={self.overflows})"
        )


class AsyncBoundedBuffer(Generic[T]):
    """
    An asynchronous bounded buffer with configurable overflow strategies.

    This is a fixed-size FIFO buffer that supports different strategies when
    the buffer is full: block the producer, drop the oldest item, drop the
    newest item, or raise an exception.

    The buffer provides backpressure to producers when using BLOCK strategy,
    preventing unbounded memory growth in high-throughput scenarios.

    Example:
        buffer = AsyncBoundedBuffer[int](capacity=10)

        # Producer
        async def producer():
            for i in range(100):
                await buffer.put(i)

        # Consumer
        async def consumer():
            async for item in buffer:
                print(f"Got: {item}")

    With drop strategy:
        buffer = AsyncBoundedBuffer[str](
            capacity=5,
            overflow_strategy=OverflowStrategy.DROP_OLDEST
        )

        # When full, oldest items are automatically dropped
        for i in range(100):
            await buffer.put(f"item_{i}")
        # Buffer contains only the last 5 items
    """

    def __init__(
        self,
        capacity: int,
        overflow_strategy: str = OverflowStrategy.BLOCK,
    ) -> None:
        """
        Initialize the bounded buffer.

        Args:
            capacity: Maximum number of items in the buffer. Must be > 0.
            overflow_strategy: Strategy for handling items when buffer is full.

        Raises:
            ValueError: If capacity is less than 1.
        """
        if capacity < 1:
            raise ValueError("capacity must be at least 1")

        self._capacity = capacity
        self._strategy = overflow_strategy
        self._buffer: list[T] = []
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)
        self._not_full = asyncio.Condition(self._lock)
        self._closed = False
        self._stats = BufferStatistics()

    async def put(self, item: T) -> bool:
        """
        Put an item into the buffer.

        Behavior depends on the overflow strategy:
        - BLOCK: Wait until space is available
        - DROP_OLDEST: Drop oldest item and add new one
        - DROP_NEWEST: Return False without adding
        - RAISE: Raise BufferFullError immediately

        Args:
            item: The item to add to the buffer.

        Returns:
            True if the item was added, False if dropped (DROP_NEWEST strategy).

        Raises:
            BufferFullError: If buffer is full and strategy is RAISE.
            RuntimeError: If buffer is closed.
        """
        if self._closed:
            raise RuntimeError("Cannot put to closed buffer")

        self._stats.puts += 1

        if self._strategy == OverflowStrategy.BLOCK:
            async with self._not_full:
                # Wait while buffer is full and not closed
                while len(self._buffer) >= self._capacity and not self._closed:
                    await self._not_full.wait()

                if self._closed:
                    raise RuntimeError("Buffer closed while waiting to put")

                self._buffer.append(item)
                self._not_empty.notify(1)
                return True

        elif self._strategy == OverflowStrategy.DROP_OLDEST:
            async with self._lock:
                if len(self._buffer) >= self._capacity:
                    self._buffer.pop(0)  # Drop oldest
                    self._stats.drops += 1
                    self._stats.overflows += 1

                self._buffer.append(item)
                self._not_empty.notify(1)
                return True

        elif self._strategy == OverflowStrategy.DROP_NEWEST:
            async with self._lock:
                if len(self._buffer) >= self._capacity:
                    self._stats.drops += 1
                    self._stats.overflows += 1
                    return False

                self._buffer.append(item)
                self._not_empty.notify(1)
                return True

        elif self._strategy == OverflowStrategy.RAISE:
            async with self._lock:
                if len(self._buffer) >= self._capacity:
                    raise BufferFullError(
                        f"Buffer is full (capacity={self._capacity})"
                    )

                self._buffer.append(item)
                self._not_empty.notify(1)
                return True

        return False

    def put_nowait(self, item: T) -> bool:
        """
        Put an item into the buffer without blocking.

        Uses the configured overflow strategy but never blocks:
        - BLOCK: Return False if full
        - DROP_OLDEST: Drop oldest and add new one
        - DROP_NEWEST: Return False if full
        - RAISE: Raise BufferFullError if full

        Args:
            item: The item to add to the buffer.

        Returns:
            True if the item was added, False if buffer was full.

        Raises:
            BufferFullError: If buffer is full and strategy is RAISE.
        """
        if self._closed:
            raise RuntimeError("Cannot put to closed buffer")

        self._stats.puts += 1

        if self._strategy == OverflowStrategy.BLOCK:
            if len(self._buffer) >= self._capacity:
                return False

            self._buffer.append(item)
            self._not_empty.notify(1)
            return True

        elif self._strategy == OverflowStrategy.DROP_OLDEST:
            if len(self._buffer) >= self._capacity:
                self._buffer.pop(0)
                self._stats.drops += 1
                self._stats.overflows += 1

            self._buffer.append(item)
            self._not_empty.notify(1)
            return True

        elif self._strategy == OverflowStrategy.DROP_NEWEST:
            if len(self._buffer) >= self._capacity:
                self._stats.drops += 1
                self._stats.overflows += 1
                return False

            self._buffer.append(item)
            self._not_empty.notify(1)
            return True

        elif self._strategy == OverflowStrategy.RAISE:
            if len(self._buffer) >= self._capacity:
                raise BufferFullError(
                    f"Buffer is full (capacity={self._capacity})"
                )

            self._buffer.append(item)
            self._not_empty.notify(1)
            return True

        return False

    async def get(self) -> T:
        """
        Get an item from the buffer.

        Blocks if the buffer is empty until an item is available.

        Returns:
            The oldest item from the buffer (FIFO).

        Raises:
            RuntimeError: If buffer is closed and empty.
        """
        async with self._not_empty:
            # Wait while buffer is empty and not closed
            while len(self._buffer) == 0 and not self._closed:
                await self._not_empty.wait()

            if len(self._buffer) == 0:
                raise RuntimeError("Buffer is closed and empty")

            item = self._buffer.pop(0)
            self._stats.gets += 1
            self._not_full.notify(1)
            return item

    def get_nowait(self) -> T | None:
        """
        Get an item from the buffer without blocking.

        Returns:
            The oldest item from the buffer, or None if empty.
        """
        if len(self._buffer) == 0:
            return None

        item = self._buffer.pop(0)
        self._stats.gets += 1
        self._not_full.notify(1)
        return item

    async def get_or_wait(
        self, timeout: float | None = None
    ) -> T | None:
        """
        Get an item from the buffer with optional timeout.

        Args:
            timeout: Maximum time to wait in seconds. None means wait indefinitely.

        Returns:
            The item from the buffer, or None if timeout is reached.
        """
        if timeout is None:
            return await self.get()

        try:
            return await asyncio.wait_for(self.get(), timeout=timeout)
        except TimeoutError:
            return None

    def close(self) -> None:
        """
        Close the buffer.

        No more items can be put after closing, but pending items
        can still be retrieved. The buffer can be iterated over to
        drain remaining items.
        """
        self._closed = True
        # Wake up any waiting tasks
        self._not_empty.notify_all()
        self._not_full.notify_all()

    @property
    def closed(self) -> bool:
        """Check if the buffer is closed."""
        return self._closed

    @property
    def empty(self) -> bool:
        """Check if the buffer is empty."""
        return len(self._buffer) == 0

    @property
    def full(self) -> bool:
        """Check if the buffer is full."""
        return len(self._buffer) >= self._capacity

    @property
    def capacity(self) -> int:
        """Get the buffer capacity."""
        return self._capacity

    @property
    def size(self) -> int:
        """Get the current number of items in the buffer."""
        return len(self._buffer)

    @property
    def available(self) -> int:
        """Get the number of available slots."""
        return max(0, self._capacity - len(self._buffer))

    @property
    def overflow_strategy(self) -> str:
        """Get the overflow strategy."""
        return self._strategy

    @property
    def statistics(self) -> BufferStatistics:
        """Get buffer statistics."""
        return self._stats

    def clear(self) -> None:
        """Clear all items from the buffer."""
        self._buffer.clear()
        self._not_full.notify_all()

    def __aiter__(self) -> "AsyncBoundedBufferIterator[T]":
        """
        Create an async iterator for consuming items from the buffer.

        Example:
            async for item in buffer:
                print(item)

        The iterator stops when the buffer is closed and empty.
        """
        return AsyncBoundedBufferIterator(self)

    def __len__(self) -> int:
        """Get the current number of items in the buffer."""
        return len(self._buffer)

    def __repr__(self) -> str:
        return (
            f"AsyncBoundedBuffer(capacity={self._capacity}, "
            f"size={len(self._buffer)}, "
            f"strategy={self._strategy}, "
            f"closed={self._closed})"
        )


class AsyncBoundedBufferIterator(Generic[T]):
    """
    Async iterator for AsyncBoundedBuffer.

    Automatically handles buffer closing and yields items
    until the buffer is closed and empty.
    """

    def __init__(self, buffer: AsyncBoundedBuffer[T]) -> None:
        self._buffer = buffer

    def __aiter__(self) -> "AsyncBoundedBufferIterator[T]":
        return self

    async def __anext__(self) -> T:
        """Get the next item from the buffer."""
        if self._buffer.closed and self._buffer.empty:
            raise StopAsyncIteration

        try:
            # Use a small timeout to allow checking closed status
            item = await asyncio.wait_for(self._buffer.get(), timeout=0.1)
            return item
        except TimeoutError:
            # Check if we should stop
            if self._buffer.closed and self._buffer.empty:
                raise StopAsyncIteration from None
            # Continue waiting via the loop
            raise


__all__ = [
    "AsyncBoundedBuffer",
    "AsyncBoundedBufferIterator",
    "BufferFullError",
    "BufferStatistics",
    "OverflowStrategy",
]
