from __future__ import annotations

import asyncio
from collections import deque
from typing import Any, AsyncGenerator


class AsyncBufferFullError(Exception):
    """Raised when the buffer is full and blocking behavior is configured."""

    pass


class AsyncBuffer:
    """
    A bounded async buffer for streaming data with backpressure support.

    Provides a thread-safe, asyncio-compatible buffer that:
    - Limits memory usage with configurable max size
    - Supports blocking and non-blocking put operations
    - Handles backpressure by blocking or raising when full
    - Supports flushing and graceful shutdown

    Example:
        ```python
        buffer = AsyncBuffer(max_size=100)

        async def producer():
            for i in range(1000):
                await buffer.put(i)

        async def consumer():
            while True:
                item = await buffer.get()
                if item is None:  # Sentinel for flush
                    break
                process(item)

        await asyncio.gather(producer(), consumer())
        ```
    """

    def __init__(
        self,
        max_size: int,
        *,
        block_on_full: bool = True,
        timeout: float | None = None,
    ) -> None:
        """
        Initialize the async buffer.

        Args:
            max_size: Maximum number of items the buffer can hold.
            block_on_full: If True, put() blocks when full until space available.
                          If False, raises AsyncBufferFullError when full.
            timeout: Optional timeout for blocking put operations.

        Raises:
            ValueError: If max_size is not a positive integer.
        """
        if not isinstance(max_size, int) or max_size <= 0:
            raise ValueError("max_size must be a positive integer")

        self._max_size = max_size
        self._block_on_full = block_on_full
        self._timeout = timeout
        self._buffer: deque[Any] = deque()
        self._put_event = asyncio.Event()
        self._closed = False
        self._put_lock = asyncio.Lock()
        self._get_lock = asyncio.Lock()

    @property
    def max_size(self) -> int:
        """Maximum buffer capacity."""
        return self._max_size

    @property
    def size(self) -> int:
        """Current number of items in the buffer."""
        return len(self._buffer)

    @property
    def available(self) -> int:
        """Number of available slots in the buffer."""
        return self._max_size - len(self._buffer)

    @property
    def is_full(self) -> bool:
        """Check if the buffer is at capacity."""
        return len(self._buffer) >= self._max_size

    @property
    def is_empty(self) -> bool:
        """Check if the buffer is empty."""
        return len(self._buffer) == 0

    @property
    def is_closed(self) -> bool:
        """Check if the buffer has been closed."""
        return self._closed

    async def put(self, item: Any) -> bool:
        """
        Add an item to the buffer.

        Args:
            item: The item to add to the buffer.

        Returns:
            True if the item was added successfully.

        Raises:
            AsyncBufferFullError: If buffer is full and block_on_full is False.
            asyncio.TimeoutError: If timeout is exceeded while blocking.
            BufferClosedError: If the buffer has been closed.
        """
        if self._closed:
            raise BufferClosedError("Cannot add to closed buffer")

        if self._block_on_full:
            while self.is_full:
                try:
                    await asyncio.wait_for(self._put_event.wait(), timeout=self._timeout)
                except TimeoutError:
                    raise TimeoutError("Buffer put operation timed out") from None
                self._put_event.clear()
        else:
            if self.is_full:
                raise AsyncBufferFullError(f"Buffer is full (max_size={self._max_size})")

        async with self._put_lock:
            self._buffer.append(item)
            if len(self._buffer) == 1:
                self._put_event.set()

        return True

    async def get(self) -> Any:
        """
        Remove and return an item from the buffer.

        Returns:
            The oldest item in the buffer.

        Raises:
            asyncio.CancelledError: If the buffer is empty and closed.
        """
        async with self._get_lock:
            while not self._buffer and not self._closed:
                await asyncio.sleep(0.01)

            if not self._buffer and self._closed:
                raise BufferClosedError("Buffer is closed and empty")

            item = self._buffer.popleft()
            if self.is_full:
                self._put_event.set()

            return item

    async def get_batch(self, batch_size: int) -> list[Any]:
        """
        Get multiple items from the buffer.

        Args:
            batch_size: Maximum number of items to retrieve.

        Returns:
            List of up to batch_size items, or empty list if buffer is empty.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        batch = []
        for _ in range(batch_size):
            try:
                item = await asyncio.wait_for(self.get(), timeout=0.01)
                batch.append(item)
            except TimeoutError:
                break
            except BufferClosedError:
                if batch:
                    break
                raise

        return batch

    async def flush(self) -> list[Any]:
        """
        Flush all items from the buffer atomically.

        Returns:
            List of all items that were in the buffer.

        Raises:
            BufferClosedError: If the buffer is already closed.
        """
        if self._closed:
            raise BufferClosedError("Cannot flush closed buffer")

        async with self._put_lock, self._get_lock:
            items = list(self._buffer)
            self._buffer.clear()
            if self.is_full:
                self._put_event.set()

        return items

    async def aclose(self) -> None:
        """
        Close the buffer and signal consumers to stop.

        Consumers will receive BufferClosedError on get() when empty.
        Pending puts will raise BufferClosedError.
        """
        self._closed = True
        self._put_event.set()

    async def __aenter__(self) -> AsyncBuffer:
        """Support async context manager."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager, closing the buffer."""
        await self.aclose()

    def __aiter__(self) -> AsyncGenerator[Any, None]:
        """Support async iteration over buffer items."""
        return self._async_iter()

    async def _async_iter(self) -> AsyncGenerator[Any, None]:
        """Async iterator implementation."""
        try:
            while True:
                try:
                    item = await self.get()
                    yield item
                except BufferClosedError:
                    break
        except asyncio.CancelledError:
            pass


class BufferClosedError(Exception):
    """Raised when operating on a closed buffer."""

    pass
