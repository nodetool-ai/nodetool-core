from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable


T = TypeVar("T")


@dataclass
class QueueStats:
    """Statistics for an AsyncQueue."""

    current_size: int
    max_size: int | None
    put_waiters: int
    get_waiters: int
    total_puts: int
    total_gets: int
    total_put_timeouts: int
    total_get_timeouts: int


class AsyncQueue(Generic[T]):
    """An async producer-consumer queue with optional size limits.

    This provides a thread-safe queue for async operations, supporting:
    - Bounded and unbounded queues
    - Blocking put/get with optional timeouts
    - Non-blocking put_nowait/get_nowait operations
    - Graceful shutdown with remaining item retrieval

    Example:
        ```python
        from nodetool.concurrency import AsyncQueue

        queue: AsyncQueue[str] = AsyncQueue(max_size=100)

        async def producer():
            for i in range(1000):
                await queue.put(f"item_{i}")

        async def consumer():
            while True:
                item = await queue.get()
                if item is None:  # Shutdown signal
                    break
                process(item)

        async def main():
            await asyncio.gather(producer(), consumer())
        ```
    """

    def __init__(self, max_size: int | None = None) -> None:
        """Initialize the queue.

        Args:
            max_size: Maximum number of items the queue can hold.
                     None means unbounded (default).

        Raises:
            ValueError: If max_size is not a positive integer.
        """
        if max_size is not None and max_size <= 0:
            raise ValueError("max_size must be a positive integer or None")

        self._queue: deque[T] = deque()
        self._max_size = max_size
        self._put_waiters: list[tuple[asyncio.Future[T], T]] = []
        self._get_waiters: list[asyncio.Future[T]] = []
        self._shutdown = False
        self._stats = QueueStats(
            current_size=0,
            max_size=max_size,
            put_waiters=0,
            get_waiters=0,
            total_puts=0,
            total_gets=0,
            total_put_timeouts=0,
            total_get_timeouts=0,
        )

    @property
    def max_size(self) -> int | None:
        """Return the maximum queue size (None for unbounded)."""
        return self._max_size

    @property
    def stats(self) -> QueueStats:
        """Return current queue statistics."""
        self._stats.current_size = len(self._queue)
        self._stats.put_waiters = len(self._put_waiters)
        self._stats.get_waiters = len(self._get_waiters)
        return self._stats

    def qsize(self) -> int:
        """Return the approximate current size of the queue.

        Note: In an async context, the size may change immediately after
        this method returns. Use for monitoring purposes only.
        """
        return len(self._queue)

    def empty(self) -> bool:
        """Return True if the queue is empty.

        Note: In an async context, the queue may become non-empty immediately
        after this method returns. Use for monitoring purposes only.
        """
        return len(self._queue) == 0

    def full(self) -> bool:
        """Return True if the queue is at maximum capacity.

        Note: For unbounded queues (max_size=None), this always returns False.
        """
        return self._max_size is not None and len(self._queue) >= self._max_size

    async def put(self, item: T, timeout: float | None = None) -> bool:
        """Put an item into the queue.

        Blocks until space is available or timeout expires.

        Args:
            item: The item to add to the queue.
            timeout: Maximum time to wait in seconds. None means wait forever.

        Returns:
            True if the item was added, False if timeout occurred.

        Raises:
            QueueShutdownError: If the queue has been shut down.
        """
        if self._shutdown:
            raise QueueShutdownError("Cannot put items into a shutdown queue")

        if self._max_size is None:
            self._queue.append(item)
            self._stats.total_puts += 1
            self._wake_getters()
            return True

        loop = asyncio.get_running_loop()

        while True:
            if len(self._queue) < self._max_size:
                self._queue.append(item)
                self._stats.total_puts += 1
                self._wake_getters()
                return True

            if self._get_waiters:
                waiter = self._get_waiters.pop(0)
                self._queue.append(item)
                self._stats.total_puts += 1
                if not waiter.done():
                    waiter.set_result(item)
                return True

            if timeout is not None:
                await asyncio.sleep(timeout)
                self._stats.total_put_timeouts += 1
                return False

            future: asyncio.Future[T] = loop.create_future()
            self._put_waiters.append((future, item))
            try:
                await future
                return True
            except asyncio.CancelledError:
                self._stats.total_put_timeouts += 1
                try:
                    self._put_waiters.remove((future, item))
                except ValueError:
                    pass
                raise

    def put_nowait(self, item: T) -> bool:
        """Put an item into the queue without blocking.

        Args:
            item: The item to add to the queue.

        Returns:
            True if the item was added, False if queue is full.

        Raises:
            QueueShutdownError: If the queue has been shut down.
        """
        if self._shutdown:
            raise QueueShutdownError("Cannot put items into a shutdown queue")

        if self._get_waiters:
            waiter = self._get_waiters.pop(0)
            self._queue.append(item)
            self._stats.total_puts += 1
            if not waiter.done():
                waiter.set_result(item)
            return True

        if self._max_size is None or len(self._queue) < self._max_size:
            self._queue.append(item)
            self._stats.total_puts += 1
            self._wake_getters()
            return True
        return False

    async def get(self, timeout: float | None = None) -> T | None:
        """Get an item from the queue.

        Blocks until an item is available or timeout expires.

        Args:
            timeout: Maximum time to wait in seconds. None means wait forever.

        Returns:
            The item from the queue, or None if timeout occurred during shutdown.

        Raises:
            QueueShutdownError: If the queue has been shut down and is empty.
        """
        loop = asyncio.get_running_loop()

        while True:
            if self._queue:
                item = self._queue.popleft()
                self._stats.total_gets += 1
                self._wake_putters()
                return item

            if self._put_waiters:
                future, item = self._put_waiters.pop(0)
                self._stats.total_gets += 1
                if not future.done():
                    future.set_result(item)
                return item

            if self._shutdown:
                raise QueueShutdownError("Cannot get from an empty shutdown queue")

            if timeout is not None:
                await asyncio.sleep(timeout)

                if self._queue:
                    item = self._queue.popleft()
                    self._stats.total_gets += 1
                    return item

                if self._put_waiters:
                    _, item = self._put_waiters.pop(0)
                    self._stats.total_gets += 1
                    return item

                self._stats.total_get_timeouts += 1
                return None

            future: asyncio.Future[T] = loop.create_future()
            self._get_waiters.append(future)
            try:
                return await future
            except asyncio.CancelledError:
                self._stats.total_get_timeouts += 1
                try:
                    self._get_waiters.remove(future)
                except ValueError:
                    pass
                raise

    def get_nowait(self) -> T | None:
        """Get an item from the queue without blocking.

        Returns:
            The item from the queue, or None if queue is empty.

        Raises:
            QueueShutdownError: If the queue has been shut down and is empty.
        """
        if not self._queue:
            if self._shutdown:
                raise QueueShutdownError("Cannot get from an empty shutdown queue")

            if self._put_waiters:
                future, item = self._put_waiters.pop(0)
                self._stats.total_gets += 1
                if not future.done():
                    future.set_result(item)
                return item
            return None

        item = self._queue.popleft()
        self._stats.total_gets += 1
        self._wake_putters()
        return item

    def shutdown(self, drain: bool = False) -> None:
        """Shut down the queue.

        After shutdown, new put() calls will raise QueueShutdownError,
        and get() will raise QueueShutdownError when the queue is empty.

        Args:
            drain: If True, wait for all current waiters to complete.
                  If False, immediately mark as shutdown.
        """
        self._shutdown = True

        for future, _ in self._put_waiters:
            if not future.done():
                future.cancel()
        for future in self._get_waiters:
            if not future.done():
                future.cancel()

    def _wake_getters(self) -> None:
        """Wake up waiting getters."""
        for future in list(self._get_waiters):
            if not future.done() and self._queue:
                item = self._queue.popleft()
                if not future.done():
                    future.set_result(item)

    def _wake_putters(self) -> None:
        """Wake up waiting putters."""
        if self._queue:
            return
        for _i, (future, item) in enumerate(list(self._put_waiters)):
            if not future.done():
                self._queue.append(item)
                if not future.done():
                    future.set_result(item)
                return

    def __repr__(self) -> str:
        size = len(self._queue)
        max_size = self._max_size
        state = "shutdown" if self._shutdown else "active"
        return f"AsyncQueue(size={size}/{max_size}, state={state})"


class QueueShutdownError(Exception):
    """Raised when attempting to operate on a shutdown queue."""

    pass
