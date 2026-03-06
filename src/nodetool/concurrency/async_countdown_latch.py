"""
Async CountDownLatch for coordinating multiple async tasks.

A CountDownLatch is a synchronization primitive that allows one or more
tasks to wait until a set of operations being performed in other tasks completes.

This is useful for scenarios like:
- Waiting for multiple concurrent operations to complete before proceeding
- Coordinating the start of multiple tasks (wait for all to be ready)
- Ensuring cleanup happens only after all dependent tasks finish
"""

import asyncio
from typing import Literal


class AsyncCountDownLatch:
    """
    An async countdown latch that allows tasks to wait until a count reaches zero.

    The latch maintains a counter that is decremented each time count_down()
    is called. Tasks can wait for the counter to reach zero using wait().

    Once the count reaches zero, all waiting tasks are unblocked and any
    subsequent calls to wait() return immediately.

    Example:
        >>> async def worker(latch, worker_id):
        ...     print(f"Worker {worker_id} starting")
        ...     await asyncio.sleep(0.1)
        ...     print(f"Worker {worker_id} done")
        ...     latch.count_down()
        >>> # Create latch with count of 3
        >>> latch = AsyncCountDownLatch(3)
        >>> # Launch 3 workers
        >>> tasks = [asyncio.create_task(worker(latch, i)) for i in range(3)]
        >>> # Wait for all workers to complete
        >>> await latch.wait()
        >>> print("All workers completed")
    """

    _count: int
    _event: asyncio.Event

    def __init__(self, count: int):
        """
        Initialize the countdown latch with the given count.

        Args:
            count: The initial count value. Must be non-negative.

        Raises:
            ValueError: If count is negative.
        """
        if count < 0:
            raise ValueError("count must be non-negative")
        self._count = count
        self._event = asyncio.Event()
        if count == 0:
            self._event.set()

    @property
    def count(self) -> int:
        """Return the current count value."""
        return self._count

    def count_down(self, n: int = 1) -> None:
        """
        Decrement the count by n.

        If the count reaches zero, all waiting tasks are notified.

        Args:
            n: The amount to decrement the count by. Must be positive.
               Defaults to 1.

        Raises:
            ValueError: If n is not positive or would make count negative.
            RuntimeError: If called after count has already reached zero.
        """
        if n <= 0:
            raise ValueError("decrement amount must be positive")
        if self._count == 0:
            raise RuntimeError("countdown latch already at zero")
        if n > self._count:
            raise ValueError(f"cannot decrement by {n}, current count is {self._count}")

        self._count -= n
        if self._count == 0:
            self._event.set()

    async def wait(self) -> None:
        """
        Wait until the count reaches zero.

        If the count is already zero, returns immediately.
        Otherwise, blocks until count_down() is called enough times
        to reduce the count to zero.

        Args:
            timeout: Optional timeout in seconds. If specified and the
                    timeout expires before the count reaches zero,
                    TimeoutError is raised.

        Raises:
            TimeoutError: If timeout is specified and expires before count reaches zero.
        """
        await self._event.wait()

    async def wait_timeout(self, timeout: float) -> Literal[True]:
        """
        Wait until the count reaches zero, with a timeout.

        Args:
            timeout: Maximum time to wait in seconds. Must be positive.

        Returns:
            True if the count reached zero before the timeout.

        Raises:
            ValueError: If timeout is not positive.
            TimeoutError: If the timeout expires before count reaches zero.
        """
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
        except TimeoutError:
            raise
        return True

    def is_done(self) -> bool:
        """
        Check if the count has reached zero.

        Returns:
            True if the count is zero, False otherwise.
        """
        return self._count == 0
