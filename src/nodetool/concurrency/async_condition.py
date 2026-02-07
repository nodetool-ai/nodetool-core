import asyncio
from collections.abc import Callable
from typing import Any


class AsyncCondition:
    """
    An async condition variable for coordinating between tasks.

    A condition variable allows one or more tasks to wait until they are
    notified by another task. Unlike a simple event, a condition variable
    is typically used with a lock to ensure that checks and waits happen
    atomically.

    The key pattern is:
    1. Acquire the lock
    2. Check a condition
    3. If not met, wait() (which releases the lock and waits)
    4. When notified, re-acquire the lock and check condition again

    This prevents race conditions where the condition might change between
    checking and waiting.

    Example:
        condition = AsyncCondition()
        data_ready = False

        async def consumer():
            async with condition:
                while not data_ready:
                    await condition.wait()
                # Process data

        async def producer():
            nonlocal data_ready
            async with condition:
                data_ready = True
                condition.notify_all()

        await asyncio.gather(consumer(), producer())
    """

    def __init__(self, lock: asyncio.Lock | None = None) -> None:
        """
        Initialize the condition variable.

        Args:
            lock: Optional lock to use. If None, creates a new Lock.
                  Using a shared lock allows conditions to share
                  synchronization state.
        """
        self._condition = asyncio.Condition(lock)

    @property
    def lock(self) -> asyncio.Lock:
        """Return the underlying lock."""
        return self._condition._lock  # type: ignore[attr-defined]

    @property
    def waiters(self) -> int:
        """Return the number of tasks waiting on this condition."""
        return len(self._condition._waiters)  # type: ignore[attr-defined]

    def locked(self) -> bool:
        """Return True if the lock is acquired."""
        return self._condition.locked()

    async def acquire(self) -> None:
        """Acquire the underlying lock."""
        await self._condition.acquire()

    def release(self) -> None:
        """Release the underlying lock."""
        self._condition.release()

    async def __aenter__(self) -> "AsyncCondition":
        """Acquire the lock when entering a context."""
        await self._condition.acquire()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Release the lock when exiting a context."""
        self._condition.release()

    def notify(self, n: int = 1) -> None:
        """
        Notify one or more waiting tasks.

        Args:
            n: Number of tasks to notify. Must be >= 1.

        Raises:
            RuntimeError: If the lock is not held when calling notify.
        """
        self._condition.notify(n)

    def notify_all(self) -> None:
        """
        Notify all waiting tasks.

        Raises:
            RuntimeError: If the lock is not held when calling notify_all.
        """
        self._condition.notify_all()

    async def wait(self) -> None:
        """
        Wait until notified.

        This method releases the underlying lock while waiting, then
        re-acquires it before returning. The lock must be held when
        calling this method.

        Warning:
            Always use wait() in a loop that checks the condition,
            as spurious wakeups can occur.

        Raises:
            RuntimeError: If the lock is not held when calling wait.
            asyncio.CancelledError: If the wait is cancelled.

        Example:
            async with condition:
                while not predicate():
                    await condition.wait()
        """
        await self._condition.wait()

    async def wait_for(self, predicate: Callable[[], bool]) -> None:
        """
        Wait until a predicate becomes True.

        This is a convenience method that combines the common pattern of
        checking a condition in a loop.

        Args:
            predicate: A callable that returns True when the condition is met.

        Raises:
            RuntimeError: If the lock is not held when calling wait_for.
            asyncio.CancelledError: If the wait is cancelled.

        Example:
            async with condition:
                await condition.wait_for(lambda: data_ready)
        """
        while not predicate():
            await self._condition.wait()

    async def wait_for_timeout(self, predicate: Callable[[], bool], timeout: float) -> bool:
        """
        Wait until a predicate becomes True, with a timeout.

        Args:
            predicate: A callable that returns True when the condition is met.
            timeout: Maximum time to wait in seconds.

        Returns:
            True if the predicate became True, False if timeout occurred.

        Raises:
            RuntimeError: If the lock is not held when calling wait_for_timeout.
            asyncio.CancelledError: If the wait is cancelled.

        Example:
            async with condition:
                success = await condition.wait_for_timeout(
                    lambda: data_ready, 5.0
                )
        """
        try:
            async with asyncio.timeout(timeout):
                await self.wait_for(predicate)
                return True
        except TimeoutError:
            return False

    def __repr__(self) -> str:
        lock_state = "locked" if self.locked() else "unlocked"
        return f"AsyncCondition({lock_state}, waiters={self.waiters})"


__all__ = ["AsyncCondition"]
