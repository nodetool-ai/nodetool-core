import asyncio
from typing import Optional


class AsyncLock:
    """
    An async lock with timeout support for exclusive resource access.

    This provides a more convenient interface than asyncio.Lock by adding:
    - Built-in timeout support via the `acquire` method
    - Context manager support for automatic release
    - Ability to check locked state without blocking

    Example:
        lock = AsyncLock()

        # Using async context manager
        async with lock:
            await do_exclusive_work()

        # Using acquire with timeout
        if await lock.acquire(timeout=10.0):
            try:
                await do_exclusive_work()
            finally:
                lock.release()
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    @property
    def locked(self) -> bool:
        """Return True if the lock is currently held."""
        return self._lock.locked()

    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire the lock with an optional timeout.

        Args:
            timeout (Optional[float]): Maximum time to wait in seconds. If None (default),
                                      wait indefinitely. If <= 0, attempt to acquire
                                      without waiting.

        Returns:
            bool: True if the lock was acquired, False if the timeout expired.
        """
        if timeout is None:
            await self._lock.acquire()
            return True

        if timeout <= 0:
            if self._lock.locked():
                return False
            await self._lock.acquire()
            return True

        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=timeout)
            return True
        except (TimeoutError, asyncio.CancelledError):
            return False

    def release(self) -> None:
        """Release the lock, allowing another task to acquire it."""
        self._lock.release()

    async def __aenter__(self) -> "AsyncLock":
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def __repr__(self) -> str:
        state = "locked" if self.locked else "unlocked"
        return f"AsyncLock({state})"
