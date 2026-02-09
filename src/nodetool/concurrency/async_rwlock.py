"""
Async Reader-Writer Lock (Read-Write Lock) for nodetool.

A read-write lock allows multiple concurrent readers but ensures exclusive
access for writers. This is useful when you have data that is read frequently
but written occasionally.

Key features:
- Multiple readers can hold the lock simultaneously
- Only one writer can hold the lock at a time
- Writers have priority over new readers (writer-preference to prevent starvation)
- Timeout support for all acquire operations

This implementation uses writer-preference semantics: when a writer is waiting,
new readers will block. This prevents writer starvation.

Both acquire and release methods are async, consistent with asyncio patterns.
"""

import asyncio
from typing import Any


class AsyncReaderWriterLock:
    """
    An async read-write lock with timeout support.

    Uses writer-preference semantics to prevent writer starvation.
    When a writer is waiting, new readers will block.

    Example:
        lock = AsyncReaderWriterLock()

        # Reading - multiple readers can hold the lock
        async with lock.read_lock():
            data = await read_shared_data()

        # Writing - exclusive access
        async with lock.write_lock():
            await update_shared_data(new_value)
    """

    def __init__(self) -> None:
        """Initialize the read-write lock."""
        self._readers: int = 0
        self._writers: int = 0  # Will be 0 or 1
        self._writers_waiting: int = 0
        self._cond = asyncio.Condition()

    @property
    def readers(self) -> int:
        """Return the current number of active readers."""
        return self._readers

    @property
    def writers(self) -> int:
        """Return the current number of active writers (0 or 1)."""
        return self._writers

    @property
    def write_pending(self) -> int:
        """Return the number of writers waiting to acquire the lock."""
        return self._writers_waiting

    @property
    def locked(self) -> bool:
        """Return True if the lock is held by either readers or a writer."""
        return self._readers > 0 or self._writers > 0

    @property
    def write_locked(self) -> bool:
        """Return True if the lock is held by a writer."""
        return self._writers > 0

    async def acquire_read(self, timeout: float | None = None) -> bool:
        """
        Acquire the lock for reading.

        Multiple readers can hold the lock simultaneously. If a writer is
        currently holding the lock or writers are waiting, the reader will
        wait until all writers complete.

        Args:
            timeout: Maximum time to wait in seconds. If None, wait indefinitely.

        Returns:
            True if the lock was acquired for reading, False if timeout expired.
        """
        async with self._cond:
            # Wait for no active writers and no waiting writers (writer preference)
            if timeout is None:
                while self._writers > 0 or self._writers_waiting > 0:
                    await self._cond.wait()
            elif timeout <= 0:
                if self._writers > 0 or self._writers_waiting > 0:
                    return False
            else:
                try:
                    async with asyncio.timeout(timeout):
                        while self._writers > 0 or self._writers_waiting > 0:
                            await self._cond.wait()
                except TimeoutError:
                    return False

            self._readers += 1
            return True

    async def acquire_write(self, timeout: float | None = None) -> bool:
        """
        Acquire the lock for writing.

        Only one writer can hold the lock at a time. Writers wait for all
        readers and any current writer to release before acquiring.

        Args:
            timeout: Maximum time to wait in seconds. If None, wait indefinitely.

        Returns:
            True if the lock was acquired for writing, False if timeout expired.
        """
        async with self._cond:
            self._writers_waiting += 1
            try:
                # Wait for no readers and no active writer
                if timeout is None:
                    while self._readers > 0 or self._writers > 0:
                        await self._cond.wait()
                elif timeout <= 0:
                    if self._readers > 0 or self._writers > 0:
                        return False
                else:
                    try:
                        async with asyncio.timeout(timeout):
                            while self._readers > 0 or self._writers > 0:
                                await self._cond.wait()
                    except TimeoutError:
                        return False

                self._writers = 1
                return True
            finally:
                self._writers_waiting -= 1

    async def release_read(self) -> None:
        """
        Release the lock held for reading.

        Raises:
            RuntimeError: If no readers are currently holding the lock.
        """
        async with self._cond:
            if self._readers == 0:
                raise RuntimeError("Cannot release read lock: no readers holding the lock")
            self._readers -= 1
            if self._readers == 0:
                # Notify waiting writers and readers
                self._cond.notify_all()

    async def release_write(self) -> None:
        """
        Release the lock held for writing.

        Raises:
            RuntimeError: If no writer is currently holding the lock.
        """
        async with self._cond:
            if self._writers == 0:
                raise RuntimeError("Cannot release write lock: no writer holding the lock")
            self._writers = 0
            # Notify all waiting tasks (both readers and writers)
            self._cond.notify_all()

    def read_lock(self) -> "_ReadLockContext":
        """Return a context manager for acquiring the read lock."""
        return _ReadLockContext(self)

    def write_lock(self) -> "_WriteLockContext":
        """Return a context manager for acquiring the write lock."""
        return _WriteLockContext(self)

    def __repr__(self) -> str:
        """Return a string representation of the lock state."""
        if self._writers > 0:
            return f"AsyncReaderWriterLock(write_locked, readers=0, pending_writers={self._writers_waiting})"
        elif self._readers > 0:
            return f"AsyncReaderWriterLock(read_locked, readers={self._readers}, pending_writers={self._writers_waiting})"
        else:
            return f"AsyncReaderWriterLock(unlocked, readers=0, pending_writers={self._writers_waiting})"


class _ReadLockContext:
    """Context manager for read lock acquisition."""

    def __init__(self, lock: AsyncReaderWriterLock) -> None:
        self._lock: AsyncReaderWriterLock = lock

    async def __aenter__(self) -> "_ReadLockContext":
        await self._lock.acquire_read()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._lock.release_read()


class _WriteLockContext:
    """Context manager for write lock acquisition."""

    def __init__(self, lock: AsyncReaderWriterLock) -> None:
        self._lock: AsyncReaderWriterLock = lock

    async def __aenter__(self) -> "_WriteLockContext":
        await self._lock.acquire_write()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._lock.release_write()


__all__ = ["AsyncReaderWriterLock"]
