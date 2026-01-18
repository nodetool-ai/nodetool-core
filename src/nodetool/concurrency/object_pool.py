import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Generic, TypeVar

T = TypeVar("T")


class ObjectPoolError(Exception):
    """Base exception for object pool errors."""

    pass


class PoolClosedError(ObjectPoolError):
    """Raised when trying to acquire from a closed pool."""

    pass


class PoolAcquireTimeoutError(ObjectPoolError):
    """Raised when acquire times out."""

    pass


class AsyncObjectPool(Generic[T]):
    """
    A generic async object pool for reusing expensive-to-create resources.

    This pool manages a collection of reusable objects with lazy initialization,
    validation on borrow, and automatic self-healing for dead objects. It's useful
    for managing resources like HTTP sessions, ML model instances, database connections,
    or any object that is expensive to create.

    Features:
    - Lazy initialization (objects created only when needed)
    - Validation on borrow with configurable health checks
    - Self-healing: dead objects are replaced automatically
    - Configurable pool size and acquire timeouts
    - Context manager support for automatic release
    - Thread-safe async operations

    Example:
        # Create a pool for HTTP sessions
        pool = AsyncObjectPool(
            factory=create_http_session,
            validator=validate_session,
            destructor=close_http_session,
            max_size=10,
        )

        # Use with context manager
        async with pool.acquire() as session:
            await session.get("https://api.example.com")

        # Use with direct acquire/release
        session = await pool.acquire(timeout=5.0)
        try:
            await session.get("https://api.example.com")
        finally:
            pool.release(session)
    """

    def __init__(
        self,
        factory: Callable[[], Awaitable[T]],
        validator: Callable[[T], Awaitable[bool]] | None = None,
        destructor: Callable[[T], Awaitable[None]] | None = None,
        max_size: int = 10,
        initial_size: int = 0,
    ):
        """
        Initialize the object pool.

        Args:
            factory: Async function to create new objects.
            validator: Async function to validate objects on borrow. Should return
                      True if valid, False otherwise. If None, no validation is performed.
            destructor: Async function to clean up objects on close. If None,
                        objects are simply discarded.
            max_size: Maximum number of objects in the pool. Must be > 0.
            initial_size: Number of objects to pre-create. Must be <= max_size.

        Raises:
            ValueError: If max_size <= 0 or initial_size > max_size.
        """
        if max_size <= 0:
            raise ValueError("max_size must be a positive integer")
        if initial_size < 0:
            raise ValueError("initial_size must be non-negative")
        if initial_size > max_size:
            raise ValueError("initial_size cannot exceed max_size")

        self._factory = factory
        self._validator = validator
        self._destructor = destructor
        self._max_size = max_size
        self._pool: asyncio.Queue[T | None] = asyncio.Queue(maxsize=max_size)
        self._closed = False

        for _ in range(max_size):
            self._pool.put_nowait(None)

        self._lock = asyncio.Lock()

    @property
    def max_size(self) -> int:
        """Return the maximum number of objects in the pool."""
        return self._max_size

    @property
    def available(self) -> int:
        """Return the number of available slots in the pool."""
        return self._pool.qsize()

    async def _create_object(self) -> T:
        """Create a new object using the factory."""
        return await self._factory()

    async def _validate_object(self, obj: T) -> bool:
        """Validate an object using the validator if provided."""
        if self._validator is None:
            return True
        return await self._validator(obj)

    async def _close_object(self, obj: T) -> None:
        """Close an object using the destructor if provided."""
        if self._destructor is not None:
            await self._destructor(obj)

    async def _acquire_object(self) -> T:
        """
        Internal method to acquire an object from the pool.

        Implements the "Lazy Slot" algorithm:
        - Pop a slot from the queue
        - If slot is None: create a fresh object
        - If slot has an object: validate it, self-heal if invalid

        Returns:
            A valid object from the pool.

        Raises:
            PoolClosedError: If the pool is closed.
        """
        if self._closed:
            raise PoolClosedError("Pool is closed")

        slot: T | None = await self._pool.get()

        try:
            if slot is None:
                return await self._create_object()
            else:
                if await self._validate_object(slot):
                    return slot
                else:
                    await self._close_object(slot)
                    return await self._create_object()
        except Exception:
            await self._pool.put(None)
            raise

    async def acquire(self, timeout: float | None = None) -> T | None:
        """
        Acquire an object from the pool.

        Args:
            timeout: Maximum time to wait in seconds. If None (default), wait
                     indefinitely. If <= 0, try to acquire without waiting
                     and return None if no object is available.

        Returns:
            A valid object from the pool, or None if timeout=0 and pool is empty.

        Raises:
            PoolAcquireTimeoutError: If timeout > 0 expires before an object is available.
            PoolClosedError: If the pool is closed.
        """
        if self._closed:
            raise PoolClosedError("Pool is closed")

        if timeout is None:
            return await self._acquire_object()

        if timeout <= 0:
            try:
                return await asyncio.wait_for(self._acquire_object(), timeout=0)
            except (TimeoutError, asyncio.CancelledError):
                return None

        try:
            return await asyncio.wait_for(self._acquire_object(), timeout=timeout)
        except TimeoutError as err:
            raise PoolAcquireTimeoutError(f"Failed to acquire object within {timeout} seconds") from err

    def release(self, obj: T) -> None:
        """
        Release an object back to the pool.

        The object is returned to the pool without validation. Validation
        will be performed on the next acquire (self-healing pattern).

        Args:
            obj: The object to release.
        """
        if self._closed:
            asyncio.get_event_loop().create_task(self._close_object(obj))
            return

        try:
            self._pool.put_nowait(obj)
        except asyncio.QueueFull:
            asyncio.get_event_loop().create_task(self._close_object(obj))

    async def prewarm(self, count: int | None = None) -> int:
        """
        Pre-create objects to warm the pool.

        Args:
            count: Number of objects to create. If None, creates up to max_size
                   or the number of available slots.

        Returns:
            The number of objects actually created.
        """
        if self._closed:
            raise PoolClosedError("Pool is closed")

        count = min(count or self._max_size, self._max_size)
        created = 0

        async with self._lock:
            for _ in range(count):
                slot = await self._pool.get()
                if slot is None:
                    try:
                        obj = await self._create_object()
                        await self._pool.put(obj)
                        created += 1
                    except Exception:
                        await self._pool.put(None)
                        break
                else:
                    await self._pool.put(slot)

        return created

    def try_acquire(self) -> T | None:
        """
        Try to acquire an object without blocking.

        Returns:
            An object if available, None otherwise.
        """
        if self._closed:
            return None

        try:
            slot: T | None = self._pool.get_nowait()
            return slot
        except asyncio.QueueEmpty:
            return None

    async def close(self) -> None:
        """
        Close the pool and all pooled objects.

        After calling this, the pool cannot be used.
        """
        self._closed = True

        while True:
            try:
                slot = self._pool.get_nowait()
                if slot is not None:
                    try:
                        await self._close_object(slot)
                    except Exception:
                        pass
            except asyncio.QueueEmpty:
                break

    async def __aenter__(self) -> "AsyncObjectPool[T]":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    def __repr__(self) -> str:
        status = "closed" if self._closed else "open"
        return f"AsyncObjectPool(max_size={self._max_size}, available={self.available}, status={status})"


@asynccontextmanager
async def pooled(
    pool: AsyncObjectPool[T],
    timeout: float | None = None,
) -> AsyncIterator[T]:
    """
    Context manager for acquiring and releasing objects from a pool.

    This is a convenience function that provides a cleaner syntax for using
    the object pool with a context manager.

    Args:
        pool: The object pool to acquire from.
        timeout: Maximum time to wait for an object. Use None (default) to wait
                 indefinitely, or 0 to return None immediately if pool is empty.

    Yields:
        An acquired object from the pool.

    Raises:
        PoolAcquireTimeoutError: If timeout expires and pool is empty.
        PoolClosedError: If the pool is closed.
        ValueError: If timeout=0 and pool is empty (cannot yield None).

    Example:
        pool = AsyncObjectPool(factory=create_session, max_size=5)

        async with pooled(pool) as session:
            await session.get("https://api.example.com")
    """
    obj = await pool.acquire(timeout=timeout)
    if obj is None:
        if timeout is not None and timeout <= 0:
            raise ValueError(
                "Pool is empty and timeout=0, cannot acquire object. Use a positive timeout or acquire() directly."
            )
        raise PoolAcquireTimeoutError(f"Failed to acquire object within {timeout} seconds")
    try:
        yield obj
    finally:
        pool.release(obj)


__all__ = [
    "AsyncObjectPool",
    "ObjectPoolError",
    "PoolAcquireTimeoutError",
    "PoolClosedError",
    "pooled",
]
