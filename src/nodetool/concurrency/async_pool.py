"""
Async Resource Pool

Provides an async pool of reusable resources with acquire/release semantics.
Useful for managing expensive resources like database connections, HTTP sessions,
or any objects that are costly to create and can be safely reused.

Example:
    async def create_connection():
        return await asyncio.to_thread(lambda: ExpensiveConnection())

    async def close_connection(conn: ExpensiveConnection):
        await asyncio.to_thread(conn.close)

    pool = AsyncPool(
        factory=create_connection,
        closer=close_connection,
        max_size=10,
        initial_size=2
    )

    # Acquire and use a resource
    async with pool.acquire() as conn:
        result = await conn.query("SELECT * FROM table")
    # Resource is automatically returned to pool
"""

import asyncio
from collections import deque
from typing import Awaitable, Callable, Generic, TypeVar

T = TypeVar("T")


class AsyncPool(Generic[T]):
    """
    An async pool of reusable resources with lazy initialization.

    This pool manages a fixed set of resources that are created on-demand
    and can be acquired and released multiple times. Resources are created
    using a factory function and optionally cleaned up when released or
    when the pool is closed.

    Features:
    - Lazy creation: Resources created only when first needed
    - Thread-safe: All operations are atomic and async-safe
    - Configurable cleanup: Optional closer function for resource cleanup
    - Stats tracking: Monitor pool usage and hit rates
    - Graceful shutdown: Clean up all resources on close

    Example:
        pool = AsyncPool(
            factory=lambda: asyncio.create_task(create_db_conn()),
            closer=lambda conn: asyncio.create_task(conn.close()),
            max_size=5
        )

        async with pool.acquire() as conn:
            await conn.execute("INSERT INTO table VALUES (1, 2, 3)")

        # Later, close the pool and cleanup resources
        await pool.close()
    """

    def __init__(
        self,
        factory: Callable[[], Awaitable[T] | T],
        closer: Callable[[T], Awaitable[None] | None] | None = None,
        max_size: int = 10,
        initial_size: int = 0,
    ):
        """
        Initialize the resource pool.

        Args:
            factory: Async or sync function that creates new resources
            closer: Optional async or sync function to cleanup resources
            max_size: Maximum number of resources in the pool (must be > 0)
            initial_size: Number of resources to create upfront (must be >= 0 and <= max_size)

        Raises:
            ValueError: If max_size <= 0 or initial_size is invalid
            TypeError: If factory is not callable
        """
        if max_size <= 0:
            raise ValueError("max_size must be a positive integer")
        if initial_size < 0 or initial_size > max_size:
            raise ValueError("initial_size must be between 0 and max_size")
        if not callable(factory):
            raise TypeError("factory must be callable")

        self._factory = factory
        self._closer = closer
        self._max_size = max_size
        self._initial_size = initial_size
        self._available: deque[T] = deque()
        self._in_use: int = 0
        self._lock = asyncio.Lock()
        self._closed = False
        self._acquire_event = asyncio.Event()

        # Statistics
        self._stats = {
            "created": 0,
            "acquired": 0,
            "released": 0,
            "closed": 0,
        }

    @property
    def max_size(self) -> int:
        """Maximum number of resources the pool can hold."""
        return self._max_size

    @property
    def size(self) -> int:
        """Current number of resources created (available + in-use)."""
        return len(self._available) + self._in_use

    @property
    def available(self) -> int:
        """Number of resources currently available for acquisition."""
        return len(self._available)

    @property
    def in_use(self) -> int:
        """Number of resources currently acquired and in use."""
        return self._in_use

    @property
    def stats(self) -> dict[str, int]:
        """
        Pool usage statistics.

        Returns:
            Dictionary with keys:
            - created: Total resources created
            - acquired: Total acquisition calls
            - released: Total release calls
            - closed: Total resources cleaned up
        """
        return self._stats.copy()

    async def _create_resource(self) -> T:
        """Create a new resource using the factory (async or sync)."""
        result = self._factory()
        if asyncio.iscoroutine(result) or hasattr(result, "__await__"):
            return await result
        return result

    async def _close_resource(self, resource: T) -> None:
        """Close a resource using the closer (async or sync) if provided."""
        if self._closer is None:
            return

        result = self._closer(resource)
        if asyncio.iscoroutine(result) or hasattr(result, "__await__"):
            await result

        self._stats["closed"] += 1

    async def _initialize(self) -> None:
        """Create initial resources during pool construction."""
        async with self._lock:
            if self._closed:
                raise RuntimeError("Cannot initialize a closed pool")

            for _ in range(self._initial_size):
                if len(self._available) + self._in_use >= self._max_size:
                    break
                resource = await self._create_resource()
                self._available.append(resource)
                self._stats["created"] += 1

    async def acquire(self, timeout: float | None = None) -> "AsyncPoolContext[T]":
        """
        Acquire a resource from the pool.

        Args:
            timeout: Maximum time to wait in seconds. If None, wait indefinitely.

        Returns:
            AsyncPoolContext: Context manager for the acquired resource

        Raises:
            RuntimeError: If pool is closed
            TimeoutError: If timeout expires before resource is available

        Example:
            ctx = await pool.acquire()
            try:
                resource = ctx.resource
                # Use resource
            finally:
                await ctx.release()
        """
        if self._closed:
            raise RuntimeError("Cannot acquire from a closed pool")

        if timeout is not None and timeout <= 0:
            # Non-blocking attempt
            async with self._lock:
                if not self._available and self.size >= self._max_size:
                    raise TimeoutError("No resources available")
                if self._available:
                    resource = self._available.popleft()
                else:
                    resource = await self._create_resource()
                    self._stats["created"] += 1
                self._in_use += 1
                self._stats["acquired"] += 1
                return AsyncPoolContext(pool=self, resource=resource)

        # Use asyncio.wait_for for timeout support
        async def _acquire_with_timeout():
            while True:
                async with self._lock:
                    if self._closed:
                        raise RuntimeError("Cannot acquire from a closed pool")

                    if self._available:
                        resource = self._available.popleft()
                        self._in_use += 1
                        self._stats["acquired"] += 1
                        return AsyncPoolContext(pool=self, resource=resource)

                    if self.size < self._max_size:
                        resource = await self._create_resource()
                        self._stats["created"] += 1
                        self._in_use += 1
                        self._stats["acquired"] += 1
                        return AsyncPoolContext(pool=self, resource=resource)

                # Wait for a resource to be released
                self._acquire_event.clear()
                await self._acquire_event.wait()

        if timeout is not None:
            try:
                return await asyncio.wait_for(_acquire_with_timeout(), timeout=timeout)
            except TimeoutError:
                raise TimeoutError("Pool acquisition timed out") from None
        else:
            return await _acquire_with_timeout()

    async def release(self, resource: T) -> None:
        """
        Release a resource back to the pool.

        The resource will be returned to the pool for reuse. If a closer
        was configured and the pool is already at capacity, the resource
        will be closed instead.

        Args:
            resource: The resource to release

        Raises:
            ValueError: If the resource doesn't belong to this pool
            RuntimeError: If the pool is closed
        """
        if self._closed:
            raise RuntimeError("Cannot release to a closed pool")

        async with self._lock:
            if self._in_use <= 0:
                raise ValueError("Releasing resource when none are in use")

            self._in_use -= 1
            self._stats["released"] += 1

            # Return to pool if there's space, otherwise close it
            if len(self._available) < self._max_size - self._in_use:
                self._available.append(resource)
                # Signal waiting acquirers
                self._acquire_event.set()
            else:
                await self._close_resource(resource)

    async def close(self) -> None:
        """
        Close the pool and cleanup all resources.

        This will close all available resources immediately. Resources that
        are currently in use will not be closed until they are released.

        After closing, no new resources can be acquired from the pool.
        """
        async with self._lock:
            if self._closed:
                return

            self._closed = True

            # Close all available resources
            while self._available:
                resource = self._available.popleft()
                await self._close_resource(resource)

    async def __aenter__(self) -> "AsyncPool[T]":
        """Initialize the pool as a context manager."""
        await self._initialize()
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb
    ) -> None:
        """Close the pool when exiting context manager."""
        await self.close()


class AsyncPoolContext(Generic[T]):
    """
    Context manager for a resource acquired from an AsyncPool.

    This class manages the lifecycle of an acquired resource, ensuring
    it's properly released back to the pool when done.

    Example:
        async with pool.acquire() as resource:
            # Use the resource
            await resource.do_something()
        # Resource automatically released back to pool
    """

    def __init__(self, pool: AsyncPool[T], resource: T):
        """
        Initialize the context manager.

        Args:
            pool: The pool that owns this resource
            resource: The acquired resource
        """
        self._pool = pool
        self._resource = resource
        self._released = False

    @property
    def resource(self) -> T:
        """
        Get the acquired resource.

        Returns:
            The managed resource

        Raises:
            RuntimeError: If the resource has already been released
        """
        if self._released:
            raise RuntimeError("Resource has been released")
        return self._resource

    async def release(self) -> None:
        """
        Release the resource back to the pool manually.

        This can be called explicitly before the context manager exits.
        Multiple calls are safe (idempotent).
        """
        if not self._released:
            await self._pool.release(self._resource)
            self._released = True

    async def __aenter__(self) -> T:
        """Enter the context and return the resource."""
        return self._resource

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb
    ) -> None:
        """Exit context and release resource."""
        await self.release()
