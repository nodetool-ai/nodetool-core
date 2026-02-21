"""
Async Resource Pool for managing reusable async resources.

This module provides AsyncResourcePool, a generic pool for managing reusable
resources like HTTP clients, database connections, or any expensive-to-create
objects that need to be shared across async tasks.

Example:
    from nodetool.concurrency import AsyncResourcePool

    # Create a pool of HTTP clients
    async def create_client():
        return aiohttp.ClientSession()

    pool = AsyncResourcePool(
        factory=create_client,
        max_size=10,
        max_idle_time=300.0,
    )

    async with pool.acquire() as client:
        response = await client.get("https://example.com")
        data = await response.text()
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Awaitable, Callable, Generic, TypeVar

T = TypeVar("T")


@dataclass
class _PooledResource(Generic[T]):
    """A wrapped resource with metadata for pool management.

    Attributes:
        resource: The actual pooled resource
        created_at: When the resource was created
        last_used_at: When the resource was last acquired
        acquire_count: How many times the resource has been acquired
    """

    resource: T
    created_at: datetime = field(default_factory=datetime.now)
    last_used_at: datetime = field(default_factory=datetime.now)
    acquire_count: int = 0

    def __hash__(self) -> int:
        # Use id of the resource for hashing, since the wrapper itself
        # may be recreated but the resource object is the same
        return id(self.resource)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _PooledResource):
            return False
        return self.resource is other.resource

    def is_expired(self, max_age: timedelta | None) -> bool:
        """Check if the resource has exceeded its maximum age.

        Args:
            max_age: Maximum age before resource is considered expired.
                    None means resources never expire due to age.

        Returns:
            True if the resource is expired, False otherwise.
        """
        if max_age is None:
            return False
        return datetime.now() - self.created_at > max_age

    def is_idle_too_long(self, max_idle: timedelta | None) -> bool:
        """Check if the resource has been idle too long.

        Args:
            max_idle: Maximum idle time before resource is considered stale.
                     None means resources never expire due to idle time.

        Returns:
            True if the resource has been idle too long, False otherwise.
        """
        if max_idle is None:
            return False
        return datetime.now() - self.last_used_at > max_idle

    def mark_used(self) -> None:
        """Update the last used timestamp and increment acquire count."""
        self.last_used_at = datetime.now()
        self.acquire_count += 1


class ResourcePoolError(Exception):
    """Base exception for resource pool errors."""


class ResourcePoolClosedError(ResourcePoolError):
    """Raised when attempting to acquire from a closed pool."""


class AsyncResourcePool(Generic[T]):
    """An async resource pool for managing reusable resources.

    This pool manages a collection of reusable resources that are expensive
    to create, such as HTTP clients, database connections, or other objects
    that should be shared across multiple async tasks.

    Features:
        - Lazy creation: Resources are created on-demand
        - Maximum size: Limits the number of resources in the pool
        - Resource expiration: Removes resources based on age and idle time
        - Graceful shutdown: Waits for resources to be returned on close
        - Statistics: Tracks pool usage and performance metrics

    Example:
        ```python
        from nodetool.concurrency import AsyncResourcePool

        # Create a pool of HTTP clients
        async def create_client():
            return aiohttp.ClientSession()

        async def close_client(client):
            await client.close()

        pool = AsyncResourcePool(
            factory=create_client,
            closer=close_client,
            max_size=10,
            max_idle_time=300.0,
        )

        # Acquire and use a resource
        async with pool.acquire() as client:
            response = await client.get("https://example.com")
            data = await response.text()

        # Or use explicit acquire/release
        resource = await pool.acquire()
        try:
            await use_resource(resource)
        finally:
            await pool.release(resource)
        ```
    """

    def __init__(
        self,
        factory: Callable[[], Awaitable[T] | T],
        closer: Callable[[T], Awaitable[None]] | Callable[[T], None] | None = None,
        *,
        max_size: int = 10,
        min_size: int = 0,
        max_age: timedelta | float | None = None,
        max_idle_time: timedelta | float | None = None,
        acquisition_timeout: float = 30.0,
    ):
        """Initialize the resource pool.

        Args:
            factory: Async or sync function that creates new resources.
            closer: Optional async or sync function that closes resources.
                   If None, resources are not explicitly closed.
            max_size: Maximum number of resources in the pool. Must be >= 1.
            min_size: Minimum number of resources to keep ready. Must be >= 0.
            max_age: Maximum age for a resource before it's expired.
                    Can be timedelta or seconds (float). None = no age limit.
            max_idle_time: Maximum idle time before a resource is pruned.
                          Can be timedelta or seconds (float). None = no idle limit.
            acquisition_timeout: Seconds to wait for resource acquisition.

        Raises:
            ValueError: If max_size < 1, min_size < 0, or min_size > max_size.
        """
        if max_size < 1:
            raise ValueError("max_size must be at least 1")
        if min_size < 0:
            raise ValueError("min_size must be non-negative")
        if min_size > max_size:
            raise ValueError("min_size cannot exceed max_size")

        self._factory = factory
        self._closer = closer
        self._max_size = max_size
        self._min_size = min_size
        self._max_age = self._to_timedelta(max_age)
        self._max_idle_time = self._to_timedelta(max_idle_time)
        self._acquisition_timeout = acquisition_timeout

        self._available: deque[_PooledResource[T]] = deque()
        self._in_use: set[_PooledResource[T]] = set()
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)
        self._closed = False
        self._total_created = 0
        self._total_acquired = 0
        self._total_released = 0
        self._total_expired = 0
        self._total_pruned = 0

    @staticmethod
    def _to_timedelta(value: timedelta | float | None) -> timedelta | None:
        """Convert float seconds to timedelta if needed.

        Args:
            value: timedelta, float seconds, or None.

        Returns:
            timedelta or None.
        """
        if isinstance(value, (int, float)):
            return timedelta(seconds=value)
        return value

    @property
    def size(self) -> int:
        """Current number of resources in the pool (both available and in-use)."""
        return len(self._available) + len(self._in_use)

    @property
    def available_count(self) -> int:
        """Number of resources currently available for acquisition."""
        return len(self._available)

    @property
    def in_use_count(self) -> int:
        """Number of resources currently in use."""
        return len(self._in_use)

    @property
    def max_size(self) -> int:
        """Maximum pool size."""
        return self._max_size

    @property
    def min_size(self) -> int:
        """Minimum pool size (maintained when possible)."""
        return self._min_size

    @property
    def closed(self) -> bool:
        """Whether the pool is closed."""
        return self._closed

    @property
    def stats(self) -> dict[str, int | float]:
        """Get pool statistics.

        Returns:
            Dictionary with pool metrics including creation, acquisition,
            release, expiration, and pruning counts.
        """
        return {
            "size": self.size,
            "available": self.available_count,
            "in_use": self.in_use_count,
            "max_size": self._max_size,
            "min_size": self._min_size,
            "total_created": self._total_created,
            "total_acquired": self._total_acquired,
            "total_released": self._total_released,
            "total_expired": self._total_expired,
            "total_pruned": self._total_pruned,
            "closed": self._closed,
        }

    async def acquire(self, timeout: float | None = None) -> T:
        """Acquire a resource from the pool.

        Args:
            timeout: Maximum time to wait for a resource. None = use pool default.

        Returns:
            A resource from the pool.

        Raises:
            ResourcePoolClosedError: If the pool is closed.
            asyncio.TimeoutError: If acquisition times out.

        Note:
            The resource must be returned to the pool using `release()` or
            by using the pool as a context manager.
        """
        timeout = timeout if timeout is not None else self._acquisition_timeout

        async with self._not_empty:
            # Wait for an available slot or resource
            end_time = asyncio.get_event_loop().time() + timeout

            while self._available and self._available[0].is_expired(self._max_age):
                expired = self._available.popleft()
                await self._close_resource(expired)
                self._total_expired += 1

            while not self._available and self.size >= self._max_size:
                remaining = end_time - asyncio.get_event_loop().time()
                if remaining <= 0:
                    raise TimeoutError(f"Acquisition timed out after {timeout}s")
                try:
                    await asyncio.wait_for(
                        self._not_empty.wait(),
                        timeout=remaining,
                    )
                except TimeoutError as err:
                    raise TimeoutError(f"Acquisition timed out after {timeout}s") from err

            # Try to get an available resource that isn't expired
            resource: _PooledResource[T] | None = None
            while self._available:
                candidate = self._available.popleft()
                if not candidate.is_expired(self._max_age):
                    resource = candidate
                    break
                await self._close_resource(candidate)
                self._total_expired += 1

            # Create new resource if needed
            if resource is None:
                resource = await self._create_resource()

            resource.mark_used()
            self._in_use.add(resource)
            self._total_acquired += 1
            self._not_empty.notify_all()

            return resource.resource

    async def release(self, resource: T) -> None:
        """Return a resource to the pool.

        Args:
            resource: The resource to return. Must have been acquired from this pool.

        Raises:
            ValueError: If the resource was not acquired from this pool.
        """
        async with self._not_empty:
            # Find the pooled wrapper for this resource
            pooled = None
            for p in self._in_use:
                if p.resource is resource:
                    pooled = p
                    break

            if pooled is None:
                raise ValueError(
                    "Resource was not acquired from this pool or already released"
                )

            self._in_use.remove(pooled)

            if self._closed:
                await self._close_resource(pooled)
            elif pooled.is_idle_too_long(self._max_idle_time):
                await self._close_resource(pooled)
                self._total_pruned += 1
            else:
                self._available.append(pooled)

            self._total_released += 1
            self._not_empty.notify_all()

    async def _create_resource(self) -> _PooledResource[T]:
        """Create a new resource using the factory.

        Returns:
            A new pooled resource wrapper.
        """
        result: T
        if asyncio.iscoroutinefunction(self._factory):
            result = await self._factory()  # type: ignore[arg-type]
        else:
            result = self._factory()  # type: ignore[arg-type]
        pooled = _PooledResource(resource=result)
        self._total_created += 1
        return pooled

    async def _close_resource(self, pooled: _PooledResource[T]) -> None:
        """Close a resource using the closer function.

        Args:
            pooled: The pooled resource wrapper to close.
        """
        if self._closer is not None:
            try:
                if asyncio.iscoroutinefunction(self._closer):
                    await self._closer(pooled.resource)  # type: ignore[arg-type]
                else:
                    self._closer(pooled.resource)  # type: ignore[arg-type]
            except Exception:
                # Log but don't raise - closing errors shouldn't break the pool
                pass

    async def prune(self) -> int:
        """Remove idle and expired resources from the pool.

        This method is called automatically during resource operations,
        but can also be called manually to force cleanup.

        Returns:
            The number of resources pruned.
        """
        async with self._not_empty:
            pruned = 0

            # Keep enough resources to satisfy min_size
            while self._available and (self.size - pruned) > self._min_size:
                candidate = self._available[-1]
                if (
                    candidate.is_expired(self._max_age)
                    or candidate.is_idle_too_long(self._max_idle_time)
                ):
                    candidate = self._available.pop()
                    await self._close_resource(candidate)
                    pruned += 1
                else:
                    break

            self._total_pruned += pruned
            return pruned

    async def close(self) -> None:
        """Close the pool and all resources.

        This method:
        1. Marks the pool as closed (no new acquisitions)
        2. Waits for in-use resources to be returned (with timeout)
        3. Closes all available resources

        Raises:
            ResourcePoolClosedError: If the pool is already closed.
        """
        if self._closed:
            raise ResourcePoolClosedError("Pool is already closed")

        self._closed = True

        # Wait for in-use resources to be returned (up to 30 seconds)
        deadline = 60
        start = asyncio.get_event_loop().time()

        while self._in_use:
            await asyncio.sleep(0.1)
            if asyncio.get_event_loop().time() - start > deadline:
                break

        # Close all available resources
        async with self._not_empty:
            while self._available:
                pooled = self._available.popleft()
                await self._close_resource(pooled)
            self._not_empty.notify_all()

    async def __aenter__(self) -> AsyncResourcePool[T]:
        """Enter the pool as a context manager.

        Returns:
            The pool instance itself.
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the pool context manager, closing all resources."""
        await self.close()

    def acquire_context(self, timeout: float | None = None) -> _ResourceContext[T]:
        """Get a context manager for acquiring and releasing a resource.

        Args:
            timeout: Maximum time to wait for a resource. None = use pool default.

        Returns:
            A context manager that acquires on enter and releases on exit.

        Example:
            ```python
            pool = AsyncResourcePool(factory=create_resource)

            async with pool.acquire_context() as resource:
                await use_resource(resource)
            # resource is automatically released
            ```
        """
        return _ResourceContext(self, timeout)


class _ResourceContext(Generic[T]):
    """Context manager for acquiring and releasing pool resources.

    This is returned by AsyncResourcePool.acquire_context() and provides
    automatic resource release when exiting the context.
    """

    def __init__(self, pool: AsyncResourcePool[T], timeout: float | None):
        self._pool = pool
        self._timeout = timeout
        self._resource: T | None = None

    async def __aenter__(self) -> T:
        """Acquire a resource from the pool.

        Returns:
            The acquired resource.
        """
        self._resource = await self._pool.acquire(self._timeout)
        return self._resource

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Release the resource back to the pool."""
        if self._resource is not None:
            await self._pool.release(self._resource)
