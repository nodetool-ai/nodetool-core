"""
Async cache with TTL (time-to-live) support.

Provides an asynchronous in-memory cache with automatic expiration,
useful for caching API responses, database queries, and expensive computations.
"""

import asyncio
import time
from collections.abc import Callable, Hashable
from typing import Any, TypeVar, cast

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class CacheEntry:
    """Internal cache entry with value and expiration time."""

    __slots__ = ("expires_at", "value")

    def __init__(self, value: V, expires_at: float | None):
        self.value = value
        self.expires_at = expires_at

    def is_expired(self, now: float) -> bool:
        """Check if this entry has expired."""
        return self.expires_at is not None and now >= self.expires_at


class AsyncCache:
    """
    Async in-memory cache with TTL support.

    Provides thread-safe caching with automatic expiration of entries.
    Useful for caching expensive operations like API calls, database queries,
    or complex computations.

    Example:
        cache = AsyncCache(ttl=60.0, max_size=1000)

        # Simple get/set
        await cache.set("user:123", user_data)
        user = await cache.get("user:123")

        # Get or compute (cache miss)
        result = await cache.get_or_compute(
            "expensive_result",
            lambda: compute_expensive_result(),
        )

        # Check if key exists
        if await cache.has("user:123"):
            ...

        # Delete specific key
        await cache.delete("user:123")

        # Clear all cache entries
        await cache.clear()

        # Get cache statistics
        stats = await cache.get_stats()
        print(f"Hit rate: {stats['hit_rate']:.1%}")
    """

    def __init__(
        self,
        ttl: float | None = 300.0,
        max_size: int | None = 1000,
        cleanup_interval: float = 60.0,
    ):
        """
        Initialize an async cache.

        Args:
            ttl: Default time-to-live in seconds for cache entries.
                 None means entries don't expire by default.
            max_size: Maximum number of entries in the cache.
                      None means unlimited size (use with caution).
            cleanup_interval: Interval in seconds for background cleanup
                            of expired entries.

        Raises:
            ValueError: If ttl, max_size, or cleanup_interval are negative.
        """
        if ttl is not None and ttl < 0:
            raise ValueError("ttl must be non-negative")
        if max_size is not None and max_size <= 0:
            raise ValueError("max_size must be positive or None")
        if cleanup_interval <= 0:
            raise ValueError("cleanup_interval must be positive")

        self._ttl = ttl
        self._max_size = max_size
        self._cleanup_interval = cleanup_interval
        self._cache: dict[K, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[None] | None = None
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        # Start background cleanup task if TTL is enabled
        if ttl is not None:
            self._start_cleanup_task()

    def _start_cleanup_task(self) -> None:
        """Start the background cleanup task."""
        try:
            loop = asyncio.get_running_loop()
            self._cleanup_task = loop.create_task(self._cleanup_loop())
        except RuntimeError:
            # No event loop running, will be started later
            pass

    async def _cleanup_loop(self) -> None:
        """Background task that periodically removes expired entries."""
        try:
            while True:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_expired()
        except asyncio.CancelledError:
            # Task was cancelled, exit gracefully
            pass

    async def _cleanup_expired(self) -> int:
        """Remove expired entries from the cache. Returns count removed."""
        async with self._lock:
            now = time.monotonic()
            expired_keys = [
                key for key, entry in self._cache.items() if entry.is_expired(now)
            ]

            for key in expired_keys:
                del self._cache[key]

            if expired_keys:
                log.debug(
                    f"Cleaned up {len(expired_keys)} expired cache entries",
                    extra={"count": len(expired_keys)},
                )

            return len(expired_keys)

    async def get(self, key: K, default: V | None = None) -> V | None:
        """
        Get a value from the cache.

        Args:
            key: The cache key.
            default: Value to return if key is not found or expired.

        Returns:
            The cached value, or default if not found/expired.
        """
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return default

            if entry.is_expired(time.monotonic()):
                # Entry expired, remove it
                del self._cache[key]
                self._misses += 1
                return default

            self._hits += 1
            return entry.value

    async def set(
        self,
        key: K,
        value: V,
        ttl: float | None = None,
    ) -> None:
        """
        Set a value in the cache.

        Args:
            key: The cache key.
            value: The value to cache.
            ttl: Time-to-live in seconds for this entry.
                 None uses the default TTL from constructor.
        """
        async with self._lock:
            # Enforce max size by evicting oldest entries (simple FIFO)
            if self._max_size is not None and key not in self._cache:
                while len(self._cache) >= self._max_size:
                    # Evict the first key (simple FIFO eviction)
                    evicted_key = next(iter(self._cache))
                    del self._cache[evicted_key]
                    self._evictions += 1

            # Calculate expiration time
            entry_ttl = ttl if ttl is not None else self._ttl
            expires_at = (
                time.monotonic() + entry_ttl if entry_ttl is not None else None
            )

            self._cache[key] = CacheEntry(value, expires_at)

    async def get_or_compute(
        self,
        key: K,
        compute_fn: Callable[[], Any] | Callable[[], Any],
        ttl: float | None = None,
    ) -> V:
        """
        Get a value from cache, or compute and cache if not present.

        This is the most convenient method for caching expensive operations.

        Args:
            key: The cache key.
            compute_fn: Async function to compute the value if cache miss.
            ttl: Time-to-live in seconds for cached result.
                 None uses the default TTL from constructor.

        Returns:
            The cached or newly computed value.

        Example:
            async def fetch_user(user_id: str) -> dict:
                return await api.get_user(user_id)

            # First call computes and caches
            user1 = await cache.get_or_compute("user:123", lambda: fetch_user("123"))
            # Second call returns cached value
            user2 = await cache.get_or_compute("user:123", lambda: fetch_user("123"))
        """
        # Try to get from cache first
        value = await self.get(key, default=None)
        if value is not None:
            return cast("V", value)

        # Cache miss - compute the value
        result = compute_fn()
        if asyncio.iscoroutine(result):
            result = await result  # type: ignore[misc]

        # Cache the computed value
        await self.set(key, result, ttl=ttl)

        return cast("V", result)

    async def has(self, key: K) -> bool:
        """
        Check if a key exists in the cache and is not expired.

        Args:
            key: The cache key.

        Returns:
            True if key exists and is not expired, False otherwise.
        """
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                return False

            if entry.is_expired(time.monotonic()):
                del self._cache[key]
                return False

            return True

    async def delete(self, key: K) -> bool:
        """
        Delete a key from the cache.

        Args:
            key: The cache key.

        Returns:
            True if the key was deleted, False if it didn't exist.
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> None:
        """Clear all entries from the cache."""
        async with self._lock:
            self._cache.clear()
            # Reset statistics
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    async def size(self) -> int:
        """
        Get the current number of entries in the cache.

        Returns:
            The number of cached entries (including expired ones).
        """
        async with self._lock:
            return len(self._cache)

    async def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            A dictionary containing:
            - size: Current number of entries
            - hits: Number of cache hits
            - misses: Number of cache misses
            - evictions: Number of evictions due to max_size
            - hit_rate: Cache hit rate (0.0 to 1.0)
        """
        async with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            return {
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": hit_rate,
            }

    async def reset_stats(self) -> None:
        """Reset cache statistics without clearing the cache."""
        async with self._lock:
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    async def cleanup(self) -> int:
        """
        Manually trigger cleanup of expired entries.

        Returns:
            The number of entries removed.

        Note:
            This is normally done automatically by the background cleanup task,
            but can be called manually if needed.
        """
        return await self._cleanup_expired()

    async def close(self) -> None:
        """
        Close the cache and stop the background cleanup task.

        Call this when the cache is no longer needed to properly clean up resources.
        """
        if self._cleanup_task is not None and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    async def __aenter__(self) -> "AsyncCache":
        """Context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Context manager exit - ensures cleanup task is stopped."""
        await self.close()


__all__ = ["AsyncCache"]
