"""
Async caching utilities with TTL and size-based eviction.

Provides a flexible caching system for async operations with support for:
- Time-based expiration (TTL)
- Size-based eviction (LRU)
- Async factory functions
- Thread-safe operations
- Statistics tracking
"""

import asyncio
import time
from collections.abc import Callable, Hashable
from typing import Any, Generic, TypeVar, cast

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class CacheEntry(Generic[V]):
    """A single cache entry with value and expiration metadata."""

    __slots__ = ("access_count", "created_at", "expires_at", "last_accessed", "value")

    def __init__(self, value: V, ttl: float | None):
        """
        Initialize a cache entry.

        Args:
            value: The cached value.
            ttl: Time-to-live in seconds, or None for no expiration.
        """
        self.value = value
        self.created_at = time.monotonic()
        self.access_count = 0
        self.last_accessed = self.created_at
        self.expires_at = self.created_at + ttl if ttl else None

    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        if self.expires_at is None:
            return False
        return time.monotonic() > self.expires_at

    def touch(self) -> None:
        """Update access time and count for LRU tracking."""
        self.access_count += 1
        self.last_accessed = time.monotonic()

    def age(self) -> float:
        """Get the age of this entry in seconds."""
        return time.monotonic() - self.created_at


class AsyncCache(Generic[K, V]):
    """
    An async cache with TTL and LRU eviction policies.

    This cache provides thread-safe operations for async workloads with
    configurable expiration and eviction policies. It's designed for
    caching expensive async operations like API calls, database queries,
    or computation results.

    Example:
        cache = AsyncCache[str, int](max_size=100, default_ttl=60.0)

        # Get or compute with cache
        result = await cache.get_or_compute(
            "key1",
            lambda: expensive_computation()
        )

        # Manual operations
        await cache.put("key2", 42, ttl=30.0)
        value = await cache.get("key2")

        # Check statistics
        stats = await cache.get_stats()
        print(f"Hit rate: {stats['hit_rate']:.2%}")
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float | None = None,
        cleanup_interval: float = 60.0,
    ):
        """
        Initialize the async cache.

        Args:
            max_size: Maximum number of entries before LRU eviction (default: 1000).
            default_ttl: Default time-to-live in seconds for entries (default: None).
            cleanup_interval: Interval between automatic cleanup cycles (default: 60.0).
        """
        if max_size <= 0:
            raise ValueError("max_size must be a positive integer")
        if default_ttl is not None and default_ttl <= 0:
            raise ValueError("default_ttl must be positive or None")

        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cleanup_interval = cleanup_interval
        self._cache: dict[K, CacheEntry[V]] = {}
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._cleanup_task: asyncio.Task[None] | None = None

    async def get(self, key: K) -> V | None:
        """
        Get a value from the cache.

        Args:
            key: The cache key.

        Returns:
            The cached value, or None if not found or expired.
        """
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None

            entry.touch()
            self._hits += 1
            return entry.value

    async def get_or_compute(
        self,
        key: K,
        factory: Callable[[], Any],
        ttl: float | None = None,
    ) -> V:
        """
        Get a value from the cache, or compute it using the factory function.

        This is the primary method for using the cache. It provides a
        cache-aside pattern with async factory support.

        Args:
            key: The cache key.
            factory: An async callable that produces the value.
            ttl: Time-to-live in seconds, or None to use default.

        Returns:
            The cached or newly computed value.

        Example:
            result = await cache.get_or_compute(
                "user_123",
                lambda: fetch_user_from_db("123")
            )
        """
        # Try to get from cache first
        value = await self.get(key)
        if value is not None:
            return value

        # Compute the value
        if asyncio.iscoroutinefunction(factory):
            computed_value = await factory()
        elif asyncio.iscoroutine(factory):
            computed_value = await factory
        else:
            computed_value = factory()

        # Cast to V to satisfy type checker - we trust the factory returns the right type
        typed_value: V = cast("V", computed_value)

        # Store in cache
        await self.put(key, typed_value, ttl=ttl)

        return typed_value

    async def put(
        self,
        key: K,
        value: V,
        ttl: float | None = None,
    ) -> None:
        """
        Put a value into the cache.

        Args:
            key: The cache key.
            value: The value to cache.
            ttl: Time-to-live in seconds, or None to use default.
        """
        ttl = ttl if ttl is not None else self._default_ttl

        async with self._lock:
            # Check if we need to evict before adding
            if len(self._cache) >= self._max_size and key not in self._cache:
                self._evict_lru()

            self._cache[key] = CacheEntry(value, ttl)

    async def delete(self, key: K) -> bool:
        """
        Delete a value from the cache.

        Args:
            key: The cache key.

        Returns:
            True if the key was found and deleted, False otherwise.
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
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    async def size(self) -> int:
        """Get the current number of entries in the cache."""
        async with self._lock:
            return len(self._cache)

    async def has_key(self, key: K) -> bool:
        """
        Check if a key exists in the cache and is not expired.

        Args:
            key: The cache key.

        Returns:
            True if the key exists and is not expired.
        """
        async with self._lock:
            entry = self._cache.get(key)
            return entry is not None and not entry.is_expired()

    async def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            A dictionary with cache statistics including:
            - size: Current number of entries
            - max_size: Maximum cache size
            - hits: Number of cache hits
            - misses: Number of cache misses
            - evictions: Number of evicted entries
            - hit_rate: Cache hit rate as a percentage
        """
        async with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0

            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": hit_rate,
            }

    async def cleanup_expired(self) -> int:
        """
        Remove all expired entries from the cache.

        Returns:
            The number of entries removed.
        """
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items() if entry.is_expired()
            ]

            for key in expired_keys:
                del self._cache[key]

            return len(expired_keys)

    async def start_auto_cleanup(self) -> None:
        """Start automatic periodic cleanup of expired entries."""
        if self._cleanup_task is not None:
            return

        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(self._cleanup_interval)
                    removed = await self.cleanup_expired()
                    if removed > 0:
                        log.debug(
                            f"Cache cleanup removed {removed} expired entries",
                            extra={"removed": removed},
                        )
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    log.error(f"Error in cache cleanup: {e}")

        self._cleanup_task = asyncio.create_task(cleanup_loop())

    async def stop_auto_cleanup(self) -> None:
        """Stop automatic periodic cleanup."""
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    def _evict_lru(self) -> None:
        """
        Evict the least recently used entry.

        This method is not thread-safe and should only be called
        while holding the lock.
        """
        if not self._cache:
            return

        # Find the entry with the oldest last_accessed time
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed,
        )

        del self._cache[lru_key]
        self._evictions += 1

    async def keys(self) -> list[K]:
        """
        Get all non-expired keys in the cache.

        Returns:
            List of keys.
        """
        async with self._lock:
            return [key for key, entry in self._cache.items() if not entry.is_expired()]

    async def items(self) -> list[tuple[K, V]]:
        """
        Get all non-expired entries in the cache.

        Returns:
            List of (key, value) tuples.
        """
        async with self._lock:
            return [
                (key, entry.value)
                for key, entry in self._cache.items()
                if not entry.is_expired()
            ]

    def __len__(self) -> int:
        """
        Get the approximate size of the cache.

        Note: This is not thread-safe and may return an approximate value.
        For accurate results, use the size() method.
        """
        return len(self._cache)


__all__ = [
    "AsyncCache",
    "CacheEntry",
]
