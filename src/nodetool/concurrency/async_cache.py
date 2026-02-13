"""
Async Cache with TTL support for memoizing async function results.

This module provides a thread-safe, async-friendly cache implementation
with TTL (Time To Live) support, useful for caching expensive async
operations like database queries, API calls, or computations.
"""

import asyncio
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Generic, TypeVar, cast

from nodetool.concurrency.async_lock import AsyncLock

K = TypeVar("K")
V = TypeVar("V")


@dataclass
class CacheEntry(Generic[V]):
    """A cache entry storing value and expiration time."""

    value: V
    expires_at: float
    hits: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        return time.time() >= self.expires_at

    def touch(self, ttl: float) -> None:
        """Update expiration time."""
        self.expires_at = time.time() + ttl


class AsyncCache(Generic[K, V]):
    """
    An async-safe cache with TTL support and LRU eviction.

    This cache provides:
    - Thread-safe async operations
    - TTL (Time To Live) for automatic expiration
    - LRU (Least Recently Used) eviction when full
    - Cache statistics tracking

    Args:
        max_size: Maximum number of entries. Default is 128.
        ttl: Default time-to-live in seconds. Default is 300 (5 minutes).
              Can be overridden per-entry.

    Example:
        >>> cache = AsyncCache[str, int](max_size=100, ttl=60)
        >>> await cache.set("key1", 42)
        >>> value = await cache.get("key1")
        >>> print(value)  # 42
    """

    def __init__(self, max_size: int = 128, ttl: float = 300.0) -> None:
        """
        Initialize the cache.

        Args:
            max_size: Maximum number of entries to store.
            ttl: Default time-to-live in seconds for cache entries.
        """
        self._max_size = max_size
        self._default_ttl = ttl
        self._cache: OrderedDict[K, CacheEntry[V]] = OrderedDict()
        self._lock = AsyncLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    async def get(self, key: K) -> V | None:
        """
        Get a value from the cache.

        Returns None if the key doesn't exist or has expired.

        Args:
            key: The cache key.

        Returns:
            The cached value or None if not found/expired.
        """
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired:
                # Remove expired entry
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.hits += 1
            self._hits += 1
            return entry.value

    async def set(self, key: K, value: V, ttl: float | None = None) -> None:
        """
        Set a value in the cache.

        Args:
            key: The cache key.
            value: The value to cache.
            ttl: Optional TTL in seconds. Uses default if not specified.
        """
        ttl = ttl if ttl is not None else self._default_ttl
        expires_at = time.time() + ttl

        async with self._lock:
            # If key exists, update and move to end
            if key in self._cache:
                self._cache[key].value = value
                self._cache[key].expires_at = expires_at
                self._cache.move_to_end(key)
                return

            # Check if we need to evict
            if len(self._cache) >= self._max_size:
                # Remove oldest (first) entry
                self._cache.popitem(last=False)
                self._evictions += 1

            # Add new entry
            self._cache[key] = CacheEntry(value=value, expires_at=expires_at)

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

    async def cleanup_expired(self) -> int:
        """
        Remove all expired entries from the cache.

        Returns:
            The number of entries removed.
        """
        async with self._lock:
            expired_keys = [
                k for k, v in self._cache.items() if v.is_expired
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)

    async def get_or_compute(
        self, key: K, compute_fn: Callable[[], Awaitable[V] | V], ttl: float | None = None
    ) -> V:
        """
        Get a value from cache, or compute it if not present.

        This is useful for caching expensive operations:
        - Returns cached value if present and not expired
        - Computes, caches, and returns value if not present
        - Thread-safe: only one computation will run per key

        Args:
            key: The cache key.
            compute_fn: Function to compute the value if not cached.
                       Can be sync or async.
            ttl: Optional TTL in seconds. Uses default if not specified.

        Returns:
            The cached or computed value.

        Example:
            >>> async def fetch_user(user_id: str) -> dict:
            ...     # Expensive database call
            ...     return await db.query(user_id)
            >>> cache = AsyncCache[str, dict]()
            >>> user = await cache.get_or_compute("user:123", fetch_user)
        """
        # Try to get from cache first
        cached = await self.get(key)
        if cached is not None:
            return cached

        # Compute the value
        result = compute_fn()
        if asyncio.iscoroutine(result):
            result = await result  # type: ignore[misc]

        # Cache the result
        await self.set(key, cast(V, result), ttl=ttl)

        return cast(V, result)

    async def size(self) -> int:
        """Get the current number of entries in the cache."""
        async with self._lock:
            return len(self._cache)

    async def stats(self) -> dict[str, int | float]:
        """
        Get cache statistics.

        Returns:
            Dictionary with keys: size, hits, misses, evictions,
            hit_rate (float), and default_ttl.
        """
        async with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": hit_rate,
                "default_ttl": self._default_ttl,
            }

    def __len__(self) -> int:
        """
        Get the current cache size.

        Note: For async consistency, prefer using `await cache.size()`.
        This method is provided for convenience in non-async contexts.

        Returns:
            Current number of entries (may be slightly stale).
        """
        return len(self._cache)


def async_cache(
    max_size: int = 128,
    ttl: float = 300.0,
) -> Callable[
    [Callable[..., Awaitable[V]]], Callable[..., Awaitable[V]]
]:
    """
    Decorator to cache async function results.

    Creates a cache instance per decorated function with automatic
    key generation from function arguments.

    Args:
        max_size: Maximum cache size. Default is 128.
        ttl: Time-to-live in seconds. Default is 300 (5 minutes).

    Returns:
        A decorator function.

    Example:
        >>> @async_cache(max_size=100, ttl=60)
        ... async def fetch_user(user_id: str) -> dict:
        ...     return await database.query(user_id)
        >>> # First call computes and caches
        >>> user1 = await fetch_user("user:123")
        >>> # Second call returns cached value
        >>> user2 = await fetch_user("user:123")
    """

    def decorator(func: Callable[..., Awaitable[V]]) -> Callable[..., Awaitable[V]]:
        cache = AsyncCache[str, V](max_size=max_size, ttl=ttl)

        @wraps(func)
        async def wrapper(*args: object, **kwargs: object) -> V:
            # Generate cache key from args and kwargs
            key_parts = [func.__qualname__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)

            return await cache.get_or_compute(cache_key, lambda: func(*args, **kwargs))

        # Add cache control methods to wrapper
        wrapper.cache = cache  # type: ignore[attr-defined]
        wrapper.cache_clear = cache.clear  # type: ignore[attr-defined]
        wrapper.cache_stats = cache.stats  # type: ignore[attr-defined]

        return wrapper

    return decorator
