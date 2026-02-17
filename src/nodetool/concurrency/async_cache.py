"""Async LRU (Least Recently Used) cache utilities.

This module provides an asynchronous LRU cache implementation with TTL support,
designed for caching async function results in high-performance applications.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Callable, Generic, TypeVar, cast

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """A cache entry with value and expiration metadata.

    Attributes:
        value: The cached value
        expires_at: When the entry expires (None for no expiration)
        created_at: When the entry was created
        access_count: Number of times this entry was accessed
        last_accessed: Last time this entry was accessed
    """

    value: T
    expires_at: datetime | None = None
    created_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)

    def is_expired(self) -> bool:
        """Check if this cache entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() >= self.expires_at

    def touch(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = datetime.now()


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring.

    Attributes:
        hits: Number of cache hits
        misses: Number of cache misses
        evictions: Number of entries evicted
        size: Current cache size
        max_size: Maximum cache size
    """

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "size": self.size,
            "max_size": self.max_size,
            "hit_rate": self.hit_rate,
        }


class AsyncLRUCache(Generic[T]):
    """Thread-safe async LRU cache with TTL support.

    This cache provides:
    - LRU eviction when the cache is full
    - TTL (time-to-live) support for automatic expiration
    - Thread-safe operations using asyncio lock
    - Statistics tracking for monitoring cache performance
    - Decorator support for easy async function memoization

    Example:
        ```python
        # Create a cache
        cache = AsyncLRUCache[str](max_size=100, ttl=300)

        # Use the cache directly
        result = await cache.get_or_compute("key", lambda: expensive_computation())

        # Use as a decorator
        @cache.cache_result
        async def fetch_user(user_id: str) -> User:
            return await db.get_user(user_id)

        # Manual operations
        await cache.put("key", "value", ttl=60)
        value = await cache.get("key")
        await cache.clear()
        ```
    """

    def __init__(
        self,
        max_size: int = 128,
        ttl: int | None = None,
        default_ttl: int | None = None,
    ) -> None:
        """Initialize the async LRU cache.

        Args:
            max_size: Maximum number of entries in the cache
            ttl: Default time-to-live for entries in seconds (None for no expiration)
            default_ttl: Alias for ttl parameter
        """
        self._max_size = max_size
        self._default_ttl = default_ttl or ttl
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._stats = CacheStats(max_size=max_size)

    @property
    def max_size(self) -> int:
        """Get the maximum cache size."""
        return self._max_size

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics (snapshot)."""
        return CacheStats(
            hits=self._stats.hits,
            misses=self._stats.misses,
            evictions=self._stats.evictions,
            size=len(self._cache),
            max_size=self._max_size,
        )

    async def get(self, key: str) -> T | None:
        """Get a value from the cache.

        Args:
            key: The cache key

        Returns:
            The cached value or None if not found/expired
        """
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return None

            if entry.is_expired():
                # Remove expired entry
                del self._cache[key]
                self._stats.misses += 1
                self._stats.size = len(self._cache)
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._stats.hits += 1
            return entry.value

    async def put(
        self,
        key: str,
        value: T,
        ttl: int | None = None,
    ) -> None:
        """Put a value into the cache.

        Args:
            key: The cache key
            value: The value to cache
            ttl: Time-to-live in seconds (uses default_ttl if not specified)
        """
        async with self._lock:
            # Calculate expiration
            expires_at = None
            if ttl is not None:
                expires_at = datetime.now() + timedelta(seconds=ttl)
            elif self._default_ttl is not None:
                expires_at = datetime.now() + timedelta(seconds=self._default_ttl)

            # Update existing entry
            if key in self._cache:
                self._cache[key] = CacheEntry(
                    value=value,
                    expires_at=expires_at,
                )
                self._cache.move_to_end(key)
                return

            # Evict LRU entry if cache is full
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
                self._stats.evictions += 1

            # Add new entry
            self._cache[key] = CacheEntry(
                value=value,
                expires_at=expires_at,
            )
            self._stats.size = len(self._cache)

    async def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[..., T],
        ttl: int | None = None,
    ) -> T:
        """Get a value from cache or compute it if not present.

        Args:
            key: The cache key
            compute_fn: Async function to compute the value if not cached
            ttl: Time-to-live in seconds for the cached result

        Returns:
            The cached or computed value
        """
        # Try to get from cache
        value = await self.get(key)
        if value is not None:
            return value

        # Compute the value
        result = compute_fn()
        if asyncio.iscoroutine(result):
            result = await result

        # Cache the result
        await self.put(key, cast(T, result), ttl=ttl)

        return cast(T, result)

    async def delete(self, key: str) -> bool:
        """Delete a value from the cache.

        Args:
            key: The cache key

        Returns:
            True if the key was deleted, False if it didn't exist
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.size = len(self._cache)
                return True
            return False

    async def clear(self) -> None:
        """Clear all entries from the cache."""
        async with self._lock:
            self._cache.clear()
            self._stats.size = 0

    async def cleanup_expired(self) -> int:
        """Remove all expired entries from the cache.

        Returns:
            The number of entries removed
        """
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items() if entry.is_expired()
            ]

            for key in expired_keys:
                del self._cache[key]

            self._stats.size = len(self._cache)
            return len(expired_keys)

    async def keys(self) -> list[str]:
        """Get all cache keys.

        Returns:
            List of cache keys
        """
        async with self._lock:
            return list(self._cache.keys())

    async def items(self) -> AsyncGenerator[tuple[str, T], None]:
        """Iterate over cache items.

        Yields:
            Tuples of (key, value) for each non-expired cache entry
        """
        async with self._lock:
            # Make a copy to avoid holding lock during iteration
            items = list(self._cache.items())

        for key, entry in items:
            if not entry.is_expired():
                yield key, entry.value

    def cache_result(
        self,
        ttl: int | None = None,
        key_fn: Callable[..., str] | None = None,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator to cache async function results.

        Args:
            ttl: Time-to-live in seconds for cached results
            key_fn: Optional function to generate cache keys from arguments.
                    If not provided, uses function name and argument string representation.

        Example:
            ```python
            cache = AsyncLRUCache[str](max_size=100)

            @cache.cache_result(ttl=300)
            async def get_user(user_id: str) -> User:
                return await db.fetch_user(user_id)
            ```

        Returns:
            Decorator function
        """

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            async def wrapper(*args: Any, **kwargs: Any) -> T:
                # Generate cache key
                if key_fn is not None:
                    cache_key = key_fn(*args, **kwargs)
                else:
                    # Use function name and args for key
                    func_name = getattr(func, "__name__", str(func))
                    args_str = ",".join(str(a) for a in args)
                    kwargs_str = ",".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
                    cache_key = f"{func_name}:{args_str}:{kwargs_str}"

                return await self.get_or_compute(
                    cache_key,
                    lambda: func(*args, **kwargs),
                    ttl=ttl,
                )

            return cast(Callable[..., T], wrapper)

        return decorator
