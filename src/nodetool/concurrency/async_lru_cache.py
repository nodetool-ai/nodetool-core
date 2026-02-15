"""
Async LRU Cache implementation for caching async function results.

Provides a thread-safe LRU cache designed specifically for async functions,
with support for TTL (time-to-live), max size limits, and cache statistics.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import wraps
from typing import Any, Awaitable, Callable, Generic, TypeVar, cast

K = TypeVar("K")
V = TypeVar("V")


@dataclass
class CacheEntry(Generic[V]):
    """A single cache entry with value and metadata."""

    value: V
    created_at: float
    expires_at: float | None = None
    hits: int = 0


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate the cache hit rate as a percentage."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0


class AsyncLRUCache(Generic[K, V]):
    """
    A thread-safe LRU cache for async functions.

    Features:
    - LRU (Least Recently Used) eviction policy
    - Optional TTL (time-to-live) for entries
    - Thread-safe operations using async locks
    - Cache statistics for monitoring
    - Support for clear, get, set, and delete operations

    Example:
        ```python
        cache = AsyncLRUCache[str, int](max_size=100, ttl_seconds=60.0)

        # Get or compute a value
        async def get_data(key: str) -> int:
            return await cache.get_or_compute(key, expensive_async_func)

        # Manual operations
        await cache.set("key", 42)
        value = await cache.get("key")
        await cache.delete("key")
        ```
    """

    def __init__(
        self,
        max_size: int = 128,
        ttl_seconds: float | None = None,
    ) -> None:
        """
        Initialize the async LRU cache.

        Args:
            max_size: Maximum number of entries to store. When exceeded,
                     the least recently used entry is evicted.
            ttl_seconds: Optional time-to-live in seconds. Entries older
                        than this are considered stale and will be refreshed.
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive if specified")

        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._cache: OrderedDict[K, CacheEntry[V]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._stats = CacheStats(max_size=max_size)
        # Track in-flight computations to prevent duplicate work
        self._pending: dict[K, asyncio.Future[V]] = {}

    @property
    def stats(self) -> CacheStats:
        """Get current cache statistics."""
        return self._stats

    @property
    def max_size(self) -> int:
        """Get the maximum cache size."""
        return self._max_size

    @property
    def ttl_seconds(self) -> float | None:
        """Get the TTL in seconds, or None if not set."""
        return self._ttl_seconds

    def _is_expired(self, entry: CacheEntry[V]) -> bool:
        """Check if a cache entry has expired."""
        if entry.expires_at is None:
            return False
        return time.monotonic() > entry.expires_at

    async def get(self, key: K) -> V | None:
        """
        Get a value from the cache if it exists and is not expired.

        Args:
            key: The cache key to look up.

        Returns:
            The cached value if found and valid, None otherwise.
        """
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._stats.misses += 1
                return None

            if self._is_expired(entry):
                # Remove expired entry
                del self._cache[key]
                self._stats.misses += 1
                self._stats.evictions += 1
                self._stats.size = len(self._cache)
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.hits += 1
            self._stats.hits += 1
            return entry.value

    async def set(self, key: K, value: V) -> None:
        """
        Set a value in the cache.

        If the cache is full, the least recently used entry is evicted.

        Args:
            key: The cache key.
            value: The value to cache.
        """
        now = time.monotonic()
        expires_at = now + self._ttl_seconds if self._ttl_seconds else None

        async with self._lock:
            if key in self._cache:
                # Update existing entry and move to end
                self._cache[key] = CacheEntry(
                    value=value,
                    created_at=now,
                    expires_at=expires_at,
                )
                self._cache.move_to_end(key)
            else:
                # Add new entry
                if len(self._cache) >= self._max_size:
                    # Evict least recently used
                    self._cache.popitem(last=False)
                    self._stats.evictions += 1

                self._cache[key] = CacheEntry(
                    value=value,
                    created_at=now,
                    expires_at=expires_at,
                )

            self._stats.size = len(self._cache)

    async def delete(self, key: K) -> bool:
        """
        Delete a value from the cache.

        Args:
            key: The cache key to delete.

        Returns:
            True if the key was found and deleted, False otherwise.
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
            evicted = len(self._cache)
            self._cache.clear()
            self._stats.evictions += evicted
            self._stats.size = 0

    async def contains(self, key: K) -> bool:
        """
        Check if a key exists in the cache and is not expired.

        Args:
            key: The cache key to check.

        Returns:
            True if the key exists and is valid, False otherwise.
        """
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if self._is_expired(entry):
                del self._cache[key]
                self._stats.evictions += 1
                self._stats.size = len(self._cache)
                return False
            return True

    async def get_or_compute(
        self,
        key: K,
        compute: Callable[[], Awaitable[V]] | Callable[[], V],
    ) -> V:
        """
        Get a value from the cache, or compute it if not present.

        This method prevents the "thundering herd" problem by ensuring
        that only one computation runs for a given key at a time.

        Args:
            key: The cache key.
            compute: An async function or coroutine to compute the value
                    if not in cache.

        Returns:
            The cached or computed value.
        """
        # Check cache first (without lock to allow concurrent reads)
        cached = await self.get(key)
        if cached is not None:
            return cached

        # Variables to track computation state
        should_compute = False
        future: asyncio.Future[V] | None = None

        # Check if there's already a pending computation
        async with self._lock:
            # Double-check cache after acquiring lock
            entry = self._cache.get(key)
            if entry is not None and not self._is_expired(entry):
                entry.hits += 1
                self._stats.hits += 1
                self._cache.move_to_end(key)
                return entry.value

            if key in self._pending:
                # Wait for existing computation
                future = self._pending[key]
            else:
                # Start new computation
                should_compute = True
                loop = asyncio.get_event_loop()
                future = loop.create_future()
                self._pending[key] = future

        if should_compute and future is not None:
            # We're responsible for computing
            try:
                raw_result = compute()
                result = cast("V", await raw_result) if asyncio.iscoroutine(raw_result) else cast("V", raw_result)
                await self.set(key, result)
                async with self._lock:
                    future.set_result(result)
                    self._pending.pop(key, None)
                return result
            except Exception as e:
                async with self._lock:
                    future.set_exception(e)
                    self._pending.pop(key, None)
                raise

        # Wait for the pending computation
        if future is not None:
            return await future

        # This should never happen, but satisfies type checker
        raise RuntimeError("Unexpected state in get_or_compute")

    async def get_many(self, keys: list[K]) -> dict[K, V]:
        """
        Get multiple values from the cache.

        Args:
            keys: List of cache keys to look up.

        Returns:
            Dictionary mapping found keys to their values.
        """
        result: dict[K, V] = {}
        async with self._lock:
            for key in keys:
                entry = self._cache.get(key)
                if entry is not None and not self._is_expired(entry):
                    self._cache.move_to_end(key)
                    entry.hits += 1
                    self._stats.hits += 1
                    result[key] = entry.value
                else:
                    self._stats.misses += 1
        return result

    async def set_many(self, items: dict[K, V]) -> None:
        """
        Set multiple values in the cache.

        Args:
            items: Dictionary of key-value pairs to cache.
        """
        now = time.monotonic()
        expires_at = now + self._ttl_seconds if self._ttl_seconds else None

        async with self._lock:
            for key, value in items.items():
                if key in self._cache:
                    self._cache[key] = CacheEntry(
                        value=value,
                        created_at=now,
                        expires_at=expires_at,
                    )
                    self._cache.move_to_end(key)
                else:
                    if len(self._cache) >= self._max_size:
                        self._cache.popitem(last=False)
                        self._stats.evictions += 1
                    self._cache[key] = CacheEntry(
                        value=value,
                        created_at=now,
                        expires_at=expires_at,
                    )
            self._stats.size = len(self._cache)

    async def delete_many(self, keys: list[K]) -> int:
        """
        Delete multiple values from the cache.

        Args:
            keys: List of cache keys to delete.

        Returns:
            Number of keys that were found and deleted.
        """
        deleted = 0
        async with self._lock:
            for key in keys:
                if key in self._cache:
                    del self._cache[key]
                    deleted += 1
            self._stats.size = len(self._cache)
        return deleted


def async_cached(
    max_size: int = 128,
    ttl_seconds: float | None = None,
    key_func: Callable[..., Any] | None = None,
) -> Callable[[Callable[..., Awaitable[V]]], Callable[..., Awaitable[V]]]:
    """
    Decorator to cache async function results.

    Args:
        max_size: Maximum number of entries to cache.
        ttl_seconds: Optional time-to-live for cached entries.
        key_func: Optional function to generate cache keys from arguments.
                 If not provided, arguments are converted to a string key.

    Returns:
        A decorator that caches the function's results.

    Example:
        ```python
        @async_cached(max_size=100, ttl_seconds=60.0)
        async def fetch_user(user_id: str) -> User:
            return await api.get_user(user_id)

        # With custom key function
        def make_key(user_id: str, **kwargs) -> str:
            return f"user:{user_id}"

        @async_cached(key_func=make_key)
        async def fetch_user_with_key(user_id: str) -> User:
            return await api.get_user(user_id)
        ```
    """

    def decorator(func: Callable[..., Awaitable[V]]) -> Callable[..., Awaitable[V]]:
        cache: AsyncLRUCache[Any, V] = AsyncLRUCache(
            max_size=max_size,
            ttl_seconds=ttl_seconds,
        )

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> V:
            if key_func is not None:
                key = key_func(*args, **kwargs)
            else:
                # Create a hashable key from args and kwargs
                key_parts = [repr(args)]
                for k in sorted(kwargs.keys()):
                    key_parts.append(f"{k}={repr(kwargs[k])}")
                key_str = ":".join(key_parts)
                key = hashlib.md5(key_str.encode()).hexdigest()

            return await cache.get_or_compute(key, lambda: func(*args, **kwargs))

        # Expose cache for inspection/clearing
        wrapper.cache = cache  # type: ignore[attr-defined]
        wrapper.cache_clear = cache.clear  # type: ignore[attr-defined]
        wrapper.cache_stats = cache.stats  # type: ignore[attr-defined]

        return wrapper  # type: ignore[return-value]

    return decorator
