"""
Async cache and memoization utilities for caching expensive async operations.

These utilities provide thread-safe (async-safe) caching with TTL (time-to-live)
support, LRU eviction, and decorator-based memoization for async functions.
"""

import asyncio
import functools
import time
from collections import OrderedDict
from collections.abc import Callable, Coroutine
from typing import Any, Generic, TypeVar

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)

K = TypeVar("K")
V = TypeVar("V")


class _CacheEntry(Generic[V]):
    """Internal cache entry with value and expiration."""

    __slots__: tuple[str, str, str, str] = ("expires_at", "hits", "last_access", "value")

    value: V
    expires_at: float | None
    hits: int
    last_access: float

    def __init__(self, value: V, ttl: float | None) -> None:
        """
        Initialize a cache entry.

        Args:
            value: The cached value.
            ttl: Time-to-live in seconds, or None for no expiration.
        """
        self.value = value
        self.expires_at = time.time() + ttl if ttl else None
        self.hits = 0
        self.last_access = time.time()

    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        return self.expires_at is not None and time.time() > self.expires_at

    def touch(self) -> None:
        """Update the last access time."""
        self.last_access = time.time()
        self.hits += 1


class AsyncCache(Generic[K, V]):
    """
    Thread-safe (async-safe) LRU cache with TTL support.

    This cache provides automatic expiration of entries based on TTL,
    LRU eviction when capacity is reached, and statistics tracking.

    Example:
        cache = AsyncCache(max_size=100, default_ttl=60.0)

        # Set a value with default TTL
        await cache.set("key1", "value1")

        # Set a value with custom TTL
        await cache.set("key2", "value2", ttl=300.0)

        # Get a value (returns None if not found or expired)
        value = await cache.get("key1")

        # Get or compute (cache miss calls the function)
        value = await cache.get_or_compute("key3", lambda: expensive_computation())

        # Clear all entries
        await cache.clear()

        # Get cache statistics
        stats = await cache.get_stats()
        print(f"Hits: {stats['hits']}, Misses: {stats['misses']}")
    """

    _max_size: int
    _default_ttl: float | None
    _cache: OrderedDict[Any, _CacheEntry[V]]
    _lock: asyncio.Lock
    _hits: int
    _misses: int
    _evictions: int

    def __init__(self, max_size: int = 128, default_ttl: float | None = None) -> None:
        """
        Initialize an async cache.

        Args:
            max_size: Maximum number of entries to store (default: 128).
                     When exceeded, least recently used entries are evicted.
            default_ttl: Default time-to-live in seconds for entries (default: None).
                        None means entries don't expire by default.
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")

        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cache = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

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
                # Remove expired entry
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._hits += 1
            return entry.value

    async def set(self, key: K, value: V, ttl: float | None = None) -> None:
        """
        Set a value in the cache.

        Args:
            key: The cache key.
            value: The value to cache.
            ttl: Time-to-live in seconds, or None to use default_ttl.
        """
        async with self._lock:
            # Use provided TTL or default
            entry_ttl = ttl if ttl is not None else self._default_ttl
            entry = _CacheEntry[V](value, entry_ttl)

            # Check if we're replacing an existing key
            is_update = key in self._cache

            self._cache[key] = entry
            self._cache.move_to_end(key)

            # Evict LRU if at capacity
            if len(self._cache) > self._max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._evictions += 1
                log.debug(
                    "Cache entry evicted",
                    extra={"key": str(oldest_key), "is_update": is_update},
                )

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

    async def get_or_compute(
        self, key: K, compute_func: Callable[[], Coroutine[Any, Any, V]], ttl: float | None = None
    ) -> V:
        """
        Get a value from cache, or compute and cache it if not present.

        This is the primary method for cache-aside pattern usage.

        Args:
            key: The cache key.
            compute_func: Async function to compute the value if not cached.
            ttl: Time-to-live in seconds for the cached value.

        Returns:
            The cached or newly computed value.

        Example:
            async def fetch_user(user_id):
                return await db.fetch_user(user_id)

            user = await cache.get_or_compute(f"user:{user_id}", lambda: fetch_user(user_id))
        """
        # Try to get from cache first
        value = await self.get(key)
        if value is not None:
            return value

        # Compute the value
        computed_value = await compute_func()

        # Cache it
        await self.set(key, computed_value, ttl)

        return computed_value

    async def clear(self) -> None:
        """Clear all entries from the cache."""
        async with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    async def cleanup_expired(self) -> int:
        """
        Remove all expired entries from the cache.

        Returns:
            The number of entries removed.
        """
        async with self._lock:
            expired_keys = [key for key, entry in self._cache.items() if entry.is_expired()]

            for key in expired_keys:
                del self._cache[key]

            if expired_keys:
                log.debug("Cleaned up expired cache entries", extra={"count": len(expired_keys)})

            return len(expired_keys)

    async def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            A dictionary with cache statistics including:
            - size: Current number of entries
            - max_size: Maximum capacity
            - hits: Number of cache hits
            - misses: Number of cache misses
            - evictions: Number of evictions
            - hit_rate: Hit rate as a percentage (or None if no requests)
        """
        async with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else None

            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": hit_rate,
            }

    async def contains(self, key: Any) -> bool:
        """
        Check if a key exists in the cache and is not expired.

        Args:
            key: The cache key.

        Returns:
            True if the key exists and is not expired, False otherwise.
        """
        async with self._lock:
            entry = self._cache.get(key)
            return entry is not None and not entry.is_expired()

    async def size(self) -> int:
        """
        Get the current number of entries in the cache.

        Returns:
            The current cache size.
        """
        async with self._lock:
            return len(self._cache)


class cached_async:
    """
    Decorator for memoizing async functions with TTL support.

    This decorator caches the results of async function calls based on
    their arguments, with optional TTL for cache expiration.

    Example:
        @cached_async(ttl=60.0)
        async def fetch_user(user_id: str) -> dict:
            return await database.query(user_id)

        # First call executes the function
        user1 = await fetch_user("user123")

        # Subsequent calls within TTL return cached result
        user2 = await fetch_user("user123")  # Returns cached result

        # Different arguments result in separate cache entries
        user3 = await fetch_user("user456")  # Executes function

        # Clear the cache
        await fetch_user.cache.clear()
    """

    def __init__(
        self,
        ttl: float | None = None,
        max_size: int = 128,
        key_func: Callable[..., tuple[Any, ...]] | None = None,
    ) -> None:
        """
        Initialize a cached_async decorator.

        Args:
            ttl: Time-to-live in seconds for cached results (default: None).
                 None means cached results don't expire.
            max_size: Maximum number of results to cache (default: 128).
            key_func: Optional function to generate cache keys from arguments.
                      If None, uses the function arguments as the key.
                      Signature: key_func(*args, **kwargs) -> tuple

        Example:
            # Custom key function
            def user_key(user_id: str, include_history: bool = False):
                return (user_id, include_history)

            @cached_async(ttl=300, key_func=user_key)
            async def get_user_data(user_id: str, include_history: bool = False):
                ...
        """
        self._ttl = ttl
        self._max_size = max_size
        self._key_func = key_func
        self._cache: AsyncCache[Any, Any] | None = None

    def __call__(
        self, func: Callable[..., Coroutine[Any, Any, V]]
    ) -> Callable[..., Coroutine[Any, Any, V]]:
        """
        Apply the caching decorator to a function.

        Args:
            func: The async function to decorate.

        Returns:
            The decorated function with caching.
        """
        # Create the cache for this function
        self._cache = AsyncCache[Any, Any](max_size=self._max_size, default_ttl=self._ttl)

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> V:
            # Generate cache key
            if self._key_func is not None:
                cache_key = self._key_func(*args, **kwargs)
            else:
                # Use args and kwargs as key (kwargs must be sorted for consistency)
                cache_key = (
                    args,
                    tuple(sorted(kwargs.items())),
                )

            # Ensure cache is initialized
            assert self._cache is not None

            # Get or compute the result
            result = await self._cache.get_or_compute(
                cache_key,
                lambda: func(*args, **kwargs),
                ttl=self._ttl,
            )
            return result  # type: ignore[return-value]

        # Attach cache control methods to the wrapped function
        wrapper.cache = self._cache  # type: ignore[attr-defined]
        wrapper.cache_clear = self._cache.clear  # type: ignore[attr-defined]
        wrapper.cache_stats = self._cache.get_stats  # type: ignore[attr-defined]

        return wrapper


__all__ = ["AsyncCache", "cached_async"]
