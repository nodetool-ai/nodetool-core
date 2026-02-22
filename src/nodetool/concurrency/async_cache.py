"""Async caching utilities for expensive async operations.

This module provides caching decorators and classes for memoizing async function
results, with support for TTL (time-to-live), size limits, and custom cache keys.
"""

import asyncio
import time
import warnings
from collections.abc import Callable, Hashable
from functools import wraps
from typing import Any, Generic, ParamSpec, TypeVar

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class CachedValue(Generic[T]):
    """Container for cached values with expiration tracking."""

    __slots__ = ("_value", "expires_at")

    def __init__(self, value: T, ttl: float | None) -> None:
        """Initialize a cached value.

        Args:
            value: The cached result value.
            ttl: Time-to-live in seconds, or None for no expiration.
        """
        self._value = value
        self.expires_at: float | None = time.time() + ttl if ttl else None

    def is_expired(self) -> bool:
        """Check if this cached value has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def get(self) -> T:
        """Get the cached value, checking expiration first."""
        if self.is_expired():
            raise KeyError("Cache entry has expired")
        return self._value


class AsyncCache:
    """
    Async-safe cache with TTL and size limits.

    This cache provides thread-safe (for async tasks) caching with configurable
    time-to-live and maximum size. It uses asyncio locks to ensure safety
    in concurrent async environments.

    Example:
        cache = AsyncCache(max_size=100, ttl=300)

        # Get or compute
        result = await cache.get_or_compute("user:123", lambda: fetch_user("123"))

        # Invalidate specific entry
        await cache.invalidate("user:123")

        # Clear all
        await cache.clear()

        # Get stats
        stats = await cache.get_stats()
        print(f"Hit rate: {stats['hit_rate']:.1%}")
    """

    def __init__(
        self,
        max_size: int = 128,
        ttl: float | None = None,
        default_ttl: float | None = None,
    ) -> None:
        """Initialize the cache.

        Args:
            max_size: Maximum number of entries to store (default: 128).
            ttl: Default time-to-live in seconds for all entries (default: None = no expiration).
            default_ttl: Alias for ttl (deprecated).
        """
        if default_ttl is not None:
            warnings.warn(
                "default_ttl is deprecated, use ttl instead",
                DeprecationWarning,
                stacklevel=2,
            )
            ttl = default_ttl

        self._max_size: int = max_size
        self._ttl: float | None = ttl
        self._cache: dict[Hashable, CachedValue] = {}
        self._lock: asyncio.Lock = asyncio.Lock()
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

    async def get(self, key: Hashable) -> Any:
        """
        Get a value from the cache if it exists and hasn't expired.

        Args:
            key: The cache key to look up.

        Returns:
            The cached value, or None if not found or expired.
        """
        async with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]
            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None

            self._hits += 1
            return entry.get()

    async def set(
        self, key: Hashable, value: Any, ttl: float | None = None
    ) -> None:
        """
        Set a value in the cache.

        Args:
            key: The cache key.
            value: The value to cache.
            ttl: Time-to-live in seconds (overrides cache default).
        """
        async with self._lock:
            # Use provided TTL or fall back to cache default
            entry_ttl = ttl if ttl is not None else self._ttl

            # Evict oldest entry if at capacity
            if len(self._cache) >= self._max_size and key not in self._cache:
                # Simple FIFO eviction - could be enhanced with LRU
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._evictions += 1

            self._cache[key] = CachedValue(value, entry_ttl)

    async def get_or_compute(
        self, key: Hashable, compute_func: Any
    ) -> Any:
        """
        Get a value from cache, or compute and cache it if not present.

        Args:
            key: The cache key.
            compute_func: Async function to compute the value if cached value is missing/expired.

        Returns:
            The cached or newly computed value.
        """
        # Try to get from cache first
        value = await self.get(key)
        if value is not None:
            return value

        # Compute the value
        result = compute_func()
        if asyncio.iscoroutine(result):
            computed = await result
        else:
            computed = result

        # Cache and return
        await self.set(key, computed)
        return computed

    async def invalidate(self, key: Hashable) -> bool:
        """
        Invalidate a specific cache entry.

        Args:
            key: The cache key to invalidate.

        Returns:
            True if the key was found and removed, False otherwise.
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
            # Reset stats but not configuration
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    async def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats including size, hits, misses, evictions, and hit rate.
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
                "ttl": self._ttl,
            }

    async def cleanup_expired(self) -> int:
        """
        Remove all expired entries from the cache.

        Returns:
            Number of entries removed.
        """
        async with self._lock:
            expired_keys = [
                key
                for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)


def cached_async(
    ttl: float | None = None,
    max_size: int = 128,
    key_func: Any = None,
) -> Any:
    """
    Decorator to cache async function results.

    This decorator memoizes the results of async function calls, with optional
    TTL and custom cache key generation. Useful for expensive operations like
    API calls, database queries, or complex computations.

    Args:
        ttl: Time-to-live in seconds for cached results (default: None = no expiration).
        max_size: Maximum number of entries to cache per function (default: 128).
        key_func: Custom function to generate cache keys from arguments.
                  If None, uses function arguments as key (requires hashable args).

    Returns:
        Decorated function with caching enabled.

    Example:
        @cached_async(ttl=60)  # Cache for 60 seconds
        async def fetch_user(user_id: str) -> dict:
            return await api.get_user(user_id)

        # First call fetches from API
        user1 = await fetch_user("user123")

        # Subsequent calls within 60 seconds return cached value
        user2 = await fetch_user("user123")

        # Custom key function for complex arguments
        def make_key(url: str, params: dict) -> str:
            return f"{url}:{json.dumps(params, sort_keys=True)}"

        @cached_async(ttl=300, key_func=make_key)
        async def fetch_data(url: str, params: dict) -> dict:
            return await api.get(url, params=params)

    Note:
        Functions decorated with @cached_async must have hashable arguments
        unless a custom key_func is provided.
    """
    # Create a cache instance for this function
    cache = AsyncCache(max_size=max_size, ttl=ttl)

    def decorator(func: Any) -> Any:
        if key_func is not None:

            @wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                cache_key = key_func(*args, **kwargs)
                return await cache.get_or_compute(cache_key, lambda: func(*args, **kwargs))
        else:

            @wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Use args and kwargs as cache key (must be hashable)
                try:
                    # Try to create hashable key from args and sorted kwargs
                    cache_key = (args, tuple(sorted(kwargs.items())))
                except TypeError as e:
                    raise TypeError(
                        f"Function arguments must be hashable for caching, "
                        f"or provide a custom key_func: {e}"
                    ) from e

                return await cache.get_or_compute(cache_key, lambda: func(*args, **kwargs))

        # Add cache control methods to the wrapper
        wrapper.cache = cache  # type: ignore[attr-defined]
        wrapper.cache.invalidate = cache.invalidate  # type: ignore[attr-defined]
        wrapper.cache.clear = cache.clear  # type: ignore[attr-defined]
        wrapper.cache.get_stats = cache.get_stats  # type: ignore[attr-defined]
        wrapper.cache.cleanup_expired = cache.cleanup_expired  # type: ignore[attr-defined]

        return wrapper

    return decorator


__all__ = [
    "AsyncCache",
    "CachedValue",
    "cached_async",
]
