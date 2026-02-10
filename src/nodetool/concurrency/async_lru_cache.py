"""
Async LRU Cache for caching expensive async operations.

This module provides an LRU (Least Recently Used) cache decorator specifically
designed for async functions. It supports TTL (time-to-live) expiration,
maxsize limits, and custom key functions.
"""

import asyncio
import hashlib
import time
import functools
from collections.abc import Callable, Hashable
from typing import Any, TypeVar, ParamSpec

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


class _CacheEntry:
    """Internal cache entry storing value and metadata."""

    __slots__ = ("value", "expires_at", "accessed_at", "hits")

    def __init__(self, value: R, ttl: float | None):
        self.value: R = value
        self.expires_at: float | None = time.time() + ttl if ttl else None
        self.accessed_at: float = time.time()
        self.hits: int = 0

    def is_expired(self) -> bool:
        """Check if the entry has expired."""
        return self.expires_at is not None and time.time() > self.expires_at

    def touch(self) -> None:
        """Update the access time for LRU tracking."""
        self.accessed_at = time.time()
        self.hits += 1


class AsyncLRUCache:
    """
    LRU cache decorator for async functions with TTL support.

    This decorator caches the results of async functions based on their arguments,
    automatically evicting the least recently used entries when the cache is full.
    It supports time-to-live (TTL) expiration for cache entries.

    Features:
    - LRU eviction when maxsize is reached
    - TTL (time-to-live) for cache entries
    - Custom key functions for complex argument types
    - Thread-safe for async operations
    - Cache statistics (hits, misses, eviction count)

    Example:
        @AsyncLRUCache(maxsize=128, ttl=300)
        async def fetch_user(user_id: str):
            return await db.fetch_user(user_id)

        # First call hits the database
        user = await fetch_user("user123")

        # Second call returns cached result
        user = await fetch_data("user123")

        # Custom key function for complex arguments
        @AsyncLRUCache(
            maxsize=64,
            ttl=600,
            key_func=lambda req: req.url + str(req.params)
        )
        async def fetch_api(request: Request):
            return await http_client.fetch(request)
    """

    def __init__(
        self,
        maxsize: int = 128,
        ttl: float | None = None,
        key_func: Callable[..., Hashable] | None = None,
    ):
        """
        Initialize the AsyncLRUCache.

        Args:
            maxsize: Maximum number of entries to store. Must be > 0.
            ttl: Time-to-live in seconds. Entries older than this are evicted.
                 If None, entries never expire based on time.
            key_func: Optional function to compute cache keys from arguments.
                      If None, uses args + kwargs (must be hashable).

        Raises:
            ValueError: If maxsize is not a positive integer.
        """
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")

        self._maxsize: int = maxsize
        self._ttl: float | None = ttl
        self._key_func: Callable[..., Hashable] | None = key_func
        self._cache: dict[Hashable, _CacheEntry] = {}
        self._lock: asyncio.Lock = asyncio.Lock()
        # Track pending computations to avoid duplicate work
        self._pending: dict[Hashable, list[asyncio.Future]] = {}
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

    @property
    def hits(self) -> int:
        """Return the number of cache hits."""
        return self._hits

    @property
    def misses(self) -> int:
        """Return the number of cache misses."""
        return self._misses

    @property
    def evictions(self) -> int:
        """Return the number of cache evictions."""
        return self._evictions

    @property
    def size(self) -> int:
        """Return the current cache size."""
        return len(self._cache)

    @property
    def stats(self) -> dict[str, int | float]:
        """
        Return cache statistics.

        Returns:
            Dictionary with hits, misses, evictions, size, and hit_rate.
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "size": len(self._cache),
            "hit_rate": hit_rate,
        }

    def _make_key(self, args: tuple, kwargs: dict) -> Hashable:
        """
        Create a cache key from function arguments.

        Args:
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            Hashable cache key.
        """
        if self._key_func is not None:
            return self._key_func(*args, **kwargs)

        # Normalize kwargs to ensure consistent ordering
        # Positional args come first, then sorted keyword args
        try:
            kwitems = tuple(sorted(kwargs.items()))
            key = (args, kwitems)
            # Test if it's hashable
            hash(key)
            return key
        except TypeError:
            # Fall back to string hashing for unhashable types
            key_str = str(args) + str(sorted(kwargs.items()))
            return hashlib.md5(key_str.encode()).hexdigest()

    async def _evict_lru(self) -> None:
        """Evict the least recently used entry from the cache."""
        if not self._cache:
            return

        # Find the LRU entry (oldest accessed_at)
        lru_key = min(self._cache.keys(), key=lambda k: self._cache[k].accessed_at)
        del self._cache[lru_key]
        self._evictions += 1
        log.debug(f"Evicted LRU cache entry. Key: {lru_key}")

    async def _cleanup_expired(self) -> None:
        """Remove all expired entries from the cache."""
        if self._ttl is None:
            return

        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]

        for key in expired_keys:
            del self._cache[key]
            self._evictions += 1

        if expired_keys:
            log.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    def clear(self) -> None:
        """Clear all entries from the cache."""
        self._cache.clear()
        self._pending.clear()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        log.debug("Cache cleared")

    async def invalidate(self, *args, **kwargs) -> bool:
        """
        Invalidate a specific cache entry.

        Args:
            *args: Function arguments to compute the cache key.
            **kwargs: Function keyword arguments.

        Returns:
            True if the entry was found and removed, False otherwise.
        """
        key = self._make_key(args, kwargs)

        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                log.debug(f"Invalidated cache entry. Key: {key}")
                return True
            return False

    async def get_or_compute(self, func: Callable[P, Any], *args: P.args, **kwargs: P.kwargs) -> R:
        """
        Get a cached value or compute it using the provided function.

        Args:
            func: Async function to compute the value if not cached.
            *args: Arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The cached or computed result.
        """
        key = self._make_key(args, kwargs)
        loop = asyncio.get_running_loop()
        
        # Create a future for this call
        result_future: asyncio.Future = loop.create_future()

        async with self._lock:
            # Check cache for valid (non-expired) entry
            if key in self._cache:
                entry = self._cache[key]
                if entry.is_expired():
                    # Entry expired, remove it
                    del self._cache[key]
                    self._evictions += 1
                else:
                    # Cache hit
                    entry.touch()
                    self._hits += 1
                    result_future.set_result(entry.value)
                    log.debug(f"Cache hit. Key: {key}")
                    return await result_future

            # Cache miss - check if there's already a pending computation
            self._misses += 1
            if key in self._pending:
                # Add our future to the waiting list
                self._pending[key].append(result_future)
                log.debug(f"Cache miss - waiting for pending computation. Key: {key}")
                return await result_future
            else:
                # We're the first to request this key - start computation
                self._pending[key] = [result_future]
                log.debug(f"Cache miss - starting computation. Key: {key}")

        # Release lock while computing
        try:
            # Compute the value
            result = await func(*args, **kwargs)

            # Store the result in cache and notify waiters
            async with self._lock:
                # Remove from pending
                if key in self._pending:
                    waiting_futures = self._pending.pop(key)
                else:
                    waiting_futures = []

                # Clean up expired entries before inserting
                await self._cleanup_expired()

                # Evict LRU if necessary
                if len(self._cache) >= self._maxsize and key not in self._cache:
                    await self._evict_lru()

                self._cache[key] = _CacheEntry(result, self._ttl)

            # Notify all waiting futures
            for future in waiting_futures:
                if not future.done():
                    future.set_result(result)

            return result
        except Exception as e:
            # Clean up pending on error
            async with self._lock:
                if key in self._pending:
                    waiting_futures = self._pending.pop(key)
                else:
                    waiting_futures = []
            
            # Notify all waiting futures of the error
            for future in waiting_futures:
                if not future.done():
                    future.set_exception(e)
            raise

    def __call__(self, func: Callable[P, Any]) -> Callable[P, Any]:
        """
        Decorator application.

        Args:
            func: Async function to decorate.

        Returns:
            Decorated async function with caching.
        """

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return await self.get_or_compute(func, *args, **kwargs)

        # Attach cache control methods to the wrapper
        wrapper.cache = self  # type: ignore[attr-defined]
        wrapper.clear = self.clear  # type: ignore[attr-defined]
        wrapper.invalidate = self.invalidate  # type: ignore[attr-defined]

        return wrapper


def async_lru_cache(
    maxsize: int = 128,
    ttl: float | None = None,
    key_func: Callable[..., Hashable] | None = None,
) -> Callable[[Callable[P, Any]], Callable[P, Any]]:
    """
    Decorator factory for creating async LRU caches.

    This is a convenience function that creates an AsyncLRUCache instance
    and applies it to a function.

    Args:
        maxsize: Maximum number of entries to store (default: 128).
        ttl: Time-to-live in seconds. If None, entries don't expire.
        key_func: Optional function to compute cache keys from arguments.

    Returns:
        Decorator function for caching async results.

    Example:
        @async_lru_cache(maxsize=64, ttl=300)
        async def fetch_data(id: str):
            return await api.fetch(id)
    """
    return AsyncLRUCache(maxsize=maxsize, ttl=ttl, key_func=key_func)


__all__ = ["AsyncLRUCache", "async_lru_cache"]
