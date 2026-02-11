"""
Async TTL cache for caching expensive async operations.

This module provides a thread-safe, async-aware cache with TTL (time-to-live)
support and LRU (Least Recently Used) eviction. It's designed for caching
results from expensive async operations like API calls, database queries,
or complex computations.

Example:
    cache = AsyncTTLCache(maxsize=100, ttl=300)

    # Cache miss - function executes
    result1 = await cache.get_or_compute("key1", lambda: expensive_api_call())

    # Cache hit - returns cached value
    result2 = await cache.get_or_compute("key1", lambda: expensive_api_call())

    # Wait for TTL to expire
    await asyncio.sleep(301)

    # Cache expired - function executes again
    result3 = await cache.get_or_compute("key1", lambda: expensive_api_call())
"""

import asyncio
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Hashable
from typing import Generic, TypeVar, cast

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class _CacheEntry(Generic[V]):
    """Internal cache entry storing value and metadata."""

    __slots__: tuple[str, ...] = ("computing", "expires_at", "value")

    value: V | None
    expires_at: float
    computing: bool

    def __init__(self, value: V | None, expires_at: float, computing: bool = False):
        self.value = value
        self.expires_at = expires_at
        self.computing = computing


class AsyncTTLCache(Generic[K, V]):
    """
    Async cache with TTL and LRU eviction.

    This cache provides:
    - TTL-based expiration
    - LRU eviction when maxsize is reached
    - Async-safe operations
    - Prevention of cache stampede (thundering herd)
    - Stale-while-revalidate support

    Example:
        cache = AsyncTTLCache(maxsize=100, ttl=60.0)

        # Basic usage
        result = await cache.get_or_compute(
            "user:123",
            lambda: fetch_user_from_db("123")
        )

        # Stale-while-revalidate
        result = await cache.get_or_compute(
            "user:123",
            lambda: fetch_user_from_db("123"),
            allow_stale=True
        )

        # Check if key exists and is fresh
        if await cache.has_fresh("key"):
            value = await cache.get("key")

        # Invalidate specific key
        await cache.invalidate("key")

        # Clear all cache
        await cache.clear()
    """

    _maxsize: int
    _default_ttl: float
    _cache: OrderedDict[K, _CacheEntry[V]]
    _lock: asyncio.Lock
    _waiters: dict[K, list[asyncio.Future[None]]]

    def __init__(
        self,
        maxsize: int = 128,
        ttl: float = 300.0,
        default_ttl: float | None = None,
    ):
        """
        Initialize the cache.

        Args:
            maxsize: Maximum number of items to store. Use 0 for unlimited.
            ttl: Default time-to-live in seconds for cache entries.
            default_ttl: Alias for ttl (deprecated, kept for compatibility).

        Raises:
            ValueError: If maxsize < 0 or ttl <= 0.
        """
        if maxsize < 0:
            raise ValueError("maxsize must be >= 0")
        if ttl <= 0:
            raise ValueError("ttl must be > 0")

        self._maxsize = maxsize
        self._default_ttl = ttl if default_ttl is None else default_ttl
        self._cache: "OrderedDict[K, _CacheEntry[V]]" = OrderedDict()
        self._lock = asyncio.Lock()
        self._waiters: "dict[K, list[asyncio.Future[None]]]" = {}

    async def get(
        self, key: K, default: V | None = None
    ) -> V | None:
        """
        Get a value from the cache.

        Returns None if the key doesn't exist or has expired.

        Args:
            key: Cache key.
            default: Value to return if key not found or expired.

        Returns:
            Cached value or default.
        """
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return default

            if entry.computing:
                # Entry is being computed
                return default

            if self._is_expired(entry):
                # Entry expired, remove it
                del self._cache[key]
                return default

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return entry.value

    async def get_or_compute(
        self,
        key: K,
        compute_fn: Callable[[], Awaitable[V]],
        ttl: float | None = None,
        allow_stale: bool = False,
    ) -> V:
        """
        Get value from cache or compute it if not present/expired.

        This method prevents cache stampede by ensuring only one coroutine
        computes the value for a given key at a time. Other coroutines wait
        for the result.

        Args:
            key: Cache key.
            compute_fn: Async function to compute the value if not cached.
            ttl: Time-to-live for this entry. Uses default_ttl if None.
            allow_stale: If True, return stale value and recompute in background.

        Returns:
            Cached or computed value.

        Raises:
            Exception: If compute_fn raises an exception.
        """
        ttl = ttl if ttl is not None else self._default_ttl

        # Fast path: try to get fresh value without lock
        async with self._lock:
            entry = self._cache.get(key)

            if entry is not None and not entry.computing:
                if not self._is_expired(entry):
                    # Fresh cache hit
                    self._cache.move_to_end(key)
                    # When computing=False, value is never None
                    return cast(V, entry.value)
                elif allow_stale:
                    # Stale-while-revalidate
                    self._cache.move_to_end(key)
                    # Trigger background refresh (fire-and-forget task)
                    asyncio.create_task(  # noqa: RUF006
                        self._compute_and_store(key, compute_fn, ttl)
                    )
                    # When computing=False, value is never None
                    return cast(V, entry.value)

            # Check if computation is in progress
            if entry is not None and entry.computing:
                # Wait for existing computation
                fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
                self._waiters.setdefault(key, []).append(fut)
                release_lock = True
            else:
                # We need to compute
                release_lock = False
                fut = asyncio.get_running_loop().create_future()  # For type checker

        if release_lock:
            # Wait for computation to complete
            try:
                await fut
                # Try again - should now be in cache
                return await self.get_or_compute(
                    key, compute_fn, ttl=ttl, allow_stale=allow_stale
                )
            finally:
                async with self._lock:
                    waiters = self._waiters.get(key, [])
                    if fut in waiters:
                        waiters.remove(fut)
                    if not waiters and key in self._waiters:
                        del self._waiters[key]

        # Compute the value
        return await self._compute_and_store(key, compute_fn, ttl)

    async def _compute_and_store(
        self, key: K, compute_fn: Callable[[], Awaitable[V]], ttl: float
    ) -> V:
        """Compute value and store in cache."""
        # Mark as computing
        async with self._lock:
            self._cache[key] = _CacheEntry[V](
                value=None, expires_at=0, computing=True
            )

        try:
            result = await compute_fn()

            # Store result
            async with self._lock:
                expires_at = time.time() + ttl
                self._cache[key] = _CacheEntry[V](
                    value=result, expires_at=expires_at, computing=False
                )
                self._cache.move_to_end(key)

                # Evict oldest if over limit
                if self._maxsize > 0 and len(self._cache) > self._maxsize:
                    _ = self._cache.popitem(last=False)

                # Notify waiters
                for fut in self._waiters.get(key, []):
                    if not fut.done():
                        fut.set_result(None)

            return result
        except Exception as e:
            # Remove entry and notify waiters of failure
            async with self._lock:
                if key in self._cache and self._cache[key].computing:
                    del self._cache[key]

                for fut in self._waiters.get(key, []):
                    if not fut.done():
                        fut.set_exception(e)

            raise

    async def set(self, key: K, value: V, ttl: float | None = None) -> None:
        """
        Set a value in the cache.

        Args:
            key: Cache key.
            value: Value to store.
            ttl: Time-to-live for this entry. Uses default_ttl if None.
        """
        ttl = ttl if ttl is not None else self._default_ttl
        expires_at = time.time() + ttl

        async with self._lock:
            self._cache[key] = _CacheEntry[V](
                value=value, expires_at=expires_at, computing=False
            )
            self._cache.move_to_end(key)

            # Evict oldest if over limit
            if self._maxsize > 0 and len(self._cache) > self._maxsize:
                _ = self._cache.popitem(last=False)

    async def has_fresh(self, key: K) -> bool:
        """
        Check if key exists and is not expired.

        Args:
            key: Cache key.

        Returns:
            True if key exists and is fresh.
        """
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None or entry.computing:
                return False
            if self._is_expired(entry):
                # Remove expired entry
                del self._cache[key]
                return False
            return True

    async def invalidate(self, key: K) -> bool:
        """
        Invalidate a specific cache entry.

        Args:
            key: Cache key to invalidate.

        Returns:
            True if key was in cache, False otherwise.
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._waiters.clear()

    async def cleanup_expired(self) -> int:
        """
        Remove all expired entries from cache.

        Returns:
            Number of entries removed.
        """
        async with self._lock:
            expired_keys = [
                key
                for key, entry in self._cache.items()
                if not entry.computing and self._is_expired(entry)
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)

    def _is_expired(self, entry: _CacheEntry[V]) -> bool:
        """Check if an entry has expired."""
        return time.time() > entry.expires_at

    async def size(self) -> int:
        """
        Get current cache size.

        Returns:
            Number of entries in cache.
        """
        async with self._lock:
            return len(self._cache)

    async def stats(self) -> dict[str, object]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats (size, maxsize, etc.).
        """
        async with self._lock:
            expired_count = sum(
                1 for entry in self._cache.values() if self._is_expired(entry)
            )
            computing_count = sum(
                1 for entry in self._cache.values() if entry.computing
            )
            return {
                "size": len(self._cache),
                "maxsize": self._maxsize if self._maxsize > 0 else "unlimited",
                "expired": expired_count,
                "computing": computing_count,
                "waiters": sum(len(w) for w in self._waiters.values()),
            }


__all__ = ["AsyncTTLCache"]
