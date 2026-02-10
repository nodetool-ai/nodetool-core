"""
Async LRU Cache with automatic expiration based on access time.

Provides a thread-safe, async-friendly LRU (Least Recently Used) cache
that automatically expires entries based on a configurable TTL (time-to-live).
Perfect for caching expensive async operations like API calls, database queries,
or computation results.
"""
import asyncio
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from typing import Any, Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")

# Sentinel value to distinguish "not in cache" from "cached None"
_NOT_FOUND = object()


class AsyncLRUCache(Generic[K, V]):
    """
    An async-friendly LRU cache with TTL-based expiration.

    This cache automatically tracks access time and expires entries that
    haven't been accessed within the configured TTL. Thread-safe and
    designed for use with async code.

    Example:
        cache = AsyncLRUCache[str, int](max_size=100, ttl_seconds=60)

        # Set a value
        await cache.set("key1", 42)

        # Get a value (returns None if not found or expired)
        value = await cache.get("key1")

        # Get or compute (compute if not in cache)
        value = await cache.get_or_compute("key2", lambda: expensive_computation())

        # Invalidate specific entry
        await cache.invalidate("key1")

        # Clear all entries
        await cache.clear()

        # Get cache statistics
        stats = await cache.get_stats()
        print(f"Hit rate: {stats['hit_rate']:.2%}")
    """

    def __init__(
        self, max_size: int = 128, ttl_seconds: float | None = None
    ) -> None:
        """
        Initialize the LRU cache.

        Args:
            max_size: Maximum number of entries to store. When exceeded,
                     the least recently used entry is evicted.
            ttl_seconds: Time-to-live for entries in seconds. If None (default),
                        entries never expire based on time.

        Raises:
            ValueError: If max_size is not positive.
        """
        if max_size <= 0:
            raise ValueError("max_size must be a positive integer")

        self._max_size: int = max_size
        self._ttl: timedelta | None = (
            timedelta(seconds=ttl_seconds) if ttl_seconds is not None else None
        )
        self._cache: OrderedDict[K, tuple[V, datetime]] = OrderedDict()
        self._lock: asyncio.Lock = asyncio.Lock()
        self._hits: int = 0
        self._misses: int = 0

    @property
    def max_size(self) -> int:
        """Return the maximum cache size."""
        return self._max_size

    @property
    def ttl_seconds(self) -> float | None:
        """Return the TTL in seconds, or None if no TTL."""
        return self._ttl.total_seconds() if self._ttl is not None else None

    @property
    def size(self) -> int:
        """Return the current number of entries in the cache."""
        return len(self._cache)

    async def get(self, key: K) -> V | None:
        """
        Get a value from the cache.

        Returns None if the key is not found or has expired.
        Expired entries are automatically removed.
        Accessing a value refreshes its TTL.

        Args:
            key: The key to look up.

        Returns:
            The cached value, or None if not found or expired.
        """
        async with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            value, last_access = self._cache[key]

            # Check expiration
            if self._ttl is not None and datetime.now() - last_access > self._ttl:
                # Entry expired
                del self._cache[key]
                self._misses += 1
                return None

            # Update access time and order (move to end = most recently used)
            self._cache[key] = (value, datetime.now())
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    async def set(self, key: K, value: V) -> None:
        """
        Set a value in the cache.

        If the key already exists, its value is updated and it becomes
        the most recently used. If the cache is full, the least recently
        used entry is evicted.

        Args:
            key: The key to store.
            value: The value to cache.
        """
        async with self._lock:
            # Update existing or add new entry
            self._cache[key] = (value, datetime.now())
            self._cache.move_to_end(key)

            # Evict LRU entry if over capacity
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    async def get_or_compute(
        self, key: K, compute_fn: Callable[[], V | Awaitable[V]]
    ) -> V:
        """
        Get a value from cache, or compute it if not present.

        This is a convenience method that combines cache lookup with
        computation of missing values. The compute_fn can be either
        a regular function or a coroutine function.

        Args:
            key: The key to look up.
            compute_fn: Function to compute the value if not cached.
                       Can be sync or async.

        Returns:
            The cached or newly computed value.

        Example:
            async def fetch_user(user_id: str) -> User:
                return await api.get_user(user_id)

            # Cache users for 5 minutes
            cache = AsyncLRUCache[str, User](ttl_seconds=300)

            # First call fetches from API
            user1 = await cache.get_or_compute("user_123", fetch_user)

            # Second call returns cached value
            user2 = await cache.get_or_compute("user_123", fetch_user)
        """
        # Try to get from cache - use sentinel to distinguish None from not found
        async with self._lock:
            if key in self._cache:
                value, last_access = self._cache[key]

                # Check expiration
                if self._ttl is not None and datetime.now() - last_access > self._ttl:
                    # Entry expired - remove and continue to compute
                    del self._cache[key]
                else:
                    # Cache hit - update access time and return
                    self._cache[key] = (value, datetime.now())
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return value  # type: ignore[return-value]

        # Cache miss - compute the value
        result = compute_fn()
        if asyncio.iscoroutine(result):
            result = await result

        # Store in cache and return
        await self.set(key, result)  # type: ignore[arg-type]
        return result  # type: ignore[return-value]

    async def invalidate(self, key: K) -> bool:
        """
        Invalidate a specific cache entry.

        Args:
            key: The key to invalidate.

        Returns:
            True if the entry was found and removed, False otherwise.
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> None:
        """Clear all entries from the cache and reset statistics."""
        async with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    async def cleanup_expired(self) -> int:
        """
        Remove all expired entries from the cache.

        This is useful for manual cleanup, though expired entries
        are automatically removed on access.

        Returns:
            The number of entries removed.
        """
        if self._ttl is None:
            return 0

        async with self._lock:
            now = datetime.now()
            expired_keys = [
                k
                for k, (_, last_access) in self._cache.items()
                if now - last_access > self._ttl
            ]

            for key in expired_keys:
                del self._cache[key]

            return len(expired_keys)

    async def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            A dictionary with statistics:
            - size: Current number of entries
            - max_size: Maximum cache size
            - hits: Number of cache hits
            - misses: Number of cache misses
            - hit_rate: Ratio of hits to total lookups (0.0 to 1.0)
        """
        async with self._lock:
            total_lookups = self._hits + self._misses
            hit_rate = self._hits / total_lookups if total_lookups > 0 else 0.0

            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }

    def contains(self, key: K) -> bool:
        """
        Check if a key is in the cache without updating access time.

        This is a non-async check that doesn't affect the LRU order
        and doesn't check expiration. Useful for existence checks
        where you don't need the value.

        Args:
            key: The key to check.

        Returns:
            True if the key exists in the cache, False otherwise.

        Note:
            This doesn't check expiration and doesn't update access time.
            For a full check, use `get()` and compare to None.
        """
        return key in self._cache

    async def keys(self) -> list[K]:
        """
        Get all keys in the cache, ordered from least to most recently used.

        Returns:
            A list of cache keys in LRU order.

        Note:
            This doesn't update access times or check expiration.
        """
        async with self._lock:
            return list(self._cache.keys())

    async def values(self) -> list[V]:
        """
        Get all values in the cache, ordered from least to most recently used.

        Returns:
            A list of cached values in LRU order.

        Note:
            This doesn't update access times or check expiration.
        """
        async with self._lock:
            return [value for value, _ in self._cache.values()]

    async def items(self) -> list[tuple[K, V]]:
        """
        Get all key-value pairs in the cache, ordered from least to most recently used.

        Returns:
            A list of (key, value) tuples in LRU order.

        Note:
            This doesn't update access times or check expiration.
        """
        async with self._lock:
            return [(key, value) for key, (value, _) in self._cache.items()]


__all__ = ["AsyncLRUCache"]
