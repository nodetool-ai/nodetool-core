from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from typing import Any, Callable, Generic, TypeVar, cast

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


class CacheEntry(Generic[V]):
    """A single entry in the async cache."""

    __slots__ = ("access_count", "expires_at", "last_accessed", "value")

    def __init__(self, value: V, expires_at: float | None = None) -> None:
        self.value = value
        self.expires_at = expires_at
        self.access_count = 0
        self.last_accessed: float = time.time()


class AsyncLRUCache(Generic[K, V]):
    """
    An async-aware LRU (Least Recently Used) cache with TTL (Time To Live) support.

    This cache is thread-safe and suitable for use in async contexts. It supports:
    - Configurable maximum size (evicts least recently used items when full)
    - Configurable TTL (items expire after specified seconds)
    - Thread-safe operations using asyncio.Lock
    - Statistics tracking

    Example:
        ```python
        cache = AsyncLRUCache(max_size=100, ttl=300)  # 100 items, 5 min TTL

        async def fetch_data(key: str) -> Data:
            # Expensive operation
            return await some_api.call(key)

        # Get from cache or fetch
        result = await cache.get_or_set(key, fetch_data, key)

        # Manual set
        await cache.set("key", value)
        ```

    Args:
        max_size: Maximum number of items in the cache. Must be positive.
        ttl: Time-to-live in seconds. None means no expiration. Defaults to None.
        ttl_reset_on_access: Whether to reset TTL when item is accessed. Defaults to True.
    """

    def __init__(
        self,
        max_size: int,
        ttl: float | None = None,
        ttl_reset_on_access: bool = True,
    ) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be a positive integer")

        if ttl is not None and ttl <= 0:
            raise ValueError("ttl must be None or a positive number")

        self._max_size = max_size
        self._ttl = ttl
        self._ttl_reset_on_access = ttl_reset_on_access
        self._lock = asyncio.Lock()
        self._cache: dict[K, CacheEntry[V]] = {}
        self._order: OrderedDict[K, float] = OrderedDict()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0,
            "expirations": 0,
        }

    @property
    def max_size(self) -> int:
        """Maximum cache size."""
        return self._max_size

    @property
    def ttl(self) -> float | None:
        """Time-to-live in seconds, or None for no expiration."""
        return self._ttl

    @property
    def size(self) -> int:
        """Current number of items in the cache."""
        return len(self._cache)

    @property
    def stats(self) -> dict[str, int]:
        """Cache statistics (hits, misses, sets, evictions, expirations)."""
        return self._stats.copy()

    def clear(self) -> None:
        """Clear all items from the cache."""
        self._cache.clear()
        self._order.clear()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0,
            "expirations": 0,
        }

    async def get(self, key: K, default: V | None = None) -> V | None:
        """
        Get a value from the cache.

        Args:
            key: The key to look up.
            default: Value to return if key not found. Defaults to None.

        Returns:
            The cached value, or default if not found/expired.
        """
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats["misses"] += 1
                return default

            now = time.time()
            expires_at = entry.expires_at
            if expires_at is not None and now >= expires_at:
                self._remove_entry(key)
                self._stats["misses"] += 1
                self._stats["expirations"] += 1
                return default

            entry.access_count += 1
            entry.last_accessed = now
            if self._ttl_reset_on_access:
                entry.expires_at = None if self._ttl is None else now + self._ttl

            self._order.move_to_end(key)
            self._stats["hits"] += 1
            return entry.value

    async def set(self, key: K, value: V, ttl: float | None = None) -> None:
        """
        Set a value in the cache.

        Args:
            key: The key to store.
            value: The value to cache.
            ttl: Optional TTL override for this specific entry.
        """
        async with self._lock:
            now = time.time()
            effective_ttl = ttl if ttl is not None else self._ttl

            if key in self._cache:
                entry = self._cache[key]
                entry.value = value
                entry.expires_at = None if effective_ttl is None else now + effective_ttl
                entry.access_count += 1
                entry.last_accessed = now
                self._order.move_to_end(key)
            else:
                while len(self._cache) >= self._max_size:
                    self._evict_lru()

                entry = CacheEntry(
                    value=value,
                    expires_at=None if effective_ttl is None else now + effective_ttl,
                )
                self._cache[key] = entry
                self._order[key] = now

            self._stats["sets"] += 1

    async def get_or_set(
        self,
        key: K,
        fetch_func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> V:
        """
        Get from cache or set using the provided fetch function.

        This is atomic - if multiple coroutines call this with the same key
        concurrently, only one will execute the fetch_func.

        Args:
            key: The cache key.
            fetch_func: Async function to call if key not found/cached.
            *args: Positional arguments passed to fetch_func.
            **kwargs: Keyword arguments passed to fetch_func.

        Returns:
            The cached or freshly fetched value.
        """
        cached = await self.get(key)
        if cached is not None:
            return cached

        async with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                now = time.time()
                expires_at = entry.expires_at
                if expires_at is not None and now >= expires_at:
                    self._remove_entry(key)
                else:
                    entry.access_count += 1
                    entry.last_accessed = now
                    if self._ttl_reset_on_access:
                        entry.expires_at = None if self._ttl is None else now + self._ttl
                    self._order.move_to_end(key)
                    self._stats["hits"] += 1
                    return entry.value

            value = await fetch_func(*args, **kwargs)
            await self.set(key, value)
            return value

    async def delete(self, key: K) -> bool:
        """
        Delete a key from the cache.

        Args:
            key: The key to delete.

        Returns:
            True if key was present, False otherwise.
        """
        async with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False

    async def contains(self, key: K) -> bool:
        """
        Check if key is in cache (and not expired).

        Args:
            key: The key to check.

        Returns:
            True if key exists and is not expired.
        """
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False

            expires_at = entry.expires_at
            if expires_at is not None and time.time() >= expires_at:
                self._remove_entry(key)
                return False

            return True

    async def clear_expired(self) -> int:
        """
        Remove all expired entries from the cache.

        Returns:
            Number of expired entries removed.
        """
        async with self._lock:
            now = time.time()
            expired_keys = [
                key for key, entry in self._cache.items() if entry.expires_at is not None and now >= entry.expires_at
            ]

            for key in expired_keys:
                self._remove_entry(key)

            count = len(expired_keys)
            self._stats["expirations"] += count
            return count

    async def peek(self, key: K, default: V | None = None) -> V | None:
        """
        Get value without updating access time or TTL.

        Args:
            key: The key to look up.
            default: Value to return if not found.

        Returns:
            The cached value, or default if not found/expired.
        """
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                return default

            now = time.time()
            expires_at = entry.expires_at
            if expires_at is not None and now >= expires_at:
                self._remove_entry(key)
                return default

            return entry.value

    def _remove_entry(self, key: K) -> None:
        """Remove an entry from both cache and order dict (must hold lock)."""
        self._cache.pop(key, None)
        self._order.pop(key, None)

    def _evict_lru(self) -> None:
        """Evict the least recently used entry (must hold lock)."""
        if not self._order:
            return

        lru_key, _ = self._order.popitem(last=False)
        self._cache.pop(lru_key, None)
        self._stats["evictions"] += 1

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics including hit rate."""
        stats: dict[str, Any] = self._stats.copy()
        total = stats["hits"] + stats["misses"]
        stats["hit_rate"] = stats["hits"] / total if total > 0 else 0.0
        stats["size"] = self.size
        stats["max_size"] = self._max_size
        return stats


def async_lru_cache(
    max_size: int = 128,
    ttl: float | None = None,
    ttl_reset_on_access: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., Any]]:
    """
    Decorator that provides async LRU caching with TTL support.

    Args:
        max_size: Maximum number of entries to cache.
        ttl: Time-to-live in seconds. None means no expiration.
        ttl_reset_on_access: Whether to reset TTL on cache hit.

    Returns:
        A decorator that wraps functions with LRU caching.

    Example:
        ```python
        @async_lru_cache(max_size=100, ttl=300)
        async def fetch_user(user_id: int) -> User:
            return await database.get_user(user_id)
        ```
    """
    cache: dict[Any, tuple[float, Any]] = {}
    lock = asyncio.Lock()
    order: OrderedDict[Any, float] = OrderedDict()

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = (args, tuple(sorted(kwargs.items())))

            async with lock:
                now = time.time()

                if key in cache:
                    cached_time, cached_value = cache[key]
                    if ttl is None or now - cached_time < ttl:
                        order[key] = now
                        if ttl_reset_on_access and ttl is not None:
                            cache[key] = (now, cached_value)
                        return cached_value
                    else:
                        order.pop(key, None)
                        cache.pop(key, None)

                result = await func(*args, **kwargs)
                cache[key] = (now, result)
                order[key] = now

                while len(cache) > max_size:
                    lru_key, _ = order.popitem(last=False)
                    cache.pop(lru_key, None)

                return result

        return wrapper

    return decorator
