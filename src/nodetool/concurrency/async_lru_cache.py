"""
Async LRU (Least Recently Used) Cache implementation.

This module provides an async-safe LRU cache with configurable size, TTL,
and async factory functions for loading cached values.
"""

import asyncio
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)

K = TypeVar("K")
V = TypeVar("V")


class AsyncLRUCache(Generic[K, V]):
    """
    An async-safe LRU (Least Recently Used) cache with optional TTL support.

    This cache provides thread-safe operations for concurrent async access,
    automatic eviction of least recently used items when capacity is reached,
    and optional time-to-live (TTL) for cache entries.

    Example:
        # Simple cache with async factory
        cache = AsyncLRUCache(maxsize=100)

        async def fetch_user(user_id: str) -> dict:
            return await database.get_user(user_id)

        # Get or load with factory
        user = await cache.get_or_load("user:123", fetch_user)

        # With TTL (entries expire after 60 seconds)
        cache = AsyncLRUCache(maxsize=100, ttl=60.0)

        # Manual operations
        await cache.put("key", "value")
        value = await cache.get("key")  # Returns "value" or None
    """

    def __init__(
        self,
        maxsize: int = 128,
        ttl: float | None = None,
        default_factory: Callable[[K], Awaitable[V]] | None = None,
    ):
        """
        Initialize the async LRU cache.

        Args:
            maxsize: Maximum number of items to cache (default: 128).
            ttl: Time-to-live for cache entries in seconds (default: None = no expiration).
            default_factory: Default async factory function for loading values (optional).

        Raises:
            ValueError: If maxsize <= 0.
        """
        if maxsize <= 0:
            raise ValueError("maxsize must be a positive integer")

        self._maxsize: int = maxsize
        self._ttl: float | None = ttl
        self._default_factory: Callable[[K], Awaitable[V]] | None = default_factory
        self._cache: OrderedDict[K, tuple[V, float | None]] = OrderedDict()
        self._lock: asyncio.Lock = asyncio.Lock()

    @property
    def maxsize(self) -> int:
        """Return the maximum cache size."""
        return self._maxsize

    @property
    def ttl(self) -> float | None:
        """Return the time-to-live in seconds."""
        return self._ttl

    @property
    def size(self) -> int:
        """Return the current number of cached items."""
        return len(self._cache)

    async def get(self, key: K) -> V | None:
        """
        Get a value from the cache by key.

        Args:
            key: The cache key.

        Returns:
            The cached value, or None if not found or expired.

        Example:
            value = await cache.get("my_key")
            if value is not None:
                print(f"Found: {value}")
        """
        async with self._lock:
            return self._get_sync(key)

    def _get_sync(self, key: K) -> V | None:
        """Internal synchronous get (must be called with lock held)."""
        if key not in self._cache:
            return None

        value, expiry = self._cache[key]

        # Check TTL expiration
        if expiry is not None and time.monotonic() > expiry:
            del self._cache[key]
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        return value

    async def put(self, key: K, value: V) -> None:
        """
        Put a value into the cache.

        Args:
            key: The cache key.
            value: The value to cache.

        Example:
            await cache.put("user:123", user_data)
        """
        async with self._lock:
            expiry = time.monotonic() + self._ttl if self._ttl is not None else None

            if key in self._cache:
                # Update existing entry
                self._cache[key] = (value, expiry)
                self._cache.move_to_end(key)
            else:
                # Add new entry, evict if necessary
                self._cache[key] = (value, expiry)
                if len(self._cache) > self._maxsize:
                    self._cache.popitem(last=False)  # Remove least recently used

    async def get_or_load(
        self,
        key: K,
        factory: Callable[[K], Awaitable[V]] | None = None,
    ) -> V:
        """
        Get a value from cache or load it using the factory function.

        If the key is not in cache or has expired, the factory function
        is called asynchronously to load the value, which is then cached.

        Args:
            key: The cache key.
            factory: Async factory function to load the value. If None, uses default_factory.

        Returns:
            The cached or loaded value.

        Raises:
            ValueError: If no factory is provided and default_factory is not set.

        Example:
            async def fetch_data(id: str) -> dict:
                return await api.fetch(id)

            # Get from cache or fetch
            data = await cache.get_or_load("item:1", fetch_data)
        """
        # Check cache first
        value = await self.get(key)
        if value is not None:
            return value

        # Load using factory
        load_factory = factory or self._default_factory
        if load_factory is None:
            raise ValueError("No factory function provided and default_factory is not set")

        loaded_value = await load_factory(key)
        await self.put(key, loaded_value)
        return loaded_value

    async def invalidate(self, key: K) -> bool:
        """
        Invalidate a cache entry.

        Args:
            key: The cache key to invalidate.

        Returns:
            True if the key was found and removed, False otherwise.

        Example:
            removed = await cache.invalidate("user:123")
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> None:
        """
        Clear all cache entries.

        Example:
            await cache.clear()
        """
        async with self._lock:
            self._cache.clear()

    async def cleanup_expired(self) -> int:
        """
        Remove all expired entries from the cache.

        Returns:
            The number of expired entries removed.

        Example:
            removed = await cache.cleanup_expired()
        """
        if self._ttl is None:
            return 0

        async with self._lock:
            now = time.monotonic()
            expired_keys = [
                key for key, (_, expiry) in self._cache.items()
                if expiry is not None and now > expiry
            ]

            for key in expired_keys:
                del self._cache[key]

            return len(expired_keys)

    def __len__(self) -> int:
        """Return the current cache size."""
        return len(self._cache)

    async def __aenter__(self) -> "AsyncLRUCache[K, V]":
        """Enter context manager (for potential future use)."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager."""
        pass


def async_lru_cache(
    maxsize: int = 128,
    ttl: float | None = None,
) -> Callable[[Callable[..., Awaitable[V]]], Callable[..., Awaitable[V]]]:
    """
    Decorator to create an async LRU cache for async functions.

    This decorator creates a cache that stores the results of async function calls
    and returns cached results on subsequent calls with the same arguments.

    Args:
        maxsize: Maximum number of entries to cache (default: 128).
        ttl: Time-to-live for cached results in seconds (default: None = no expiration).

    Returns:
        A decorator function.

    Example:
        @async_lru_cache(maxsize=100, ttl=60.0)
        async def fetch_user(user_id: str) -> dict:
            return await database.query(user_id)

        # First call fetches from database
        user1 = await fetch_user("user:123")

        # Subsequent calls return cached result
        user2 = await fetch_user("user:123")  # Returns cached value
    """
    def decorator(func: Callable[..., Awaitable[V]]) -> Callable[..., Awaitable[V]]:
        cache = AsyncLRUCache[object, V](maxsize=maxsize, ttl=ttl)

        async def wrapper(*args: object, **kwargs: object) -> V:
            # Create a cache key from args and kwargs
            # For simplicity, use args[0] as key if it's a single-arg function
            # Otherwise use a tuple of (args, frozenset(kwargs.items()))
            key = args[0] if len(args) == 1 and not kwargs else (args, frozenset(kwargs.items()))

            return await cache.get_or_load(key, lambda _: func(*args, **kwargs))  # type: ignore[arg-type]

        # Add cache control methods to the wrapped function
        wrapper.cache = cache  # type: ignore[attr-defined]
        wrapper.cache_clear = cache.clear  # type: ignore[attr-defined]
        wrapper.cache_info = lambda: {  # type: ignore[attr-defined]
            "size": cache.size,
            "maxsize": cache.maxsize,
            "ttl": cache.ttl,
        }

        return wrapper

    return decorator


__all__ = ["AsyncLRUCache", "async_lru_cache"]
