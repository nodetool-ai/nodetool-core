"""
Async TTL (Time-To-Live) cache with LRU eviction.

This module provides an asynchronous cache with time-based expiration and
size-based eviction using the Least Recently Used (LRU) policy.
"""

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)

K = TypeVar("K")
V = TypeVar("V")


@dataclass
class CacheEntry(Generic[V]):
    """
    A cache entry containing a value and its expiration time.

    Attributes:
        value: The cached value.
        expires_at: The monotonic time when this entry expires.
    """

    value: V
    expires_at: float


class AsyncTTLCache(Generic[K, V]):
    """
    An asynchronous cache with TTL expiration and LRU eviction.

    This cache supports:
    - Time-to-live (TTL) based expiration of entries
    - LRU (Least Recently Used) eviction when the cache is full
    - Async-safe operations with proper locking
    - Background cleanup of expired entries
    - Get-or-compute pattern for cache-aside loading

    Example:
        cache = AsyncTTLCache[str, int](max_size=100, ttl=60.0)

        # Basic usage
        await cache.set("key1", 42)
        value = await cache.get("key1")  # Returns 42

        # Get or compute
        value = await cache.get_or_compute(
            "key2",
            lambda: expensive_computation()
        )

        # Use as context manager for automatic background cleanup
        async with cache:
            await cache.set("key3", 100)
            # Background cleanup task runs automatically
    """

    def __init__(self, max_size: int, ttl: float, cleanup_interval: float = 1.0):
        """
        Initialize the AsyncTTLCache.

        Args:
            max_size: Maximum number of entries in the cache.
            ttl: Time-to-live in seconds for each cache entry.
            cleanup_interval: Interval in seconds for background cleanup.

        Raises:
            ValueError: If max_size <= 0 or ttl <= 0.
        """
        if max_size <= 0:
            raise ValueError("max_size must be a positive integer")
        if ttl <= 0:
            raise ValueError("ttl must be a positive number")
        if cleanup_interval <= 0:
            raise ValueError("cleanup_interval must be a positive number")

        self._max_size = max_size
        self._ttl = ttl
        self._cleanup_interval = cleanup_interval
        self._cache: OrderedDict[K, CacheEntry[V]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[None] | None = None
        self._is_running = False

    @property
    def max_size(self) -> int:
        """Return the maximum cache size."""
        return self._max_size

    @property
    def ttl(self) -> float:
        """Return the time-to-live in seconds."""
        return self._ttl

    @property
    def size(self) -> int:
        """Return the current number of entries in the cache."""
        return len(self._cache)

    async def get(self, key: K) -> V | None:
        """
        Get a value from the cache.

        If the key exists but the entry has expired, it will be removed
        and None will be returned.

        Args:
            key: The cache key.

        Returns:
            The cached value, or None if the key doesn't exist or has expired.
        """
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                return None

            if self._is_expired(entry):
                # Remove expired entry
                del self._cache[key]
                log.debug("Cache entry expired", extra={"key": str(key)})
                return None

            # Move to end to mark as recently used
            self._cache.move_to_end(key)
            return entry.value

    async def set(self, key: K, value: V) -> None:
        """
        Set a value in the cache.

        If the key already exists, it will be updated. If the cache is full,
        the least recently used entry will be evicted.

        Args:
            key: The cache key.
            value: The value to cache.
        """
        async with self._lock:
            now = asyncio.get_running_loop().time()
            expires_at = now + self._ttl

            # If key already exists, update it
            if key in self._cache:
                self._cache[key] = CacheEntry(value=value, expires_at=expires_at)
                self._cache.move_to_end(key)
                return

            # Check if we need to evict
            if len(self._cache) >= self._max_size:
                # Evict the oldest (first) entry
                evicted_key, _ = self._cache.popitem(last=False)
                log.debug(
                    "Cache entry evicted (LRU)",
                    extra={"evicted_key": str(evicted_key)},
                )

            # Add new entry
            self._cache[key] = CacheEntry(value=value, expires_at=expires_at)

    async def has(self, key: K) -> bool:
        """
        Check if a key exists in the cache and is not expired.

        Args:
            key: The cache key.

        Returns:
            True if the key exists and has not expired, False otherwise.
        """
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                return False

            if self._is_expired(entry):
                del self._cache[key]
                return False

            return True

    async def invalidate(self, key: K) -> bool:
        """
        Invalidate (remove) a specific key from the cache.

        Args:
            key: The cache key to invalidate.

        Returns:
            True if the key was found and removed, False otherwise.
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                log.debug("Cache entry invalidated", extra={"key": str(key)})
                return True
            return False

    async def clear(self) -> None:
        """
        Clear all entries from the cache.
        """
        async with self._lock:
            self._cache.clear()
            log.debug("Cache cleared")

    async def cleanup_expired(self) -> int:
        """
        Manually remove all expired entries from the cache.

        This is useful if you're not using the background cleanup task.

        Returns:
            The number of entries that were removed.
        """
        async with self._lock:
            expired_keys = [
                key
                for key, entry in self._cache.items()
                if self._is_expired(entry)
            ]

            for key in expired_keys:
                del self._cache[key]

            if expired_keys:
                log.debug(
                    "Cleaned up expired entries",
                    extra={"count": len(expired_keys)},
                )

            return len(expired_keys)

    async def get_or_compute(
        self,
        key: K,
        compute_fn: Callable[[], Any] | Callable[[], Any] | None = None,
    ) -> V:
        """
        Get a value from the cache, or compute it if not present.

        This is useful for the cache-aside pattern where you want to
        load values on-demand.

        Args:
            key: The cache key.
            compute_fn: An async function that computes the value if missing.
                       If None, returns None on cache miss.

        Returns:
            The cached or computed value, or None if not found and no compute_fn.

        Raises:
            ValueError: If compute_fn is None and key is not in cache.
        """
        # Try to get from cache first
        value = await self.get(key)
        if value is not None:
            return value

        # Compute the value
        if compute_fn is None:
            raise ValueError(f"Key '{key}' not found in cache and no compute_fn provided")

        # Compute outside the lock
        computed_value: V = await compute_fn()

        # Store in cache
        await self.set(key, computed_value)

        return computed_value

    def _is_expired(self, entry: CacheEntry[V]) -> bool:
        """
        Check if a cache entry has expired.

        Args:
            entry: The cache entry to check.

        Returns:
            True if the entry has expired, False otherwise.
        """
        now = asyncio.get_running_loop().time()
        return now >= entry.expires_at

    async def _cleanup_loop(self) -> None:
        """
        Internal background task that periodically cleans up expired entries.
        """
        try:
            while self._is_running:
                await asyncio.sleep(self._cleanup_interval)
                await self.cleanup_expired()
        except asyncio.CancelledError:
            log.debug("Cache cleanup task cancelled")
            raise

    async def __aenter__(self) -> "AsyncTTLCache[K, V]":
        """
        Enter the context manager and start the background cleanup task.

        Returns:
            The cache instance.
        """
        self._is_running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        log.debug("Cache cleanup task started")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit the context manager and stop the background cleanup task.
        """
        self._is_running = False

        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        log.debug("Cache cleanup task stopped")

    def __len__(self) -> int:
        """
        Return the current number of entries in the cache.

        Note: This is a synchronous operation and may not reflect
        expired entries that haven't been cleaned up yet.
        """
        return len(self._cache)


__all__ = ["AsyncTTLCache", "CacheEntry"]
