"""Tests for AsyncCache functionality."""

import asyncio
import time

import pytest

from nodetool.concurrency import AsyncCache, async_cache


class TestAsyncCache:
    """Test suite for AsyncCache class."""

    @pytest.mark.asyncio
    async def test_set_and_get(self) -> None:
        """Test basic set and get operations."""
        cache = AsyncCache[str, int](max_size=10, ttl=60.0)
        await cache.set("key1", 42)
        value = await cache.get("key1")
        assert value == 42

    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self) -> None:
        """Test getting a key that doesn't exist."""
        cache = AsyncCache[str, int]()
        value = await cache.get("nonexistent")
        assert value is None

    @pytest.mark.asyncio
    async def test_delete_key(self) -> None:
        """Test deleting a key."""
        cache = AsyncCache[str, int]()
        await cache.set("key1", 42)
        deleted = await cache.delete("key1")
        assert deleted is True
        value = await cache.get("key1")
        assert value is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_key(self) -> None:
        """Test deleting a key that doesn't exist."""
        cache = AsyncCache[str, int]()
        deleted = await cache.delete("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_clear_cache(self) -> None:
        """Test clearing the entire cache."""
        cache = AsyncCache[str, int]()
        await cache.set("key1", 42)
        await cache.set("key2", 100)
        await cache.clear()
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
        assert len(cache) == 0

    @pytest.mark.asyncio
    async def test_expiration(self) -> None:
        """Test that entries expire after TTL."""
        cache = AsyncCache[str, int](ttl=0.1)  # 100ms TTL
        await cache.set("key1", 42)
        assert await cache.get("key1") == 42
        await asyncio.sleep(0.15)
        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_custom_ttl(self) -> None:
        """Test custom TTL per entry."""
        cache = AsyncCache[str, int](ttl=10.0)  # Long default TTL
        await cache.set("key1", 42, ttl=0.1)  # Override with short TTL
        assert await cache.get("key1") == 42
        await asyncio.sleep(0.15)
        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_lru_eviction(self) -> None:
        """Test LRU eviction when cache is full."""
        cache = AsyncCache[str, int](max_size=3)
        await cache.set("key1", 1)
        await cache.set("key2", 2)
        await cache.set("key3", 3)
        await cache.set("key4", 4)  # Should evict key1
        assert await cache.get("key1") is None
        assert await cache.get("key2") == 2
        assert await cache.get("key3") == 3
        assert await cache.get("key4") == 4

    @pytest.mark.asyncio
    async def test_lru_updates_on_access(self) -> None:
        """Test that accessing an entry updates its LRU position."""
        cache = AsyncCache[str, int](max_size=3)
        await cache.set("key1", 1)
        await cache.set("key2", 2)
        await cache.set("key3", 3)
        # Access key1 to make it recently used
        await cache.get("key1")
        await cache.set("key4", 4)  # Should evict key2 (least recently used)
        assert await cache.get("key1") == 1
        assert await cache.get("key2") is None
        assert await cache.get("key3") == 3
        assert await cache.get("key4") == 4

    @pytest.mark.asyncio
    async def test_update_existing_key(self) -> None:
        """Test updating an existing key."""
        cache = AsyncCache[str, int]()
        await cache.set("key1", 42)
        await cache.set("key1", 100)
        assert await cache.get("key1") == 100
        # Should still only have one entry
        assert await cache.size() == 1

    @pytest.mark.asyncio
    async def test_size(self) -> None:
        """Test getting cache size."""
        cache = AsyncCache[str, int]()
        assert await cache.size() == 0
        await cache.set("key1", 1)
        assert await cache.size() == 1
        await cache.set("key2", 2)
        await cache.set("key3", 3)
        assert await cache.size() == 3

    @pytest.mark.asyncio
    async def test_stats(self) -> None:
        """Test cache statistics."""
        cache = AsyncCache[str, int](max_size=100, ttl=60.0)
        await cache.set("key1", 42)
        await cache.get("key1")  # Hit
        await cache.get("key2")  # Miss

        stats = await cache.stats()
        assert stats["size"] == 1
        assert stats["max_size"] == 100
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["evictions"] == 0
        assert stats["hit_rate"] == 0.5
        assert stats["default_ttl"] == 60.0

    @pytest.mark.asyncio
    async def test_cleanup_expired(self) -> None:
        """Test cleanup of expired entries."""
        cache = AsyncCache[str, int](ttl=0.1)
        await cache.set("key1", 1)
        await cache.set("key2", 2)
        await cache.set("key3", 3, ttl=1.0)  # Longer TTL

        await asyncio.sleep(0.15)
        removed = await cache.cleanup_expired()
        assert removed == 2
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
        assert await cache.get("key3") == 3

    @pytest.mark.asyncio
    async def test_get_or_compute_with_async_function(self) -> None:
        """Test get_or_compute with async compute function."""
        cache = AsyncCache[str, int]()
        compute_count = 0

        async def compute_value() -> int:
            nonlocal compute_count
            compute_count += 1
            await asyncio.sleep(0.01)
            return 42

        # First call computes
        result1 = await cache.get_or_compute("key1", compute_value)
        assert result1 == 42
        assert compute_count == 1

        # Second call uses cache
        result2 = await cache.get_or_compute("key1", compute_value)
        assert result2 == 42
        assert compute_count == 1  # Not called again

    @pytest.mark.asyncio
    async def test_get_or_compute_with_sync_function(self) -> None:
        """Test get_or_compute with sync compute function."""
        cache = AsyncCache[str, int]()
        compute_count = 0

        def compute_value() -> int:
            nonlocal compute_count
            compute_count += 1
            return 42

        # First call computes
        result1 = await cache.get_or_compute("key1", compute_value)
        assert result1 == 42
        assert compute_count == 1

        # Second call uses cache
        result2 = await cache.get_or_compute("key1", compute_value)
        assert result2 == 42
        assert compute_count == 1

    @pytest.mark.asyncio
    async def test_get_or_compute_with_custom_ttl(self) -> None:
        """Test get_or_compute with custom TTL."""
        cache = AsyncCache[str, int](ttl=10.0)

        async def compute_value() -> int:
            return 42

        result1 = await cache.get_or_compute("key1", compute_value, ttl=0.1)
        assert result1 == 42
        await asyncio.sleep(0.15)

        # Should recompute due to expiration
        result2 = await cache.get_or_compute("key1", compute_value, ttl=0.1)
        assert result2 == 42

    @pytest.mark.asyncio
    async def test_len_dunder(self) -> None:
        """Test __len__ method."""
        cache = AsyncCache[str, int]()
        assert len(cache) == 0
        await cache.set("key1", 1)
        assert len(cache) == 1


class TestAsyncCacheDecorator:
    """Test suite for async_cache decorator."""

    @pytest.mark.asyncio
    async def test_decorator_caches_results(self) -> None:
        """Test that decorator caches function results."""
        call_count = 0

        @async_cache(max_size=10, ttl=60.0)
        async def expensive_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return x * 2

        # First call
        result1 = await expensive_function(5)
        assert result1 == 10
        assert call_count == 1

        # Second call with same args (cached)
        result2 = await expensive_function(5)
        assert result2 == 10
        assert call_count == 1

        # Third call with different args
        result3 = await expensive_function(10)
        assert result3 == 20
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_decorator_with_kwargs(self) -> None:
        """Test decorator with keyword arguments."""
        call_count = 0

        @async_cache(max_size=10, ttl=60.0)
        async def func(a: int, b: int = 5) -> int:
            nonlocal call_count
            call_count += 1
            return a + b

        # Call with default arg
        result1 = await func(10)
        assert result1 == 15
        assert call_count == 1

        # Call with explicit arg (should have different cache key)
        result2 = await func(10, b=5)
        assert result2 == 15
        # Note: Due to sorted kwargs, these might have same key

    @pytest.mark.asyncio
    async def test_decorator_cache_clear(self) -> None:
        """Test clearing cache via decorated function."""
        call_count = 0

        @async_cache(max_size=10, ttl=60.0)
        async def func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        await func(5)
        await func(5)
        assert call_count == 1

        await func.cache_clear()  # type: ignore[attr-defined]

        await func(5)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_decorator_cache_stats(self) -> None:
        """Test getting stats via decorated function."""
        @async_cache(max_size=100, ttl=60.0)
        async def func(x: int) -> int:
            return x * 2

        await func(5)
        await func(5)
        await func(10)

        stats = await func.cache_stats()  # type: ignore[attr-defined]
        assert stats["hits"] == 1
        assert stats["misses"] == 2

    @pytest.mark.asyncio
    async def test_decorator_with_expiration(self) -> None:
        """Test that decorator cache respects TTL."""
        call_count = 0

        @async_cache(max_size=10, ttl=0.1)
        async def func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        await func(5)
        assert call_count == 1

        await asyncio.sleep(0.15)

        await func(5)
        assert call_count == 2  # Recomputed after expiration
