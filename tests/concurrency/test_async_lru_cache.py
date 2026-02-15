"""Tests for AsyncLRUCache and async_cached decorator."""

import asyncio
import time

import pytest

from nodetool.concurrency.async_lru_cache import (
    AsyncLRUCache,
    CacheStats,
    async_cached,
)


class TestAsyncLRUCache:
    """Tests for AsyncLRUCache class."""

    @pytest.mark.asyncio
    async def test_basic_set_and_get(self):
        """Test basic set and get operations."""
        cache = AsyncLRUCache[str, int](max_size=10)

        await cache.set("key1", 42)
        result = await cache.get("key1")

        assert result == 42
        assert cache.stats.hits == 1
        assert cache.stats.misses == 0
        assert cache.stats.size == 1

    @pytest.mark.asyncio
    async def test_get_missing_key(self):
        """Test getting a missing key returns None."""
        cache = AsyncLRUCache[str, int](max_size=10)

        result = await cache.get("nonexistent")

        assert result is None
        assert cache.stats.misses == 1

    @pytest.mark.asyncio
    async def test_delete_key(self):
        """Test deleting a key from cache."""
        cache = AsyncLRUCache[str, int](max_size=10)

        await cache.set("key1", 42)
        deleted = await cache.delete("key1")
        result = await cache.get("key1")

        assert deleted is True
        assert result is None
        assert cache.stats.size == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent_key(self):
        """Test deleting a nonexistent key returns False."""
        cache = AsyncLRUCache[str, int](max_size=10)

        deleted = await cache.delete("nonexistent")

        assert deleted is False

    @pytest.mark.asyncio
    async def test_clear_cache(self):
        """Test clearing all entries from cache."""
        cache = AsyncLRUCache[str, int](max_size=10)

        await cache.set("key1", 1)
        await cache.set("key2", 2)
        await cache.set("key3", 3)

        await cache.clear()

        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
        assert await cache.get("key3") is None
        assert cache.stats.size == 0

    @pytest.mark.asyncio
    async def test_contains_key(self):
        """Test checking if key exists in cache."""
        cache = AsyncLRUCache[str, int](max_size=10)

        await cache.set("key1", 42)

        assert await cache.contains("key1") is True
        assert await cache.contains("nonexistent") is False

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """Test that LRU eviction removes the least recently used entry."""
        cache = AsyncLRUCache[str, int](max_size=3)

        await cache.set("a", 1)
        await cache.set("b", 2)
        await cache.set("c", 3)

        # Access 'a' to make it more recently used
        await cache.get("a")

        # Add new item, should evict 'b' (LRU)
        await cache.set("d", 4)

        assert await cache.get("a") == 1  # Still present
        assert await cache.get("b") is None  # Evicted
        assert await cache.get("c") == 3  # Still present
        assert await cache.get("d") == 4  # New item
        assert cache.stats.evictions == 1

    @pytest.mark.asyncio
    async def test_update_existing_key(self):
        """Test updating an existing key moves it to most recent."""
        cache = AsyncLRUCache[str, int](max_size=3)

        await cache.set("a", 1)
        await cache.set("b", 2)
        await cache.set("c", 3)

        # Update 'a' to make it most recent
        await cache.set("a", 10)

        # Add new item, should evict 'b' (now LRU)
        await cache.set("d", 4)

        assert await cache.get("a") == 10  # Updated value
        assert await cache.get("b") is None  # Evicted
        assert await cache.get("c") == 3
        assert await cache.get("d") == 4

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Test that entries expire after TTL."""
        cache = AsyncLRUCache[str, int](max_size=10, ttl_seconds=0.1)

        await cache.set("key1", 42)

        # Should be present immediately
        assert await cache.get("key1") == 42

        # Wait for TTL to expire
        await asyncio.sleep(0.15)

        # Should be expired now
        result = await cache.get("key1")
        assert result is None
        assert cache.stats.evictions == 1

    @pytest.mark.asyncio
    async def test_contains_respects_ttl(self):
        """Test that contains() respects TTL."""
        cache = AsyncLRUCache[str, int](max_size=10, ttl_seconds=0.1)

        await cache.set("key1", 42)

        assert await cache.contains("key1") is True

        await asyncio.sleep(0.15)

        assert await cache.contains("key1") is False

    @pytest.mark.asyncio
    async def test_get_or_compute_caches_result(self):
        """Test get_or_compute caches the computed result."""
        cache = AsyncLRUCache[str, int](max_size=10)
        call_count = 0

        async def compute():
            nonlocal call_count
            call_count += 1
            return 42

        # First call should compute
        result1 = await cache.get_or_compute("key1", compute)
        assert result1 == 42
        assert call_count == 1

        # Second call should use cache
        result2 = await cache.get_or_compute("key1", compute)
        assert result2 == 42
        assert call_count == 1  # Not incremented

    @pytest.mark.asyncio
    async def test_get_or_compute_with_sync_function(self):
        """Test get_or_compute works with sync functions."""
        cache = AsyncLRUCache[str, int](max_size=10)
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return 42

        result = await cache.get_or_compute("key1", compute)
        assert result == 42
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_get_or_compute_prevents_thundering_herd(self):
        """Test that concurrent get_or_compute calls don't cause duplicate work."""
        cache = AsyncLRUCache[str, int](max_size=10)
        call_count = 0
        compute_event = asyncio.Event()

        async def compute():
            nonlocal call_count
            call_count += 1
            await compute_event.wait()
            return 42

        async def get_value():
            return await cache.get_or_compute("key1", compute)

        # Start multiple concurrent requests
        tasks = [get_value() for _ in range(5)]
        task_results = asyncio.gather(*tasks)

        # Let the computation complete
        compute_event.set()

        results = await task_results

        # All should get the same result
        assert all(r == 42 for r in results)
        # But computation should only run once
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_get_or_compute_propagates_exception(self):
        """Test that exceptions in compute are propagated."""
        cache = AsyncLRUCache[str, int](max_size=10)

        async def compute():
            raise ValueError("Compute failed")

        with pytest.raises(ValueError, match="Compute failed"):
            await cache.get_or_compute("key1", compute)

    @pytest.mark.asyncio
    async def test_get_many(self):
        """Test getting multiple values at once."""
        cache = AsyncLRUCache[str, int](max_size=10)

        await cache.set("a", 1)
        await cache.set("b", 2)
        await cache.set("c", 3)

        results = await cache.get_many(["a", "b", "d"])

        assert results == {"a": 1, "b": 2}
        # 'd' should not be in results (missing)

    @pytest.mark.asyncio
    async def test_set_many(self):
        """Test setting multiple values at once."""
        cache = AsyncLRUCache[str, int](max_size=10)

        await cache.set_many({"a": 1, "b": 2, "c": 3})

        assert await cache.get("a") == 1
        assert await cache.get("b") == 2
        assert await cache.get("c") == 3
        assert cache.stats.size == 3

    @pytest.mark.asyncio
    async def test_delete_many(self):
        """Test deleting multiple values at once."""
        cache = AsyncLRUCache[str, int](max_size=10)

        await cache.set("a", 1)
        await cache.set("b", 2)
        await cache.set("c", 3)

        deleted = await cache.delete_many(["a", "b", "nonexistent"])

        assert deleted == 2
        assert await cache.get("a") is None
        assert await cache.get("b") is None
        assert await cache.get("c") == 3

    @pytest.mark.asyncio
    async def test_cache_stats_hit_rate(self):
        """Test cache hit rate calculation."""
        cache = AsyncLRUCache[str, int](max_size=10)

        await cache.set("key1", 42)

        # 3 hits
        await cache.get("key1")
        await cache.get("key1")
        await cache.get("key1")

        # 2 misses
        await cache.get("missing1")
        await cache.get("missing2")

        assert cache.stats.hits == 3
        assert cache.stats.misses == 2
        assert cache.stats.hit_rate == 60.0  # 3 / 5 * 100

    @pytest.mark.asyncio
    async def test_cache_stats_empty_hit_rate(self):
        """Test hit rate when no operations performed."""
        stats = CacheStats()

        assert stats.hit_rate == 0.0

    @pytest.mark.asyncio
    async def test_invalid_max_size(self):
        """Test that invalid max_size raises ValueError."""
        with pytest.raises(ValueError, match="max_size must be positive"):
            AsyncLRUCache[str, int](max_size=0)

        with pytest.raises(ValueError, match="max_size must be positive"):
            AsyncLRUCache[str, int](max_size=-1)

    @pytest.mark.asyncio
    async def test_invalid_ttl(self):
        """Test that invalid TTL raises ValueError."""
        with pytest.raises(ValueError, match="ttl_seconds must be positive"):
            AsyncLRUCache[str, int](max_size=10, ttl_seconds=0)

        with pytest.raises(ValueError, match="ttl_seconds must be positive"):
            AsyncLRUCache[str, int](max_size=10, ttl_seconds=-1)

    @pytest.mark.asyncio
    async def test_properties(self):
        """Test cache properties."""
        cache = AsyncLRUCache[str, int](max_size=100, ttl_seconds=60.0)

        assert cache.max_size == 100
        assert cache.ttl_seconds == 60.0
        assert isinstance(cache.stats, CacheStats)

    @pytest.mark.asyncio
    async def test_different_key_types(self):
        """Test cache works with different key types."""
        # Integer keys
        int_cache = AsyncLRUCache[int, str](max_size=10)
        await int_cache.set(1, "one")
        assert await int_cache.get(1) == "one"

        # Tuple keys
        tuple_cache = AsyncLRUCache[tuple, str](max_size=10)
        await tuple_cache.set((1, 2), "tuple_key")
        assert await tuple_cache.get((1, 2)) == "tuple_key"

    @pytest.mark.asyncio
    async def test_ttl_expiration_on_contains(self):
        """Test that contains properly evicts expired entries."""
        cache = AsyncLRUCache[str, int](max_size=10, ttl_seconds=0.1)

        await cache.set("key1", 42)
        assert cache.stats.size == 1

        await asyncio.sleep(0.15)

        # contains() should evict expired entry
        result = await cache.contains("key1")
        assert result is False
        assert cache.stats.evictions == 1
        assert cache.stats.size == 0


class TestAsyncCachedDecorator:
    """Tests for async_cached decorator."""

    @pytest.mark.asyncio
    async def test_basic_decorator(self):
        """Test basic decorator usage."""
        call_count = 0

        @async_cached(max_size=10)
        async def expensive_func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call
        result1 = await expensive_func(5)
        assert result1 == 10
        assert call_count == 1

        # Second call (cached)
        result2 = await expensive_func(5)
        assert result2 == 10
        assert call_count == 1  # Not incremented

        # Different argument
        result3 = await expensive_func(10)
        assert result3 == 20
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_decorator_with_ttl(self):
        """Test decorator with TTL."""
        call_count = 0

        @async_cached(max_size=10, ttl_seconds=0.1)
        async def timed_func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = await timed_func(5)
        assert result1 == 10
        assert call_count == 1

        await asyncio.sleep(0.15)

        # Should recompute after TTL
        result2 = await timed_func(5)
        assert result2 == 10
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_decorator_with_custom_key_func(self):
        """Test decorator with custom key function."""
        call_count = 0

        def make_key(user_id: str, **kwargs) -> str:
            return f"user:{user_id}"

        @async_cached(max_size=10, key_func=make_key)
        async def fetch_user(user_id: str, extra: str = "") -> str:
            nonlocal call_count
            call_count += 1
            return f"user_{user_id}{extra}"

        result1 = await fetch_user("123", extra="_admin")
        assert result1 == "user_123_admin"
        assert call_count == 1

        # Same user_id, different extra - should use cache due to custom key
        result2 = await fetch_user("123", extra="_guest")
        assert result2 == "user_123_admin"  # Cached result
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_decorator_cache_clear(self):
        """Test that decorated function has cache_clear method."""
        call_count = 0

        @async_cached(max_size=10)
        async def func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x

        await func(1)
        assert call_count == 1

        # Clear cache
        await func.cache_clear()

        # Should recompute
        await func(1)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_decorator_cache_stats(self):
        """Test that decorated function exposes cache_stats."""
        @async_cached(max_size=10)
        async def func(x: int) -> int:
            return x

        await func(1)
        await func(1)
        await func(2)

        stats = func.cache_stats
        assert stats.hits >= 1
        assert stats.misses >= 1

    @pytest.mark.asyncio
    async def test_decorator_preserves_function_name(self):
        """Test that decorator preserves function metadata."""
        @async_cached(max_size=10)
        async def my_special_function(x: int) -> int:
            """This is my special function."""
            return x

        assert my_special_function.__name__ == "my_special_function"
        assert "special function" in my_special_function.__doc__

    @pytest.mark.asyncio
    async def test_decorator_with_kwargs(self):
        """Test decorator with keyword arguments."""
        call_count = 0

        @async_cached(max_size=10)
        async def func(a: int, b: int = 0, c: int = 0) -> int:
            nonlocal call_count
            call_count += 1
            return a + b + c

        result1 = await func(1, b=2, c=3)
        assert result1 == 6
        assert call_count == 1

        # Same arguments, different order shouldn't matter for cache
        result2 = await func(1, c=3, b=2)
        assert result2 == 6
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_decorator_concurrent_calls(self):
        """Test decorator handles concurrent calls correctly."""
        call_count = 0
        compute_delay = asyncio.Event()

        @async_cached(max_size=10)
        async def slow_func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            await compute_delay.wait()
            return x * 2

        async def call_func():
            return await slow_func(5)

        # Start concurrent calls
        tasks = [call_func() for _ in range(10)]
        gathered = asyncio.gather(*tasks)

        # Let computation complete
        compute_delay.set()

        results = await gathered

        # All should get the same result
        assert all(r == 10 for r in results)
        # But only one computation should have run
        assert call_count == 1
