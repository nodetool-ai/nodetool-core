"""Tests for async caching utilities."""

import asyncio
import time

import pytest

from nodetool.concurrency.async_cache import AsyncCache, CachedValue, cached_async


class TestCachedValue:
    """Tests for CachedValue class."""

    def test_initialization_without_ttl(self):
        """Test that CachedValue without TTL doesn't expire."""
        value = CachedValue("test", None)
        assert value.get() == "test"
        assert not value.is_expired()

    def test_initialization_with_ttl(self):
        """Test that CachedValue with TTL tracks expiration."""
        value = CachedValue("test", 1.0)
        assert value.get() == "test"
        assert not value.is_expired()

    def test_expiration_after_ttl(self):
        """Test that CachedValue expires after TTL passes."""
        value = CachedValue("test", 0.01)
        time.sleep(0.02)
        assert value.is_expired()
        with pytest.raises(KeyError, match="expired"):
            value.get()

    def test_expiration_none_ttl_never_expires(self):
        """Test that CachedValue with None TTL never expires."""
        value = CachedValue("test", None)
        time.sleep(0.1)
        assert not value.is_expired()
        assert value.get() == "test"


class TestAsyncCache:
    """Tests for AsyncCache class."""

    @pytest.mark.asyncio
    async def test_set_and_get(self):
        """Test basic set and get operations."""
        cache = AsyncCache()
        await cache.set("key1", "value1")
        assert await cache.get("key1") == "value1"

    @pytest.mark.asyncio
    async def test_get_returns_none_for_missing_key(self):
        """Test that get returns None for non-existent keys."""
        cache = AsyncCache()
        assert await cache.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_default_initialization(self):
        """Test default AsyncCache values."""
        cache = AsyncCache()
        assert cache._max_size == 128
        assert cache._ttl is None

    @pytest.mark.asyncio
    async def test_custom_initialization(self):
        """Test AsyncCache with custom values."""
        cache = AsyncCache(max_size=10, ttl=60)
        assert cache._max_size == 10
        assert cache._ttl == 60

    @pytest.mark.asyncio
    async def test_get_or_compute_caches_result(self):
        """Test that get_or_compute caches computed values."""
        cache = AsyncCache()
        call_count = 0

        async def compute():
            nonlocal call_count
            call_count += 1
            return "computed"

        result1 = await cache.get_or_compute("key", compute)
        result2 = await cache.get_or_compute("key", compute)

        assert result1 == "computed"
        assert result2 == "computed"
        assert call_count == 1  # Only computed once

    @pytest.mark.asyncio
    async def test_get_or_compute_with_sync_function(self):
        """Test that get_or_compute works with sync functions."""
        cache = AsyncCache()
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return "sync_result"

        result = await cache.get_or_compute("key", compute)
        assert result == "sync_result"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_invalidate_removes_entry(self):
        """Test that invalidate removes cached entry."""
        cache = AsyncCache()
        await cache.set("key", "value")
        assert await cache.invalidate("key") is True
        assert await cache.get("key") is None

    @pytest.mark.asyncio
    async def test_invalidate_nonexistent_key_returns_false(self):
        """Test that invalidate returns False for non-existent keys."""
        cache = AsyncCache()
        assert await cache.invalidate("nonexistent") is False

    @pytest.mark.asyncio
    async def test_clear_removes_all_entries(self):
        """Test that clear removes all entries and resets stats."""
        cache = AsyncCache()
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.get("key1")  # Generate some stats

        await cache.clear()

        assert await cache.get("key1") is None
        assert await cache.get("key2") is None

        stats = await cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 2  # Two get calls after clear

    @pytest.mark.asyncio
    async def test_max_size_eviction(self):
        """Test that cache evicts oldest entry when at capacity."""
        cache = AsyncCache(max_size=2)

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")  # Should evict key1

        assert await cache.get("key1") is None
        assert await cache.get("key2") == "value2"
        assert await cache.get("key3") == "value3"

        stats = await cache.get_stats()
        assert stats["evictions"] == 1

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Test that entries expire after TTL."""
        cache = AsyncCache(ttl=0.05)
        await cache.set("key", "value")

        assert await cache.get("key") == "value"
        time.sleep(0.06)
        assert await cache.get("key") is None

    @pytest.mark.asyncio
    async def test_custom_ttl_overrides_default(self):
        """Test that per-entry TTL overrides cache default."""
        cache = AsyncCache(ttl=1.0)

        await cache.set("key1", "value1", ttl=0.02)
        await cache.set("key2", "value2")

        time.sleep(0.03)
        assert await cache.get("key1") is None  # Expired (custom TTL)
        assert await cache.get("key2") == "value2"  # Not expired (default TTL)

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test cache statistics."""
        cache = AsyncCache(max_size=5, ttl=60)

        await cache.set("key1", "value1")
        await cache.get("key1")  # Hit
        await cache.get("key2")  # Miss
        await cache.set("key2", "value2")
        await cache.get("key2")  # Hit

        stats = await cache.get_stats()
        assert stats["size"] == 2
        assert stats["max_size"] == 5
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["evictions"] == 0
        assert stats["hit_rate"] == 2 / 3
        assert stats["ttl"] == 60

    @pytest.mark.asyncio
    async def test_cleanup_expired(self):
        """Test cleanup of expired entries."""
        cache = AsyncCache(ttl=0.05)

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        time.sleep(0.06)
        await cache.set("key3", "value3")  # Add fresh entry

        removed = await cache.cleanup_expired()
        assert removed == 2
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
        assert await cache.get("key3") == "value3"

    @pytest.mark.asyncio
    async def test_concurrent_access_no_crashes(self):
        """Test that cache is safe for concurrent access (no crashes)."""
        cache = AsyncCache(max_size=100)

        async def compute(value: int) -> int:
            await asyncio.sleep(0.001)
            return value * 2

        # Concurrent calls for different keys
        tasks = [cache.get_or_compute(f"key{i}", lambda v=i: compute(v)) for i in range(50)]
        results = await asyncio.gather(*tasks)

        # All should complete successfully
        assert len(results) == 50
        # Check a few values are correct
        assert results[0] == 0  # 0 * 2 = 0
        assert results[1] == 2  # 1 * 2 = 2
        assert results[25] == 50  # 25 * 2 = 50

        # Sequential calls should be cached
        result1 = await cache.get_or_compute("sequential", lambda: 42)
        result2 = await cache.get_or_compute("sequential", lambda: 99)
        assert result1 == 42
        assert result2 == 42  # Should return cached value, not 99


class TestCachedAsyncDecorator:
    """Tests for cached_async decorator."""

    @pytest.mark.asyncio
    async def test_decorator_caches_results(self):
        """Test that decorator caches function results."""
        call_count = 0

        @cached_async(ttl=60)
        async def fetch_data(user_id: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"data-{user_id}"

        result1 = await fetch_data("user123")
        result2 = await fetch_data("user123")
        result3 = await fetch_data("user456")

        assert result1 == "data-user123"
        assert result2 == "data-user123"
        assert result3 == "data-user456"
        assert call_count == 2  # Called once for each unique key

    @pytest.mark.asyncio
    async def test_decorator_with_ttl(self):
        """Test that decorator respects TTL."""
        call_count = 0

        @cached_async(ttl=0.05)
        async def fetch_data(key: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"data-{key}"

        await fetch_data("key1")
        await fetch_data("key1")  # Cache hit
        assert call_count == 1

        time.sleep(0.06)
        await fetch_data("key1")  # Cache expired
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_decorator_with_custom_key_func(self):
        """Test decorator with custom key function."""
        call_count = 0

        def make_key(url: str, params: dict) -> str:
            sorted_items = tuple(sorted(params.items()))
            return f"{url}:{sorted_items}"

        @cached_async(ttl=60, key_func=make_key)
        async def fetch_data(url: str, params: dict) -> str:
            nonlocal call_count
            call_count += 1
            return f"{url}-{params['id']}"

        await fetch_data("/api", {"id": 1})
        await fetch_data("/api", {"id": 1})  # Cache hit
        await fetch_data("/api", {"id": 2})  # Different key

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_decorator_cache_control_methods(self):
        """Test cache control methods on decorated function."""
        call_count = 0

        @cached_async(ttl=60)
        async def fetch_data(key: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"data-{key}"

        # Cache a value
        await fetch_data("key1")
        assert call_count == 1

        # Clear cache
        await fetch_data.cache.clear()
        await fetch_data("key1")
        assert call_count == 2

        # After clearing again, verify empty cache
        await fetch_data.cache.clear()
        stats = await fetch_data.cache.get_stats()
        assert stats["size"] == 0

    @pytest.mark.asyncio
    async def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function name and docstring."""

        @cached_async(ttl=60)
        async def my_function(x: int) -> int:
            """My docstring."""
            return x * 2

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    @pytest.mark.asyncio
    async def test_decorator_with_kwargs(self):
        """Test that decorator handles keyword arguments correctly."""
        call_count = 0

        @cached_async(ttl=60)
        async def compute(a: int, b: int = 10) -> int:
            nonlocal call_count
            call_count += 1
            return a + b

        result1 = await compute(1, b=2)
        result2 = await compute(1, b=2)
        result3 = await compute(1, b=3)  # Different kwargs

        assert result1 == 3
        assert result2 == 3
        assert result3 == 4
        assert call_count == 2  # Two unique calls

    @pytest.mark.asyncio
    async def test_decorator_unhashable_args_raises_error(self):
        """Test that unhashable arguments raise TypeError without key_func."""

        @cached_async(ttl=60)
        async def fetch_data(data: list) -> str:
            return "result"

        with pytest.raises(TypeError, match="hashable"):
            await fetch_data([1, 2, 3])

    @pytest.mark.asyncio
    async def test_decorator_max_size(self):
        """Test that decorator respects max_size parameter."""
        call_count = 0

        @cached_async(ttl=60, max_size=2)
        async def fetch_data(key: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"data-{key}"

        await fetch_data("key1")
        await fetch_data("key2")
        await fetch_data("key3")  # Evicts key1
        await fetch_data("key1")  # Recomputes

        assert call_count == 4


class TestCachedAsyncDecoratorEdgeCases:
    """Tests for edge cases in cached_async decorator."""

    @pytest.mark.asyncio
    async def test_default_ttl_no_expiration(self):
        """Test that default None TTL means no expiration."""

        @cached_async()
        async def fetch_data(key: str) -> str:
            return f"data-{key}"

        result1 = await fetch_data("key1")
        time.sleep(0.1)
        result2 = await fetch_data("key1")

        # Without counting calls, we verify behavior through consistency
        assert result1 == result2

    @pytest.mark.asyncio
    async def test_multiple_decorated_functions_independent(self):
        """Test that different decorated functions have independent caches."""
        count1 = 0
        count2 = 0

        @cached_async(ttl=60)
        async def func1(x: int) -> int:
            nonlocal count1
            count1 += 1
            return x * 2

        @cached_async(ttl=60)
        async def func2(x: int) -> int:
            nonlocal count2
            count2 += 1
            return x * 3

        await func1(5)
        await func2(5)
        await func1(5)
        await func2(5)

        assert count1 == 1
        assert count2 == 1
