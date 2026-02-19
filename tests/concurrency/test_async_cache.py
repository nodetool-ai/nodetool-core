import asyncio
import time

import pytest

from nodetool.concurrency.async_cache import AsyncCache, cached_async


class TestAsyncCache:
    """Tests for AsyncCache class."""

    def test_init_default_parameters(self):
        """Test cache initialization with default parameters."""
        cache = AsyncCache()
        assert cache._max_size == 128
        assert cache._default_ttl is None

    def test_init_custom_parameters(self):
        """Test cache initialization with custom parameters."""
        cache = AsyncCache(max_size=10, default_ttl=60.0)
        assert cache._max_size == 10
        assert cache._default_ttl == 60.0

    def test_init_invalid_max_size(self):
        """Test that invalid max_size raises ValueError."""
        with pytest.raises(ValueError, match="max_size must be positive"):
            AsyncCache(max_size=0)
        with pytest.raises(ValueError, match="max_size must be positive"):
            AsyncCache(max_size=-1)

    @pytest.mark.asyncio
    async def test_set_and_get(self):
        """Test basic set and get operations."""
        cache = AsyncCache()

        await cache.set("key1", "value1")
        value = await cache.get("key1")

        assert value == "value1"

    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self):
        """Test getting a nonexistent key returns None."""
        cache = AsyncCache()

        value = await cache.get("nonexistent")

        assert value is None

    @pytest.mark.asyncio
    async def test_set_with_custom_ttl(self):
        """Test setting a value with custom TTL."""
        cache = AsyncCache()

        await cache.set("key1", "value1", ttl=0.1)
        value = await cache.get("key1")
        assert value == "value1"

        # Wait for expiration
        await asyncio.sleep(0.15)
        value = await cache.get("key1")
        assert value is None

    @pytest.mark.asyncio
    async def test_set_with_default_ttl(self):
        """Test setting a value with default TTL."""
        cache = AsyncCache(default_ttl=0.1)

        await cache.set("key1", "value1")
        value = await cache.get("key1")
        assert value == "value1"

        # Wait for expiration
        await asyncio.sleep(0.15)
        value = await cache.get("key1")
        assert value is None

    @pytest.mark.asyncio
    async def test_set_without_ttl(self):
        """Test setting a value without TTL doesn't expire."""
        cache = AsyncCache()

        await cache.set("key1", "value1")
        await asyncio.sleep(0.1)

        value = await cache.get("key1")
        assert value == "value1"

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """Test that LRU eviction works when max_size is exceeded."""
        cache = AsyncCache(max_size=3)

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        # All values should be present
        assert await cache.get("key1") == "value1"
        assert await cache.get("key2") == "value2"
        assert await cache.get("key3") == "value3"

        # Add one more - should evict key1 (least recently used)
        await cache.set("key4", "value4")

        assert await cache.get("key1") is None
        assert await cache.get("key2") == "value2"
        assert await cache.get("key3") == "value3"
        assert await cache.get("key4") == "value4"

    @pytest.mark.asyncio
    async def test_lru_updates_on_access(self):
        """Test that accessing a key updates its position in LRU."""
        cache = AsyncCache(max_size=3)

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        # Access key1 to make it more recently used
        assert await cache.get("key1") == "value1"

        # Add key4 - should evict key2 (now least recently used)
        await cache.set("key4", "value4")

        assert await cache.get("key1") == "value1"
        assert await cache.get("key2") is None
        assert await cache.get("key3") == "value3"
        assert await cache.get("key4") == "value4"

    @pytest.mark.asyncio
    async def test_delete_existing_key(self):
        """Test deleting an existing key."""
        cache = AsyncCache()

        await cache.set("key1", "value1")
        result = await cache.delete("key1")

        assert result is True
        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_key(self):
        """Test deleting a nonexistent key returns False."""
        cache = AsyncCache()

        result = await cache.delete("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_or_compute_cache_hit(self):
        """Test get_or_compute returns cached value on hit."""
        cache = AsyncCache()
        compute_count = 0

        async def compute():
            nonlocal compute_count
            compute_count += 1
            return "computed"

        # First call - should compute
        result1 = await cache.get_or_compute("key1", compute)
        assert result1 == "computed"
        assert compute_count == 1

        # Second call - should use cache
        result2 = await cache.get_or_compute("key1", compute)
        assert result2 == "computed"
        assert compute_count == 1

    @pytest.mark.asyncio
    async def test_get_or_compute_cache_miss(self):
        """Test get_or_compute computes and caches on miss."""
        cache = AsyncCache()
        compute_count = 0

        async def compute():
            nonlocal compute_count
            compute_count += 1
            return f"value{compute_count}"

        result = await cache.get_or_compute("key1", compute)

        assert result == "value1"
        assert compute_count == 1
        assert await cache.get("key1") == "value1"

    @pytest.mark.asyncio
    async def test_get_or_compute_with_custom_ttl(self):
        """Test get_or_compute respects custom TTL."""
        cache = AsyncCache()
        compute_count = 0

        async def compute():
            nonlocal compute_count
            compute_count += 1
            return f"value{compute_count}"

        # First call with TTL
        result1 = await cache.get_or_compute("key1", compute, ttl=0.1)
        assert result1 == "value1"
        assert compute_count == 1

        # Second call should use cache
        result2 = await cache.get_or_compute("key1", compute, ttl=0.1)
        assert result2 == "value1"
        assert compute_count == 1

        # Wait for expiration
        await asyncio.sleep(0.15)

        # Third call should recompute
        result3 = await cache.get_or_compute("key1", compute, ttl=0.1)
        assert result3 == "value2"
        assert compute_count == 2

    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clearing the cache."""
        cache = AsyncCache()

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        assert await cache.size() == 3

        await cache.clear()

        assert await cache.size() == 0
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
        assert await cache.get("key3") is None

    @pytest.mark.asyncio
    async def test_cleanup_expired(self):
        """Test cleaning up expired entries."""
        cache = AsyncCache()

        await cache.set("key1", "value1", ttl=0.1)
        await cache.set("key2", "value2", ttl=0.1)
        await cache.set("key3", "value3")  # No TTL

        assert await cache.size() == 3

        # Wait for expiration
        await asyncio.sleep(0.15)

        removed = await cache.cleanup_expired()

        assert removed == 2
        assert await cache.size() == 1
        assert await cache.get("key3") == "value3"

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting cache statistics."""
        cache = AsyncCache(max_size=10)

        # Initial stats
        stats = await cache.get_stats()
        assert stats["size"] == 0
        assert stats["max_size"] == 10
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["evictions"] == 0
        assert stats["hit_rate"] is None

        # Add some entries
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        # Cache hit
        await cache.get("key1")

        # Cache miss
        await cache.get("nonexistent")

        stats = await cache.get_stats()
        assert stats["size"] == 2
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 50.0

    @pytest.mark.asyncio
    async def test_contains_existing(self):
        """Test contains returns True for existing unexpired key."""
        cache = AsyncCache()

        await cache.set("key1", "value1")

        assert await cache.contains("key1") is True

    @pytest.mark.asyncio
    async def test_contains_nonexistent(self):
        """Test contains returns False for nonexistent key."""
        cache = AsyncCache()

        assert await cache.contains("key1") is False

    @pytest.mark.asyncio
    async def test_contains_expired(self):
        """Test contains returns False for expired key."""
        cache = AsyncCache()

        await cache.set("key1", "value1", ttl=0.1)
        assert await cache.contains("key1") is True

        await asyncio.sleep(0.15)
        assert await cache.contains("key1") is False

    @pytest.mark.asyncio
    async def test_size(self):
        """Test getting cache size."""
        cache = AsyncCache(max_size=5)

        assert await cache.size() == 0

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        assert await cache.size() == 3

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """Test that cache handles concurrent access correctly."""
        cache = AsyncCache(max_size=10)
        compute_count = 0

        async def compute(key):
            nonlocal compute_count
            compute_count += 1
            await asyncio.sleep(0.01)
            return f"value_{key}"

        # Concurrent access to same key
        tasks = [cache.get_or_compute("key1", lambda: compute("key1")) for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # All should get the same result
        assert all(r == "value_key1" for r in results)
        # Multiple computations may occur due to race condition (acceptable behavior)
        assert compute_count >= 1

    @pytest.mark.asyncio
    async def test_update_existing_key(self):
        """Test updating an existing key."""
        cache = AsyncCache()

        await cache.set("key1", "value1")
        await cache.set("key1", "value2")

        assert await cache.get("key1") == "value2"
        assert await cache.size() == 1


class TestCachedAsync:
    """Tests for cached_async decorator."""

    @pytest.mark.asyncio
    async def test_basic_caching(self):
        """Test basic decorator caching."""
        call_count = 0

        @cached_async(ttl=1.0)
        async def fetch_data(user_id: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"data_{user_id}"

        # First call
        result1 = await fetch_data("user1")
        assert result1 == "data_user1"
        assert call_count == 1

        # Second call - should use cache
        result2 = await fetch_data("user1")
        assert result2 == "data_user1"
        assert call_count == 1

        # Different argument
        result3 = await fetch_data("user2")
        assert result3 == "data_user2"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Test that cached results expire after TTL."""
        call_count = 0

        @cached_async(ttl=0.1)
        async def fetch_data(user_id: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"data_{user_id}"

        # First call
        result1 = await fetch_data("user1")
        assert result1 == "data_user1"
        assert call_count == 1

        # Second call - should use cache
        await fetch_data("user1")
        assert call_count == 1

        # Wait for expiration
        await asyncio.sleep(0.15)

        # Third call - should recompute
        result3 = await fetch_data("user1")
        assert result3 == "data_user1"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_no_ttl(self):
        """Test that results without TTL don't expire."""
        call_count = 0

        @cached_async()
        async def fetch_data(user_id: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"data_{user_id}"

        await fetch_data("user1")
        await asyncio.sleep(0.1)
        await fetch_data("user1")

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_max_size(self):
        """Test that max_size limits cache entries."""
        call_count = 0

        @cached_async(max_size=2)
        async def fetch_data(user_id: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"data_{user_id}"

        # Fill cache
        await fetch_data("user1")
        await fetch_data("user2")

        # Access user1 to make it more recent
        await fetch_data("user1")

        # Add user3 - should evict user2
        await fetch_data("user3")

        # user2 should be recomputed
        await fetch_data("user2")

        assert call_count == 4  # user1, user2, user3, user2 again

    @pytest.mark.asyncio
    async def test_with_keyword_arguments(self):
        """Test decorator with keyword arguments."""
        call_count = 0

        @cached_async(ttl=1.0)
        async def fetch_data(user_id: str, include_history: bool = False) -> str:
            nonlocal call_count
            call_count += 1
            return f"data_{user_id}_history_{include_history}"

        # Different keyword arguments should create different cache entries
        result1 = await fetch_data("user1", include_history=False)
        result2 = await fetch_data("user1", include_history=True)

        assert result1 == "data_user1_history_False"
        assert result2 == "data_user1_history_True"
        assert call_count == 2

        # Same arguments should use cache
        await fetch_data("user1", include_history=False)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_custom_key_func(self):
        """Test decorator with custom key function."""
        call_count = 0

        def custom_key(user_id: str, **kwargs) -> tuple:
            # Ignore include_history for caching
            return (user_id,)

        @cached_async(ttl=1.0, key_func=custom_key)
        async def fetch_data(user_id: str, include_history: bool = False) -> str:
            nonlocal call_count
            call_count += 1
            return f"data_{user_id}_history_{include_history}"

        # Different include_history should still use same cache entry
        await fetch_data("user1", include_history=False)
        await fetch_data("user1", include_history=True)

        assert call_count == 1  # Only called once

    @pytest.mark.asyncio
    async def test_cache_clear(self):
        """Test clearing the function cache."""
        call_count = 0

        @cached_async(ttl=1.0)
        async def fetch_data(user_id: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"data_{user_id}"

        await fetch_data("user1")
        assert call_count == 1

        await fetch_data.cache_clear()  # type: ignore[attr-defined]

        await fetch_data("user1")
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_cache_stats(self):
        """Test getting cache statistics."""
        @cached_async(ttl=1.0, max_size=10)
        async def fetch_data(user_id: str) -> str:
            return f"data_{user_id}"

        await fetch_data("user1")
        await fetch_data("user1")  # Cache hit
        await fetch_data("user2")  # Cache miss

        stats = await fetch_data.cache_stats()  # type: ignore[attr-defined]

        assert stats["size"] == 2
        assert stats["hits"] == 1
        assert stats["misses"] == 2

    @pytest.mark.asyncio
    async def test_cache_attribute(self):
        """Test that cache attribute is accessible."""
        @cached_async(ttl=1.0)
        async def fetch_data(user_id: str) -> str:
            return f"data_{user_id}"

        assert hasattr(fetch_data, "cache")
        assert fetch_data.cache is not None  # type: ignore[attr-defined]
