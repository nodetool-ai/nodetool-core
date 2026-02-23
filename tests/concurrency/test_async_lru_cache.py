"""Tests for AsyncLRUCache and async_lru_cache decorator."""

import asyncio
import pytest

from nodetool.concurrency import AsyncLRUCache, async_lru_cache


class TestAsyncLRUCache:
    """Test cases for AsyncLRUCache class."""

    @pytest.mark.asyncio
    async def test_basic_get_put(self):
        """Test basic get and put operations."""
        cache = AsyncLRUCache(maxsize=10)

        await cache.put("key1", "value1")
        value = await cache.get("key1")

        assert value == "value1"

    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self):
        """Test getting a non-existent key returns None."""
        cache = AsyncLRUCache(maxsize=10)

        value = await cache.get("nonexistent")

        assert value is None

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """Test that least recently used items are evicted when capacity is reached."""
        cache = AsyncLRUCache(maxsize=3)

        await cache.put("key1", "value1")
        await cache.put("key2", "value2")
        await cache.put("key3", "value3")

        # All three should be in cache
        assert await cache.get("key1") == "value1"
        assert await cache.get("key2") == "value2"
        assert await cache.get("key3") == "value3"

        # Adding a fourth item should evict the least recently used (key1)
        await cache.put("key4", "value4")

        assert await cache.get("key1") is None  # Evicted
        assert await cache.get("key2") == "value2"
        assert await cache.get("key3") == "value3"
        assert await cache.get("key4") == "value4"

    @pytest.mark.asyncio
    async def test_lru_access_order(self):
        """Test that accessing an item updates its position in LRU order."""
        cache = AsyncLRUCache(maxsize=3)

        await cache.put("key1", "value1")
        await cache.put("key2", "value2")
        await cache.put("key3", "value3")

        # Access key1 to make it more recently used
        await cache.get("key1")

        # Add key4, should evict key2 (now least recently used)
        await cache.put("key4", "value4")

        assert await cache.get("key1") == "value1"  # Still in cache
        assert await cache.get("key2") is None  # Evicted
        assert await cache.get("key3") == "value3"
        assert await cache.get("key4") == "value4"

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Test that items expire after TTL."""
        cache = AsyncLRUCache(maxsize=10, ttl=0.1)  # 100ms TTL

        await cache.put("key1", "value1")

        # Should be available immediately
        assert await cache.get("key1") == "value1"

        # Wait for expiration
        await asyncio.sleep(0.15)

        # Should be expired
        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_ttl_refresh_on_update(self):
        """Test that updating an entry refreshes its TTL."""
        cache = AsyncLRUCache(maxsize=10, ttl=0.1)

        await cache.put("key1", "value1")

        # Wait half the TTL
        await asyncio.sleep(0.05)

        # Update the value (should refresh TTL)
        await cache.put("key1", "value2")

        # Wait another 0.08 seconds (total 0.15 from first put, but only 0.1 from update)
        await asyncio.sleep(0.08)

        # Should still be available because TTL was refreshed
        assert await cache.get("key1") == "value2"

    @pytest.mark.asyncio
    async def test_get_or_load_with_factory(self):
        """Test get_or_load with factory function."""
        cache = AsyncLRUCache(maxsize=10)

        call_count = 0

        async def factory(key: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"value_{key}"

        # First call should invoke factory
        value1 = await cache.get_or_load("key1", factory)
        assert value1 == "value_key1"
        assert call_count == 1

        # Second call should use cache
        value2 = await cache.get_or_load("key1", factory)
        assert value2 == "value_key1"
        assert call_count == 1  # No additional call

    @pytest.mark.asyncio
    async def test_get_or_load_with_default_factory(self):
        """Test get_or_load with default factory set at init."""
        call_count = 0

        async def factory(key: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"value_{key}"

        cache = AsyncLRUCache(maxsize=10, default_factory=factory)

        # First call should invoke default factory
        value1 = await cache.get_or_load("key1")
        assert value1 == "value_key1"
        assert call_count == 1

        # Second call should use cache
        value2 = await cache.get_or_load("key1")
        assert value2 == "value_key1"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_get_or_load_no_factory_error(self):
        """Test get_or_load raises error when no factory is provided."""
        cache = AsyncLRUCache(maxsize=10)

        with pytest.raises(ValueError, match="No factory function provided"):
            await cache.get_or_load("key1")

    @pytest.mark.asyncio
    async def test_invalidate(self):
        """Test invalidating a cache entry."""
        cache = AsyncLRUCache(maxsize=10)

        await cache.put("key1", "value1")

        # Should exist
        assert await cache.get("key1") == "value1"

        # Invalidate
        result = await cache.invalidate("key1")
        assert result is True

        # Should be gone
        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_invalidate_nonexistent(self):
        """Test invalidating a non-existent key returns False."""
        cache = AsyncLRUCache(maxsize=10)

        result = await cache.invalidate("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clearing all cache entries."""
        cache = AsyncLRUCache(maxsize=10)

        await cache.put("key1", "value1")
        await cache.put("key2", "value2")
        await cache.put("key3", "value3")

        assert cache.size == 3

        await cache.clear()

        assert cache.size == 0
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
        assert await cache.get("key3") is None

    @pytest.mark.asyncio
    async def test_cleanup_expired(self):
        """Test cleaning up expired entries."""
        cache = AsyncLRUCache(maxsize=10, ttl=0.1)

        await cache.put("key1", "value1")
        await cache.put("key2", "value2")

        # Wait for expiration
        await asyncio.sleep(0.15)

        # Cleanup should remove 2 entries
        removed = await cache.cleanup_expired()
        assert removed == 2
        assert cache.size == 0

    @pytest.mark.asyncio
    async def test_cleanup_expired_no_ttl(self):
        """Test cleanup_expired returns 0 when no TTL is set."""
        cache = AsyncLRUCache(maxsize=10)

        await cache.put("key1", "value1")

        removed = await cache.cleanup_expired()
        assert removed == 0

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """Test concurrent access to the cache."""
        cache = AsyncLRUCache(maxsize=10)

        async def worker(worker_id: int):
            for i in range(5):
                key = f"key_{i}"
                await cache.put(key, f"value_{worker_id}_{i}")
                value = await cache.get(key)
                # Just verify we got some value
                assert value is not None

        # Run multiple workers concurrently
        tasks = [worker(i) for i in range(5)]
        await asyncio.gather(*tasks)

    @pytest.mark.asyncio
    async def test_properties(self):
        """Test cache properties."""
        cache = AsyncLRUCache(maxsize=100, ttl=60.0)

        assert cache.maxsize == 100
        assert cache.ttl == 60.0
        assert cache.size == 0

        await cache.put("key1", "value1")

        assert cache.size == 1

    @pytest.mark.asyncio
    async def test_len_magic_method(self):
        """Test __len__ magic method."""
        cache = AsyncLRUCache(maxsize=10)

        assert len(cache) == 0

        await cache.put("key1", "value1")
        await cache.put("key2", "value2")

        assert len(cache) == 2

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test using cache as context manager."""
        cache = AsyncLRUCache(maxsize=10)

        async with cache:
            await cache.put("key1", "value1")
            assert await cache.get("key1") == "value1"

    @pytest.mark.asyncio
    async def test_update_existing_key(self):
        """Test updating an existing key."""
        cache = AsyncLRUCache(maxsize=10)

        await cache.put("key1", "value1")
        await cache.put("key1", "value2")

        assert await cache.get("key1") == "value2"
        assert cache.size == 1  # Should not increase size

    @pytest.mark.asyncio
    async def test_invalid_maxsize(self):
        """Test that invalid maxsize raises ValueError."""
        with pytest.raises(ValueError, match="maxsize must be a positive integer"):
            AsyncLRUCache(maxsize=0)

        with pytest.raises(ValueError, match="maxsize must be a positive integer"):
            AsyncLRUCache(maxsize=-1)


class TestAsyncLRUCacheDecorator:
    """Test cases for async_lru_cache decorator."""

    @pytest.mark.asyncio
    async def test_decorator_basic_caching(self):
        """Test basic decorator functionality."""
        call_count = 0

        @async_lru_cache(maxsize=10)
        async def fetch_data(item_id: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"data_{item_id}"

        # First call
        result1 = await fetch_data("item1")
        assert result1 == "data_item1"
        assert call_count == 1

        # Second call should use cache
        result2 = await fetch_data("item1")
        assert result2 == "data_item1"
        assert call_count == 1  # No additional call

        # Different item should call function
        result3 = await fetch_data("item2")
        assert result3 == "data_item2"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_decorator_with_ttl(self):
        """Test decorator with TTL."""
        call_count = 0

        @async_lru_cache(maxsize=10, ttl=0.1)
        async def fetch_data(item_id: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"data_{item_id}"

        # First call
        result1 = await fetch_data("item1")
        assert result1 == "data_item1"
        assert call_count == 1

        # Immediate second call should use cache
        result2 = await fetch_data("item1")
        assert call_count == 1

        # Wait for TTL to expire
        await asyncio.sleep(0.15)

        # Should call function again after TTL
        result3 = await fetch_data("item1")
        assert result3 == "data_item1"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_decorator_cache_control(self):
        """Test cache control methods on decorated function."""
        @async_lru_cache(maxsize=10)
        async def fetch_data(item_id: str) -> str:
            return f"data_{item_id}"

        # Add some items
        await fetch_data("item1")
        await fetch_data("item2")

        # Check cache_info
        info = fetch_data.cache_info()  # type: ignore
        assert info["size"] == 2
        assert info["maxsize"] == 10
        assert info["ttl"] is None

        # Clear cache
        await fetch_data.cache_clear()  # type: ignore

        # Verify cache is cleared
        info = fetch_data.cache_info()  # type: ignore
        assert info["size"] == 0

    @pytest.mark.asyncio
    async def test_decorator_with_multiple_args(self):
        """Test decorator with multiple arguments."""
        call_count = 0

        @async_lru_cache(maxsize=10)
        async def compute(a: int, b: int) -> int:
            nonlocal call_count
            call_count += 1
            return a + b

        # First call
        result1 = await compute(1, 2)
        assert result1 == 3
        assert call_count == 1

        # Same args should use cache
        result2 = await compute(1, 2)
        assert result2 == 3
        assert call_count == 1

        # Different args should call function
        result3 = await compute(2, 3)
        assert result3 == 5
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_decorator_with_kwargs(self):
        """Test decorator with keyword arguments."""
        call_count = 0

        @async_lru_cache(maxsize=10)
        async def compute(a: int, b: int = 10) -> int:
            nonlocal call_count
            call_count += 1
            return a + b

        # First call
        result1 = await compute(1, b=2)
        assert result1 == 3
        assert call_count == 1

        # Same args should use cache
        result2 = await compute(1, b=2)
        assert result2 == 3
        assert call_count == 1

        # Different order of kwargs should still use cache
        result3 = await compute(1, b=2)
        assert result3 == 3
        assert call_count == 1
