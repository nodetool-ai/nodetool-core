"""Tests for async caching utilities."""

import asyncio
import pytest

from nodetool.concurrency.async_cache import AsyncCache, CacheEntry


class TestCacheEntry:
    """Tests for CacheEntry class."""

    def test_cache_entry_creation(self):
        """Test that cache entries are created correctly."""
        entry = CacheEntry(value="test", ttl=10.0)
        assert entry.value == "test"
        assert entry.expires_at is not None
        assert entry.access_count == 0
        assert entry.created_at > 0

    def test_cache_entry_no_expiration(self):
        """Test that entries with no TTL don't expire."""
        entry = CacheEntry(value="test", ttl=None)
        assert entry.expires_at is None
        assert not entry.is_expired()

    @pytest.mark.asyncio
    async def test_cache_entry_expiration(self):
        """Test that entries expire after TTL."""
        entry = CacheEntry(value="test", ttl=0.1)
        assert not entry.is_expired()
        await asyncio.sleep(0.15)
        assert entry.is_expired()

    def test_cache_entry_touch(self):
        """Test that touch updates access metrics."""
        entry = CacheEntry(value="test", ttl=10.0)
        initial_count = entry.access_count
        initial_time = entry.last_accessed

        entry.touch()

        assert entry.access_count == initial_count + 1
        assert entry.last_accessed > initial_time

    def test_cache_entry_age(self):
        """Test that age returns correct elapsed time."""
        entry = CacheEntry(value="test", ttl=10.0)
        age = entry.age()
        assert age >= 0


class TestAsyncCache:
    """Tests for AsyncCache class."""

    @pytest.mark.asyncio
    async def test_cache_initialization(self):
        """Test that cache initializes with correct defaults."""
        cache = AsyncCache[str, int](max_size=100, default_ttl=60.0)
        assert await cache.size() == 0
        assert cache._max_size == 100
        assert cache._default_ttl == 60.0

    @pytest.mark.asyncio
    async def test_cache_initialization_invalid_params(self):
        """Test that cache rejects invalid initialization parameters."""
        with pytest.raises(ValueError, match="max_size must be a positive integer"):
            AsyncCache[str, int](max_size=0)

        with pytest.raises(ValueError, match="default_ttl must be positive"):
            AsyncCache[str, int](max_size=100, default_ttl=-1.0)

    @pytest.mark.asyncio
    async def test_cache_put_and_get(self):
        """Test basic put and get operations."""
        cache = AsyncCache[str, int](max_size=100)

        await cache.put("key1", 42)
        value = await cache.get("key1")

        assert value == 42

    @pytest.mark.asyncio
    async def test_cache_get_miss(self):
        """Test that get returns None for missing keys."""
        cache = AsyncCache[str, int](max_size=100)

        value = await cache.get("nonexistent")

        assert value is None

    @pytest.mark.asyncio
    async def test_cache_get_expired(self):
        """Test that expired entries are not returned."""
        cache = AsyncCache[str, int](max_size=100, default_ttl=0.1)

        await cache.put("key1", 42)
        assert await cache.get("key1") == 42

        await asyncio.sleep(0.15)
        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_cache_custom_ttl(self):
        """Test that custom TTL overrides default."""
        cache = AsyncCache[str, int](max_size=100, default_ttl=0.1)

        await cache.put("key1", 42, ttl=1.0)
        await cache.put("key2", 100)  # Uses default TTL

        await asyncio.sleep(0.15)

        assert await cache.get("key1") == 42
        assert await cache.get("key2") is None

    @pytest.mark.asyncio
    async def test_cache_delete(self):
        """Test deleting entries from cache."""
        cache = AsyncCache[str, int](max_size=100)

        await cache.put("key1", 42)
        assert await cache.delete("key1") is True
        assert await cache.get("key1") is None
        assert await cache.delete("key1") is False

    @pytest.mark.asyncio
    async def test_cache_clear(self):
        """Test clearing all entries."""
        cache = AsyncCache[str, int](max_size=100)

        await cache.put("key1", 42)
        await cache.put("key2", 100)
        assert await cache.size() == 2

        await cache.clear()
        assert await cache.size() == 0
        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_cache_has_key(self):
        """Test checking if key exists."""
        cache = AsyncCache[str, int](max_size=100, default_ttl=0.1)

        assert not await cache.has_key("key1")

        await cache.put("key1", 42)
        assert await cache.has_key("key1")

        await asyncio.sleep(0.15)
        assert not await cache.has_key("key1")

    @pytest.mark.asyncio
    async def test_cache_lru_eviction(self):
        """Test that LRU eviction works when max_size is reached."""
        cache = AsyncCache[str, int](max_size=3)

        await cache.put("key1", 1)
        await cache.put("key2", 2)
        await cache.put("key3", 3)

        # Access key1 to make it more recently used
        await cache.get("key1")

        # Add key4, should evict key2 (least recently used)
        await cache.put("key4", 4)

        assert await cache.get("key1") == 1
        assert await cache.get("key2") is None  # Evicted
        assert await cache.get("key3") == 3
        assert await cache.get("key4") == 4

    @pytest.mark.asyncio
    async def test_cache_stats(self):
        """Test cache statistics tracking."""
        cache = AsyncCache[str, int](max_size=100)

        await cache.put("key1", 42)
        await cache.get("key1")  # Hit
        await cache.get("key2")  # Miss

        stats = await cache.get_stats()

        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 50.0
        assert stats["size"] == 1
        assert stats["max_size"] == 100

    @pytest.mark.asyncio
    async def test_cache_cleanup_expired(self):
        """Test manual cleanup of expired entries."""
        cache = AsyncCache[str, int](max_size=100, default_ttl=0.1)

        await cache.put("key1", 1)
        await cache.put("key2", 2)
        await cache.put("key3", 3)

        assert await cache.size() == 3

        await asyncio.sleep(0.15)

        removed = await cache.cleanup_expired()
        assert removed == 3
        assert await cache.size() == 0

    @pytest.mark.asyncio
    async def test_cache_keys_and_items(self):
        """Test getting keys and items from cache."""
        cache = AsyncCache[str, int](max_size=100)

        await cache.put("key1", 1)
        await cache.put("key2", 2)

        keys = await cache.keys()
        items = await cache.items()

        assert set(keys) == {"key1", "key2"}
        assert set(items) == {("key1", 1), ("key2", 2)}

    @pytest.mark.asyncio
    async def test_cache_get_or_compute_hit(self):
        """Test get_or_compute returns cached value."""
        cache = AsyncCache[str, int](max_size=100)

        await cache.put("key1", 42)

        call_count = 0

        async def factory():
            nonlocal call_count
            call_count += 1
            return 100

        result = await cache.get_or_compute("key1", factory)

        assert result == 42
        assert call_count == 0  # Factory not called

    @pytest.mark.asyncio
    async def test_cache_get_or_compute_miss(self):
        """Test get_or_compute computes and caches value."""
        cache = AsyncCache[str, int](max_size=100)

        call_count = 0

        async def factory():
            nonlocal call_count
            call_count += 1
            return 100

        result = await cache.get_or_compute("key1", factory)

        assert result == 100
        assert call_count == 1

        # Second call should use cache
        result2 = await cache.get_or_compute("key1", factory)
        assert result2 == 100
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_cache_get_or_compute_with_sync_factory(self):
        """Test get_or_compute works with sync factory."""
        cache = AsyncCache[str, int](max_size=100)

        def sync_factory():
            return 42

        result = await cache.get_or_compute("key1", sync_factory)

        assert result == 42

    @pytest.mark.asyncio
    async def test_cache_get_or_compute_with_custom_ttl(self):
        """Test get_or_compute respects custom TTL."""
        cache = AsyncCache[str, int](max_size=100, default_ttl=0.1)

        async def factory():
            return 42

        result = await cache.get_or_compute("key1", factory, ttl=1.0)

        assert result == 42

        await asyncio.sleep(0.15)

        # Should still be cached due to custom TTL
        result2 = await cache.get("key1")
        assert result2 == 42

    @pytest.mark.asyncio
    async def test_cache_auto_cleanup(self):
        """Test automatic cleanup of expired entries."""
        cache = AsyncCache[str, int](max_size=100, default_ttl=0.1, cleanup_interval=0.2)

        await cache.put("key1", 1)
        await cache.put("key2", 2)

        await cache.start_auto_cleanup()

        # Wait for cleanup cycle
        await asyncio.sleep(0.3)

        assert await cache.size() == 0

        await cache.stop_auto_cleanup()

    @pytest.mark.asyncio
    async def test_cache_concurrent_access(self):
        """Test that cache handles concurrent access safely."""
        cache = AsyncCache[str, int](max_size=100)

        async def worker(key: str, value: int):
            await cache.put(key, value)
            await asyncio.sleep(0.01)
            return await cache.get(key)

        tasks = [
            worker(f"key{i}", i)
            for i in range(10)
        ]

        results = await asyncio.gather(*tasks)

        assert all(r is not None for r in results)
        assert await cache.size() == 10

    @pytest.mark.asyncio
    async def test_cache_update_existing_key(self):
        """Test updating an existing key doesn't cause eviction."""
        cache = AsyncCache[str, int](max_size=2)

        await cache.put("key1", 1)
        await cache.put("key2", 2)

        # Update key1, should not evict
        await cache.put("key1", 10)

        assert await cache.get("key1") == 10
        assert await cache.get("key2") == 2

    @pytest.mark.asyncio
    async def test_cache_len_dunder(self):
        """Test __len__ returns approximate size."""
        cache = AsyncCache[str, int](max_size=100)

        await cache.put("key1", 1)
        await cache.put("key2", 2)

        assert len(cache) == 2

    @pytest.mark.asyncio
    async def test_cache_entry_with_none_value(self):
        """Test that None can be cached as a value."""
        cache = AsyncCache[str, int | None](max_size=100)

        await cache.put("key1", None)
        result = await cache.get("key1")

        assert result is None

    @pytest.mark.asyncio
    async def test_cache_statistics_hit_rate_zero_requests(self):
        """Test hit rate calculation with no requests."""
        cache = AsyncCache[str, int](max_size=100)

        stats = await cache.get_stats()

        assert stats["hit_rate"] == 0.0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    @pytest.mark.asyncio
    async def test_cache_eviction_increments_counter(self):
        """Test that eviction counter is incremented."""
        cache = AsyncCache[str, int](max_size=2)

        await cache.put("key1", 1)
        await cache.put("key2", 2)
        await cache.put("key3", 3)  # Should evict one entry

        stats = await cache.get_stats()

        assert stats["evictions"] == 1
