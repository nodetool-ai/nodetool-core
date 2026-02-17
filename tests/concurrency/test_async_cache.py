"""Tests for async LRU cache utilities."""

import asyncio
from datetime import datetime, timedelta

import pytest

from nodetool.concurrency.async_cache import AsyncLRUCache, CacheEntry, CacheStats


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_cache_entry_creation(self) -> None:
        """Test creating a cache entry."""
        entry = CacheEntry(value="test_value")
        assert entry.value == "test_value"
        assert entry.expires_at is None
        assert entry.access_count == 0
        assert isinstance(entry.created_at, datetime)
        assert isinstance(entry.last_accessed, datetime)

    def test_cache_entry_with_expiration(self) -> None:
        """Test cache entry with expiration."""
        expires_at = datetime.now() + timedelta(seconds=10)
        entry = CacheEntry(value="test_value", expires_at=expires_at)
        assert entry.expires_at == expires_at
        assert not entry.is_expired()

    def test_cache_entry_expiration_check(self) -> None:
        """Test cache entry expiration check."""
        # Not expired
        future_entry = CacheEntry(
            value="future",
            expires_at=datetime.now() + timedelta(seconds=10),
        )
        assert not future_entry.is_expired()

        # Expired
        past_entry = CacheEntry(
            value="past",
            expires_at=datetime.now() - timedelta(seconds=10),
        )
        assert past_entry.is_expired()

        # No expiration (None)
        no_exp_entry = CacheEntry(value="no_exp", expires_at=None)
        assert not no_exp_entry.is_expired()

    def test_cache_entry_touch(self) -> None:
        """Test updating cache entry access statistics."""
        entry = CacheEntry(value="test")
        initial_count = entry.access_count
        initial_last_accessed = entry.last_accessed

        # Small delay to ensure timestamp changes
        import time

        time.sleep(0.01)

        entry.touch()

        assert entry.access_count == initial_count + 1
        assert entry.last_accessed > initial_last_accessed


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_cache_stats_creation(self) -> None:
        """Test creating cache stats."""
        stats = CacheStats(hits=10, misses=5, evictions=2, size=8, max_size=10)
        assert stats.hits == 10
        assert stats.misses == 5
        assert stats.evictions == 2
        assert stats.size == 8
        assert stats.max_size == 10

    def test_hit_rate_calculation(self) -> None:
        """Test cache hit rate calculation."""
        # Normal case
        stats = CacheStats(hits=8, misses=2)
        assert stats.hit_rate == 0.8

        # All hits
        stats = CacheStats(hits=10, misses=0)
        assert stats.hit_rate == 1.0

        # All misses
        stats = CacheStats(hits=0, misses=10)
        assert stats.hit_rate == 0.0

        # No requests
        stats = CacheStats(hits=0, misses=0)
        assert stats.hit_rate == 0.0

    def test_to_dict(self) -> None:
        """Test converting stats to dictionary."""
        stats = CacheStats(hits=8, misses=2, evictions=1, size=5, max_size=10)
        stats_dict = stats.to_dict()

        assert stats_dict["hits"] == 8
        assert stats_dict["misses"] == 2
        assert stats_dict["evictions"] == 1
        assert stats_dict["size"] == 5
        assert stats_dict["max_size"] == 10
        assert stats_dict["hit_rate"] == 0.8


class TestAsyncLRUCache:
    """Tests for AsyncLRUCache."""

    @pytest.mark.asyncio
    async def test_cache_initialization(self) -> None:
        """Test cache initialization."""
        cache = AsyncLRUCache(max_size=10, ttl=60)
        assert cache.max_size == 10
        assert cache.stats.size == 0
        assert cache.stats.hits == 0
        assert cache.stats.misses == 0

    @pytest.mark.asyncio
    async def test_put_and_get(self) -> None:
        """Test basic put and get operations."""
        cache = AsyncLRUCache(max_size=10)

        await cache.put("key1", "value1")
        value = await cache.get("key1")

        assert value == "value1"
        assert cache.stats.hits == 1
        assert cache.stats.misses == 0

    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self) -> None:
        """Test getting a nonexistent key."""
        cache = AsyncLRUCache(max_size=10)

        value = await cache.get("nonexistent")

        assert value is None
        assert cache.stats.misses == 1
        assert cache.stats.hits == 0

    @pytest.mark.asyncio
    async def test_put_updates_existing_entry(self) -> None:
        """Test that put updates an existing entry."""
        cache = AsyncLRUCache(max_size=10)

        await cache.put("key1", "value1")
        await cache.put("key1", "value2")

        value = await cache.get("key1")
        assert value == "value2"
        assert cache.stats.size == 1

    @pytest.mark.asyncio
    async def test_lru_eviction(self) -> None:
        """Test LRU eviction when cache is full."""
        cache = AsyncLRUCache(max_size=3)

        # Fill cache
        await cache.put("key1", "value1")
        await cache.put("key2", "value2")
        await cache.put("key3", "value3")

        # Access key1 to make it more recently used
        await cache.get("key1")

        # Add new entry, should evict key2 (least recently used)
        await cache.put("key4", "value4")

        # Check that key2 was evicted
        assert await cache.get("key2") is None
        assert await cache.get("key1") == "value1"
        assert await cache.get("key3") == "value3"
        assert await cache.get("key4") == "value4"

        assert cache.stats.evictions == 1
        assert cache.stats.size == 3

    @pytest.mark.asyncio
    async def test_ttl_expiration(self) -> None:
        """Test TTL expiration."""
        cache = AsyncLRUCache(max_size=10)

        # Add entry with short TTL
        await cache.put("key1", "value1", ttl=1)

        # Should be available immediately
        assert await cache.get("key1") == "value1"

        # Wait for expiration
        await asyncio.sleep(1.5)

        # Should be expired now
        value = await cache.get("key1")
        assert value is None
        assert cache.stats.misses == 1  # Miss due to expiration

    @pytest.mark.asyncio
    async def test_default_ttl(self) -> None:
        """Test default TTL."""
        cache = AsyncLRUCache(max_size=10, ttl=1)

        await cache.put("key1", "value1")

        # Should be available immediately
        assert await cache.get("key1") == "value1"

        # Wait for expiration
        await asyncio.sleep(1.5)

        # Should be expired now
        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_get_or_compute(self) -> None:
        """Test get_or_compute method."""
        cache = AsyncLRUCache(max_size=10)
        compute_count = 0

        async def compute_value() -> str:
            nonlocal compute_count
            compute_count += 1
            await asyncio.sleep(0.01)  # Simulate some work
            return "computed"

        # First call should compute
        result1 = await cache.get_or_compute("key1", compute_value)
        assert result1 == "computed"
        assert compute_count == 1

        # Second call should use cache
        result2 = await cache.get_or_compute("key1", compute_value)
        assert result2 == "computed"
        assert compute_count == 1  # Should not recompute

    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        """Test deleting entries."""
        cache = AsyncLRUCache(max_size=10)

        await cache.put("key1", "value1")

        # Delete existing key
        deleted = await cache.delete("key1")
        assert deleted is True
        assert await cache.get("key1") is None

        # Delete nonexistent key
        deleted = await cache.delete("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_clear(self) -> None:
        """Test clearing the cache."""
        cache = AsyncLRUCache(max_size=10)

        await cache.put("key1", "value1")
        await cache.put("key2", "value2")
        await cache.put("key3", "value3")

        assert cache.stats.size == 3

        await cache.clear()

        assert cache.stats.size == 0
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
        assert await cache.get("key3") is None

    @pytest.mark.asyncio
    async def test_cleanup_expired(self) -> None:
        """Test cleaning up expired entries."""
        cache = AsyncLRUCache(max_size=10)

        # Add entries with different TTLs
        await cache.put("key1", "value1", ttl=1)
        await cache.put("key2", "value2", ttl=2)
        await cache.put("key3", "value3", ttl=3)

        await asyncio.sleep(1.5)

        # Cleanup should remove key1
        removed = await cache.cleanup_expired()
        assert removed == 1
        assert cache.stats.size == 2

        # key1 should be gone
        assert await cache.get("key1") is None
        assert await cache.get("key2") == "value2"
        assert await cache.get("key3") == "value3"

    @pytest.mark.asyncio
    async def test_keys(self) -> None:
        """Test getting all keys."""
        cache = AsyncLRUCache(max_size=10)

        await cache.put("key1", "value1")
        await cache.put("key2", "value2")
        await cache.put("key3", "value3")

        keys = await cache.keys()
        assert set(keys) == {"key1", "key2", "key3"}

    @pytest.mark.asyncio
    async def test_items(self) -> None:
        """Test iterating over cache items."""
        cache = AsyncLRUCache(max_size=10)

        await cache.put("key1", "value1")
        await cache.put("key2", "value2")
        await cache.put("key3", "value3")

        items = []
        async for key, value in cache.items():
            items.append((key, value))

        assert set(items) == {("key1", "value1"), ("key2", "value2"), ("key3", "value3")}

    @pytest.mark.asyncio
    async def test_items_with_expired_entries(self) -> None:
        """Test that items iterator skips expired entries."""
        cache = AsyncLRUCache(max_size=10)

        await cache.put("key1", "value1", ttl=1)
        await cache.put("key2", "value2")  # No expiration
        await cache.put("key3", "value3", ttl=1)

        await asyncio.sleep(1.5)

        items = []
        async for key, value in cache.items():
            items.append((key, value))

        # Only key2 should be returned (not expired)
        assert items == [("key2", "value2")]

    @pytest.mark.asyncio
    async def test_cache_result_decorator(self) -> None:
        """Test cache_result decorator."""
        cache = AsyncLRUCache(max_size=10, ttl=60)
        call_count = 0

        @cache.cache_result()
        async def expensive_function(x: int, y: int) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return x + y

        # First call
        result1 = await expensive_function(2, 3)
        assert result1 == 5
        assert call_count == 1

        # Second call with same args (should use cache)
        result2 = await expensive_function(2, 3)
        assert result2 == 5
        assert call_count == 1  # Should not increment

        # Different args (should not use cache)
        result3 = await expensive_function(3, 4)
        assert result3 == 7
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_cache_result_decorator_with_custom_key_fn(self) -> None:
        """Test cache_result decorator with custom key function."""
        cache = AsyncLRUCache(max_size=10)
        call_count = 0

        def custom_key_fn(user_id: str, action: str) -> str:
            return f"user:{user_id}:action:{action}"

        @cache.cache_result(key_fn=custom_key_fn)
        async def user_action(user_id: str, action: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"{user_id} performed {action}"

        # First call
        result1 = await user_action("user1", "login")
        assert result1 == "user1 performed login"
        assert call_count == 1

        # Same args should use cache
        await user_action("user1", "login")
        assert call_count == 1

        # Different args should not use cache
        await user_action("user1", "logout")
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_stats_tracking(self) -> None:
        """Test that stats are tracked correctly."""
        cache = AsyncLRUCache(max_size=5)

        await cache.put("key1", "value1")
        await cache.put("key2", "value2")
        await cache.put("key3", "value3")

        # Hits
        await cache.get("key1")
        await cache.get("key2")
        await cache.get("key1")  # key1 again

        # Misses
        await cache.get("nonexistent1")
        await cache.get("nonexistent2")

        stats = cache.stats
        assert stats.hits == 3
        assert stats.misses == 2
        assert stats.size == 3
        assert stats.evictions == 0
        assert stats.hit_rate == 0.6

    @pytest.mark.asyncio
    async def test_concurrent_access(self) -> None:
        """Test thread-safe concurrent access."""
        cache = AsyncLRUCache(max_size=100)
        results = []

        async def worker(worker_id: int) -> None:
            for i in range(10):
                key = f"worker{worker_id}_key{i}"
                await cache.put(key, f"value_{worker_id}_{i}")
                value = await cache.get(key)
                results.append(value)

        # Run multiple workers concurrently
        tasks = [worker(i) for i in range(5)]
        await asyncio.gather(*tasks)

        # Should have completed all operations without errors
        assert len(results) == 50

    @pytest.mark.asyncio
    async def test_no_expiration_when_ttl_none(self) -> None:
        """Test that entries don't expire when TTL is None."""
        cache = AsyncLRUCache(max_size=10, ttl=None)

        await cache.put("key1", "value1")

        # Should be available immediately
        assert await cache.get("key1") == "value1"

        # Should still be available after some time
        await asyncio.sleep(0.5)
        assert await cache.get("key1") == "value1"

    @pytest.mark.asyncio
    async def test_get_or_compute_with_sync_function(self) -> None:
        """Test get_or_compute with a synchronous compute function."""
        cache = AsyncLRUCache(max_size=10)
        compute_count = 0

        def compute_value_sync() -> str:
            nonlocal compute_count
            compute_count += 1
            return "computed_sync"

        # Should work with sync functions too
        result = await cache.get_or_compute("key1", compute_value_sync)
        assert result == "computed_sync"
        assert compute_count == 1

        # Second call should use cache
        result2 = await cache.get_or_compute("key1", compute_value_sync)
        assert result2 == "computed_sync"
        assert compute_count == 1
