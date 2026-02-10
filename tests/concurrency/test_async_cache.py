"""
Comprehensive tests for AsyncTTLCache.
"""

import asyncio

import pytest

from nodetool.concurrency.async_cache import AsyncTTLCache, CacheEntry


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_cache_entry_creation(self):
        """Test creating a cache entry."""
        entry = CacheEntry(value=42, expires_at=100.0)
        assert entry.value == 42
        assert entry.expires_at == 100.0

    def test_cache_entry_with_different_types(self):
        """Test cache entry with various value types."""
        entry_str = CacheEntry(value="hello", expires_at=100.0)
        assert entry_str.value == "hello"

        entry_list = CacheEntry(value=[1, 2, 3], expires_at=100.0)
        assert entry_list.value == [1, 2, 3]

        entry_dict = CacheEntry(value={"key": "value"}, expires_at=100.0)
        assert entry_dict.value == {"key": "value"}


class TestAsyncTTLCacheInit:
    """Tests for AsyncTTLCache initialization."""

    def test_init_with_valid_params(self):
        """Test initialization with valid parameters."""
        cache = AsyncTTLCache(max_size=100, ttl=60.0)
        assert cache.max_size == 100
        assert cache.ttl == 60.0
        assert cache.size == 0

    def test_init_with_custom_cleanup_interval(self):
        """Test initialization with custom cleanup interval."""
        cache = AsyncTTLCache(max_size=100, ttl=60.0, cleanup_interval=0.5)
        assert cache.max_size == 100
        assert cache.ttl == 60.0

    def test_init_with_invalid_max_size(self):
        """Test that invalid max_size raises ValueError."""
        with pytest.raises(ValueError, match="max_size must be a positive integer"):
            AsyncTTLCache(max_size=0, ttl=60.0)

        with pytest.raises(ValueError, match="max_size must be a positive integer"):
            AsyncTTLCache(max_size=-1, ttl=60.0)

    def test_init_with_invalid_ttl(self):
        """Test that invalid ttl raises ValueError."""
        with pytest.raises(ValueError, match="ttl must be a positive number"):
            AsyncTTLCache(max_size=100, ttl=0)

        with pytest.raises(ValueError, match="ttl must be a positive number"):
            AsyncTTLCache(max_size=100, ttl=-1.0)

    def test_init_with_invalid_cleanup_interval(self):
        """Test that invalid cleanup_interval raises ValueError."""
        with pytest.raises(ValueError, match="cleanup_interval must be a positive number"):
            AsyncTTLCache(max_size=100, ttl=60.0, cleanup_interval=0)

        with pytest.raises(ValueError, match="cleanup_interval must be a positive number"):
            AsyncTTLCache(max_size=100, ttl=60.0, cleanup_interval=-0.5)


class TestAsyncTTLCacheBasicOperations:
    """Tests for basic cache operations."""

    @pytest.mark.asyncio
    async def test_set_and_get(self):
        """Test basic set and get operations."""
        cache = AsyncTTLCache[str, int](max_size=100, ttl=60.0)

        await cache.set("key1", 42)
        value = await cache.get("key1")

        assert value == 42
        assert cache.size == 1

    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self):
        """Test getting a key that doesn't exist."""
        cache = AsyncTTLCache[str, int](max_size=100, ttl=60.0)

        value = await cache.get("nonexistent")

        assert value is None

    @pytest.mark.asyncio
    async def test_set_update_existing_key(self):
        """Test updating an existing key."""
        cache = AsyncTTLCache[str, int](max_size=100, ttl=60.0)

        await cache.set("key1", 42)
        await cache.set("key1", 100)

        value = await cache.get("key1")
        assert value == 100
        assert cache.size == 1

    @pytest.mark.asyncio
    async def test_has_key_exists(self):
        """Test has() with existing key."""
        cache = AsyncTTLCache[str, int](max_size=100, ttl=60.0)

        await cache.set("key1", 42)

        assert await cache.has("key1") is True

    @pytest.mark.asyncio
    async def test_has_key_not_exists(self):
        """Test has() with non-existing key."""
        cache = AsyncTTLCache[str, int](max_size=100, ttl=60.0)

        assert await cache.has("nonexistent") is False

    @pytest.mark.asyncio
    async def test_has_expired_key(self):
        """Test has() with expired key."""
        cache = AsyncTTLCache[str, int](max_size=100, ttl=0.1)

        await cache.set("key1", 42)
        assert await cache.has("key1") is True

        await asyncio.sleep(0.15)
        assert await cache.has("key1") is False

    @pytest.mark.asyncio
    async def test_invalidate_existing_key(self):
        """Test invalidating an existing key."""
        cache = AsyncTTLCache[str, int](max_size=100, ttl=60.0)

        await cache.set("key1", 42)
        assert await cache.has("key1") is True

        result = await cache.invalidate("key1")
        assert result is True
        assert await cache.has("key1") is False

    @pytest.mark.asyncio
    async def test_invalidate_nonexistent_key(self):
        """Test invalidating a non-existing key."""
        cache = AsyncTTLCache[str, int](max_size=100, ttl=60.0)

        result = await cache.invalidate("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clearing the cache."""
        cache = AsyncTTLCache[str, int](max_size=100, ttl=60.0)

        await cache.set("key1", 42)
        await cache.set("key2", 100)
        await cache.set("key3", 200)

        assert cache.size == 3

        await cache.clear()

        assert cache.size == 0
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
        assert await cache.get("key3") is None

    @pytest.mark.asyncio
    async def test_len(self):
        """Test __len__ method."""
        cache = AsyncTTLCache[str, int](max_size=100, ttl=60.0)

        assert len(cache) == 0

        await cache.set("key1", 42)
        assert len(cache) == 1

        await cache.set("key2", 100)
        assert len(cache) == 2

        await cache.invalidate("key1")
        assert len(cache) == 1


class TestAsyncTTLCacheTTLExpiration:
    """Tests for TTL-based expiration."""

    @pytest.mark.asyncio
    async def test_expiration_after_ttl(self):
        """Test that entries expire after TTL."""
        cache = AsyncTTLCache[str, int](max_size=100, ttl=0.1)

        await cache.set("key1", 42)
        assert await cache.get("key1") == 42

        await asyncio.sleep(0.15)

        value = await cache.get("key1")
        assert value is None

    @pytest.mark.asyncio
    async def test_expiration_on_get(self):
        """Test that expired entries are removed on get."""
        cache = AsyncTTLCache[str, int](max_size=100, ttl=0.1)

        await cache.set("key1", 42)
        await cache.set("key2", 100)
        await cache.set("key3", 200)

        assert cache.size == 3

        await asyncio.sleep(0.15)

        # Getting expired entry removes it
        await cache.get("key1")
        assert cache.size == 2

    @pytest.mark.asyncio
    async def test_expiration_on_has(self):
        """Test that expired entries are removed on has."""
        cache = AsyncTTLCache[str, int](max_size=100, ttl=0.1)

        await cache.set("key1", 42)
        await cache.set("key2", 100)

        assert cache.size == 2

        await asyncio.sleep(0.15)

        # Checking expired entry removes it
        await cache.has("key1")
        assert cache.size == 1

    @pytest.mark.asyncio
    async def test_cleanup_expired_manual(self):
        """Test manual cleanup of expired entries."""
        cache = AsyncTTLCache[str, int](max_size=100, ttl=0.1)

        await cache.set("key1", 42)
        await cache.set("key2", 100)
        await cache.set("key3", 200)

        assert cache.size == 3

        await asyncio.sleep(0.15)

        removed = await cache.cleanup_expired()
        assert removed == 3
        assert cache.size == 0

    @pytest.mark.asyncio
    async def test_cleanup_expired_partial(self):
        """Test cleanup with some expired and some valid entries."""
        cache = AsyncTTLCache[str, int](max_size=100, ttl=0.1)

        await cache.set("key1", 42)
        await cache.set("key2", 100)

        await asyncio.sleep(0.15)

        await cache.set("key3", 200)  # This one is still valid

        removed = await cache.cleanup_expired()
        assert removed == 2
        assert cache.size == 1
        assert await cache.get("key3") == 200


class TestAsyncTTLCacheLRUEviction:
    """Tests for LRU eviction when cache is full."""

    @pytest.mark.asyncio
    async def test_lru_eviction_when_full(self):
        """Test that LRU eviction works when cache is full."""
        cache = AsyncTTLCache[str, int](max_size=3, ttl=60.0)

        await cache.set("key1", 1)
        await cache.set("key2", 2)
        await cache.set("key3", 3)

        assert cache.size == 3

        # Adding a 4th item should evict the first (least recently used)
        await cache.set("key4", 4)

        assert cache.size == 3
        assert await cache.get("key1") is None  # Evicted
        assert await cache.get("key2") == 2
        assert await cache.get("key3") == 3
        assert await cache.get("key4") == 4

    @pytest.mark.asyncio
    async def test_lru_eviction_with_access_pattern(self):
        """Test LRU eviction with specific access pattern."""
        cache = AsyncTTLCache[str, int](max_size=3, ttl=60.0)

        await cache.set("key1", 1)
        await cache.set("key2", 2)
        await cache.set("key3", 3)

        # Access key1 to make it more recently used
        await cache.get("key1")

        # Add key4, should evict key2 (now least recently used)
        await cache.set("key4", 4)

        assert cache.size == 3
        assert await cache.get("key1") == 1  # Still present
        assert await cache.get("key2") is None  # Evicted
        assert await cache.get("key3") == 3
        assert await cache.get("key4") == 4

    @pytest.mark.asyncio
    async def test_lru_eviction_with_update(self):
        """Test that updating a key affects LRU order."""
        cache = AsyncTTLCache[str, int](max_size=3, ttl=60.0)

        await cache.set("key1", 1)
        await cache.set("key2", 2)
        await cache.set("key3", 3)

        # Update key1 to make it more recently used
        await cache.set("key1", 100)

        # Add key4, should evict key2
        await cache.set("key4", 4)

        assert cache.size == 3
        assert await cache.get("key1") == 100  # Still present, updated
        assert await cache.get("key2") is None  # Evicted
        assert await cache.get("key3") == 3
        assert await cache.get("key4") == 4

    @pytest.mark.asyncio
    async def test_lru_eviction_order(self):
        """Test LRU eviction maintains correct order."""
        cache = AsyncTTLCache[str, int](max_size=3, ttl=60.0)

        await cache.set("key1", 1)
        await cache.set("key2", 2)
        await cache.set("key3", 3)

        # Access in different order to test LRU tracking
        await cache.get("key1")  # key1: most recent
        await cache.get("key2")  # key2: second most recent
        # key3: least recent

        # Add key4, should evict key3
        await cache.set("key4", 4)

        assert await cache.get("key1") == 1
        assert await cache.get("key2") == 2
        assert await cache.get("key3") is None  # Evicted
        assert await cache.get("key4") == 4


class TestAsyncTTLCacheGetOrCompute:
    """Tests for get_or_compute method."""

    @pytest.mark.asyncio
    async def test_get_or_compute_cache_hit(self):
        """Test get_or_compute with cache hit."""
        cache = AsyncTTLCache[str, int](max_size=100, ttl=60.0)

        await cache.set("key1", 42)

        compute_called = False

        async def compute_fn():
            nonlocal compute_called
            compute_called = True
            return 100

        value = await cache.get_or_compute("key1", compute_fn)

        assert value == 42
        assert compute_called is False

    @pytest.mark.asyncio
    async def test_get_or_compute_cache_miss(self):
        """Test get_or_compute with cache miss."""
        cache = AsyncTTLCache[str, int](max_size=100, ttl=60.0)

        compute_called = False

        async def compute_fn():
            nonlocal compute_called
            compute_called = True
            return 100

        value = await cache.get_or_compute("key1", compute_fn)

        assert value == 100
        assert compute_called is True
        assert await cache.get("key1") == 100

    @pytest.mark.asyncio
    async def test_get_or_compute_no_compute_fn(self):
        """Test get_or_compute without compute_fn raises ValueError."""
        cache = AsyncTTLCache[str, int](max_size=100, ttl=60.0)

        with pytest.raises(ValueError, match="not found in cache and no compute_fn"):
            await cache.get_or_compute("key1", None)

    @pytest.mark.asyncio
    async def test_get_or_compute_expired_entry(self):
        """Test get_or_compute with expired entry."""
        cache = AsyncTTLCache[str, int](max_size=100, ttl=0.1)

        await cache.set("key1", 42)

        await asyncio.sleep(0.15)

        compute_called = False

        async def compute_fn():
            nonlocal compute_called
            compute_called = True
            return 100

        value = await cache.get_or_compute("key1", compute_fn)

        assert value == 100
        assert compute_called is True

    @pytest.mark.asyncio
    async def test_get_or_compute_concurrent(self):
        """Test get_or_compute with concurrent access."""
        cache = AsyncTTLCache[str, int](max_size=100, ttl=60.0)

        compute_count = 0

        async def compute_fn():
            nonlocal compute_count
            compute_count += 1
            await asyncio.sleep(0.1)
            return 100

        # Concurrent calls should all compute
        tasks = [
            cache.get_or_compute("key1", compute_fn) for _ in range(3)
        ]

        results = await asyncio.gather(*tasks)

        assert all(r == 100 for r in results)
        # Note: Due to lack of locking in get_or_compute, multiple computes may happen
        # This is expected behavior for this implementation


class TestAsyncTTLCacheContextManager:
    """Tests for context manager and background cleanup."""

    @pytest.mark.asyncio
    async def test_context_manager_basic(self):
        """Test basic context manager usage."""
        cache = AsyncTTLCache[str, int](max_size=100, ttl=60.0)

        async with cache:
            await cache.set("key1", 42)
            assert await cache.get("key1") == 42

    @pytest.mark.asyncio
    async def test_context_manager_cleanup_on_exit(self):
        """Test that cleanup task is stopped on exit."""
        cache = AsyncTTLCache[str, int](max_size=100, ttl=60.0)

        async with cache:
            assert cache._cleanup_task is not None
            assert cache._is_running is True

        assert cache._is_running is False
        assert cache._cleanup_task is None

    @pytest.mark.asyncio
    async def test_background_cleanup(self):
        """Test that background cleanup removes expired entries."""
        cache = AsyncTTLCache[str, int](max_size=100, ttl=0.1, cleanup_interval=0.1)

        async with cache:
            await cache.set("key1", 42)
            await cache.set("key2", 100)

            assert cache.size == 2

            # Wait for expiration and cleanup
            await asyncio.sleep(0.25)

            # Background cleanup should have removed expired entries
            assert cache.size == 0

    @pytest.mark.asyncio
    async def test_context_manager_exception_handling(self):
        """Test context manager handles exceptions correctly."""
        cache = AsyncTTLCache[str, int](max_size=100, ttl=60.0)

        with pytest.raises(ValueError):
            async with cache:
                await cache.set("key1", 42)
                raise ValueError("test error")

        # Cleanup task should still be stopped
        assert cache._is_running is False
        assert cache._cleanup_task is None


class TestAsyncTTLCacheThreadSafety:
    """Tests for concurrent access and thread safety."""

    @pytest.mark.asyncio
    async def test_concurrent_set_operations(self):
        """Test concurrent set operations."""
        cache = AsyncTTLCache[str, int](max_size=100, ttl=60.0)

        async def set_value(key: str, value: int):
            await cache.set(key, value)

        tasks = [
            set_value(f"key{i}", i) for i in range(50)
        ]

        await asyncio.gather(*tasks)

        assert cache.size == 50

        for i in range(50):
            assert await cache.get(f"key{i}") == i

    @pytest.mark.asyncio
    async def test_concurrent_get_operations(self):
        """Test concurrent get operations."""
        cache = AsyncTTLCache[str, int](max_size=100, ttl=60.0)

        # Pre-populate cache
        for i in range(50):
            await cache.set(f"key{i}", i)

        async def get_value(key: str):
            return await cache.get(key)

        tasks = [
            get_value(f"key{i}") for i in range(50)
        ]

        results = await asyncio.gather(*tasks)

        assert results == list(range(50))

    @pytest.mark.asyncio
    async def test_concurrent_mixed_operations(self):
        """Test concurrent mixed operations."""
        cache = AsyncTTLCache[str, int](max_size=100, ttl=60.0)

        async def mixed_operations(key: str, value: int):
            await cache.set(key, value)
            await cache.get(key)
            await cache.has(key)

        tasks = [
            mixed_operations(f"key{i}", i) for i in range(50)
        ]

        await asyncio.gather(*tasks)

        assert cache.size == 50


class TestAsyncTTLCacheEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_cache_with_size_one(self):
        """Test cache with max_size=1."""
        cache = AsyncTTLCache[str, int](max_size=1, ttl=60.0)

        await cache.set("key1", 1)
        assert await cache.get("key1") == 1

        await cache.set("key2", 2)
        assert await cache.get("key1") is None
        assert await cache.get("key2") == 2

    @pytest.mark.asyncio
    async def test_very_small_ttl(self):
        """Test cache with very small TTL."""
        cache = AsyncTTLCache[str, int](max_size=100, ttl=0.001)

        await cache.set("key1", 42)
        assert await cache.get("key1") == 42

        await asyncio.sleep(0.01)

        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_different_key_types(self):
        """Test cache with different key types."""
        cache_str = AsyncTTLCache[str, int](max_size=100, ttl=60.0)
        await cache_str.set("string_key", 42)
        assert await cache_str.get("string_key") == 42

        cache_int = AsyncTTLCache[int, str](max_size=100, ttl=60.0)
        await cache_int.set(123, "value")
        assert await cache_int.get(123) == "value"

        cache_tuple = AsyncTTLCache[tuple, str](max_size=100, ttl=60.0)
        await cache_tuple.set((1, 2, 3), "tuple_value")
        assert await cache_tuple.get((1, 2, 3)) == "tuple_value"

    @pytest.mark.asyncio
    async def test_different_value_types(self):
        """Test cache with different value types."""
        cache = AsyncTTLCache[str, object](max_size=100, ttl=60.0)

        await cache.set("int", 42)
        assert await cache.get("int") == 42

        await cache.set("str", "hello")
        assert await cache.get("str") == "hello"

        await cache.set("list", [1, 2, 3])
        assert await cache.get("list") == [1, 2, 3]

        await cache.set("dict", {"key": "value"})
        assert await cache.get("dict") == {"key": "value"}

        await cache.set("none", None)
        assert await cache.get("none") is None

    @pytest.mark.asyncio
    async def test_get_expired_updates_lru(self):
        """Test that getting an expired entry removes it and doesn't affect LRU of other entries."""
        cache = AsyncTTLCache[str, int](max_size=3, ttl=0.1)

        await cache.set("key1", 1)
        await cache.set("key2", 2)
        await cache.set("key3", 3)

        await asyncio.sleep(0.15)

        # All entries are expired, getting key1 removes it
        await cache.get("key1")

        # Add a fresh entry (key4)
        await cache.set("key4", 4)

        # All original entries are expired and should be gone
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None  # Expired
        assert await cache.get("key3") is None  # Expired
        assert await cache.get("key4") == 4  # Only the fresh entry remains

    @pytest.mark.asyncio
    async def test_clear_empty_cache(self):
        """Test clearing an empty cache."""
        cache = AsyncTTLCache[str, int](max_size=100, ttl=60.0)

        assert cache.size == 0
        await cache.clear()
        assert cache.size == 0

    @pytest.mark.asyncio
    async def test_has_with_expired_entry(self):
        """Test that has() properly handles expired entries."""
        cache = AsyncTTLCache[str, int](max_size=100, ttl=0.1)

        await cache.set("key1", 42)

        assert await cache.has("key1") is True

        await asyncio.sleep(0.15)

        assert await cache.has("key1") is False
        assert cache.size == 0  # Should be removed

    @pytest.mark.asyncio
    async def test_invalidate_expired_entry(self):
        """Test invalidating an expired entry removes it."""
        cache = AsyncTTLCache[str, int](max_size=100, ttl=0.1)

        await cache.set("key1", 42)

        await asyncio.sleep(0.15)

        # Invalidating expired entry - it still exists in cache but is expired
        # The invalidate method checks if key exists, not if it's expired
        result = await cache.invalidate("key1")
        # The entry exists in the OrderedDict, so it returns True
        assert result is True
        # After invalidation, it should be gone
        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_multiple_entries_same_expiration(self):
        """Test multiple entries expiring at the same time."""
        cache = AsyncTTLCache[str, int](max_size=100, ttl=0.1)

        # Add all entries quickly so they have similar expiration times
        for i in range(10):
            await cache.set(f"key{i}", i)

        assert cache.size == 10

        await asyncio.sleep(0.15)

        # All should be expired
        for i in range(10):
            assert await cache.get(f"key{i}") is None

    @pytest.mark.asyncio
    async def test_update_key_extends_ttl(self):
        """Test that updating a key extends its TTL."""
        cache = AsyncTTLCache[str, int](max_size=100, ttl=0.1)

        await cache.set("key1", 42)

        # Wait a bit, then update
        await asyncio.sleep(0.05)
        await cache.set("key1", 100)

        # Wait for original TTL to expire
        await asyncio.sleep(0.07)

        # Should still be valid because update extended TTL
        value = await cache.get("key1")
        assert value == 100
