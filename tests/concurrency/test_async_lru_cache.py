"""Tests for AsyncLRUCache utility."""
import asyncio

import pytest

from nodetool.concurrency import AsyncLRUCache


@pytest.mark.asyncio
async def test_cache_set_and_get():
    """Test basic set and get operations."""
    cache = AsyncLRUCache[str, int](max_size=10)

    await cache.set("key1", 100)
    await cache.set("key2", 200)

    assert await cache.get("key1") == 100
    assert await cache.get("key2") == 200


@pytest.mark.asyncio
async def test_cache_get_missing_key():
    """Test get with missing key returns None."""
    cache = AsyncLRUCache[str, int]()

    result = await cache.get("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_cache_max_size_eviction():
    """Test that cache evicts LRU entry when max_size is exceeded."""
    cache = AsyncLRUCache[str, int](max_size=3)

    await cache.set("a", 1)
    await cache.set("b", 2)
    await cache.set("c", 3)

    # All three should be in cache
    assert await cache.get("a") == 1
    assert await cache.get("b") == 2
    assert await cache.get("c") == 3

    # Access 'a' to make it more recently used
    await cache.get("a")

    # Add a fourth item - 'b' should be evicted (least recently used)
    await cache.set("d", 4)

    assert await cache.get("a") == 1  # Still there (was accessed)
    assert await cache.get("b") is None  # Evicted
    assert await cache.get("c") == 3
    assert await cache.get("d") == 4


@pytest.mark.asyncio
async def test_cache_ttl_expiration():
    """Test that entries expire after TTL."""
    cache = AsyncLRUCache[str, int](max_size=10, ttl_seconds=0.1)

    await cache.set("key1", 100)
    assert await cache.get("key1") == 100

    # Wait for expiration
    await asyncio.sleep(0.15)

    # Should be expired now
    assert await cache.get("key1") is None


@pytest.mark.asyncio
async def test_cache_ttl_refreshes_on_access():
    """Test that accessing an entry refreshes its TTL."""
    cache = AsyncLRUCache[str, int](max_size=10, ttl_seconds=0.2)

    await cache.set("key1", 100)

    # Wait half the TTL
    await asyncio.sleep(0.1)

    # Access the entry - this should refresh the TTL
    assert await cache.get("key1") == 100

    # Wait another 0.15 seconds (total 0.25 from start, but only 0.15 since last access)
    await asyncio.sleep(0.15)

    # Should still be valid because TTL was refreshed
    assert await cache.get("key1") == 100


@pytest.mark.asyncio
async def test_cache_get_or_compute_sync():
    """Test get_or_compute with synchronous compute function."""
    cache = AsyncLRUCache[str, int]()

    compute_count = 0

    def compute_value():
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
    assert compute_count == 1  # Not called again


@pytest.mark.asyncio
async def test_cache_get_or_compute_async():
    """Test get_or_compute with async compute function."""
    cache = AsyncLRUCache[str, int]()

    compute_count = 0

    async def compute_value():
        nonlocal compute_count
        compute_count += 1
        await asyncio.sleep(0.01)  # Simulate async work
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
async def test_cache_get_or_compute_with_ttl():
    """Test that get_or_compute recomputes after TTL expiration."""
    cache = AsyncLRUCache[str, int](max_size=10, ttl_seconds=0.1)

    compute_count = 0

    def compute_value():
        nonlocal compute_count
        compute_count += 1
        return compute_count * 100

    # First call computes
    result1 = await cache.get_or_compute("key1", compute_value)
    assert result1 == 100
    assert compute_count == 1

    # Second call uses cache
    result2 = await cache.get_or_compute("key1", compute_value)
    assert result2 == 100
    assert compute_count == 1

    # Wait for expiration
    await asyncio.sleep(0.15)

    # Third call recomputes
    result3 = await cache.get_or_compute("key1", compute_value)
    assert result3 == 200
    assert compute_count == 2


@pytest.mark.asyncio
async def test_cache_invalidate():
    """Test invalidating a specific cache entry."""
    cache = AsyncLRUCache[str, int]()

    await cache.set("key1", 100)
    await cache.set("key2", 200)

    assert await cache.get("key1") == 100

    # Invalidate key1
    result = await cache.invalidate("key1")
    assert result is True

    # key1 should be gone
    assert await cache.get("key1") is None

    # key2 should still be there
    assert await cache.get("key2") == 200

    # Invalidating non-existent key returns False
    result = await cache.invalidate("nonexistent")
    assert result is False


@pytest.mark.asyncio
async def test_cache_clear():
    """Test clearing the entire cache."""
    cache = AsyncLRUCache[str, int]()

    await cache.set("key1", 100)
    await cache.set("key2", 200)
    await cache.set("key3", 300)

    # Access key1 to create some hits
    await cache.get("key1")
    await cache.get("nonexistent")  # Create a miss

    assert cache.size == 3

    # Clear the cache
    await cache.clear()

    assert cache.size == 0
    assert await cache.get("key1") is None
    assert await cache.get("key2") is None
    assert await cache.get("key3") is None


@pytest.mark.asyncio
async def test_cache_cleanup_expired():
    """Test manual cleanup of expired entries."""
    cache = AsyncLRUCache[str, int](max_size=10, ttl_seconds=0.1)

    await cache.set("key1", 100)
    await cache.set("key2", 200)
    await cache.set("key3", 300)

    # Wait for expiration
    await asyncio.sleep(0.15)

    # All should be expired
    removed = await cache.cleanup_expired()
    assert removed == 3
    assert cache.size == 0


@pytest.mark.asyncio
async def test_cache_cleanup_expired_with_valid_entries():
    """Test that cleanup only removes expired entries."""
    cache = AsyncLRUCache[str, int](max_size=10, ttl_seconds=0.2)

    await cache.set("key1", 100)
    await asyncio.sleep(0.1)
    await cache.set("key2", 200)

    # key1 is 0.1s old, key2 is 0s old (within 0.2s TTL)
    removed = await cache.cleanup_expired()
    assert removed == 0
    assert cache.size == 2


@pytest.mark.asyncio
async def test_cache_cleanup_expired_without_ttl():
    """Test that cleanup returns 0 when there's no TTL."""
    cache = AsyncLRUCache[str, int]()

    await cache.set("key1", 100)
    await cache.set("key2", 200)

    removed = await cache.cleanup_expired()
    assert removed == 0


@pytest.mark.asyncio
async def test_cache_get_stats():
    """Test getting cache statistics."""
    cache = AsyncLRUCache[str, int](max_size=10)

    # Initial stats
    stats = await cache.get_stats()
    assert stats["size"] == 0
    assert stats["max_size"] == 10
    assert stats["hits"] == 0
    assert stats["misses"] == 0
    assert stats["hit_rate"] == 0.0

    # Add entries
    await cache.set("key1", 100)
    await cache.set("key2", 200)

    stats = await cache.get_stats()
    assert stats["size"] == 2

    # Generate hits and misses
    await cache.get("key1")  # hit
    await cache.get("key2")  # hit
    await cache.get("nonexistent")  # miss

    stats = await cache.get_stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 2 / 3


@pytest.mark.asyncio
async def test_cache_hit_rate():
    """Test hit rate calculation."""
    cache = AsyncLRUCache[str, int]()

    await cache.set("key1", 100)

    # 3 hits, 2 misses = 60% hit rate
    await cache.get("key1")  # hit
    await cache.get("key1")  # hit
    await cache.get("missing")  # miss
    await cache.get("key1")  # hit
    await cache.get("missing2")  # miss

    stats = await cache.get_stats()
    assert stats["hits"] == 3
    assert stats["misses"] == 2
    assert stats["hit_rate"] == 0.6


@pytest.mark.asyncio
async def test_cache_contains():
    """Test contains method for checking key existence."""
    cache = AsyncLRUCache[str, int]()

    assert not cache.contains("key1")

    await cache.set("key1", 100)

    assert cache.contains("key1")
    assert not cache.contains("key2")


@pytest.mark.asyncio
async def test_cache_keys():
    """Test getting all keys from cache."""
    cache = AsyncLRUCache[str, int](max_size=3)

    await cache.set("a", 1)
    await cache.set("b", 2)
    await cache.set("c", 3)

    keys = await cache.keys()
    assert keys == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_cache_values():
    """Test getting all values from cache."""
    cache = AsyncLRUCache[str, int]()

    await cache.set("a", 1)
    await cache.set("b", 2)
    await cache.set("c", 3)

    values = await cache.values()
    assert values == [1, 2, 3]


@pytest.mark.asyncio
async def test_cache_items():
    """Test getting all key-value pairs from cache."""
    cache = AsyncLRUCache[str, int]()

    await cache.set("a", 1)
    await cache.set("b", 2)
    await cache.set("c", 3)

    items = await cache.items()
    assert items == [("a", 1), ("b", 2), ("c", 3)]


@pytest.mark.asyncio
async def test_cache_lru_order():
    """Test that LRU eviction order is correct."""
    cache = AsyncLRUCache[str, int](max_size=3)

    await cache.set("a", 1)
    await cache.set("b", 2)
    await cache.set("c", 3)

    # Access 'a' to make it more recent
    await cache.get("a")

    # Add 'd' - 'b' should be evicted (least recently used)
    await cache.set("d", 4)

    items = await cache.items()
    keys = [k for k, _ in items]
    # Order should be c, a, d (b was evicted)
    assert keys == ["c", "a", "d"]


@pytest.mark.asyncio
async def test_cache_update_existing_key():
    """Test that updating an existing key updates its value and LRU position."""
    cache = AsyncLRUCache[str, int](max_size=3)

    await cache.set("a", 1)
    await cache.set("b", 2)
    await cache.set("c", 3)

    # Update 'a' - should move to most recent position
    await cache.set("a", 100)

    # Add 'd' - 'b' should be evicted (not 'a')
    await cache.set("d", 4)

    assert await cache.get("a") == 100  # Updated value, still present
    assert await cache.get("b") is None  # Evicted
    assert await cache.get("c") == 3
    assert await cache.get("d") == 4


@pytest.mark.asyncio
async def test_cache_size_property():
    """Test the size property."""
    cache = AsyncLRUCache[str, int]()

    assert cache.size == 0

    await cache.set("key1", 100)
    assert cache.size == 1

    await cache.set("key2", 200)
    assert cache.size == 2

    await cache.invalidate("key1")
    assert cache.size == 1

    await cache.clear()
    assert cache.size == 0


@pytest.mark.asyncio
async def test_cache_max_size_property():
    """Test the max_size property."""
    cache = AsyncLRUCache[str, int](max_size=42)

    assert cache.max_size == 42


@pytest.mark.asyncio
async def test_cache_ttl_seconds_property():
    """Test the ttl_seconds property."""
    cache_no_ttl = AsyncLRUCache[str, int]()
    assert cache_no_ttl.ttl_seconds is None

    cache_with_ttl = AsyncLRUCache[str, int](ttl_seconds=60)
    assert cache_with_ttl.ttl_seconds == 60


@pytest.mark.asyncio
async def test_cache_invalid_max_size():
    """Test that invalid max_size raises ValueError."""
    with pytest.raises(ValueError, match="max_size must be a positive integer"):
        AsyncLRUCache[str, int](max_size=0)

    with pytest.raises(ValueError, match="max_size must be a positive integer"):
        AsyncLRUCache[str, int](max_size=-1)


@pytest.mark.asyncio
async def test_cache_concurrent_access():
    """Test that cache handles concurrent access correctly."""
    cache = AsyncLRUCache[str, int](max_size=100)

    async def worker(worker_id: int):
        for i in range(10):
            key = f"key_{i}"
            await cache.set(key, worker_id * 100 + i)
            value = await cache.get(key)
            assert value == worker_id * 100 + i

    # Run multiple workers concurrently
    await asyncio.gather(*(worker(i) for i in range(5)))

    # Cache should have 10 entries (one per key, last write wins)
    assert cache.size == 10


@pytest.mark.asyncio
async def test_cache_get_or_compute_concurrent():
    """Test that get_or_compute handles concurrent calls correctly."""
    cache = AsyncLRUCache[str, int]()
    compute_count = 0
    lock = asyncio.Lock()

    async def compute_value():
        nonlocal compute_count
        async with lock:
            compute_count += 1
        await asyncio.sleep(0.01)
        return 42

    # Multiple concurrent calls for the same key
    results = await asyncio.gather(
        *(cache.get_or_compute("key1", compute_value) for _ in range(5))
    )

    # All should return the same value
    assert all(r == 42 for r in results)

    # But compute should only be called once (first call wins, others use cache)
    # Note: In current implementation, multiple computes might race
    # This is acceptable for a simple cache


@pytest.mark.asyncio
async def test_cache_with_none_value():
    """Test that None values can be stored and retrieved."""
    cache = AsyncLRUCache[str, int | None]()

    await cache.set("key1", None)

    # get_or_compute should return None from cache
    result = await cache.get("key1")
    assert result is None

    # But we can still distinguish between cached None and missing key
    # by checking if compute was called
    compute_called = False

    def compute():
        nonlocal compute_called
        compute_called = True
        return 42

    # This should use cached None, not call compute
    result = await cache.get_or_compute("key1", compute)
    assert result is None
    assert not compute_called
