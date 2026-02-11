"""Tests for AsyncCache with TTL support."""

import asyncio
import time

import pytest

from nodetool.concurrency import AsyncCache


@pytest.mark.asyncio
async def test_cache_basic_get_set():
    """Test basic get and set operations."""
    cache = AsyncCache(ttl=60.0)

    # Set a value
    await cache.set("key1", "value1")

    # Get the value back
    value = await cache.get("key1")
    assert value == "value1"

    # Get non-existent key
    value = await cache.get("nonexistent", default="default")
    assert value == "default"

    await cache.close()


@pytest.mark.asyncio
async def test_cache_get_or_compute():
    """Test get_or_compute method."""
    cache = AsyncCache(ttl=60.0)
    compute_count = 0

    async def compute_value():
        nonlocal compute_count
        compute_count += 1
        await asyncio.sleep(0.01)
        return "computed"

    # First call should compute
    value1 = await cache.get_or_compute("expensive_key", lambda: compute_value())
    assert value1 == "computed"
    assert compute_count == 1

    # Second call should use cached value
    value2 = await cache.get_or_compute("expensive_key", lambda: compute_value())
    assert value2 == "computed"
    assert compute_count == 1  # Should not increment

    await cache.close()


@pytest.mark.asyncio
async def test_cache_get_or_compute_with_sync_function():
    """Test get_or_compute with synchronous function."""
    cache = AsyncCache(ttl=60.0)
    compute_count = 0

    def compute_value():
        nonlocal compute_count
        compute_count += 1
        return "sync_computed"

    # First call should compute
    value1 = await cache.get_or_compute("sync_key", lambda: compute_value())
    assert value1 == "sync_computed"
    assert compute_count == 1

    # Second call should use cached value
    value2 = await cache.get_or_compute("sync_key", lambda: compute_value())
    assert value2 == "sync_computed"
    assert compute_count == 1

    await cache.close()


@pytest.mark.asyncio
async def test_cache_ttl_expiration():
    """Test TTL-based expiration."""
    cache = AsyncCache(ttl=0.1)  # 100ms TTL

    # Set a value
    await cache.set("key1", "value1")

    # Value should be available immediately
    value = await cache.get("key1")
    assert value == "value1"

    # Wait for expiration
    await asyncio.sleep(0.15)

    # Value should be expired
    value = await cache.get("key1", default="expired")
    assert value == "expired"

    await cache.close()


@pytest.mark.asyncio
async def test_cache_custom_ttl():
    """Test custom TTL per entry."""
    cache = AsyncCache(ttl=1.0)  # Default 1s TTL

    # Set with custom TTL
    await cache.set("short_lived", "value1", ttl=0.1)
    await cache.set("long_lived", "value2", ttl=1.0)

    # Both should be available immediately
    assert await cache.get("short_lived") == "value1"
    assert await cache.get("long_lived") == "value2"

    # Wait for short_lived to expire
    await asyncio.sleep(0.15)

    # short_lived should be expired
    assert await cache.get("short_lived", default="expired") == "expired"
    # long_lived should still be available
    assert await cache.get("long_lived") == "value2"

    await cache.close()


@pytest.mark.asyncio
async def test_cache_no_expiration():
    """Test cache with no expiration (None TTL)."""
    cache = AsyncCache(ttl=None)

    # Set a value
    await cache.set("key1", "value1")

    # Value should be available immediately
    assert await cache.get("key1") == "value1"

    # Wait longer than any reasonable TTL
    await asyncio.sleep(0.2)

    # Value should still be available
    assert await cache.get("key1") == "value1"

    await cache.close()


@pytest.mark.asyncio
async def test_cache_has():
    """Test has() method."""
    cache = AsyncCache(ttl=60.0)

    # Key doesn't exist
    assert await cache.has("key1") is False

    # Set a value
    await cache.set("key1", "value1")

    # Key exists
    assert await cache.has("key1") is True

    await cache.close()


@pytest.mark.asyncio
async def test_cache_has_with_expiration():
    """Test has() method respects expiration."""
    cache = AsyncCache(ttl=0.1)

    # Set a value
    await cache.set("key1", "value1")

    # Key exists
    assert await cache.has("key1") is True

    # Wait for expiration
    await asyncio.sleep(0.15)

    # Key no longer exists
    assert await cache.has("key1") is False

    await cache.close()


@pytest.mark.asyncio
async def test_cache_delete():
    """Test delete() method."""
    cache = AsyncCache(ttl=60.0)

    # Set a value
    await cache.set("key1", "value1")
    assert await cache.has("key1") is True

    # Delete the key
    result = await cache.delete("key1")
    assert result is True
    assert await cache.has("key1") is False

    # Delete non-existent key
    result = await cache.delete("nonexistent")
    assert result is False

    await cache.close()


@pytest.mark.asyncio
async def test_cache_clear():
    """Test clear() method."""
    cache = AsyncCache(ttl=60.0)

    # Set multiple values
    await cache.set("key1", "value1")
    await cache.set("key2", "value2")
    await cache.set("key3", "value3")

    assert await cache.size() == 3

    # Clear all
    await cache.clear()

    assert await cache.size() == 0
    assert await cache.get("key1", default="gone") == "gone"
    assert await cache.get("key2", default="gone") == "gone"
    assert await cache.get("key3", default="gone") == "gone"

    await cache.close()


@pytest.mark.asyncio
async def test_cache_max_size_eviction():
    """Test max_size eviction policy."""
    cache = AsyncCache(ttl=60.0, max_size=3)

    # Fill to max capacity
    await cache.set("key1", "value1")
    await cache.set("key2", "value2")
    await cache.set("key3", "value3")

    assert await cache.size() == 3

    # Add one more - should evict oldest
    await cache.set("key4", "value4")

    assert await cache.size() == 3
    # First key should be evicted (FIFO)
    assert await cache.get("key1", default="evicted") == "evicted"
    # Other keys should still exist
    assert await cache.get("key2") == "value2"
    assert await cache.get("key3") == "value3"
    assert await cache.get("key4") == "value4"

    await cache.close()


@pytest.mark.asyncio
async def test_cache_max_size_update_existing():
    """Test that updating existing key doesn't trigger eviction."""
    cache = AsyncCache(ttl=60.0, max_size=2)

    # Fill to max capacity
    await cache.set("key1", "value1")
    await cache.set("key2", "value2")

    # Update existing key - should not trigger eviction
    await cache.set("key1", "value1_updated")

    assert await cache.size() == 2
    assert await cache.get("key1") == "value1_updated"
    assert await cache.get("key2") == "value2"

    await cache.close()


@pytest.mark.asyncio
async def test_cache_stats():
    """Test cache statistics."""
    cache = AsyncCache(ttl=60.0)

    # Initial stats
    stats = await cache.get_stats()
    assert stats["size"] == 0
    assert stats["hits"] == 0
    assert stats["misses"] == 0
    assert stats["evictions"] == 0
    assert stats["hit_rate"] == 0.0

    # Set a value
    await cache.set("key1", "value1")

    # Cache hit
    await cache.get("key1")
    stats = await cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 0
    assert stats["hit_rate"] == 1.0

    # Cache miss
    await cache.get("nonexistent")
    stats = await cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5

    await cache.close()


@pytest.mark.asyncio
async def test_cache_stats_with_eviction():
    """Test cache statistics include evictions."""
    cache = AsyncCache(ttl=60.0, max_size=2)

    # Fill and evict
    await cache.set("key1", "value1")
    await cache.set("key2", "value2")
    await cache.set("key3", "value3")  # Evicts key1

    stats = await cache.get_stats()
    assert stats["evictions"] == 1

    await cache.close()


@pytest.mark.asyncio
async def test_cache_reset_stats():
    """Test reset_stats() method."""
    cache = AsyncCache(ttl=60.0)

    # Generate some stats
    await cache.set("key1", "value1")
    await cache.get("key1")  # hit
    await cache.get("nonexistent")  # miss

    stats_before = await cache.get_stats()
    assert stats_before["hits"] > 0 or stats_before["misses"] > 0

    # Reset stats
    await cache.reset_stats()

    stats_after = await cache.get_stats()
    assert stats_after["hits"] == 0
    assert stats_after["misses"] == 0
    assert stats_after["evictions"] == 0
    assert stats_after["hit_rate"] == 0.0

    # Cache should still have data
    assert await cache.get("key1") == "value1"

    await cache.close()


@pytest.mark.asyncio
async def test_cache_size():
    """Test size() method."""
    cache = AsyncCache(ttl=60.0)

    assert await cache.size() == 0

    await cache.set("key1", "value1")
    assert await cache.size() == 1

    await cache.set("key2", "value2")
    assert await cache.size() == 2

    await cache.delete("key1")
    assert await cache.size() == 1

    await cache.clear()
    assert await cache.size() == 0

    await cache.close()


@pytest.mark.asyncio
async def test_cache_cleanup_task():
    """Test background cleanup task."""
    cache = AsyncCache(ttl=0.1, cleanup_interval=0.1)

    # Add entries that will expire
    await cache.set("key1", "value1")
    await cache.set("key2", "value2")
    await cache.set("key3", "value3")

    assert await cache.size() == 3

    # Wait for cleanup task to run
    await asyncio.sleep(0.25)

    # Expired entries should be cleaned up
    # The cleanup runs every 0.1s, so after 0.25s it should have run at least once
    assert await cache.size() == 0

    await cache.close()


@pytest.mark.asyncio
async def test_cache_manual_cleanup():
    """Test manual cleanup() method."""
    cache = AsyncCache(ttl=0.1)

    # Disable background cleanup for this test
    await cache.close()
    cache_no_cleanup = AsyncCache(ttl=0.1, cleanup_interval=1000.0)

    # Add entries that will expire
    await cache_no_cleanup.set("key1", "value1")
    await cache_no_cleanup.set("key2", "value2")

    assert await cache_no_cleanup.size() == 2

    # Wait for expiration
    await asyncio.sleep(0.15)

    # Entries are still in cache (not cleaned up yet)
    assert await cache_no_cleanup.size() == 2

    # Manually trigger cleanup
    removed = await cache_no_cleanup.cleanup()
    assert removed == 2
    assert await cache_no_cleanup.size() == 0

    await cache_no_cleanup.close()


@pytest.mark.asyncio
async def test_cache_context_manager():
    """Test using cache as a context manager."""
    async with AsyncCache(ttl=60.0) as cache:
        await cache.set("key1", "value1")
        assert await cache.get("key1") == "value1"

    # Cache should be closed after exiting context
    # Background cleanup task should be cancelled


@pytest.mark.asyncio
async def test_cache_close():
    """Test close() method."""
    cache = AsyncCache(ttl=0.1)

    await cache.set("key1", "value1")
    assert await cache.get("key1") == "value1"

    # Close the cache
    await cache.close()

    # Background cleanup task should be stopped
    # Cache should still be accessible for reads (not cleared by close)
    assert await cache.get("key1") == "value1"


@pytest.mark.asyncio
async def test_cache_multiple_types():
    """Test cache with different value types."""
    cache = AsyncCache(ttl=60.0)

    # String
    await cache.set("string", "value")
    assert await cache.get("string") == "value"

    # Integer
    await cache.set("int", 42)
    assert await cache.get("int") == 42

    # Float
    await cache.set("float", 3.14)
    assert await cache.get("float") == 3.14

    # List
    await cache.set("list", [1, 2, 3])
    assert await cache.get("list") == [1, 2, 3]

    # Dict
    await cache.set("dict", {"key": "value"})
    assert await cache.get("dict") == {"key": "value"}

    # None
    await cache.set("none", None)
    assert await cache.get("none") is None

    await cache.close()


@pytest.mark.asyncio
async def test_cache_invalid_parameters():
    """Test that invalid parameters raise ValueError."""
    # Negative TTL
    with pytest.raises(ValueError, match="ttl must be non-negative"):
        AsyncCache(ttl=-1.0)

    # Invalid max_size
    with pytest.raises(ValueError, match="max_size must be positive"):
        AsyncCache(max_size=0)

    # Negative cleanup_interval
    with pytest.raises(ValueError, match="cleanup_interval must be positive"):
        AsyncCache(cleanup_interval=-1.0)


@pytest.mark.asyncio
async def test_cache_concurrent_access():
    """Test cache with concurrent access."""
    cache = AsyncCache(ttl=60.0)
    results = []

    async def worker(worker_id: int):
        for i in range(10):
            key = f"key_{i}"
            # Try to get or compute
            value = await cache.get_or_compute(
                key,
                lambda: f"worker_{worker_id}_value_{i}",
            )
            results.append((worker_id, i, value))

    # Run multiple workers concurrently
    tasks = [worker(i) for i in range(5)]
    await asyncio.gather(*tasks)

    # All workers should have gotten values
    assert len(results) == 50

    # Check that cache has the expected keys
    for i in range(10):
        key = f"key_{i}"
        value = await cache.get(key)
        assert value is not None
        assert value.endswith(f"_value_{i}")

    await cache.close()


@pytest.mark.asyncio
async def test_cache_get_or_compute_with_none_value():
    """Test get_or_compute when compute_fn returns None."""
    cache = AsyncCache(ttl=60.0)

    async def compute_none():
        return None

    # First call computes None
    value1 = await cache.get_or_compute("none_key", lambda: compute_none())
    assert value1 is None

    # Since None is a valid return value, we need to be careful
    # The current implementation will treat None as a cache miss
    # Let's verify the behavior
    compute_count = 0

    async def compute_with_counter():
        nonlocal compute_count
        compute_count += 1
        return None

    # First call
    await cache.get_or_compute("counter_key", lambda: compute_with_counter())
    assert compute_count == 1

    # Second call - since None is returned, it will re-compute
    await cache.get_or_compute("counter_key", lambda: compute_with_counter())
    assert compute_count == 2  # Increments because None triggers re-compute

    await cache.close()
