"""
Tests for AsyncTTLCache.
"""

import asyncio
import time

import pytest

from nodetool.concurrency import AsyncTTLCache


class TestAsyncTTLCache:
    """Test suite for AsyncTTLCache."""

    @pytest.mark.asyncio
    async def test_basic_get_set(self):
        """Test basic get and set operations."""
        cache: AsyncTTLCache[str, int] = AsyncTTLCache(maxsize=10, ttl=60.0)

        # Set and get
        await cache.set("key1", 100)
        value = await cache.get("key1")
        assert value == 100

        # Get non-existent key
        assert await cache.get("nonexistent") is None
        assert await cache.get("nonexistent", default=42) == 42

    @pytest.mark.asyncio
    async def test_get_or_compute_cache_miss(self):
        """Test get_or_compute on cache miss."""
        cache: AsyncTTLCache[str, int] = AsyncTTLCache(maxsize=10, ttl=60.0)
        call_count = 0

        async def compute():
            nonlocal call_count
            call_count += 1
            return 42

        # First call - cache miss, computes
        result = await cache.get_or_compute("key", compute)
        assert result == 42
        assert call_count == 1

        # Second call - cache hit, no compute
        result = await cache.get_or_compute("key", compute)
        assert result == 42
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        cache: AsyncTTLCache[str, int] = AsyncTTLCache(maxsize=10, ttl=0.1)
        call_count = 0

        async def compute():
            nonlocal call_count
            call_count += 1
            return 42

        # First call
        result = await cache.get_or_compute("key", compute)
        assert result == 42
        assert call_count == 1

        # Still cached
        result = await cache.get_or_compute("key", compute)
        assert result == 42
        assert call_count == 1

        # Wait for TTL to expire
        await asyncio.sleep(0.15)

        # Should recompute
        result = await cache.get_or_compute("key", compute)
        assert result == 42
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """Test LRU eviction when maxsize is reached."""
        cache: AsyncTTLCache[str, int] = AsyncTTLCache(maxsize=3, ttl=60.0)

        # Fill cache
        await cache.set("key1", 1)
        await cache.set("key2", 2)
        await cache.set("key3", 3)

        # Access key1 to make it recently used
        await cache.get("key1")

        # Add key4 - should evict key2 (least recently used)
        await cache.set("key4", 4)

        assert await cache.get("key1") == 1  # Still present
        assert await cache.get("key2") is None  # Evicted
        assert await cache.get("key3") == 3  # Still present
        assert await cache.get("key4") == 4  # Present

    @pytest.mark.asyncio
    async def test_has_fresh(self):
        """Test has_fresh method."""
        cache: AsyncTTLCache[str, int] = AsyncTTLCache(maxsize=10, ttl=0.1)

        # Non-existent key
        assert not await cache.has_fresh("key")

        # Set key
        await cache.set("key", 42)
        assert await cache.has_fresh("key")

        # Wait for expiration
        await asyncio.sleep(0.15)
        assert not await cache.has_fresh("key")

    @pytest.mark.asyncio
    async def test_invalidate(self):
        """Test invalidation of specific keys."""
        cache: AsyncTTLCache[str, int] = AsyncTTLCache(maxsize=10, ttl=60.0)

        await cache.set("key1", 1)
        await cache.set("key2", 2)

        assert await cache.has_fresh("key1")
        assert await cache.has_fresh("key2")

        # Invalidate key1
        assert await cache.invalidate("key1")
        assert not await cache.has_fresh("key1")
        assert await cache.has_fresh("key2")

        # Invalidate non-existent key
        assert not await cache.invalidate("nonexistent")

    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clearing all cache entries."""
        cache: AsyncTTLCache[str, int] = AsyncTTLCache(maxsize=10, ttl=60.0)

        await cache.set("key1", 1)
        await cache.set("key2", 2)
        await cache.set("key3", 3)

        assert await cache.size() == 3

        await cache.clear()

        assert await cache.size() == 0
        assert not await cache.has_fresh("key1")
        assert not await cache.has_fresh("key2")
        assert not await cache.has_fresh("key3")

    @pytest.mark.asyncio
    async def test_cleanup_expired(self):
        """Test cleanup of expired entries."""
        cache: AsyncTTLCache[str, int] = AsyncTTLCache(maxsize=10, ttl=0.1)

        await cache.set("key1", 1)
        await cache.set("key2", 2)
        await cache.set("key3", 3)

        assert await cache.size() == 3

        # Wait for expiration
        await asyncio.sleep(0.15)

        # Cleanup should remove 3 entries
        removed = await cache.cleanup_expired()
        assert removed == 3
        assert await cache.size() == 0

    @pytest.mark.asyncio
    async def test_cache_stampede_prevention(self):
        """Test that cache stampede (thundering herd) is prevented."""
        cache: AsyncTTLCache[str, int] = AsyncTTLCache(maxsize=10, ttl=60.0)
        call_count = 0
        compute_delay = 0.1

        async def compute():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(compute_delay)
            return 42

        # Launch multiple concurrent requests for the same key
        tasks = [
            cache.get_or_compute("key", compute) for _ in range(10)
        ]

        results = await asyncio.gather(*tasks)

        # All should get the same result
        assert all(r == 42 for r in results)

        # But compute should only be called once
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_stale_while_revalidate(self):
        """Test stale-while-revalidate pattern."""
        cache: AsyncTTLCache[str, int] = AsyncTTLCache(maxsize=10, ttl=0.1)
        call_count = 0

        async def compute():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)  # Simulate work
            return call_count  # Return call count as value

        # First call
        result = await cache.get_or_compute("key", compute)
        assert result == 1
        assert call_count == 1

        # Wait for expiration
        await asyncio.sleep(0.15)

        # Request with allow_stale - should return stale value immediately
        # and trigger background refresh
        start_time = asyncio.get_event_loop().time()
        result = await cache.get_or_compute("key", compute, allow_stale=True)
        elapsed = asyncio.get_event_loop().time() - start_time

        # Should return stale value quickly (without waiting for compute)
        assert result == 1  # Still returns old value
        assert elapsed < 0.05  # Should be fast (not waiting for compute sleep)

        # Wait for background refresh to complete
        await asyncio.sleep(0.2)

        # Now check that a fresh request gets the updated value
        # The background refresh should have happened
        result = await cache.get_or_compute("key", compute)
        # Result should be >= 2 since background refresh happened
        assert result >= 2

    @pytest.mark.asyncio
    async def test_compute_fn_exception(self):
        """Test handling of exceptions in compute_fn."""
        cache: AsyncTTLCache[str, int] = AsyncTTLCache(maxsize=10, ttl=60.0)

        async def failing_compute():
            raise ValueError("Compute failed")

        # Should propagate exception
        with pytest.raises(ValueError, match="Compute failed"):
            await cache.get_or_compute("key", failing_compute)

        # Key should not be in cache
        assert not await cache.has_fresh("key")

    @pytest.mark.asyncio
    async def test_custom_ttl_per_entry(self):
        """Test custom TTL for individual entries."""
        cache: AsyncTTLCache[str, int] = AsyncTTLCache(maxsize=10, ttl=60.0)

        # Set with custom TTL
        await cache.set("key1", 1, ttl=0.1)
        await cache.set("key2", 2, ttl=1.0)

        # Both should be fresh
        assert await cache.has_fresh("key1")
        assert await cache.has_fresh("key2")

        # Wait for key1 to expire
        await asyncio.sleep(0.15)

        # Only key1 should be expired
        assert not await cache.has_fresh("key1")
        assert await cache.has_fresh("key2")

    @pytest.mark.asyncio
    async def test_unlimited_cache_size(self):
        """Test cache with unlimited size (maxsize=0)."""
        cache: AsyncTTLCache[str, int] = AsyncTTLCache(maxsize=0, ttl=60.0)

        # Add many entries
        for i in range(1000):
            await cache.set(f"key{i}", i)

        # All should be present
        assert await cache.size() == 1000
        assert await cache.get("key0") == 0
        assert await cache.get("key999") == 999

    @pytest.mark.asyncio
    async def test_size(self):
        """Test size method."""
        cache: AsyncTTLCache[str, int] = AsyncTTLCache(maxsize=10, ttl=60.0)

        assert await cache.size() == 0

        await cache.set("key1", 1)
        assert await cache.size() == 1

        await cache.set("key2", 2)
        await cache.set("key3", 3)
        assert await cache.size() == 3

        await cache.invalidate("key1")
        assert await cache.size() == 2

        await cache.clear()
        assert await cache.size() == 0

    @pytest.mark.asyncio
    async def test_stats(self):
        """Test stats method."""
        cache: AsyncTTLCache[str, int] = AsyncTTLCache(maxsize=10, ttl=0.1)

        # Add some entries
        await cache.set("key1", 1)
        await cache.set("key2", 2)

        stats = await cache.stats()
        assert stats["size"] == 2
        assert stats["maxsize"] == 10
        assert stats["expired"] == 0
        assert stats["computing"] == 0
        assert stats["waiters"] == 0

        # Wait for expiration
        await asyncio.sleep(0.15)

        stats = await cache.stats()
        assert stats["expired"] == 2

    @pytest.mark.asyncio
    async def test_get_updates_lru(self):
        """Test that get operation updates LRU order."""
        cache: AsyncTTLCache[str, int] = AsyncTTLCache(maxsize=3, ttl=60.0)

        await cache.set("key1", 1)
        await cache.set("key2", 2)
        await cache.set("key3", 3)

        # Access key1 to make it recently used
        await cache.get("key1")

        # Add key4 - should evict key2 (not key1)
        await cache.set("key4", 4)

        assert await cache.get("key1") == 1  # Still present (recently used)
        assert await cache.get("key2") is None  # Evicted (least recently used)
        assert await cache.get("key3") == 3
        assert await cache.get("key4") == 4

    @pytest.mark.asyncio
    async def test_concurrent_different_keys(self):
        """Test concurrent requests for different keys."""
        cache: AsyncTTLCache[str, int] = AsyncTTLCache(maxsize=10, ttl=60.0)

        async def compute(key: str):
            await asyncio.sleep(0.05)
            return f"value_{key}"

        # Launch concurrent requests for different keys
        tasks = [
            cache.get_or_compute(f"key{i}", lambda k=i: compute(k))
            for i in range(5)
        ]

        result_values = await asyncio.gather(*tasks)

        # All should complete with different values
        assert len(set(result_values)) == 5
        assert result_values[0] == "value_0"
        assert result_values[4] == "value_4"

    @pytest.mark.asyncio
    async def test_invalid_maxsize(self):
        """Test that invalid maxsize raises error."""
        with pytest.raises(ValueError, match="maxsize must be >= 0"):
            AsyncTTLCache(maxsize=-1, ttl=60.0)

    @pytest.mark.asyncio
    async def test_invalid_ttl(self):
        """Test that invalid ttl raises error."""
        with pytest.raises(ValueError, match="ttl must be > 0"):
            AsyncTTLCache(maxsize=10, ttl=0)

        with pytest.raises(ValueError, match="ttl must be > 0"):
            AsyncTTLCache(maxsize=10, ttl=-1)

    @pytest.mark.asyncio
    async def test_set_overwrites_existing(self):
        """Test that set overwrites existing entries."""
        cache: AsyncTTLCache[str, int] = AsyncTTLCache(maxsize=10, ttl=60.0)

        await cache.set("key", 1)
        assert await cache.get("key") == 1

        await cache.set("key", 2)
        assert await cache.get("key") == 2

        # Size should still be 1
        assert await cache.size() == 1
