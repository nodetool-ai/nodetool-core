import asyncio
import time

import pytest

from nodetool.concurrency.async_cache import AsyncLRUCache, async_lru_cache


class TestAsyncLRUCache:
    """Tests for AsyncLRUCache class."""

    def test_init_with_valid_params(self):
        """Test initialization with valid parameters."""
        cache = AsyncLRUCache(max_size=10)
        assert cache.max_size == 10
        assert cache.ttl is None
        assert cache.size == 0

    def test_init_with_ttl(self):
        """Test initialization with TTL."""
        cache = AsyncLRUCache(max_size=10, ttl=300)
        assert cache.ttl == 300
        assert cache._ttl_reset_on_access is True

    def test_init_with_custom_ttl_reset(self):
        """Test initialization with custom TTL reset setting."""
        cache = AsyncLRUCache(max_size=10, ttl=300, ttl_reset_on_access=False)
        assert cache._ttl_reset_on_access is False

    def test_init_with_invalid_max_size(self):
        """Test that invalid max_size raises ValueError."""
        with pytest.raises(ValueError, match="max_size must be a positive integer"):
            AsyncLRUCache(max_size=0)

        with pytest.raises(ValueError, match="max_size must be a positive integer"):
            AsyncLRUCache(max_size=-1)

    def test_init_with_invalid_ttl(self):
        """Test that invalid ttl raises ValueError."""
        with pytest.raises(ValueError, match="ttl must be None or a positive number"):
            AsyncLRUCache(max_size=10, ttl=0)

        with pytest.raises(ValueError, match="ttl must be None or a positive number"):
            AsyncLRUCache(max_size=10, ttl=-1)

    @pytest.mark.asyncio
    async def test_set_and_get(self):
        """Test basic set and get operations."""
        cache = AsyncLRUCache(max_size=10)
        await cache.set("key1", "value1")
        result = await cache.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_get_missing_key(self):
        """Test get with missing key returns default."""
        cache = AsyncLRUCache(max_size=10)
        result = await cache.get("missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_with_default(self):
        """Test get with custom default value."""
        cache = AsyncLRUCache(max_size=10)
        result = await cache.get("missing", default="default")
        assert result == "default"

    @pytest.mark.asyncio
    async def test_set_updates_existing(self):
        """Test that set updates existing key."""
        cache = AsyncLRUCache(max_size=10)
        await cache.set("key", "value1")
        await cache.set("key", "value2")
        result = await cache.get("key")
        assert result == "value2"

    @pytest.mark.asyncio
    async def test_delete(self):
        """Test delete operation."""
        cache = AsyncLRUCache(max_size=10)
        await cache.set("key", "value")
        result = await cache.delete("key")
        assert result is True
        assert await cache.get("key") is None

    @pytest.mark.asyncio
    async def test_delete_missing_key(self):
        """Test delete of missing key returns False."""
        cache = AsyncLRUCache(max_size=10)
        result = await cache.delete("missing")
        assert result is False

    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clear operation."""
        cache = AsyncLRUCache(max_size=10)
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        assert cache.size == 2

        cache.clear()
        assert cache.size == 0
        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_contains(self):
        """Test contains operation."""
        cache = AsyncLRUCache(max_size=10)
        await cache.set("key", "value")

        assert await cache.contains("key") is True
        assert await cache.contains("missing") is False

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """Test that LRU eviction works correctly."""
        cache = AsyncLRUCache(max_size=3)
        await cache.set("a", 1)
        await cache.set("b", 2)
        await cache.set("c", 3)

        await cache.get("a")  # Access 'a' to make it recently used

        await cache.set("d", 4)  # Should evict 'b' (LRU)

        assert await cache.get("a") == 1
        assert await cache.get("b") is None  # Evicted
        assert await cache.get("c") == 3
        assert await cache.get("d") == 4

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Test that TTL expiration works."""
        cache = AsyncLRUCache(max_size=10, ttl=0.1)

        await cache.set("key", "value")
        assert await cache.get("key") == "value"

        await asyncio.sleep(0.2)
        assert await cache.get("key") is None

    @pytest.mark.asyncio
    async def test_clear_expired(self):
        """Test clear_expired removes expired entries."""
        cache = AsyncLRUCache(max_size=10, ttl=0.1)

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        await asyncio.sleep(0.2)

        removed = await cache.clear_expired()
        assert removed == 2
        assert cache.size == 0

    @pytest.mark.asyncio
    async def test_clear_expired_partial(self):
        """Test clear_expired with some expired entries."""
        cache = AsyncLRUCache(max_size=10, ttl=0.1)

        await cache.set("key1", "value1")
        await asyncio.sleep(0.05)
        await cache.set("key2", "value2")

        await asyncio.sleep(0.06)

        removed = await cache.clear_expired()
        assert removed == 1
        assert cache.size == 1
        assert await cache.get("key1") is None
        assert await cache.get("key2") == "value2"

    @pytest.mark.asyncio
    async def test_peek(self):
        """Test peek doesn't update access time."""
        cache = AsyncLRUCache(max_size=3)

        await cache.set("a", 1)
        await cache.set("b", 2)
        await cache.set("c", 3)

        await cache.get("a")  # Make 'a' recently used
        await cache.peek("a")

        await cache.set("d", 4)  # Should evict 'b' since 'a' was accessed

        assert await cache.get("a") == 1
        assert await cache.get("b") is None
        assert await cache.get("c") == 3
        assert await cache.get("d") == 4

    @pytest.mark.asyncio
    async def test_get_or_set(self):
        """Test get_or_set fetches and caches value."""
        cache = AsyncLRUCache(max_size=10)
        call_count = 0

        async def fetch():
            nonlocal call_count
            call_count += 1
            return "fetched"

        result = await cache.get_or_set("key", fetch)
        assert result == "fetched"
        assert call_count == 1

        result = await cache.get_or_set("key", fetch)
        assert result == "fetched"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_stats(self):
        """Test statistics tracking."""
        cache = AsyncLRUCache(max_size=10)

        await cache.set("key1", "value1")
        await cache.get("key1")
        await cache.get("missing")
        await cache.get("missing")

        stats = cache.stats
        assert stats["sets"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 2

    @pytest.mark.asyncio
    async def test_stats_with_evictions(self):
        """Test statistics with evictions."""
        cache = AsyncLRUCache(max_size=2)

        await cache.set("a", 1)
        await cache.set("b", 2)
        await cache.set("c", 3)  # Evicts 'a'

        stats = cache.stats
        assert stats["sets"] == 3
        assert stats["evictions"] == 1

    @pytest.mark.asyncio
    async def test_get_stats_includes_hit_rate(self):
        """Test get_stats includes calculated hit rate."""
        cache = AsyncLRUCache(max_size=10)

        await cache.set("key", "value")
        await cache.get("key")
        await cache.get("missing")

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert abs(stats["hit_rate"] - 0.333) < 0.01


class TestAsyncLruCacheDecorator:
    """Tests for async_lru_cache decorator."""

    @pytest.mark.asyncio
    async def test_basic_caching(self):
        """Test basic decorator functionality."""
        call_count = 0

        @async_lru_cache(max_size=10)
        async def fetch_data(key: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"data-{key}"

        result1 = await fetch_data("test")
        assert result1 == "data-test"
        assert call_count == 1

        result2 = await fetch_data("test")
        assert result2 == "data-test"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_different_keys(self):
        """Test that different keys cache separately."""
        call_count = 0

        @async_lru_cache(max_size=10)
        async def fetch_data(key: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"data-{key}"

        await fetch_data("a")
        await fetch_data("b")
        await fetch_data("a")

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Test that decorator respects TTL."""
        call_count = 0

        @async_lru_cache(max_size=10, ttl=0.1)
        async def fetch_data(key: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"data-{key}"

        await fetch_data("test")
        assert call_count == 1

        await asyncio.sleep(0.2)

        await fetch_data("test")
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_max_size_eviction(self):
        """Test that decorator respects max_size."""
        call_count = 0

        @async_lru_cache(max_size=3)
        async def fetch_data(key: int) -> int:
            nonlocal call_count
            call_count += 1
            return key * 2

        for i in range(5):
            await fetch_data(i)

        assert call_count == 5

        await fetch_data(0)  # Cache hit
        assert call_count == 5

    @pytest.mark.asyncio
    async def test_kwargs_caching(self):
        """Test that kwargs are included in cache key."""
        call_count = 0

        @async_lru_cache(max_size=10)
        async def fetch_data(key: str, offset: int = 0) -> str:
            nonlocal call_count
            call_count += 1
            return f"data-{key}-{offset}"

        await fetch_data("test", offset=10)
        await fetch_data("test", offset=10)  # Cache hit
        await fetch_data("test", offset=20)  # Different kwarg

        assert call_count == 2
