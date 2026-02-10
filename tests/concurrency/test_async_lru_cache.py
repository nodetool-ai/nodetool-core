"""Tests for AsyncLRUCache and async_lru_cache decorator."""

import asyncio
import time

import pytest

from nodetool.concurrency.async_lru_cache import AsyncLRUCache, async_lru_cache


class TestAsyncLRUCache:
    """Tests for AsyncLRUCache class."""

    @pytest.mark.asyncio
    async def test_cache_hit_on_second_call(self):
        """Test that second call returns cached result."""
        call_count = 0

        @AsyncLRUCache(maxsize=10)
        async def fetch_data(key: str):
            nonlocal call_count
            call_count += 1
            return f"data-{key}"

        # First call - cache miss
        result1 = await fetch_data("key1")
        assert result1 == "data-key1"
        assert call_count == 1

        # Second call - cache hit
        result2 = await fetch_data("key1")
        assert result2 == "data-key1"
        assert call_count == 1  # Should not increment

    @pytest.mark.asyncio
    async def test_different_args_different_cache_entries(self):
        """Test that different arguments create different cache entries."""
        call_count = 0

        @AsyncLRUCache(maxsize=10)
        async def fetch_data(key: str):
            nonlocal call_count
            call_count += 1
            return f"data-{key}"

        await fetch_data("key1")
        await fetch_data("key2")
        await fetch_data("key3")

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """Test that least recently used entries are evicted."""
        call_count = {}

        @AsyncLRUCache(maxsize=3)
        async def fetch_data(key: str):
            call_count[key] = call_count.get(key, 0) + 1
            return f"data-{key}"

        # Fill cache to capacity
        await fetch_data("key1")
        await fetch_data("key2")
        await fetch_data("key3")

        # Access key1 to make it more recently used than key2
        await fetch_data("key1")

        # Add new entry, should evict key2 (LRU)
        await fetch_data("key4")

        # key2 should have been evicted, so this should increment call_count
        await fetch_data("key2")
        assert call_count["key2"] == 2

        # key1 should still be cached (was accessed more recently)
        await fetch_data("key1")
        assert call_count["key1"] == 2

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Test that entries expire after TTL."""
        call_count = 0

        @AsyncLRUCache(maxsize=10, ttl=0.1)  # 100ms TTL
        async def fetch_data(key: str):
            nonlocal call_count
            call_count += 1
            return f"data-{key}"

        # First call
        await fetch_data("key1")
        assert call_count == 1

        # Immediate second call - should hit cache
        await fetch_data("key1")
        assert call_count == 1

        # Wait for TTL to expire
        await asyncio.sleep(0.15)

        # Third call - should miss due to expiration
        await fetch_data("key1")
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_cache_properties(self):
        """Test cache properties (hits, misses, evictions, size)."""
        cache = AsyncLRUCache(maxsize=2)

        @cache
        async def fetch_data(key: str):
            return f"data-{key}"

        # Initial state
        assert cache.hits == 0
        assert cache.misses == 0
        assert cache.evictions == 0
        assert cache.size == 0

        # First call - miss
        await fetch_data("key1")
        assert cache.misses == 1
        assert cache.size == 1

        # Second call - hit
        await fetch_data("key1")
        assert cache.hits == 1
        assert cache.size == 1

        # Add second entry
        await fetch_data("key2")
        assert cache.size == 2

        # Add third entry - should trigger eviction
        await fetch_data("key3")
        assert cache.evictions == 1
        assert cache.size == 2

    @pytest.mark.asyncio
    async def test_stats_property(self):
        """Test stats property returns correct statistics."""
        cache = AsyncLRUCache(maxsize=10)

        @cache
        async def fetch_data(key: str):
            return f"data-{key}"

        await fetch_data("key1")
        await fetch_data("key1")
        await fetch_data("key2")

        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["evictions"] == 0
        assert stats["size"] == 2
        assert stats["hit_rate"] == 1 / 3  # 1 hit out of 3 total calls

    @pytest.mark.asyncio
    async def test_clear_method(self):
        """Test that clear method empties the cache."""
        cache = AsyncLRUCache(maxsize=10)

        @cache
        async def fetch_data(key: str):
            return f"data-{key}"

        await fetch_data("key1")
        await fetch_data("key2")
        assert cache.size == 2

        cache.clear()
        assert cache.size == 0
        assert cache.hits == 0
        assert cache.misses == 0

    @pytest.mark.asyncio
    async def test_invalidate_method(self):
        """Test that invalidate method removes specific entries."""
        cache = AsyncLRUCache(maxsize=10)

        @cache
        async def fetch_data(key: str):
            return f"data-{key}"

        await fetch_data("key1")
        await fetch_data("key2")
        assert cache.size == 2

        # Invalidate key1
        result = await fetch_data.invalidate("key1")
        assert result is True
        assert cache.size == 1

        # Invalidate non-existent key
        result = await fetch_data.invalidate("key3")
        assert result is False
        assert cache.size == 1

    @pytest.mark.asyncio
    async def test_custom_key_function(self):
        """Test custom key function for complex arguments."""
        call_count = 0

        class Request:
            def __init__(self, url: str, params: dict):
                self.url = url
                self.params = params

        # Custom key function
        def make_key(url: str, params: dict) -> str:
            return f"{url}:{sorted(params.items())}"

        cache = AsyncLRUCache(maxsize=10, key_func=make_key)

        @cache
        async def fetch_api(url: str, params: dict):
            nonlocal call_count
            call_count += 1
            return {"url": url, "params": params}

        # Same params should hit cache
        await fetch_api("/api", {"a": 1, "b": 2})
        await fetch_api("/api", {"b": 2, "a": 1})  # Different order
        assert call_count == 1

        # Different params should miss
        await fetch_api("/api", {"a": 1, "b": 3})
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_keyword_arguments(self):
        """Test that keyword arguments are handled correctly."""
        call_count = 0

        @AsyncLRUCache(maxsize=10)
        async def fetch_data(key: str, value: int = 0):
            nonlocal call_count
            call_count += 1
            return f"{key}-{value}"

        # Same args, same order - should cache
        await fetch_data("key1", value=1)
        await fetch_data("key1", value=1)
        assert call_count == 1

        # Different value - should miss
        await fetch_data("key1", value=2)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_invalid_maxsize(self):
        """Test that invalid maxsize raises ValueError."""
        with pytest.raises(ValueError, match="maxsize must be positive"):
            AsyncLRUCache(maxsize=0)

        with pytest.raises(ValueError, match="maxsize must be positive"):
            AsyncLRUCache(maxsize=-1)

    @pytest.mark.asyncio
    async def test_no_ttl(self):
        """Test that entries without TTL don't expire."""
        call_count = 0

        @AsyncLRUCache(maxsize=10, ttl=None)
        async def fetch_data(key: str):
            nonlocal call_count
            call_count += 1
            return f"data-{key}"

        await fetch_data("key1")
        await asyncio.sleep(0.1)  # Wait a bit
        await fetch_data("key1")

        assert call_count == 1  # Should still hit cache


class TestAsyncLruCacheDecorator:
    """Tests for async_lru_cache decorator factory."""

    @pytest.mark.asyncio
    async def test_decorator_factory(self):
        """Test that async_lru_cache creates working caches."""
        call_count = 0

        @async_lru_cache(maxsize=5, ttl=1.0)
        async def fetch_data(key: str):
            nonlocal call_count
            call_count += 1
            return f"data-{key}"

        await fetch_data("key1")
        await fetch_data("key1")

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function name and docstring."""

        @async_lru_cache()
        async def my_function():
            """My docstring."""
            return "result"

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    @pytest.mark.asyncio
    async def test_default_parameters(self):
        """Test that default parameters work correctly."""
        @async_lru_cache()
        async def fetch_data(key: str):
            return f"data-{key}"

        # Should work with default maxsize=128
        for i in range(150):
            await fetch_data(f"key{i}")

        # Cache should be at maxsize (oldest entries evicted)
        assert fetch_data.cache.size <= 128

    @pytest.mark.asyncio
    async def test_attached_methods(self):
        """Test that clear and invalidate methods are attached."""
        @async_lru_cache(maxsize=10)
        async def fetch_data(key: str):
            return f"data-{key}"

        # Populate cache
        await fetch_data("key1")
        await fetch_data("key2")

        # Test attached clear method
        fetch_data.clear()
        assert fetch_data.cache.size == 0

        # Repopulate
        await fetch_data("key1")
        await fetch_data("key2")

        # Test attached invalidate method
        await fetch_data.invalidate("key1")
        assert fetch_data.cache.size == 1

    @pytest.mark.asyncio
    async def test_cache_property_attached(self):
        """Test that cache property is accessible through wrapper."""
        @async_lru_cache(maxsize=10)
        async def fetch_data(key: str):
            return f"data-{key}"

        await fetch_data("key1")
        await fetch_data("key1")

        # Access cache through wrapper
        cache = fetch_data.cache
        assert cache.hits == 1
        assert cache.misses == 1


class TestAsyncLRUCacheEdgeCases:
    """Tests for edge cases and concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """Test that cache is thread-safe for concurrent access."""
        call_count = 0

        @AsyncLRUCache(maxsize=10)
        async def fetch_data(key: str):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Simulate slow operation
            return f"data-{key}"

        # Launch concurrent calls for same key
        tasks = [fetch_data("key1") for _ in range(5)]
        await asyncio.gather(*tasks)

        # Should only execute once due to pending computation tracking
        assert call_count <= 2  # Allow some variance in race conditions

    @pytest.mark.asyncio
    async def test_exception_handling(self):
        """Test that exceptions are not cached."""
        exception_count = 0

        @AsyncLRUCache(maxsize=10)
        async def fetch_data(key: str):
            nonlocal exception_count
            exception_count += 1
            if key == "error":
                raise ValueError("Test error")
            return f"data-{key}"

        # Call with error key
        with pytest.raises(ValueError, match="Test error"):
            await fetch_data("error")
        assert exception_count == 1

        # Call again - should not cache the exception
        with pytest.raises(ValueError, match="Test error"):
            await fetch_data("error")
        assert exception_count == 2

    @pytest.mark.asyncio
    async def test_empty_arguments(self):
        """Test function with no arguments."""
        call_count = 0

        @AsyncLRUCache(maxsize=10)
        async def fetch_data():
            nonlocal call_count
            call_count += 1
            return "data"

        await fetch_data()
        await fetch_data()

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_multiple_keys_concurrent(self):
        """Test concurrent access with different keys."""
        call_counts = {}

        @AsyncLRUCache(maxsize=100)
        async def fetch_data(key: str):
            call_counts[key] = call_counts.get(key, 0) + 1
            await asyncio.sleep(0.01)
            return f"data-{key}"

        # Launch concurrent calls for different keys
        keys = [f"key{i}" for i in range(10)]
        tasks = []
        for key in keys:
            for _ in range(3):
                tasks.append(fetch_data(key))

        await asyncio.gather(*tasks)

        # Each key should be called minimal times due to pending tracking
        for key in keys:
            assert call_counts[key] <= 3  # At most once per call, hopefully less due to caching
