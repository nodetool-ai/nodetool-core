import asyncio
import time

import pytest

from nodetool.concurrency.cache import AsyncCache, _generate_cache_key, _make_hashable, cached


class TestMakeHashable:
    """Tests for the _make_hashable utility function."""

    def test_make_hashable_primitives(self):
        """Test that primitive types are returned as-is."""
        assert _make_hashable("string") == "string"
        assert _make_hashable(42) == 42
        assert _make_hashable(3.14) == 3.14
        assert _make_hashable(True) is True
        assert _make_hashable(None) is None

    def test_make_hashable_tuple(self):
        """Test that tuples are converted to hashable form."""
        result = _make_hashable((1, 2, 3))
        assert result == (1, 2, 3)

    def test_make_hashable_list(self):
        """Test that lists are converted to tuples."""
        result = _make_hashable([1, 2, 3])
        assert result == (1, 2, 3)
        assert isinstance(result, tuple)

    def test_make_hashable_dict(self):
        """Test that dicts are converted to sorted tuples."""
        result = _make_hashable({"b": 2, "a": 1})
        assert result == (("a", 1), ("b", 2))

    def test_make_hashable_set(self):
        """Test that sets are converted to frozensets."""
        result = _make_hashable({1, 2, 3})
        assert isinstance(result, frozenset)
        assert set(result) == {1, 2, 3}

    def test_make_hashable_nested(self):
        """Test nested structures."""
        result = _make_hashable({"key": [1, 2, {"nested": "value"}]})
        assert result == (("key", (1, 2, (("nested", "value"),))),)


class TestGenerateCacheKey:
    """Tests for the _generate_cache_key utility function."""

    def test_generate_cache_key_consistency(self):
        """Test that same inputs produce same key."""

        def dummy_func(x: int, y: int, a: int = 3):
            pass

        key1 = _generate_cache_key(dummy_func, (1, 2), {"a": 3})
        key2 = _generate_cache_key(dummy_func, (1, 2), {"a": 3})
        assert key1 == key2

    def test_generate_cache_key_different_args(self):
        """Test that different inputs produce different keys."""

        def dummy_func(x: int, y: int, a: int = 3):
            pass

        key1 = _generate_cache_key(dummy_func, (1, 2), {"a": 3})
        key2 = _generate_cache_key(dummy_func, (1, 3), {"a": 3})
        assert key1 != key2

    def test_generate_cache_key_different_kwargs_order(self):
        """Test that kwargs order doesn't affect key."""

        def dummy_func(x: int, a: int = 2, b: int = 3):
            pass

        key1 = _generate_cache_key(dummy_func, (1,), {"a": 2, "b": 3})
        key2 = _generate_cache_key(dummy_func, (1,), {"b": 3, "a": 2})
        assert key1 == key2


class TestAsyncCache:
    """Tests for the AsyncCache class."""

    @pytest.mark.asyncio
    async def test_cache_set_and_get(self):
        """Test basic set and get operations."""
        cache = AsyncCache()
        await cache.set("key1", "value1")
        assert await cache.get("key1") == "value1"

    @pytest.mark.asyncio
    async def test_cache_get_missing_key(self):
        """Test getting a non-existent key returns None."""
        cache = AsyncCache()
        assert await cache.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self):
        """Test that entries expire after TTL."""
        cache = AsyncCache(default_ttl=0.1)  # 100ms TTL
        await cache.set("key1", "value1")
        assert await cache.get("key1") == "value1"

        # Wait for expiration
        await asyncio.sleep(0.15)
        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_cache_custom_ttl(self):
        """Test custom TTL per entry."""
        cache = AsyncCache(default_ttl=10.0)
        await cache.set("key1", "value1", ttl=0.1)

        assert await cache.get("key1") == "value1"
        await asyncio.sleep(0.15)
        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_cache_delete(self):
        """Test deleting a cache entry."""
        cache = AsyncCache()
        await cache.set("key1", "value1")
        assert await cache.delete("key1") is True
        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_cache_delete_nonexistent(self):
        """Test deleting a non-existent key returns False."""
        cache = AsyncCache()
        assert await cache.delete("nonexistent") is False

    @pytest.mark.asyncio
    async def test_cache_clear(self):
        """Test clearing all entries."""
        cache = AsyncCache()
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.clear()
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
        assert cache.size == 0

    @pytest.mark.asyncio
    async def test_cache_max_size_eviction(self):
        """Test FIFO eviction when max_size is reached."""
        cache = AsyncCache(max_size=3)
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")
        assert cache.size == 3

        # Adding a 4th entry should evict the first
        await cache.set("key4", "value4")
        assert cache.size == 3
        assert await cache.get("key1") is None
        assert await cache.get("key4") == "value4"

    @pytest.mark.asyncio
    async def test_cache_cleanup_expired(self):
        """Test cleanup of expired entries."""
        cache = AsyncCache(default_ttl=0.1)
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3", ttl=10.0)  # Longer TTL

        await asyncio.sleep(0.15)
        removed = await cache.cleanup_expired()

        assert removed == 2
        assert cache.size == 1
        assert await cache.get("key3") == "value3"

    @pytest.mark.asyncio
    async def test_cache_result_decorator(self):
        """Test the cache_result decorator."""
        cache = AsyncCache()
        call_count = 0

        @cache.cache_result(ttl=1.0)
        async def fetch_data(user_id: int):
            nonlocal call_count
            call_count += 1
            return f"data_{user_id}"

        # First call executes the function
        result = await fetch_data(1)
        assert result == "data_1"
        assert call_count == 1

        # Second call uses cache
        result = await fetch_data(1)
        assert result == "data_1"
        assert call_count == 1  # Not incremented

        # Different args bypass cache
        result = await fetch_data(2)
        assert result == "data_2"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_cache_result_decorator_expiration(self):
        """Test that cached results expire after TTL."""
        cache = AsyncCache()
        call_count = 0

        @cache.cache_result(ttl=0.1)
        async def fetch_data(user_id: int):
            nonlocal call_count
            call_count += 1
            return f"data_{user_id}"

        await fetch_data(1)
        assert call_count == 1

        # Wait for expiration
        await asyncio.sleep(0.15)

        await fetch_data(1)
        assert call_count == 2  # Function called again

    @pytest.mark.asyncio
    async def test_invalidate(self):
        """Test invalidating a cached function call."""
        cache = AsyncCache()
        call_count = 0

        @cache.cache_result(ttl=10.0)
        async def fetch_data(user_id: int):
            nonlocal call_count
            call_count += 1
            return f"data_{user_id}"

        # Cache the result
        await fetch_data(1)
        assert call_count == 1

        # Invalidate the cache
        await cache.invalidate(fetch_data, 1)

        # Next call executes the function again
        await fetch_data(1)
        assert call_count == 2


class TestCachedDecorator:
    """Tests for the @cached decorator."""

    @pytest.mark.asyncio
    async def test_cached_decorator_basic(self):
        """Test basic @cached decorator functionality."""
        call_count = 0

        @cached(ttl=1.0)
        async def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        result = await compute(5)
        assert result == 10
        assert call_count == 1

        # Cached call
        result = await compute(5)
        assert result == 10
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_cached_decorator_with_different_args(self):
        """Test that different arguments create separate cache entries."""
        call_count = 0

        @cached(ttl=1.0)
        async def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        await compute(1)
        await compute(2)
        await compute(3)

        assert call_count == 3

        # Cached calls
        await compute(1)
        await compute(2)
        await compute(3)

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_cached_decorator_expiration(self):
        """Test that cached results expire."""
        call_count = 0

        @cached(ttl=0.1)
        async def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        await compute(5)
        assert call_count == 1

        await asyncio.sleep(0.15)

        await compute(5)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_cached_decorator_with_kwargs(self):
        """Test caching with keyword arguments."""
        call_count = 0

        @cached(ttl=1.0)
        async def compute(x: int, y: int = 10) -> int:
            nonlocal call_count
            call_count += 1
            return x + y

        await compute(1, y=2)
        assert call_count == 1

        # Same call, should hit cache
        await compute(1, y=2)
        assert call_count == 1

        # Different kwarg order, should still hit cache
        await compute(x=1, y=2)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_cached_decorator_custom_cache(self):
        """Test using a custom cache instance."""
        custom_cache = AsyncCache(default_ttl=1.0)
        call_count = 0

        @cached(ttl=1.0, cache_instance=custom_cache)
        async def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        await compute(5)
        assert call_count == 1

        await compute(5)
        assert call_count == 1

        assert custom_cache.size == 1

    @pytest.mark.asyncio
    async def test_cached_invalidate_method(self):
        """Test the invalidate method added by @cached."""
        call_count = 0

        @cached(ttl=10.0)
        async def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        await compute(5)
        assert call_count == 1

        # Invalidate the cache
        await compute.invalidate(5)

        await compute(5)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_cached_cache_key_method(self):
        """Test the cache_key method added by @cached."""

        @cached(ttl=10.0)
        async def compute(x: int, y: int = 10) -> int:
            return x + y

        key1 = compute.cache_key(1, 2)
        key2 = compute.cache_key(1, y=2)

        # Should be the same regardless of kwarg order
        assert key1 == key2

    @pytest.mark.asyncio
    async def test_cached_with_complex_args(self):
        """Test caching with complex argument types."""
        call_count = 0

        @cached(ttl=1.0)
        async def process_data(data: list, config: dict) -> str:
            nonlocal call_count
            call_count += 1
            return f"processed_{len(data)}_{len(config)}"

        # First call
        await process_data([1, 2, 3], {"key": "value"})
        assert call_count == 1

        # Same args, should hit cache
        await process_data([1, 2, 3], {"key": "value"})
        assert call_count == 1

        # Different list order
        await process_data([3, 2, 1], {"key": "value"})
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_cached_cache_property(self):
        """Test that the cache property is accessible."""

        @cached(ttl=10.0)
        async def compute(x: int) -> int:
            return x * 2

        assert hasattr(compute, "cache")
        assert isinstance(compute.cache, AsyncCache)
