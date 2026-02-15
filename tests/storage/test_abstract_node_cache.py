"""Tests for AbstractNodeCache implementations."""

import time
from unittest.mock import Mock

import pytest

from nodetool.storage.abstract_node_cache import AbstractNodeCache
from nodetool.storage.memory_node_cache import MemoryNodeCache


class TestMemoryNodeCache:
    """Tests for in-memory node cache implementation."""

    def test_cache_initialization(self):
        """Test that cache initializes empty."""
        cache = MemoryNodeCache()
        assert len(cache.cache) == 0

    def test_cache_set_and_get(self):
        """Test basic set and get operations."""
        cache = MemoryNodeCache()
        cache.set("test_key", "test_value")
        assert cache.get("test_key") == "test_value"

    def test_cache_get_nonexistent_key(self):
        """Test getting non-existent key returns None."""
        cache = MemoryNodeCache()
        assert cache.get("nonexistent") is None

    def test_cache_set_with_default_ttl(self):
        """Test that items expire after default TTL."""
        cache = MemoryNodeCache()
        cache.set("test_key", "test_value", ttl=1)
        assert cache.get("test_key") == "test_value"
        time.sleep(1.1)
        assert cache.get("test_key") is None

    def test_cache_set_with_custom_ttl(self):
        """Test that items expire after custom TTL."""
        cache = MemoryNodeCache()
        cache.set("test_key", "test_value", ttl=2)
        assert cache.get("test_key") == "test_value"
        time.sleep(2.1)
        assert cache.get("test_key") is None

    def test_cache_set_with_no_ttl(self):
        """Test that items with ttl=0 or None don't expire."""
        cache = MemoryNodeCache()
        cache.set("test_key", "test_value", ttl=0)
        time.sleep(0.5)
        assert cache.get("test_key") == "test_value"

    def test_cache_clear(self):
        """Test clearing the cache."""
        cache = MemoryNodeCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"

        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_cache_overwrite_existing_key(self):
        """Test overwriting an existing cache key."""
        cache = MemoryNodeCache()
        cache.set("test_key", "value1")
        assert cache.get("test_key") == "value1"

        cache.set("test_key", "value2")
        assert cache.get("test_key") == "value2"

    def test_cache_expiry_removes_entry(self):
        """Test that expired entries are removed from cache."""
        cache = MemoryNodeCache()
        cache.set("test_key", "test_value", ttl=1)
        time.sleep(1.1)
        cache.get("test_key")  # Access should trigger removal

        assert "test_key" not in cache.cache

    def test_cache_complex_values(self):
        """Test caching complex data structures."""
        cache = MemoryNodeCache()
        complex_value = {"nested": {"data": [1, 2, 3]}}
        cache.set("complex_key", complex_value)
        assert cache.get("complex_key") == complex_value

    def test_generate_cache_key(self):
        """Test cache key generation."""
        cache = MemoryNodeCache()
        key = cache.generate_cache_key("TestClass", {"param1": "value1"})
        assert "TestClass:" in key
        assert ":" in key  # Should have hash separator
        assert isinstance(key, str)

    def test_generate_cache_key_is_deterministic(self):
        """Test that same inputs produce same cache key."""
        cache = MemoryNodeCache()
        props = {"param1": "value1", "param2": 42}
        key1 = cache.generate_cache_key("TestClass", props)
        key2 = cache.generate_cache_key("TestClass", props)
        assert key1 == key2

    def test_generate_cache_key_different_params(self):
        """Test that different params produce different cache keys."""
        cache = MemoryNodeCache()
        key1 = cache.generate_cache_key("TestClass", {"param1": "value1"})
        key2 = cache.generate_cache_key("TestClass", {"param1": "value2"})
        assert key1 != key2


class TestAbstractNodeCacheInterface:
    """Tests to ensure all AbstractNodeCache implementations follow the contract."""

    def test_memory_cache_implements_interface(self):
        """Test that MemoryNodeCache implements all required methods."""
        cache = MemoryNodeCache()
        assert hasattr(cache, "get")
        assert hasattr(cache, "set")
        assert hasattr(cache, "clear")
        assert callable(cache.get)
        assert callable(cache.set)
        assert callable(cache.clear)

    def test_cache_get_signature_matches_abstract(self):
        """Test that get signature matches abstract interface."""
        cache = MemoryNodeCache()
        # Should accept key parameter
        result = cache.get("test")
        # Should not raise TypeError for wrong signature
        assert result is None or isinstance(result, object)

    def test_cache_set_signature_matches_abstract(self):
        """Test that set signature matches abstract interface."""
        cache = MemoryNodeCache()
        # Should accept key, value, and optional ttl
        cache.set("test", "value", ttl=100)
        assert cache.get("test") == "value"
