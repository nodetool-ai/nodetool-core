import time

from nodetool.storage.memory_uri_cache import MemoryUriCache


def test_memory_uri_cache_set_get_and_expiry():
    cache = MemoryUriCache(default_ttl=1)
    key = "memory://test-object"
    value = {"a": 1}

    # Initially empty
    assert cache.get(key) is None

    # Set and get
    cache.set(key, value)
    assert cache.get(key) == value

    # After TTL it should expire
    time.sleep(1.1)
    assert cache.get(key) is None
