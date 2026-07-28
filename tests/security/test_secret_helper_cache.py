"""Regression tests for the environment-derived secret cache.

The cache used to be consulted before the ``check_env`` branch, so a value
cached from ``os.environ`` was returned even to callers that explicitly asked
not to consult the environment. It was also unbounded.
"""

import pytest

from nodetool.security.secret_helper import (
    _SECRET_CACHE,
    _SECRET_CACHE_MAX_SIZE,
    clear_secret_cache,
    get_secret,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    _SECRET_CACHE.clear()
    yield
    _SECRET_CACHE.clear()


async def test_check_env_false_does_not_serve_env_cached_value(monkeypatch):
    monkeypatch.setenv("TEST_SECRET_KEY", "from-env")

    assert await get_secret("TEST_SECRET_KEY", "u1") == "from-env"
    assert ("u1", "TEST_SECRET_KEY") in _SECRET_CACHE

    assert await get_secret("TEST_SECRET_KEY", "u1", default="fallback", check_env=False) == "fallback"
    assert await get_secret("TEST_SECRET_KEY", "u1", check_env=False) is None


async def test_check_env_true_still_caches(monkeypatch):
    monkeypatch.setenv("TEST_SECRET_KEY", "from-env")
    assert await get_secret("TEST_SECRET_KEY", "u1") == "from-env"

    monkeypatch.delenv("TEST_SECRET_KEY")
    # Still served from the cache until explicitly cleared.
    assert await get_secret("TEST_SECRET_KEY", "u1") == "from-env"

    clear_secret_cache("u1", "TEST_SECRET_KEY")
    assert await get_secret("TEST_SECRET_KEY", "u1", default="d") == "d"


async def test_cache_is_bounded(monkeypatch):
    for i in range(_SECRET_CACHE_MAX_SIZE + 50):
        key = f"TEST_BOUNDED_SECRET_{i}"
        monkeypatch.setenv(key, f"value-{i}")
        assert await get_secret(key, "u1") == f"value-{i}"

    assert len(_SECRET_CACHE) <= _SECRET_CACHE_MAX_SIZE
    # Oldest entries were evicted, newest retained.
    assert ("u1", "TEST_BOUNDED_SECRET_0") not in _SECRET_CACHE
    last = _SECRET_CACHE_MAX_SIZE + 49
    assert ("u1", f"TEST_BOUNDED_SECRET_{last}") in _SECRET_CACHE


async def test_missing_secret_returns_default_and_is_not_cached(monkeypatch):
    monkeypatch.delenv("TEST_ABSENT_SECRET", raising=False)
    assert await get_secret("TEST_ABSENT_SECRET", "u1", default="d") == "d"
    assert ("u1", "TEST_ABSENT_SECRET") not in _SECRET_CACHE
