# Async TTL Cache with LRU Eviction

**Insight**: Added `AsyncTTLCache` for time-based caching with automatic expiration and LRU eviction.

**Rationale**: Async operations like API calls and database queries are expensive. Caching results with TTL improves performance and reduces load, while automatic expiration prevents stale data. LRU eviction ensures bounded memory usage.

**Example**:
```python
from nodetool.concurrency import AsyncTTLCache

# Create cache with 5-minute TTL and max 1000 entries
cache = AsyncTTLCache(ttl=300.0, max_size=1000)

async with cache:
    # Cache-aside pattern: get or compute
    user = await cache.get_or_compute(
        f"user:{user_id}",
        lambda: fetch_user_from_db(user_id)
    )

    # Manual get/set
    await cache.set("key", value, ttl=60.0)  # Custom TTL
    value = await cache.get("key")

    # Check existence
    if await cache.has("key"):
        print("Cache hit!")
```

**Impact**: Reduces redundant async operations, improves response times, and prevents memory leaks with automatic cleanup and LRU eviction. Ideal for API response caching, database query caching, and expensive computation memoization.

**Files**:
- `src/nodetool/concurrency/async_cache.py`
- `tests/concurrency/test_async_cache.py`
- `src/nodetool/concurrency/__init__.py`

**Date**: 2026-02-10
