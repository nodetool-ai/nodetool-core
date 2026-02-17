# Async LRU Cache

**Insight**: Added `AsyncLRUCache` utility in `src/nodetool/concurrency/async_cache.py` for thread-safe async LRU caching with TTL support.

**Rationale**: The codebase has various caching needs scattered throughout (workflow results, API responses, expensive computations), but no unified async-safe caching utility. The new `AsyncLRUCache` provides:
- LRU eviction when cache is full
- TTL (time-to-live) support for automatic expiration
- Thread-safe operations using asyncio locks
- Statistics tracking for monitoring cache performance (hits, misses, evictions, hit rate)
- Decorator support for easy async function memoization
- `get_or_compute()` method for cache-aside pattern

**Example**:

```python
from nodetool.concurrency import AsyncLRUCache

# Create a cache with max 100 entries, 5-minute TTL
cache = AsyncLRUCache[str, User](max_size=100, ttl=300)

# Use cache-aside pattern
user = await cache.get_or_compute(
    "user:123",
    lambda: fetch_user_from_db("123")
)

# Use as decorator
@cache.cache_result(ttl=60)
async def get_user(user_id: str) -> User:
    return await db.fetch_user(user_id)

# Monitor performance
stats = cache.stats
print(f"Hit rate: {stats.hit_rate:.2%}")
print(f"Evictions: {stats.evictions}")
```

**Impact**: Provides a reusable, type-safe async caching utility that follows the codebase's async-first patterns. Enables efficient memoization of expensive async operations (DB queries, API calls, model inference) with automatic memory management via LRU eviction.

**Files**:
- `src/nodetool/concurrency/async_cache.py`
- `src/nodetool/concurrency/__init__.py`
- `tests/concurrency/test_async_cache.py`

**Date**: 2026-02-17
