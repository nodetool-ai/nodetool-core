# Async Cache and Memoization

**Insight**: Added `AsyncCache` and `cached_async` decorator for thread-safe async memoization with TTL support.

**Rationale**: The nodetool-core concurrency module had many utilities (debounce, throttle, rate limiting, retry, etc.) but lacked a general-purpose async caching mechanism. Caching expensive async operations (like API calls, database queries, or ML model inference) is essential for performance.

**Implementation**:
- `AsyncCache[K, V]`: Generic LRU cache with TTL, eviction, and statistics
- `cached_async`: Decorator for memoizing async functions with custom key functions
- Thread-safe via `asyncio.Lock`
- Automatic expiration based on TTL
- LRU eviction when capacity exceeded
- Cache statistics (hits, misses, evictions, hit rate)

**Example Usage**:
```python
from nodetool.concurrency import AsyncCache, cached_async

# Direct cache usage
cache = AsyncCache(max_size=100, default_ttl=60.0)
result = await cache.get_or_compute("key", lambda: expensive_api_call())

# Decorator usage
@cached_async(ttl=300.0, max_size=50)
async def fetch_user(user_id: str):
    return await db.query(user_id)
```

**Impact**: Provides a reusable, efficient caching mechanism for async operations throughout the codebase, reducing redundant computations and API calls.

**Files**: 
- `src/nodetool/concurrency/async_cache.py`
- `tests/concurrency/test_async_cache.py`

**Date**: 2026-02-19
