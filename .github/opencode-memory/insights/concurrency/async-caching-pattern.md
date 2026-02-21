# Async Caching Pattern

**Insight**: Async-safe caching with TTL and size limits for memoizing expensive async operations.

**Rationale**: In async workflows, operations like API calls, database queries, and complex computations are often expensive. Caching their results can significantly improve performance and reduce resource usage.

**Example**:
```python
from nodetool.concurrency import cached_async, AsyncCache

# Using the decorator
@cached_async(ttl=60)  # Cache for 60 seconds
async def fetch_user(user_id: str) -> dict:
    return await api.get_user(user_id)

# Using AsyncCache directly
cache = AsyncCache(max_size=100, ttl=300)
result = await cache.get_or_compute("key123", lambda: expensive_operation())

# Custom key function for complex arguments
def make_key(url: str, params: dict) -> str:
    return f"{url}:{json.dumps(params, sort_keys=True)}"

@cached_async(ttl=300, key_func=make_key)
async def fetch_data(url: str, params: dict) -> dict:
    return await http_get(url, params=params)
```

**Impact**: Provides a simple, async-safe caching mechanism that reduces redundant async operations, improving performance and resource efficiency in workflows.

**Files**:
- `src/nodetool/concurrency/async_cache.py`
- `tests/concurrency/test_async_cache.py`

**Date**: 2026-02-21
