# Async LRU Cache

**Insight**: AsyncLRUCache provides async-safe least-recently-used caching with TTL support for expensive async operations.

**Rationale**:
- Python's built-in `functools.lru_cache` is synchronous and not safe for concurrent async access
- Many async operations (API calls, database queries) are expensive and benefit from caching
- TTL support ensures stale data is automatically expired
- LRU eviction prevents unbounded memory growth

**Implementation**:
- Uses `OrderedDict` for O(1) LRU tracking via `move_to_end()` and `popitem(last=False)`
- `asyncio.Lock` ensures thread-safe concurrent access
- TTL is tracked per-entry as absolute expiration timestamps
- `get_or_load` method provides cache-aside pattern with factory functions
- Decorator `@async_lru_cache` provides drop-in caching for async functions

**Example**:
```python
from nodetool.concurrency import async_lru_cache

@async_lru_cache(maxsize=100, ttl=60.0)
async def fetch_user(user_id: str) -> dict:
    return await database.query(user_id)

# First call: fetches from database
user = await fetch_user("user:123")

# Subsequent calls within 60 seconds: returns cached result
user = await fetch_user("user:123")
```

**Files**:
- `src/nodetool/concurrency/async_lru_cache.py`
- `tests/concurrency/test_async_lru_cache.py`

**Date**: 2026-02-23
