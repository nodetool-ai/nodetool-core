# AsyncCache Implementation

**Insight**: AsyncCache provides thread-safe, async-friendly caching with TTL and LRU eviction for expensive async operations.

**Rationale**: Workflow engines frequently need to cache expensive operations like database queries, API calls, or model inferences. A dedicated async cache prevents redundant computations while maintaining async semantics and thread safety.

**Key Features**:
- **TTL Support**: Automatic expiration of cache entries with configurable time-to-live per entry
- **LRU Eviction**: Least Recently Used eviction when cache reaches max size
- **Thread Safety**: Uses AsyncLock for safe concurrent access
- **Statistics Tracking**: Hits, misses, evictions, and hit rate for monitoring cache effectiveness
- **Decorator Support**: `@async_cache` decorator for easy function memoization
- **get_or_compute**: Atomic "get or compute" pattern prevents cache stampede

**Example Usage**:

```python
# Direct cache usage
cache = AsyncCache[str, dict](max_size=100, ttl=60.0)
user = await cache.get_or_compute("user:123", fetch_user_from_db)

# Decorator usage
@async_cache(max_size=100, ttl=60.0)
async def fetch_user(user_id: str) -> dict:
    return await database.query(user_id)
```

**Performance Considerations**:
- Cache operations are O(1) for get/set/delete
- LRU updates on access maintain freshness without overhead
- TTL checks are performed on read to avoid background tasks
- Statistics counters use atomic operations for accuracy

**Files**:
- `src/nodetool/concurrency/async_cache.py`: Implementation
- `tests/concurrency/test_async_cache.py`: Test suite (22 tests)

**Date**: 2026-02-13
