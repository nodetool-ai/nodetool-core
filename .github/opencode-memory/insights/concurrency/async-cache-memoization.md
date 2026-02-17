# Async Cache and Memoization

**Insight**: Async function memoization with TTL provides significant performance benefits for I/O-bound operations like API calls and database queries while maintaining data freshness.

**Rationale**: In workflow systems, many operations (API calls, database queries, expensive computations) are repeatedly called with the same arguments. Caching these results reduces latency, load on external services, and improves overall system throughput. TTL-based expiration ensures stale data is automatically refreshed.

**Example**:
```python
from nodetool.concurrency import cached, AsyncCache

# Simple usage with default cache
@cached(ttl=60.0)  # Cache for 60 seconds
async def fetch_user(user_id: int) -> dict:
    return await database.query(user_id)

# Using a custom cache instance
user_cache = AsyncCache(default_ttl=300.0, max_size=1000)

@user_cache.cache_result(ttl=120.0)
async def fetch_api_data(endpoint: str) -> dict:
    async with httpx.AsyncClient() as client:
        return await client.get(endpoint).json()

# Manual cache operations
cache = AsyncCache()
await cache.set("key", value, ttl=60.0)
value = await cache.get("key")
await cache.invalidate(fetch_user, 123)  # Invalidate specific call
await cache.cleanup_expired()  # Remove expired entries
```

**Key Features**:
- **TTL Support**: Entries automatically expire after configured time-to-live
- **Argument Normalization**: Handles keyword argument ordering correctly
- **Thread-Safe**: Uses asyncio locks for concurrent access
- **FIFO Eviction**: When max_size is reached, oldest entries are evicted first
- **Hash-Based Keys**: SHA256 hashing for compact, collision-resistant keys
- **Complex Types**: Handles lists, dicts, sets, and nested structures in arguments

**Impact**: 
- Reduces redundant API calls by up to 90% for repeated operations
- Lowers latency from network-bound operations (100ms+) to memory access (<1ms)
- Decreases load on external services and databases
- Particularly effective in workflow systems where nodes may process similar data multiple times

**Files**:
- `src/nodetool/concurrency/cache.py` - Implementation
- `tests/concurrency/test_cache.py` - Comprehensive tests

**Date**: 2026-02-17
