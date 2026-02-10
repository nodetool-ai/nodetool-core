# Async LRU Cache Implementation

**Insight**: Async LRU caches require careful handling of concurrent access to avoid redundant computations and race conditions. Using a pending computation tracker with futures allows multiple concurrent calls for the same key to wait for a single computation.

**Rationale**: When caching async operations, multiple concurrent requests for the same uncached key can result in redundant expensive operations. A naive cache implementation would execute each request independently, wasting resources. The solution is to track pending computations and make subsequent requests wait for the first one to complete.

**Example**:
```python
from nodetool.concurrency import async_lru_cache

@async_lru_cache(maxsize=128, ttl=300)  # 5 minute TTL
async def fetch_user_profile(user_id: str) -> dict:
    # Expensive database or API call
    return await db.fetch_user(user_id)

# First call executes the function
profile1 = await fetch_user_profile("user123")  # Cache miss - executes

# Concurrent calls wait for the first result
import asyncio
tasks = [fetch_user_profile("user123") for _ in range(10)]
results = await asyncio.gather(*tasks)  # Only executes once

# Later calls return cached result
profile2 = await fetch_user_profile("user123")  # Cache hit
```

**Key Implementation Details**:

1. **Pending Computation Tracking**: Use a dict of key -> list[Future] to track in-flight computations. When a cache miss occurs:
   - First caller creates a computation future
   - Subsequent callers append their futures to the waiting list
   - All waiters get the same result when computation completes

2. **LRU Eviction**: Track `accessed_at` timestamp on each cache entry. When capacity is reached, evict the entry with the oldest `accessed_at` (not `created_at`). This ensures frequently-used entries stay in cache.

3. **TTL Expiration**: Check entry expiration on cache hit, not just on background cleanup. Expired entries are removed immediately when accessed.

4. **Lock Granularity**: Use a single lock for cache operations but release it during actual computation to avoid blocking other cache operations.

**Impact**: 
- Reduces redundant API/database calls under concurrent load
- Improves response times for repeated expensive operations
- Prevents thundering herd problems with pending computation tracking
- Cache hit rate monitoring via `stats` property helps identify caching opportunities

**Files**: 
- `src/nodetool/concurrency/async_lru_cache.py`
- `tests/concurrency/test_async_lru_cache.py`

**Date**: 2026-02-10
