# Async LRU Cache Implementation

**Insight**: Implemented `AsyncLRUCache` class and `async_lru_cache` decorator in `src/nodetool/concurrency/async_cache.py` for async-aware LRU caching with TTL support.

**Rationale**: An async LRU cache is essential for caching AI API responses, memoizing expensive async operations, and reducing redundant computations. The implementation complements existing async utilities (semaphore, rate limiter, etc.) by providing caching capabilities with configurable size limits and time-to-live expiration.

**Example**:
```python
from nodetool.concurrency import AsyncLRUCache, async_lru_cache

# Class-based usage
cache = AsyncLRUCache(max_size=100, ttl=300)  # 100 items, 5 min TTL
await cache.set("key", value)
result = await cache.get("key")

# Get-or-set pattern for atomic fetch
result = await cache.get_or_set(key, fetch_function, arg1, arg2)

# Decorator usage
@async_lru_cache(max_size=128, ttl=600)
async def fetch_user(user_id: int) -> User:
    return await database.get_user(user_id)
```

**Key Features**:
- Thread-safe async operations using `asyncio.Lock`
- Configurable maximum size with LRU eviction on overflow
- Optional TTL (time-to-live) with configurable reset-on-access
- Statistics tracking (hits, misses, sets, evictions, expirations)
- `get_or_set` method for atomic fetch-if-missing pattern
- Decorator support for function-level caching

**Impact**: Complements the async utilities suite with caching capabilities essential for AI workload optimization.

**Files**:
- `src/nodetool/concurrency/async_cache.py` - Implementation
- `src/nodetool/concurrency/__init__.py` - Exports
- `tests/concurrency/test_async_cache.py` - Tests

**Date**: 2026-01-21
