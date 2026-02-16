# Async Cache Implementation

**Insight**: Implemented a comprehensive async caching utility with TTL and LRU eviction policies for caching expensive async operations.

**Rationale**: The codebase had extensive async concurrency utilities (rate limiting, retry, timeout, circuit breaker, etc.) but lacked a caching mechanism. Caching is essential for:
- Reducing redundant API calls (especially LLM providers)
- Avoiding expensive database queries
- Caching decoded media assets
- Improving workflow performance by caching node computations

**Example**:
```python
from nodetool.concurrency import AsyncCache

# Create cache with 100 max entries and 60s default TTL
cache = AsyncCache[str, int](max_size=100, default_ttl=60.0)

# Use cache-aside pattern
result = await cache.get_or_compute(
    "user_123",
    lambda: fetch_user_from_db("123")
)

# Manual operations
await cache.put("key", value, ttl=30.0)
value = await cache.get("key")

# Check statistics
stats = await cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
```

**Key Features**:
- **TTL Expiration**: Time-based expiration with configurable default and per-entry TTL
- **LRU Eviction**: Least-recently-used eviction when cache reaches max_size
- **Thread-Safe**: All operations protected by asyncio.Lock for concurrent access
- **Async Factory Support**: `get_or_compute()` method supports both sync and async factory functions
- **Statistics Tracking**: Tracks hits, misses, evictions, and hit rate
- **Auto Cleanup**: Optional background task to periodically remove expired entries

**Implementation Details**:
- Uses `time.monotonic()` for reliable time measurements
- `CacheEntry` class uses `__slots__` for memory efficiency
- Type-safe with generic `AsyncCache[K, V]` parameters
- Follows existing codebase patterns (similar to `RetryPolicy`, `TimeoutPolicy`)

**Testing**: Comprehensive test suite with 29 tests covering:
- Basic operations (put, get, delete, clear)
- TTL expiration and custom TTL
- LRU eviction behavior
- Statistics tracking
- Concurrent access safety
- Auto cleanup functionality
- Edge cases (None values, cache-aside pattern)

**Impact**: Provides a unified, async-native caching solution that integrates seamlessly with existing concurrency utilities in the `nodetool.concurrency` module.

**Files**: `src/nodetool/concurrency/async_cache.py`, `tests/concurrency/test_async_cache.py`

**Date**: 2026-02-16
