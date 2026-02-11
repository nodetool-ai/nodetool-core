# AsyncTTLCache Design Insights

**Insight**: When building async caches with TTL support, use `asyncio.Lock` for thread-safety and `OrderedDict` for O(1) LRU eviction via `move_to_end()`/`popitem(last=False)`.

**Rationale**: The cache must handle concurrent access from multiple async tasks. Using `asyncio.Lock` ensures atomic operations for cache reads/writes/eviction. `OrderedDict` provides efficient LRU tracking without maintaining a separate data structure.

**Implementation patterns**:

1. **Cache stampede prevention**: Use a "computing" flag and waiter futures to ensure only one task computes a value while others wait for the result.
   ```python
   if entry is not None and entry.computing:
       fut = asyncio.get_running_loop().create_future()
       self._waiters.setdefault(key, []).append(fut)
       await fut  # Wait for computation
   ```

2. **Type-safe generic class**: Use `Generic[K, V]` with proper `Awaitable[V]` for compute_fn signatures. The internal `_CacheEntry` stores `V | None` but uses `cast(V, entry.value)` when `computing=False` since we never store None in non-computing entries.

3. **Stale-while-revalidate**: Return stale value immediately and trigger background refresh with `asyncio.create_task()` - must use `# noqa: RUF006` to satisfy linters for fire-and-forget tasks.

4. **LRU with OrderedDict**: Always call `move_to_end(key)` on access to maintain recency order. Evict with `popitem(last=False)` when at capacity.

**Impact**: Provides a production-ready async cache that reduces redundant expensive operations (API calls, DB queries) while preventing thundering herd problems under concurrent load.

**Files**: `src/nodetool/concurrency/async_ttl_cache.py`, `tests/concurrency/test_async_ttl_cache.py`

**Date**: 2026-02-11
