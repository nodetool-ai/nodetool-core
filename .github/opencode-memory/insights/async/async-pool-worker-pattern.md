# AsyncPool Worker Pool Pattern

**Insight**: Added `AsyncPool` class for bounded concurrent task execution.

**Rationale**: While `AsyncTaskGroup` provides unbounded task spawning with result tracking, `AsyncPool` offers controlled parallelism with:
- Fixed number of worker threads
- Configurable work queue size
- Built-in timeout support
- Context manager support

**Example**:
```python
async with AsyncPool(max_workers=4) as pool:
    futures = [pool.submit(process_item, i) for i in range(100)]
    results = await pool.gather_results(futures)
```

**Impact**: Provides a clean abstraction for parallel processing workloads where resource bounds are important (e.g., API calls, file processing).

**Files**:
- `src/nodetool/concurrency/async_pool.py`
- `tests/concurrency/test_async_pool.py`

**Date**: 2026-01-22
