# Async Concurrency Utilities

**Insight**: Added `AsyncSemaphore` and `gather_with_concurrency` utilities in `src/nodetool/concurrency/async_utils.py` for managing async concurrency control.

**Rationale**: The codebase heavily uses async patterns, and these utilities provide common patterns for:
- Rate limiting concurrent operations with timeout support
- Batched execution of async tasks with configurable concurrency limits
- Clean context manager support for resource management

**Example**:

```python
from nodetool.concurrency.async_utils import AsyncSemaphore, gather_with_concurrency

# Rate limiting with timeout
sem = AsyncSemaphore(max_tasks=5)
if await sem.acquire(timeout=10.0):
    try:
        await do_concurrent_work()
    finally:
        sem.release()

# Context manager pattern
async with AsyncSemaphore(max_tasks=3):
    await do_work()

# Batched execution with concurrency limit
results = await gather_with_concurrency(
    [fetch_url(url) for url in urls],
    max_concurrent=10
)
```

**Impact**: Provides reusable async utilities that follow the codebase's async-first patterns and DI conventions.

**Files**:
- `src/nodetool/concurrency/async_utils.py`
- `src/nodetool/concurrency/__init__.py`
- `tests/concurrency/test_async_utils.py`

**Date**: 2026-01-12
