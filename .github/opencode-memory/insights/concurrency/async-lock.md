# AsyncLock Implementation

**Insight**: Added `AsyncLock` class to complement `AsyncSemaphore` for exclusive resource access in async code.

**Rationale**: While `AsyncSemaphore` limits concurrent access to N tasks, `AsyncLock` (N=1) ensures exclusive access to a single resource. The implementation wraps `asyncio.Lock` with timeout support using `asyncio.wait_for`.

**Example**:
```python
from nodetool.concurrency import AsyncLock

lock = AsyncLock()

# Using context manager
async with lock:
    await do_exclusive_work()

# Using acquire with timeout
if await lock.acquire(timeout=10.0):
    try:
        await do_exclusive_work()
    finally:
        lock.release()
```

**Implementation Notes**:
- Uses `asyncio.wait_for` for timeout support
- Handles both `TimeoutError` and `asyncio.CancelledError` (Python 3.12+ behavior)
- Follows same patterns as `AsyncSemaphore` for consistency

**Files**:
- `src/nodetool/concurrency/async_lock.py`
- `src/nodetool/concurrency/__init__.py`
- `tests/concurrency/test_async_lock.py`

**Date**: 2026-01-17
