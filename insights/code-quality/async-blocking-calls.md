# Async/Blocking Call Pattern for `concurrent.futures.Future`

**Insight**: When mixing `concurrent.futures.Future` (from `ThreadedEventLoop.run_coroutine()`) with async code, you cannot await the future directly because it doesn't have `__await__`.

**Rationale**: `concurrent.futures.Future` is designed for thread-based concurrency and doesn't implement the async protocol. To await it in an async context, you must convert it using `asyncio.wrap_future()`.

**Example**:
```python
from concurrent.futures import Future
import asyncio

# Blocking (bad in async context):
result = future.result()  # Blocks the event loop

# Non-blocking (good):
wrapped = asyncio.wrap_future(future)
await asyncio.wait_for(wrapped, timeout=30.0)
```

**Impact**: Prevents event loop blocking and allows proper timeout handling for threaded operations.

**Files**:
- `src/nodetool/workflows/run_workflow.py`

**Date**: 2026-01-15
