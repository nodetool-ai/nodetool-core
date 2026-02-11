# Async Cancellation Scope

**Insight**: Added `AsyncCancellationScope` for structured cancellation handling across concurrent async tasks, providing cooperative cancellation patterns with cleanup callbacks.

**Rationale**: The concurrency module lacked a clean way to coordinate cancellation across multiple workers. While asyncio has built-in task cancellation, it requires careful handling and doesn't provide a cooperative pattern where tasks can check for cancellation at safe points.

**Example**:
```python
async def worker(scope: AsyncCancellationScope, task_id: int):
    try:
        while not scope.is_cancelled():
            # Do work
            await asyncio.sleep(0.1)
    except CancellationError:
        print(f"Task {task_id} cleaned up")

scope = AsyncCancellationScope()

async with scope:
    # Launch workers
    tasks = [
        asyncio.create_task(worker(scope, i))
        for i in range(5)
    ]

    # Cancel all workers after some time
    await asyncio.sleep(1.0)
    scope.cancel()

    # Wait for cleanup
    await asyncio.gather(*tasks, return_exceptions=True)
```

**Impact**: Provides a safer, cooperative cancellation pattern compared to raw `asyncio.Task.cancel()`, with proper cleanup handling through registered callbacks.

**Files**:
- `src/nodetool/concurrency/cancellation.py` - New module
- `src/nodetool/concurrency/__init__.py` - Updated exports
- `tests/concurrency/test_cancellation.py` - 15 comprehensive tests

**Date**: 2026-02-11
