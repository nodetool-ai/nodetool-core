# AsyncCountDownLatch for Task Coordination

**Insight**: AsyncCountDownLatch provides a simple synchronization primitive for coordinating multiple async tasks by allowing tasks to wait until a count reaches zero.

**Rationale**: When working with concurrent async operations, it's common to need to wait for multiple tasks to complete before proceeding. The CountDownLatch pattern (from Java) fills this gap elegantly in the async/await context. Unlike `asyncio.gather` which requires all tasks upfront, a latch allows tasks to count down independently as they complete.

**Example**: Coordinating worker completion
```python
async def worker(latch: AsyncCountDownLatch, worker_id: int):
    await do_work(worker_id)
    latch.count_down()  # Each worker counts down when done

latch = AsyncCountDownLatch(num_workers)
tasks = [asyncio.create_task(worker(latch, i)) for i in range(num_workers)]
await latch.wait()  # Blocks until all workers complete
```

**Impact**: Provides a clean, composable way to coordinate async task completion without complex callback chains or manual event tracking.

**Files**: 
- `src/nodetool/concurrency/async_countdown_latch.py`
- `tests/concurrency/test_async_countdown_latch.py`

**Date**: 2026-02-23
