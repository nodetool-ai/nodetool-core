# Async Barrier Synchronization Primitive

**Insight**: Added `AsyncBarrier` and `BrokenBarrierError` for synchronizing groups of async tasks at a checkpoint.

**Rationale**: The existing concurrency primitives (`AsyncEvent`, `AsyncLock`, `AsyncSemaphore`) cover signaling and mutual exclusion, but not coordinated waiting. `AsyncBarrier` fills this gap by allowing multiple tasks to wait for each other before proceeding, essential for phased parallel operations.

**Example**:
```python
from nodetool.concurrency import AsyncBarrier

barrier = AsyncBarrier(3)

async def phase_worker(worker_id):
    # Phase 1: Do some work
    await do_work(worker_id)
    
    # Wait for all workers to complete phase 1
    await barrier.wait()
    
    # Phase 2: All workers proceed together
    await continue_work(worker_id)

# All workers run in parallel, synchronizing between phases
tasks = [phase_worker(i) for i in range(3)]
await asyncio.gather(*tasks)
```

**Key Features**:
- **Phased Operations**: All tasks must reach the barrier before any can proceed
- **Reusable**: Barrier automatically resets after all parties arrive
- **Timeout Support**: Optional timeout prevents indefinite blocking
- **Abort Capability**: Force break the barrier for error handling
- **Thread Safety**: All operations are async-safe

**Impact**: Enables implementing parallel algorithms that require synchronization points, such as bulk data processing, distributed consensus, and test synchronization.

**Files**:
- `src/nodetool/concurrency/async_barrier.py`
- `tests/concurrency/test_async_barrier.py`

**Date**: 2026-01-22
