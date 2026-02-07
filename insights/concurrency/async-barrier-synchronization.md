# Async Barrier Synchronization

**Insight**: AsyncBarrier provides a synchronization primitive for coordinating multiple coroutines to reach a common point before proceeding.

**Rationale**: Barrier synchronization is essential for phased concurrent operations where multiple tasks must complete one phase before any can start the next. Unlike locks (mutual exclusion) or semaphores (resource limiting), barriers ensure all parties arrive before any proceed.

**Example**:
```python
from nodetool.concurrency import AsyncBarrier

# Phased computation where each task must complete phase N before any start phase N+1
barrier = AsyncBarrier(parties=3)

async def worker(task_id: int):
    for phase in range(3):
        # Do work for this phase
        await process_phase(task_id, phase)
        
        # Wait for all tasks to complete this phase
        await barrier.wait()
        
        # All tasks now proceed to next phase together

await asyncio.gather(*[worker(i) for i in range(3)])
```

**Use Cases**:
- Phased parallel algorithms (e.g., parallel iterative methods)
- Coordinated batch processing
- Synchronized checkpointing in distributed workflows
- Ensuring data consistency across multiple producers/consumers

**Key Features**:
- Reusable (automatically resets after all parties pass)
- Properties for visibility (`parties`, `waiting`, `remaining`)
- Leader selection (one coroutine returns False from `wait()`)
- Manual `reset()` for exceptional circumstances

**Files**: `src/nodetool/concurrency/async_barrier.py`

**Date**: 2026-02-07
