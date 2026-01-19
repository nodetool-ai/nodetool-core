# AsyncQueue Implementation Challenges

**Insight**: Implementing a correct async producer-consumer queue requires careful handling of race conditions between putters and getters.

**Rationale**: The AsyncQueue implementation uses futures to coordinate between blocking put() and get() operations. When the queue is full, put() creates a future and waits. When get() is called, it resolves the oldest waiting putter's future and returns the item. This pattern avoids the race conditions that occur when using simple events or conditions.

**Key Implementation Details**:
- Bounded queues use a max_size parameter; unbounded queues have max_size=None
- put() blocks when queue is full, get() blocks when queue is empty
- Timeout variants (put(timeout=...), get(timeout=...)) return False/None on timeout
- Shutdown prevents new puts and raises QueueShutdownError on empty queue gets
- Statistics tracking via QueueStats dataclass

**Files**:
- `src/nodetool/concurrency/async_queue.py`
- `tests/concurrency/test_async_queue.py`

**Date**: 2026-01-19
