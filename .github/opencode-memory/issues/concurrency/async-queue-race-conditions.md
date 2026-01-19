# Async Queue Race Conditions

**Problem**: Concurrent producer-consumer queues are difficult to implement correctly due to subtle race conditions between put and get operations.

**Solution**: Use a future-based approach where:
- Putters create a future and wait when queue is full
- Getters resolve the oldest waiting putter's future directly
- This avoids the "add to queue then signal" pattern which can cause duplicates

**Why**: Traditional approaches using asyncio.Event or Condition can lead to:
- Items being added multiple times
- Futures never being resolved
- Producers/consumers hanging indefinitely

**Files**:
- `src/nodetool/concurrency/async_queue.py`
- `tests/concurrency/test_async_queue.py`

**Date**: 2026-01-19
