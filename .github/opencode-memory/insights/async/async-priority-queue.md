# AsyncPriorityQueue Utility

**Insight**: Added `AsyncPriorityQueue` to the concurrency module for priority-based task scheduling in async workflows.

**Rationale**: Workflow engines often need to process tasks with different priorities. The new `AsyncPriorityQueue` provides:
- Priority-based ordering (lower values = higher priority)
- FIFO ordering for equal priorities (using creation time)
- Optional maximum capacity
- Timeout support for `get()` operations
- Sync `put_nowait()` and `get_nowait()` methods for non-blocking operations
- Async iterator support for convenient consumption

**Example**:
```python
from nodetool.concurrency import AsyncPriorityQueue

queue = AsyncPriorityQueue(max_size=100)

# Add items with different priorities (0 = highest)
await queue.put(0, "urgent_task")
await queue.put(2, "background_task")
await queue.put(1, "important_task")

# Items are returned in priority order
item = await queue.get()  # "urgent_task"
item = await queue.get()  # "important_task"
item = await queue.get()  # "background_task"
```

**Impact**: Complements existing concurrency utilities (`AsyncSemaphore`, `AsyncLock`, `AsyncTaskGroup`) for building sophisticated async workflow execution patterns.

**Files**:
- `src/nodetool/concurrency/async_priority_queue.py` - Implementation
- `tests/concurrency/test_async_priority_queue.py` - Tests (31 tests)

**Date**: 2026-01-20
