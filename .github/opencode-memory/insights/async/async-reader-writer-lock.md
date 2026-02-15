# Async Reader-Writer Lock

**Insight**: A read-write lock allows multiple concurrent readers but exclusive writer access, essential for shared data structures that are read frequently but written occasionally.

**Rationale**:
- Standard locks (like `AsyncLock`) allow only one task at a time, which is inefficient for read-heavy workloads
- Reader-writer locks optimize for the common case where many tasks read data while writes are rare
- Writer-preference semantics prevent writer starvation by blocking new readers when a writer is waiting

**Implementation Details**:
- Uses `asyncio.Condition` for efficient task coordination and notification
- Both acquire and release methods are async, consistent with Python's asyncio patterns
- Supports timeout for both read and write acquisition
- Context managers (`read_lock()`, `write_lock()`) for automatic cleanup
- Properties for introspection: `readers`, `writers`, `write_pending`, `locked`, `write_locked`

**Example**:
```python
from nodetool.concurrency import AsyncReaderWriterLock

lock = AsyncReaderWriterLock()
shared_data = []

# Multiple readers can hold the lock simultaneously
async def reader():
    async with lock.read_lock():
        # Safe to read - multiple readers can be here at once
        print(len(shared_data))

# Only one writer at a time, exclusive access
async def writer():
    async with lock.write_lock():
        # Safe to modify - no readers or other writers
        shared_data.append("data")
```

**When to Use**:
- Caches and shared state with many reads, few writes
- Configuration data that is read often but updated rarely
- Collections where iteration happens frequently but modifications are rare

**When NOT to Use**:
- Write-heavy workloads (a standard lock may be more efficient)
- Simple critical sections (use `AsyncLock` instead)

**Files**:
- `src/nodetool/concurrency/async_rwlock.py`
- `tests/concurrency/test_async_rwlock.py`

**Date**: 2026-02-07
