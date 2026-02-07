# Async Condition Variable

**Insight**: Condition variables enable tasks to wait for specific state changes atomically, preventing race conditions between checking a condition and waiting for notification.

**Rationale**: Unlike simple events or barriers, condition variables combine a lock with a wait/notify mechanism. This ensures that between checking a condition and beginning to wait, the condition cannot change. Without this atomic check-and-wait pattern, tasks could miss notifications or wait indefinitely.

**Example**:
```python
condition = AsyncCondition()
state = {"ready": False}

async def consumer():
    async with condition:  # Acquires lock
        while not state["ready"]:  # Check condition atomically
            await condition.wait()  # Releases lock, waits, re-acquires
        # Process data

async def producer():
    async with condition:  # Same lock
        state["ready"] = True
        condition.notify_all()
```

**Use Cases**:
- Producer-consumer patterns with bounded buffers
- Waiting for state changes that depend on complex conditions
- Coordinating multiple tasks that need to check conditions safely
- Implementing thread-safe data structures with async operations

**Key Methods**:
- `wait()`: Release lock, wait for notification, re-acquire lock
- `wait_for(predicate)`: Wait until predicate is true (handles spurious wakeups)
- `wait_for_timeout(predicate, timeout)`: Wait with timeout, returns bool
- `notify(n=1)`: Wake n waiting tasks
- `notify_all()`: Wake all waiting tasks

**Impact**: Prevents race conditions in producer-consumer scenarios and provides a structured way to coordinate async tasks based on complex state conditions.

**Files**:
- `src/nodetool/concurrency/async_condition.py`
- `tests/concurrency/test_async_condition.py`

**Date**: 2026-02-07
