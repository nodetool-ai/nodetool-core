# AsyncEvent for Inter-Task Signaling

**Insight**: `AsyncEvent` provides a primitive for signaling between async tasks, enabling producer-consumer patterns and coordination without busy-waiting.

**Rationale**: When one task needs to notify others that something has happened, using asyncio.Event with value passing and proper reset semantics provides a clean, efficient solution compared to polling or complex callback chains.

**Example**:
```python
from nodetool.concurrency import AsyncEvent, AsyncTaskGroup

event = AsyncEvent()

async def producer():
    await asyncio.sleep(1)
    event.set("data_ready")

async def consumer():
    value = await event.wait()
    print(f"Received: {value}")

group = AsyncTaskGroup()
group.spawn("producer", producer())
group.spawn("consumer", consumer())
await group.run()
```

**Modes**:
- **Manual reset (default)**: Event stays set until explicitly cleared. All waiters are notified when set() is called.
- **Auto-reset**: Event automatically clears after each waiter receives the value. Useful for one-shot notifications.

**Use Cases**:
- Producer-consumer workflows
- Coordinated task startup/shutdown
- State change notifications
- Broadcasting values to multiple consumers

**Files**: `src/nodetool/concurrency/async_event.py`, `tests/concurrency/test_async_event.py`

**Date**: 2026-01-20
