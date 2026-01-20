# Async Debounce Utility

**Insight**: Added `DebouncedFunc` class and `debounce` decorator for limiting async function call frequency.

**Rationale**: Debouncing is a common pattern for rate limiting at the function call level. Unlike `AsyncRateLimiter` which controls the rate of operations, debouncing ensures a function is only called after a quiet period where no new calls are made. This is useful for:
- Preventing duplicate API requests from rapid user actions
- Rate limiting UI event handlers
- Coalescing multiple rapid updates into a single operation

**Implementation**:
- `DebouncedFunc`: A class that wraps an async function and implements debouncing behavior
- `debounce`: A decorator that creates a debounced version of an async function
- Key methods: `__call__` for scheduling, `flush()` for immediate execution, `cancel()` for discarding pending calls
- Uses execution ID tracking to handle race conditions between rapid calls

**Example**:
```python
from nodetool.concurrency import debounce

@debounce(wait=0.5)
async def save_data(data: dict) -> None:
    await database.save(data)

# Multiple calls within 500ms only trigger one execution
await save_data({"key": "value1"})
await save_data({"key": "value2"})
await save_data({"key": "value3"})
# Only the last call with value3 is executed
```

**Files**:
- `src/nodetool/concurrency/debounce.py` - Implementation
- `tests/concurrency/test_debounce.py` - 19 comprehensive tests

**Date**: 2026-01-20
