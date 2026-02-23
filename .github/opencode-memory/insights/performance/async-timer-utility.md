# Async Timer Utility

**Insight**: Async timing utilities provide high-resolution performance measurement for async operations using Python's monotonic clock.

**Rationale**: Performance monitoring and debugging async code requires accurate timing that doesn't interfere with execution. The monotonic clock is immune to system clock changes and provides suitable resolution for measuring sub-millisecond operations.

**Implementation**: The `AsyncTimer` class provides three usage patterns:

1. **Context Manager**: `async with AsyncTimer(auto_start=True) as timer:` - automatically starts timing on entry, stops on exit
2. **Manual Control**: `timer.start()` / `timer.stop()` for precise control over what gets measured
3. **Decorator**: `@timer()` or `@timer(name="OpName", logger=print)` for function-level timing with optional logging

**Example**:
```python
from nodetool.concurrency import AsyncTimer, timer

# Context manager pattern
async with AsyncTimer(auto_start=True) as t:
    await some_async_operation()
print(f"Took {t.elapsed:.3f}s")

# Decorator pattern with logging
@timer(name="API Fetch", logger=logging.info)
async def fetch_data(url):
    return await http_get(url)
```

**Design Decisions**:
- Uses `time.monotonic()` instead of `time.time()` for immunity to clock adjustments
- Thread-safe timing via monotonic clock (important for async context)
- `TimerStats` class provides detailed information including start/end timestamps
- Decorator stores elapsed time as function attribute for programmatic access
- Context manager properly handles exceptions (timer stops even if exception occurs)

**Files**: `src/nodetool/concurrency/timer.py`, `tests/concurrency/test_timer.py`

**Date**: 2026-02-23
