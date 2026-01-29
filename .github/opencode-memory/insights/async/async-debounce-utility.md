# Async Debounce Utility

**Insight**: Added `DebouncePolicy` class for debouncing async function execution.

**Rationale**: Debouncing is essential for controlling rate-sensitive operations like API calls triggered by user input, event handlers, and resource cleanup. This new utility complements existing concurrency tools (`AsyncRateLimiter`, `RetryPolicy`) by providing debounce-specific functionality.

**Example**:
```python
from nodetool.concurrency import DebouncePolicy

# Debounce auto-save with 500ms wait, 5s maximum
policy = DebouncePolicy(wait=0.5, max_wait=5.0)

async def save_document():
    await database.save(document)

# Schedule saves - only last one executes after quiet period
await policy.schedule(save_document)
await policy.schedule(save_document)  # Reset timer
await policy.schedule(save_document)  # Reset timer

# Force immediate execution
await policy.flush()
```

**Key Features**:
- Configurable wait time before execution
- Optional max_wait to force execution after maximum delay
- Thread-safe with asyncio.Lock
- Context manager support
- flush() method for immediate execution

**Files**:
- `src/nodetool/concurrency/debounce.py`
- `tests/concurrency/test_debounce.py`

**Impact**: Reduces API calls, prevents resource thrashing, and improves user experience for rapid input scenarios.

**Date**: 2026-01-22
