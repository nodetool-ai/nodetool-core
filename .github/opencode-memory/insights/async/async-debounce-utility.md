# Async Debounce Utility

**Insight**: Added `debounce` and `DebouncedCall` utilities for debouncing async function calls.

**Rationale**: Debouncing complements the existing rate limiting utilities by waiting for a period of inactivity before executing a function. This is useful for:
- UI event handling (e.g., waiting for user to stop typing before searching)
- API call optimization (e.g., batching rapid updates)
- Preventing rapid successive calls to expensive functions

**Example**:
```python
from nodetool.concurrency import debounce, DebouncedCall

# Decorator usage
debounced_search = debounce(search_function, wait=0.3)

# Context manager usage for fine-grained control
async with DebouncedCall(save_data, wait=0.3) as debounced:
    await debounced.trigger(data)
```

**Features**:
- `leading` edge option: Execute immediately on first call
- `trailing` edge option: Execute after wait period of inactivity (default)
- `max_wait` option: Force execution after maximum wait time
- Both function decorator and class-based context manager APIs

**Files**:
- `src/nodetool/concurrency/debounce.py`
- `tests/concurrency/test_debounce.py`

**Date**: 2026-01-18
