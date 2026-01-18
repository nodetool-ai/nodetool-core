# Async Debounce Pattern

**Insight**: Added `AsyncDebounce` and `async_debounce` utilities to the concurrency module for debouncing rapid successive calls.

**Rationale**: Debouncing is a complementary pattern to rate limiting. While rate limiting controls throughput (N calls per time period), debouncing waits until a quiet period before executing (calls are coalesced into one). This is ideal for:

- Search-as-you-type: Wait until user stops typing before sending API request
- Auto-save: Wait until user stops editing before saving
- API request coalescing: Combine multiple rapid requests into a single batched request
- UI event handling: Throttle rapid UI updates

**Example**:
```python
from nodetool.concurrency import AsyncDebounce, async_debounce

# As a class
async def save_document(content: str):
    await save_to_database(content)

debounced_save = AsyncDebounce(save_document, wait=0.5)
await debounced_save("first")
await debounced_save("second")
# Only one call to save_document("second") will execute after 0.5s

# As a decorator
@async_debounce(wait=0.3)
async def search(query: str):
    results = await api.search(query)
    return results
```

**Features**:
- Configurable wait time for quiet period
- Automatic replacement of pending calls with latest arguments
- `flush()` method for immediate execution when needed
- `cancel()` method to discard pending calls
- `pending` property to check execution state

**Impact**: Provides a missing async utility that pairs well with existing rate limiting, circuit breaker, and retry utilities for building resilient async workflows.

**Files**: `src/nodetool/concurrency/async_debounce.py`, `tests/concurrency/test_async_debounce.py`

**Date**: 2026-01-18
