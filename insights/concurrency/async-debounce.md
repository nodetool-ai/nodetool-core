# Async Debounce Implementation

**Insight**: Implemented `AsyncDebounce` and `debounce()` for debouncing async function calls with proper cancellation and flush support.

**Rationale**: Debouncing is essential for rate-limiting bursts of events like user input, API calls, or network requests. The async implementation complements existing concurrency utilities (rate limiting, batching, timeouts) by providing a different pattern for controlling execution timing.

**Design Choices**:
1. Return `asyncio.Task` from `__call__` instead of awaiting immediately - this allows callers to track execution status
2. Use `asyncio.Lock` for thread-safe state management of pending calls
3. Provide `flush()` for immediate execution and `cancel()` to prevent execution
4. Support both decorator syntax (`@debounce(0.5)`) and class-based decorator (`@AsyncDebounce(wait_seconds=0.5)`)

**Example**:
```python
from nodetool.concurrency import debounce, AsyncDebounce

# Using the debounce function
@debounce(0.5)
async def save_document(content: str) -> None:
    await database.save(content)

# Multiple calls within 0.5s will be debounced
await save_document("draft1")
await save_document("draft2")  # First call is cancelled

# Control methods
save_document.cancel()  # Cancel pending call
save_document.flush()   # Execute immediately

# Using AsyncDebounce as decorator
@AsyncDebounce(wait_seconds=0.5)
async def process_upload(file: Upload) -> None:
    await process(file)
```

**Files**:
- `src/nodetool/concurrency/debounce.py` - Implementation
- `tests/concurrency/test_debounce.py` - 17 comprehensive tests

**Date**: 2026-01-21
