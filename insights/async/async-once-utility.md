# AsyncOnce One-Time Execution Guard

**Insight**: Added `AsyncOnce` utility for ensuring async functions execute only once, even when called concurrently from multiple coroutines.

**Rationale**: In async Python, when the same initialization or setup function might be triggered from multiple entry points, you need a way to:
1. Execute the function exactly once
2. Have all callers wait for and receive the same result
3. Cache exceptions so all callers see the same failure

This pattern is common in:
- Lazy initialization of shared resources
- Singleton-like patterns in async code
- Background setup that may be triggered from multiple places

**Example**:
```python
from nodetool.concurrency import AsyncOnce

once = AsyncOnce()

async def initialize_database():
    # This runs only once, even if called concurrently
    return await setup_connection()

# First call triggers initialization
result1 = await once.run(initialize_database())

# Subsequent calls return the cached result
result2 = await once.run(initialize_database())
assert result1 == result2
```

**Key Features**:
- Thread-safe concurrent call coalescing
- Exception caching and propagation
- Properties for `done`, `result`, and `exception` status

**Files**:
- `src/nodetool/concurrency/async_once.py`
- `tests/concurrency/test_async_once.py`

**Date**: 2026-01-19
