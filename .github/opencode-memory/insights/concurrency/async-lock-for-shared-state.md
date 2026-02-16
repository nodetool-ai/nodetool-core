# Using Async Lock for Shared Mutable State

**Insight**: Shared mutable state in async codebases must be protected with `asyncio.Lock()` to prevent race conditions, even for simple dictionary operations.

**Rationale**:
- Python dictionaries are not thread-safe for concurrent modifications
- In async code, multiple coroutines can interleave at any `await` point
- Without locks, read-modify-write operations can race and corrupt state
- Even simple operations like `if key in dict: return dict[key]` are not atomic

**Example**:
```python
# Bad - race condition
_CACHE: dict[str, str] = {}

async def get_cached(key: str) -> str | None:
    if key in _CACHE:  # Another coroutine could modify here
        return _CACHE[key]
    return None

# Good - protected with lock
_CACHE: dict[str, str] = {}
_CACHE_LOCK = asyncio.Lock()

async def get_cached(key: str) -> str | None:
    async with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key]
    return None
```

**Impact**: Prevents race conditions, data corruption, and mysterious bugs in production systems.

**Files**: `src/nodetool/security/secret_helper.py`

**Date**: 2026-02-16
