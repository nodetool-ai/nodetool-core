# AsyncObjectPool Pattern

**Insight**: Added `AsyncObjectPool` for generic resource pooling with lazy initialization and self-healing.

**Rationale**: The codebase already has specialized connection pools for SQLite and PostgreSQL. A generic object pool provides a reusable pattern for:
- HTTP session reuse
- ML model instance pooling
- Any expensive-to-create async resource
- Reducing latency and memory pressure

**Example**:

```python
from nodetool.concurrency.object_pool import AsyncObjectPool, pooled

# Create a pool for HTTP sessions
pool = AsyncObjectPool(
    factory=create_http_session,
    validator=validate_session,
    destructor=close_http_session,
    max_size=10,
)

# Use with context manager
async with pooled(pool) as session:
    await session.get("https://api.example.com")

# Use with direct acquire/release
session = await pool.acquire(timeout=5.0)
try:
    await session.get("https://api.example.com")
finally:
    pool.release(session)
```

**Key Features**:
- Lazy initialization (objects created only when needed)
- "Validate on Borrow" health checks
- Self-healing: dead objects replaced automatically
- Configurable pool size and timeouts
- Synchronous release for performance

**Files**:
- `src/nodetool/concurrency/object_pool.py`
- `src/nodetool/concurrency/__init__.py`
- `tests/concurrency/test_object_pool.py`

**Date**: 2026-01-18
