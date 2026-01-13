# Async Timeout Utilities

**Insight**: Added a comprehensive timeout utility module (`src/nodetool/concurrency/timeout.py`) that complements existing async utilities (`AsyncSemaphore`, `gather_with_exponential_backoff`) with timeout control.

**Rationale**: Async code often needs timeout handling to prevent hanging operations. The new utilities provide:
- `timeout` decorator for wrapping functions with timeouts
- `with_timeout` helper for inline timeout control
- `TimeoutPolicy` class for configurable timeout behavior
- `TimeoutContext` for context manager-based timeout control

**Example**:
```python
from nodetool.concurrency.timeout import timeout, with_timeout, TimeoutPolicy

# Decorator usage
@timeout(5.0)
async def fetch_data(url):
    ...

# Inline usage
result = await with_timeout(lambda: fetch_data(url), timeout_seconds=5.0)

# Policy-based usage
policy = TimeoutPolicy(default_timeout=30.0, timeout_exception=ApiTimeoutError)
await policy.execute(lambda: fetch_data(url))
```

**Impact**: Provides consistent, reusable timeout handling across the codebase with support for custom exceptions and messages.

**Files**: `src/nodetool/concurrency/timeout.py`, `src/nodetool/concurrency/__init__.py`

**Date**: 2026-01-13
