# Async Retry Utilities with Exponential Backoff

**Insight**: Added `retry_with_exponential_backoff` function and `RetryPolicy` class in `src/nodetool/concurrency/retry.py` for handling transient failures with configurable retry strategies.

**Rationale**: The codebase has multiple manual retry implementations scattered across different modules (e.g., `retry_on_locked` in sqlite_adapter.py, `query_with_retry` in claude_agent_message_processor.py). A centralized retry utility provides:
- Consistent retry behavior across the codebase
- Configurable exponential backoff with jitter
- Support for unlimited retries (-1) or fixed retry counts
- Customizable exception filtering
- Both functional and decorator-based usage patterns

**Example**:

```python
from nodetool.concurrency.retry import retry_with_exponential_backoff, RetryPolicy

# Simple usage with defaults
result = await retry_with_exponential_backoff(
    lambda: fetch_data(url),
    max_retries=3,
    initial_delay=1.0,
)

# Custom policy with filtering
policy = RetryPolicy(
    max_retries=5,
    initial_delay=0.5,
    max_delay=30.0,
    retryable_exceptions=(ConnectionError, TimeoutError),
)

# Decorator usage
@RetryPolicy(max_retries=3, initial_delay=1.0)
async def unreliable_api_call():
    ...
```

**Impact**: Provides a reusable pattern for handling transient failures, reducing code duplication and ensuring consistent retry behavior across the codebase.

**Files**:
- `src/nodetool/concurrency/retry.py`
- `src/nodetool/concurrency/__init__.py`
- `tests/concurrency/test_retry.py`

**Date**: 2026-01-12
