# Debounce and Throttle Patterns for Async Code

**Insight**: Debounce and throttle are essential patterns for controlling execution frequency in async systems, preventing resource exhaustion and excessive API calls.

**Rationale**: When building async systems, especially those interacting with external APIs or user interfaces, it's critical to control how frequently operations execute. Debouncing waits for a pause in calls before executing (ideal for search-as-you-type), while throttling limits execution rate (ideal for API rate limiting).

**Implementation Patterns**:

1. **AsyncDebounce**: Delays execution until a specified time has passed since the last invocation. Uses a timer loop that resets on each new call. Only the last function in a rapid series executes.

2. **AsyncThrottle**: Ensures a minimum time between executions. Supports both "skip if throttled" mode (returns None) and "wait mode" (blocks until allowed).

3. **AdaptiveThrottle**: Automatically adjusts throttle interval based on success/failure rates. Increases interval on failures (backoff) and decreases on success (recovery).

**Key Design Decisions**:

- Used `asyncio.Lock` for thread-safe state management
- Implemented non-blocking debounce using `asyncio.create_task()` pattern
- Adaptive throttle uses configurable multipliers for backoff/recovery
- All utilities support cancellation and state inspection

**Example Usage**:
```python
# Debounce user input
debounce = AsyncDebounce(delay=0.5)
task = asyncio.create_task(debounce.execute(lambda: search(query)))

# Throttle API calls
throttle = AsyncThrottle(interval=1.0)
result = await throttle.execute(lambda: api.fetch_data())

# Adaptive throttling for unreliable services
adaptive = AdaptiveThrottle(min_interval=0.1, max_interval=10.0)
result = await adaptive.execute(lambda: unreliable_api_call())
```

**Impact**: Reduces unnecessary computations, prevents API rate limit errors, and improves system responsiveness under load.

**Files**:
- `src/nodetool/concurrency/debounce_throttle.py`
- `tests/concurrency/test_debounce_throttle.py`

**Date**: 2026-02-08
