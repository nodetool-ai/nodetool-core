# Async Event Logger Usage Pattern

**Insight**: The `WorkflowEventLogger` class uses an async queue for batching event writes to improve performance. Node events use non-blocking mode (`blocking=False`) for performance, while run-level events use blocking mode.

**Rationale**: This design reduces database contention during high-frequency node execution by batching events. The background flush loop runs every 0.1 seconds by default.

**Usage Pattern**:
```python
logger = WorkflowEventLogger(run_id)
await logger.start()  # Start background flush task
# ... log events (node events are queued) ...
await logger.stop()   # Flush remaining events and stop
```

**Impact**: Tests and code using the logger must call `start()` and `stop()` to ensure events are properly flushed.

**Files**: `src/nodetool/workflows/event_logger.py`

**Date**: 2026-01-22
