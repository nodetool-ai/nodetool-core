# Event Logger Test Assertion Order (Jan 2026)

**Problem**: `test_event_logger_convenience_methods` in `tests/workflows/test_resumable_workflows.py` was asserting specific event order, but blocking and non-blocking event writes result in interleaved database writes that don't preserve call order.

**Solution**: Updated test to:
1. Call `await logger.start()` before logging non-blocking events
2. Call `await logger.stop()` after to flush pending events
3. Changed assertions to verify event types using a set rather than strict index positions

**Why**: The `WorkflowEventLogger` uses `blocking=False` by default for node events, which queues them for async flushing. Blocking calls write directly. This results in interleaved write order that differs from call sequence.

**Files**:
- `tests/workflows/test_resumable_workflows.py:106-132`

**Date**: 2026-01-21
