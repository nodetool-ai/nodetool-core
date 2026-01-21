# WorkflowEventLogger Non-Blocking Events Not Flushed

**Problem**: Tests using `WorkflowEventLogger` convenience methods (`log_node_scheduled`, `log_node_started`, `log_node_completed`) that use `blocking=False` failed because events were queued but never flushed. The logger's background flush loop requires `start()` to be called, and pending events require `stop()` to be flushed.

**Solution**: Updated `test_event_logger_convenience_methods` in `tests/workflows/test_resumable_workflows.py` to call `await logger.start()` before logging events and `await logger.stop()` after to ensure all queued events are flushed. Also updated assertions to be order-agnostic since blocking and non-blocking events have different timing.

**Files**:
- `src/nodetool/workflows/event_logger.py`
- `tests/workflows/test_resumable_workflows.py`

**Date**: 2026-01-21
