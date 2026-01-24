# WorkflowEventLogger Test Fix

**Problem**: Test `test_event_logger_convenience_methods` expected 5 events but only received 2 because the logger's background flush loop wasn't started/stopped.

**Solution**: Added `await logger.start()` before logging events and `await logger.stop()` after to ensure queued (non-blocking) events are flushed before assertions.

**Why**: Node events (`log_node_scheduled`, `log_node_started`, `log_node_completed`) use `blocking=False` by default, queuing events asynchronously. Without starting the logger, these events remain in the queue. Also removed order-dependent assertions since event order is non-deterministic when mixing blocking/non-blocking writes.

**Files**: `tests/workflows/test_resumable_workflows.py`

**Date**: 2026-01-21
