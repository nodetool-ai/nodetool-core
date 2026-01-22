# Pre-existing Test Failure

**Problem**: The test `test_event_logger_convenience_methods` in `tests/workflows/test_resumable_workflows.py` fails because it expects 5 events but only 2 are found.

**Root Cause**: The test logs events using `blocking=False` for node events (scheduled, started, completed) which queues them but doesn't write them. The test doesn't call `stop()` on the logger to flush the queued events.

**Impact**: This is a pre-existing test issue unrelated to linting/code quality improvements.

**Files**: `tests/workflows/test_resumable_workflows.py`

**Date**: 2026-01-21
