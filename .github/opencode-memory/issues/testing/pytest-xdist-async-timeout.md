# Pytest-xdist Async Teardown Timeout

**Problem**: Job execution tests timeout during teardown when run with pytest-xdist parallel execution.

**Symptoms**:
- Tests pass successfully but workers crash during fixture teardown
- Error: "Timeout waiting for tasks to finish during shutdown"
- Threads stuck in `aiosqlite/core.py` waiting for transactions
- Async finalizer in pytest-asyncio plugin hangs

**Files affected**:
- `tests/workflows/test_job_execution.py`
- `tests/workflows/test_job_execution_manager.py`

**Root cause**: Threaded event loops created by `ThreadedJobExecution` don't clean up properly when pytest-asyncio runs async finalizers during test teardown.

**Workaround**: Run tests individually or exclude these tests from parallel execution.

**Date**: 2026-01-12
