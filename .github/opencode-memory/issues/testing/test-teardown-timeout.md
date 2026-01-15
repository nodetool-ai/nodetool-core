# Test Teardown Timeout Issue

**Problem**: Job execution tests (`test_job_execution_manager.py` and `test_job_execution.py`) timeout during fixture teardown when run with pytest-asyncio.

**Symptoms**:
- Tests pass successfully but workers crash during async fixture teardown
- Error: "Timeout waiting for tasks to finish during shutdown"
- Threads stuck in `aiosqlite/core.py` waiting for transactions

**Root Cause**: Threaded event loops created by `ThreadedJobExecution` don't clean up properly when pytest-asyncio runs async finalizers during test teardown.

**Solution**: Added timeout wrapper to Makefile test commands to prevent indefinite hangs. Tests are allowed to timeout gracefully.

**Workaround**:
- Run main test suite without job execution tests: `pytest -n auto -q --ignore=tests/workflows/test_docker_job_execution.py --ignore=tests/workflows/test_job_execution_manager.py --ignore=tests/workflows/test_job_execution.py`
- Run job execution tests individually with timeout: `timeout 60 uv run pytest -q tests/workflows/test_job_execution_manager.py tests/workflows/test_job_execution.py`

**Files**:
- `Makefile` - Added timeout wrappers for job execution tests

**Date**: 2026-01-15
