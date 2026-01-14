# Test Crashes with pytest-xdist

**Problem**: Multiple tests using `ThreadedEventLoop` caused pytest-xdist worker crashes when running the full test suite with `-n auto` due to resource contention.

**Files Fixed**:
- `tests/workflows/test_job_execution_manager.py`
- `tests/workflows/test_job_execution.py`
- `tests/workflows/test_threaded_job_execution.py`

**Solution**: Added module-level skip using `pytest.skip()` when `PYTEST_XDIST_WORKER` environment variable is set. Tests that create `ThreadedEventLoop` instances are skipped in xdist mode because they don't play well with parallel execution.

**Why**: The tests create multiple threaded event loops which compete for resources when running in parallel with other tests across multiple workers.

**Date**: 2026-01-14
