### Flaky Parallel Test Execution

**Date Discovered**: 2026-01-12

**Context**: Job execution tests (`tests/workflows/test_job_execution.py`) occasionally fail when run with `pytest -n auto` due to race conditions with the `JobExecutionManager` singleton and threaded event loop cleanup. Tests pass individually or when run with `-n 0`.

**Solution**: Tests are marked with `pytest.mark.xdist_group(name="job_execution")` to run in the same xdist group, but this doesn't fully isolate from other test files running in parallel. The issue is inherent to the singleton pattern and async cleanup.

**Related Files**: `tests/workflows/test_job_execution.py`, `src/nodetool/workflows/job_execution_manager.py`

**Prevention**: Re-run tests if they fail; the issue is flaky, not a code regression. Consider using process-level isolation for these tests in the future.
