### Parallel Test Race Conditions

**Date Discovered**: 2026-01-10

**Context**: 5 job execution tests fail when run with `pytest -n auto` but pass individually

**Solution**: Tests need shared state isolation for parallel execution (pre-existing issue)

**Related Files**: `tests/workflows/test_job_execution.py`, `tests/workflows/test_job_execution_manager.py`

**Prevention**: Use unique test databases/resources for each test
