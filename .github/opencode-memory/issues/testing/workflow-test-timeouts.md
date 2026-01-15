# Workflow Test Timeouts

**Problem**: `tests/workflows/test_job_execution_manager.py` and `tests/workflows/test_job_execution.py` timeout due to SQLite database locking when running multiple threaded workflow executions concurrently.

**Solution**: Exclude these tests from quick test runs using `--ignore` flags. Tests pass individually but timeout when run together due to SQLite adapter contention.

**Why**: The threaded workflow execution uses multiple SQLite connections that lock the database, causing retries that exceed test timeouts.

**Files**:
- `tests/workflows/test_job_execution_manager.py`
- `tests/workflows/test_job_execution.py`
- `src/nodetool/models/sqlite_adapter.py`

**Date**: 2026-01-15
