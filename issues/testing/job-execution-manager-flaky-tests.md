# JobExecutionManager Test Flakiness

**Problem**: Tests in `test_job_execution_manager.py` and `test_job_execution.py` fail with worker crashes when run with pytest-xdist parallel execution (`-n auto`).

**Solution**: Run these tests serially (without `-n auto`) after the parallel test suite. Updated Makefile to:
1. Run parallel tests with these files excluded
2. Run these specific tests separately in serial mode

**Why**: The `JobExecutionManager` singleton has state that conflicts when accessed from multiple parallel test workers. Tests pass when run individually or serially.

**Files Modified**:
- `Makefile` - Updated test targets to run problematic tests separately
- `tests/workflows/test_job_execution.py` - Added `pytest.mark.serial` 
- `tests/workflows/test_job_execution_manager.py` - Added `pytest.mark.serial`

**Date**: 2026-01-14
