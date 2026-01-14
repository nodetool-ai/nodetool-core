# JobExecutionManager Test Flakiness

**Problem**: Tests in `test_job_execution_manager.py` and `test_job_execution.py` fail with timeouts when run after the parallel test suite. The Makefile was running these tests twice - first excluded from parallel tests, then run separately - which caused threading/event loop cleanup issues leading to timeouts.

**Solution**: Removed the redundant second test run from the Makefile. The parallel test suite already excludes these flaky tests and passes successfully. These tests should only be run manually when specifically testing job execution functionality.

**Files Modified**:
- `Makefile` - Removed duplicate test run for flaky tests

**Date**: 2026-01-14
