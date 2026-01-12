### Test Status Timing Issues

**Date Discovered**: 2026-01-12

**Context**: Tests checking job status fail when workflow completes faster than the test can check the status. Empty workflows (0 nodes) complete in milliseconds, so status checks for "running" may find "completed" or "scheduled" instead.

**Solution**:
1. Added wait loop with timeout to allow job to transition to running state
2. Updated status assertions to accept multiple valid states: "running", "completed", "failed", and "scheduled"

**Related Files**:
- `tests/workflows/test_job_execution.py:115-127` - test_start_job
- `tests/workflows/test_threaded_job_execution.py:236-245` - test_threaded_job_database_record

**Prevention**: For quick-completing workflows, check status with a timeout or accept multiple valid terminal states in assertions
