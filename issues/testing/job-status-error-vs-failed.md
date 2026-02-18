# Job Status "error" vs "failed"

**Problem**: The `WorkflowRunner.run()` method was setting job status to "error" when exceptions occurred, but "error" is not a valid `JobStatus`. The valid statuses are: "scheduled", "running", "suspended", "paused", "completed", "failed", "cancelled", "recovering".

**Solution**: Changed status from "error" to "failed" in both the `self.status` assignment and the `JobUpdate` message creation.

**Why**: This was causing the `test_ctrl_017_controlled_node_error_propagates_and_fails_job` test to fail because it expected status "failed" but was getting "error". The code was using an invalid status value that doesn't match the `JobStatus` type definition.

**Files**:
- `src/nodetool/workflows/workflow_runner.py:1095`
- `src/nodetool/workflows/workflow_runner.py:1110`

**Date**: 2026-02-18
