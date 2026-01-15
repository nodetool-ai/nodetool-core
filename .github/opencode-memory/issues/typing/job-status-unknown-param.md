# Job Model Unknown Parameter

**Problem**: `subprocess_job_execution.py` passed `status="running"` to `Job` model constructor, but the model has no `status` field.

**Solution**: Removed the `status` parameter from the `Job` model instantiation.

**Why**: The `Job` model (`src/nodetool/models/job.py`) doesn't define a `status` field. Job status is implicitly tracked via `started_at` and `finished_at` timestamps. The code was incorrectly assuming a `status` field existed.

**Files**:
- `src/nodetool/workflows/subprocess_job_execution.py:729`
- `src/nodetool/models/job.py`

**Date**: 2026-01-15
