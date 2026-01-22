# Autosave Max Versions Limit Bug

**Problem**: The autosave endpoint was using FIFO (First-In-First-Out) behavior instead of skipping new autosaves when the max versions limit was reached. Also, the default max_versions was 50 instead of the expected 20.

**Solution**: Changed the implementation in `src/nodetool/api/workflow.py` to:
1. Use default max_versions of 20 (matching the cleanup function and test expectations)
2. Skip new autosaves when the limit is reached, rather than deleting old ones

**Files**: 
- `src/nodetool/api/workflow.py:915-927` (autosave endpoint)
- `tests/api/test_workflow_version_api.py:341-362` (test)

**Date**: 2026-01-22
