# Test Fix: Autosave Max Versions Limit

**Problem**: Test `test_autosave_max_versions_limit` expected autosaves to be skipped when max versions is reached, but the implementation uses FIFO deletion instead.

**Solution**: Updated the test to reflect the actual FIFO (first-in-first-out) behavior where old autosaves are deleted to make room for new ones, rather than skipping new autosaves.

**Why**: The implementation at `src/nodetool/api/workflow.py:917-926` deletes oldest autosaves when count reaches `max_versions` to maintain the limit while preserving the latest data. This is better UX than silently skipping saves.

**Files**: 
- `tests/api/test_workflow_version_api.py:test_autosave_max_versions_limit`
- `src/nodetool/api/workflow.py:905-974`

**Date**: 2026-01-22
