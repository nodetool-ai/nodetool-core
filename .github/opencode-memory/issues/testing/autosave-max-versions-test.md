# Autosave Max Versions Test

**Problem**: Test `test_autosave_max_versions_limit` in `tests/api/test_workflow_version_api.py` expected max_versions=20 but code uses 50.

**Solution**: Updated test to create 55 autosaves and verify FIFO deletion maintains 50 versions.

**Files**: `tests/api/test_workflow_version_api.py`

**Date**: 2026-01-22
