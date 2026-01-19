# Exception Chaining and IO Import Issues

**Problem**: Two code quality issues were causing validation failures:
1. Lint error B904: Exception handlers were raising new exceptions without chaining with `from err`
2. UnboundLocalError: `io.BytesIO()` was used in `processing_context.py` but `BytesIO` was already imported directly from `io`, causing confusion

**Solution**:
1. Added `from err` to all `raise HTTPException` statements in `src/nodetool/api/asset.py` (lines 407, 414, 422)
2. Changed `io.BytesIO(image_bytes)` to `BytesIO(image_bytes)` in `src/nodetool/workflows/processing_context.py` (line 1126)

**Why**: Using exception chaining (`raise ... from err`) preserves the original exception traceback, making debugging easier. The `io.BytesIO` vs `BytesIO` inconsistency was a typo that caused the test to fail.

**Files**:
- `src/nodetool/api/asset.py`
- `src/nodetool/workflows/processing_context.py`

**Date**: 2026-01-19
