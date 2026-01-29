# Import Order Linting Fix

**Problem**: Unsorted import block in `src/nodetool/concurrency/retry.py` caused linting failure.

**Solution**: Reordered imports to follow project conventions - `collections.abc.Callable` now comes before `typing` imports.

**Files**:
- `src/nodetool/concurrency/retry.py`

**Date**: 2026-01-18
