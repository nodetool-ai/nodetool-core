# Type Checker Issue with hasattr and Dynamic Methods

**Problem**: The `_set_resuming_state()` method is defined on `SuspendableNode` but not on all node types. When calling this method after checking with `hasattr()`, the type checker reports a `call-non-callable` error because it cannot verify the method exists.

**Solution**: Added `# type: ignore[call-non-callable]` comment to the method call in `src/nodetool/workflows/recovery.py:113` since the `hasattr` check guarantees the method exists at runtime.

**Files**:
- `src/nodetool/workflows/recovery.py`

**Date**: 2026-01-16
