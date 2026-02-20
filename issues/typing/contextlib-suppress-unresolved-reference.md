# Unresolved Reference: `suppress` from contextlib

**Problem**: The `suppress` context manager from `contextlib` module was used in `cleanup_gpu_memory()` function without being imported, causing a type checking error (unresolved-reference).

**Solution**: Added `suppress` to the existing `from contextlib import contextmanager` statement, changing it to `from contextlib import contextmanager, suppress`.

**Why**: The `suppress` context manager is used to safely ignore exceptions from `torch.cuda.ipc_collect()` which may not be available in all PyTorch builds. Without the import, type checkers flag this as an error even though the code would work at runtime (Python would raise NameError).

**Files**:
- `src/nodetool/workflows/memory_utils.py`

**Impact**:
- Fixed type checking error (unresolved-reference)
- All typecheck, lint, and test validations pass

**Date**: 2026-02-20
