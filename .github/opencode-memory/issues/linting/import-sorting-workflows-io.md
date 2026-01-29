# Import Sorting Issue in workflows/io.py

**Problem**: Ruff reported unsorted imports in `src/nodetool/workflows/io.py` at line 8.

**Solution**: Ran `ruff check --select I --fix` to auto-fix the import sorting.

**Files**: `src/nodetool/workflows/io.py`

**Date**: 2026-01-22
