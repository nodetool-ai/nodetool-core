# Unsafe Type Assertions in workflows/io.py

**Problem**: Using `assert isinstance()` statements for type checking. Assertions can be disabled with Python's `-O` flag, bypassing validation in production.

**Locations**:
- `src/nodetool/workflows/io.py:167-169` - Constructor parameter validation
- `src/nodetool/workflows/io.py:198` - Internal state check in `emit()`
- `src/nodetool/workflows/io.py:276-278` - Internal state checks in `complete()`
- `src/nodetool/workflows/io.py:315` - Internal state check in `default()`

**Solution**: Replaced `assert isinstance()` checks with explicit `if not isinstance()` checks that raise `TypeError` with descriptive messages.

**Why**: Assertions are meant for debugging and can be disabled with `python -O`. Type checks that validate API contracts should always run in production. Using explicit `TypeError` raises ensures validation cannot be bypassed.

**Files**:
- `src/nodetool/workflows/io.py`

**Date**: 2026-02-20
