# Deprecated datetime.utcnow() in user_manager.py

**Problem**: `datetime.utcnow()` is deprecated in Python 3.12+ and emits deprecation warnings in type checking.

**Solution**: Replaced `datetime.utcnow()` with `datetime.now(UTC)` and adjusted ISO format conversion:
- Before: `datetime.utcnow().isoformat() + "Z"`
- After: `datetime.now(UTC).isoformat().replace("+00:00", "Z")`

**Why**: `datetime.utcnow()` returns naive datetime objects (no timezone info), while `datetime.now(UTC)` returns timezone-aware objects. The Python documentation recommends using timezone-aware datetime objects for representing UTC times. The `.replace("+00:00", "Z")` maintains backward compatibility with the "Z" suffix format used in the rest of the codebase.

**Files**:
- `src/nodetool/security/user_manager.py`

**Date**: 2026-02-13
