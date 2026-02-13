# API Users Type Safety and Exception Handling

**Problem**: Multiple type safety and exception handling issues in `src/nodetool/api/users.py`

**Solution**:
1. Removed empty `TYPE_CHECKING` block that served no purpose
2. Fixed `is_admin_user()` calls to check for `None` user_id before passing to function
3. Removed duplicate `HTTPException` import inside exception handler
4. Added exception chaining with `from e` to preserve stack traces

**Why**:
- Empty `TYPE_CHECKING` blocks are unnecessary code clutter (TC005 lint rule)
- `is_admin_user()` expects `str` but `getattr()` can return `None`, causing type errors
- Importing inside exception handler is redundant and causes F823 "referenced before assignment" error
- Exception chaining (B904) helps distinguish original exceptions from exception handler errors

**Files**: `src/nodetool/api/users.py`

**Date**: 2026-02-13
