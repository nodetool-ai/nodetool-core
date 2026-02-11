# User API Type Safety and Exception Chaining

**Problem**: Multiple code quality issues in `src/nodetool/api/users.py`:
1. Empty `TYPE_CHECKING` block that served no purpose
2. Missing `user_id` null check before calling `is_admin_user()` which expects `str`
3. Redundant `HTTPException` re-import inside an except block
4. Missing exception chaining in except blocks (B904 lint rule)

**Solution**:
1. Removed the unused `TYPE_CHECKING` import and empty block
2. Added `not user_id or` checks before all `is_admin_user(user_id)` calls
3. Removed the redundant `from fastapi import HTTPException` inside the except block
4. Added `from e` exception chaining to all `raise HTTPException` statements in except blocks

**Why**: These changes:
- Fix type checking warnings about `user_id` being `Any | None` instead of `str`
- Improve error debugging by preserving exception chains
- Clean up unnecessary imports and code
- Pass lint checks (TC005, F823, B904)

**Files**: `src/nodetool/api/users.py`

**Date**: 2026-02-11
