# Type Safety Issues in users.py API Endpoints

**Problem**: API endpoints in `src/nodetool/api/users.py` had several code quality issues:
1. Empty `TYPE_CHECKING` block (TC005 linting error)
2. Redundant inner import of `HTTPException` when already imported at top (F823 error)
3. `user_id` could be `None` but `is_admin_user()` requires `str` (invalid-argument-type typecheck warnings)
4. Missing exception chaining in `except` blocks (B904 linting errors)

**Solution**:
1. Removed empty `if TYPE_CHECKING: pass` block and unused `TYPE_CHECKING` import
2. Removed redundant inner import of `HTTPException`
3. Added `user_id is None or` check before calling `is_admin_user(user_id)` in all endpoints
4. Added `from e` exception chaining to all `raise HTTPException` statements in `except` blocks

**Why**:
- Empty TYPE_CHECKING blocks violate linting rules
- Redundant imports are dead code that confuses type checkers
- Type safety prevents runtime `TypeError` when None is passed to functions expecting str
- Exception chaining preserves original exception context for debugging

**Files**:
- `src/nodetool/api/users.py`

**Date**: 2026-02-13
