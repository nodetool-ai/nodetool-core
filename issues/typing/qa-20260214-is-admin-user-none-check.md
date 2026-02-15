# is_admin_user() Type Annotation Missing None

**Problem**: The `is_admin_user()` function had a parameter type of `str` but was being called with `str | None` values from `getattr(request.state, "user_id", None)`.

**Solution**: Updated the function signature to accept `str | None` and added an early return for None values.

**Why**: FastAPI request.state.user_id can be None when not authenticated, so the type signature should reflect this.

**Files**:
- `src/nodetool/security/admin_auth.py` (function signature)
- `src/nodetool/api/users.py` (5 call sites)

**Date**: 2026-02-14
