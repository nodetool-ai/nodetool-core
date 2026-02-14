# Nullable user_id Pattern in Auth Checks

**Insight**: When extracting `user_id` from request state using `getattr(request.state, "user_id", None)`, the result is always `Any | None` and requires explicit null checking before use.

**Rationale**: The authentication middleware sets user_id on successful auth, but endpoints should always validate the value exists before passing it to type-annotated functions.

**Example**:
```python
# Before (type unsafe)
user_id = getattr(request.state, "user_id", None)
if not is_admin_user(user_id):  # Type error if user_id is None
    raise HTTPException(...)

# After (type safe)
user_id = getattr(request.state, "user_id", None)
if not user_id or not is_admin_user(user_id):  # Explicit null check
    raise HTTPException(...)
```

**Impact**: Prevents runtime type errors and improves code safety in authentication-dependent endpoints.

**Files**: `src/nodetool/api/users.py`

**Date**: 2026-02-14
