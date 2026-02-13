# Exception Chaining and Type Safety Patterns

**Insight**: Always use `raise ... from e` in exception handlers and validate types before passing to functions that expect specific types.

**Rationale**: 
- Exception chaining preserves the original traceback and makes debugging easier by showing the full error chain
- Type validation before function calls prevents runtime TypeErrors and provides clearer error messages
- These patterns improve code robustness and maintainability

**Example**:
```python
# Before - lost exception context, potential type error
try:
    result = manager.add_user(username, role)
except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e))

# After - preserves context, type-safe
try:
    result = manager.add_user(username, role)
except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e)) from e

# Before - potential None passed to function expecting str
if not is_admin_user(user_id):  # user_id could be None
    raise HTTPException(status_code=403, detail="Admin required")

# After - explicit None check, type-safe
if user_id is None or not is_admin_user(user_id):
    raise HTTPException(status_code=403, detail="Admin required")
```

**Impact**: Fixed 4 endpoints with missing exception chaining and 4 endpoints with potential None type errors in users.py, improving error handling and type safety.

**Files**:
- `src/nodetool/api/users.py`

**Date**: 2026-02-13
