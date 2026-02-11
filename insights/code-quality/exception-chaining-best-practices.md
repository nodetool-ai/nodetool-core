# Exception Chaining Best Practices

**Insight**: Python's `raise ... from err` syntax preserves the full exception stack trace, making debugging significantly easier.

**Rationale**: When catching and re-raising exceptions in an `except` block, using `raise NewException() from original_exception` preserves the original traceback and shows both exceptions in the stack. Without chaining, the original exception context is lost, making debugging much harder.

**Example**:
```python
# Bad - loses original exception context
try:
    manager.add_user(username, role)
except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e))

# Good - preserves exception chain
try:
    manager.add_user(username, role)
except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e)) from e
```

**Impact**: Ruff rule B904 enforces this pattern. Using explicit chaining helps developers trace the root cause of errors through multiple layers of exception handling.

**Date**: 2026-02-11
