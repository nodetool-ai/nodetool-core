# Avoid Assertions in Production Code

**Insight**: Python's `assert` statements should not be used for runtime validation in production code because:
1. They are silently removed when Python is run with the `-O` (optimize) flag
2. `AssertionError` is a generic exception that doesn't clearly indicate the error type
3. They are intended for debugging/development, not for input validation or error handling

**Rationale**: When assertions are optimized away, critical validation checks disappear, potentially allowing invalid state to propagate through the system. This is especially dangerous in:
- Input validation (missing required fields)
- Type checking (ensuring correct data types)
- State validation (ensuring internal consistency)

**Example**:
```python
# BAD - Can be optimized away
assert tool is not None, f"Tool {name} not found"

# GOOD - Always runs and raises specific exception
if tool is None:
    raise ValueError(f"Tool {name} not found")
```

**Best Practice**: Use specific exception types:
- `ValueError` for invalid input/missing required values
- `TypeError` for type mismatches
- `RuntimeError` for internal state errors
- `KeyError` for missing dictionary keys
- Custom exceptions for domain-specific errors

**Impact**: Prevents silent failures in production environments where Python optimization may be enabled, and provides clearer error messages for debugging and logging.

**Files**: All production code, especially:
- API handlers and validators
- Chat and messaging modules
- Database adapters
- Workflow runners

**Date**: 2026-02-21
