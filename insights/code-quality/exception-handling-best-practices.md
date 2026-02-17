# Exception Handling vs Assertions

**Insight**: Use proper exception handling for runtime validation, not assert statements. Python's `assert` is a debugging tool that can be disabled with `python -O`, making it unreliable for production validation.

**Rationale**: 
- Assert statements are removed when Python runs with optimizations (`-O` flag)
- Runtime validation must always execute, regardless of Python flags
- Explicit exceptions provide better error messages and stack traces
- Type checkers understand explicit condition checks better than asserts

**Example**:

**Before (unsafe):**
```python
assert image_name, "Image name is required"
assert api_key, "API_KEY environment variable is not set"
```

**After (safe):**
```python
if not image_name:
    raise ValueError("Image name is required")
if not api_key:
    raise ValueError("API_KEY environment variable is not set")
```

**Impact**: 
- Ensures validation always runs in production
- Better error messages for users
- More maintainable code
- Type checkers can narrow types more effectively

**Date**: 2026-02-17
