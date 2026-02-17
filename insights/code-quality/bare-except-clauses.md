# Bare Except Clauses Should Be Specific

**Insight**: Bare `except Exception:` clauses should specify the actual exception types being caught to avoid masking unexpected errors and make debugging easier.

**Rationale**:
- Bare except clauses catch ALL exceptions, including SystemExit and KeyboardInterrupt
- This makes debugging difficult as unexpected errors are silently caught
- Specific exception types document what errors are expected
- Type checkers and linters can better understand code flow

**Example**:
```python
# Problem: Catches everything
try:
    py_type = slot.type.get_python_type()
except Exception:
    py_type = None

# Solution: Catch specific expected exceptions
try:
    py_type = slot.type.get_python_type()
except (AttributeError, TypeError, NotImplementedError):
    # Type doesn't support get_python_type or fails during type resolution
    py_type = None
```

**Impact**: Improved in `src/nodetool/dsl/handles.py` and `src/nodetool/io/uri_utils.py`

**Files**:
- `src/nodetool/dsl/handles.py` (line 63)
- `src/nodetool/io/uri_utils.py` (line 20)

**Date**: 2026-02-17
