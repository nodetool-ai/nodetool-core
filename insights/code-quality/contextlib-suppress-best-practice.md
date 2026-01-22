# contextlib.suppress() Best Practice

**Insight**: Using `contextlib.suppress()` is more idiomatic and cleaner than try-except-pass patterns for exception handling where the exception is truly ignorable.

**Rationale**:
- Reduces boilerplate code
- Makes the intent clearer (we explicitly want to ignore these exceptions)
- Easier to read and maintain
- Python 3.4+ standard library feature

**Example**:

Before:
```python
try:
    os.remove(temp_file)
except OSError:
    pass
```

After:
```python
with suppress(OSError):
    os.remove(temp_file)
```

**Impact**: Applied to 28 locations across the codebase, improving code readability and consistency.

**Files**: See `issues/code-quality/sim105-f401-improvements-jan-2026.md`

**Date**: 2026-01-22
