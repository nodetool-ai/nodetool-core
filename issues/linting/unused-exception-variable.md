# Unused Exception Variables in Except Clauses

**Problem**: Exception variables assigned in except clauses but never used trigger F841 linter errors.

**Solution**: Remove the variable name if the exception isn't being used, or use a logging statement to record it.

**Why**: Unused variables add noise and can confuse readers about whether the exception is actually being handled.

**Example**:
```python
# Problem: Variable 'e' assigned but never used
except (OSError, RuntimeError) as e:
    # fallback code

# Solution: Remove variable name
except (OSError, RuntimeError):
    # fallback code
```

**Files**:
- `src/nodetool/io/uri_utils.py` (line 20)

**Date**: 2026-02-17
