# TimeoutError vs asyncio.TimeoutError

**Insight**: Use the builtin `TimeoutError` exception instead of `asyncio.TimeoutError` in async code.

**Rationale**: The `asyncio.TimeoutError` is an alias for the builtin `TimeoutError` since Python 3.10. Using the builtin is simpler, more portable, and follows Python best practices. The ruff rule UP041 enforces this.

**Example**:
```python
# Instead of:
except asyncio.TimeoutError:
    pass

# Use:
except TimeoutError:
    pass
```

**Impact**:
- Cleaner code with fewer imports
- Passes ruff UP041 check
- More portable across Python versions

**Files**: `src/nodetool/workflows/event_logger.py`

**Date**: 2026-01-21
