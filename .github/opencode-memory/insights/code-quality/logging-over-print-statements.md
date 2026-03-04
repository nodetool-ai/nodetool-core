# Use Proper Logging Instead of print() Statements

**Insight**: Always use proper logging infrastructure instead of `print()` statements for error messages in production code.

**Rationale**: Using `log = get_logger(__name__)` from the logging configuration provides:
- Configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Consistent log formatting with timestamps and context
- Output control via environment variables (LOG_LEVEL)
- Easier debugging and production monitoring
- Ability to redirect logs to files or external services

**Example**:
```python
# ❌ BAD - print() bypasses logging infrastructure
except psycopg.Error as e:
    print(f"PostgreSQL error during table creation: {e}")
    raise e

# ✅ GOOD - use proper logging
except psycopg.Error as e:
    log.error(f"PostgreSQL error during table creation: {e}")
    raise e
```

**Impact**: Improved observability and log management in production environments.

**Files**: src/nodetool/models/postgres_adapter.py

**Date**: 2026-03-04
