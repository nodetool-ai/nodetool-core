# Logging Best Practices for NodeTool Core

**Insight**: When adding logging to a Python module, follow these patterns for consistency:

## Import Pattern

```python
import logging

log = logging.getLogger(__name__)
```

Or use the project's logging config:

```python
from nodetool.config.logging_config import get_logger

log = get_logger(__name__)
```

## Log Level Selection

- `log.debug()`: Detailed info for troubleshooting, only in debug mode
- `log.info()`: General operational information
- `log.warning()`: Something unexpected but recoverable
- `log.error()`: Something failed but the system can continue
- `log.critical()`: Serious error, system may need to shut down

## Avoid print() for Production

Never use print() for production code because:
- No log levels (can't filter output)
- No timestamps or context
- Hard to redirect to files or log aggregation systems
- Can't be disabled in production

## Exception Handling with Logging

```python
try:
    something()
except Exception as e:
    log.debug("Operation failed: %s", e)
    # fallback behavior
```

Always include the exception as the last argument for proper formatting.

**Impact**: Consistent logging makes debugging easier and integrates with production monitoring.

**Files**: All source files in `src/nodetool/`

**Date**: 2026-01-20
