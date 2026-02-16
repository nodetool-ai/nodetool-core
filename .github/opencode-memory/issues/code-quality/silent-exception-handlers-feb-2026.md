# Silent Exception Handlers in Workflow Runner

**Problem**: Several exception handlers in `workflow_runner.py` used bare `except Exception:` without logging, making debugging extremely difficult when errors occurred in streaming detection and edge draining operations.

**Solution**: Added `as e` to capture exception objects and `log.debug()` statements to log error details before continuing execution.

**Why**: While these are "best effort" operations where ignoring errors is intentional, completely silent exception handlers make it nearly impossible to diagnose issues in production. Adding debug logging provides observability without changing the error-handling behavior.

**Files**:
- `src/nodetool/workflows/workflow_runner.py:374, 395, 1316`

**Date**: 2026-02-16
