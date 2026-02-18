# Test Control Workflow B007 Unused Variable

**Problem**: A test in `test_control_workflow.py` had an unused loop variable `msg` in an async for loop, triggering a B007 linting rule violation.

**Solution**: Renamed the variable from `msg` to `_msg` to indicate it's intentionally unused.

**Why**: The loop was only used to consume messages from the generator, not to use the variable values. Prefixing with underscore is the Python convention for intentionally unused variables.

**Files**:
- `tests/workflows/test_control_workflow.py:190`

**Date**: 2026-02-18
