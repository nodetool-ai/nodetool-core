# Logging in Best-Effort Exception Handlers

**Insight**: Even "best effort" exception handlers that intentionally ignore errors should log debug information for observability.

**Rationale**:
- Silent exception handlers make debugging nearly impossible in production
- Debug logging provides visibility without changing error-handling behavior
- Helps identify recurring issues or problematic code paths
- Minimal performance overhead when log level is set appropriately
- Enables post-mortem analysis and monitoring

**Example**:
```python
# Bad - completely silent
try:
    detect_streaming_nodes()
except Exception:
    pass  # What went wrong? When? How often?

# Good - observable
try:
    detect_streaming_nodes()
except Exception as e:
    log.debug(f"Best-effort streaming detection failed: {e}")
    # Still ignores error, but we know it happened
```

**Impact**: Improved debuggability and operational awareness without sacrificing the "best effort" error-handling strategy.

**Files**: `src/nodetool/workflows/workflow_runner.py`

**Date**: 2026-02-16
