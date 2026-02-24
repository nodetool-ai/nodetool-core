# Logging with Context

**Insight**: Error logs should include sufficient context for debugging (user IDs, request details, timestamps) to enable effective troubleshooting.

**Rationale**: Logs without context make it difficult to trace issues back to specific requests or users. Adding structured context improves observability and debugging.

**Best Practices**:
1. **Include user context**: Add user IDs, workflow IDs, or other identifiers
2. **Request context**: Include method, path, and client information
3. **Use structured logging**: Use proper log levels and format parameters
4. **Avoid string concatenation**: Use %s formatting or f-strings for lazy evaluation

**Example Pattern**:
```python
# Before:
log.error(f"Request validation error: {exc}")

# After:
log.error(
    "Request validation error: %s | Method: %s | Path: %s | Client: %s",
    exc.errors(),
    request.method,
    request.url.path,
    request.client.host if request.client else "unknown",
)
```

**Impact**: Better traceability of issues, easier debugging, improved incident response.

**Files**: `src/nodetool/api/server.py`, `src/nodetool/api/workspace.py`

**Date**: 2026-02-24
