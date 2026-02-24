# User-Friendly Error Messages

**Insight**: Error messages should be actionable and user-facing, while detailed technical information should be logged server-side for debugging.

**Rationale**: Raw exception strings expose internal implementation details and are not helpful to end users. Users need clear, actionable error messages that explain what went wrong and what they can do about it.

**Best Practices**:
1. **User-facing messages**: Explain what went wrong and suggest next steps
2. **Server-side logs**: Include full context (user IDs, request details, full stack traces)
3. **Consistent format**: Use similar structure across all error responses
4. **Appropriate detail**: More detail in development, sanitized in production

**Example Pattern**:
```python
try:
    # operation
except SpecificException as e:
    log.error("Context about what failed: %s", e, extra={"user_id": user, "request_id": id})
    raise HTTPException(
        status_code=appropriate_code,
        detail="User-friendly message explaining what to do",
    ) from e
```

**Impact**: Improved user experience, better debugging capability, enhanced security by not exposing internals.

**Files**:
- `src/nodetool/api/server.py` - Validation error handling
- `src/nodetool/api/workspace.py` - Consistent error messages
- `src/nodetool/cli.py` - Enhanced CLI error messages

**Date**: 2026-02-24
