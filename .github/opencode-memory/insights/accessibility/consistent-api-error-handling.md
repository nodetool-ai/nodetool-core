# Consistent API Error Handling

**Insight**: API error handling should follow a consistent pattern: detailed logging server-side + user-friendly messages client-side.

**Rationale**: Raw exception strings expose internal implementation details and are not helpful to end users. Users need clear, actionable error messages that explain what went wrong and what they can do about it. Detailed technical information should be logged server-side for debugging.

**Best Practices**:
1. **Log with context**: Include user IDs, request details, and full exception information
2. **User-facing messages**: Explain what went wrong and suggest next steps in plain language
3. **Consistent format**: Use similar structure across all error responses
4. **Appropriate detail**: More detail in development, sanitized in production
5. **Proper status codes**: Use appropriate HTTP status codes (400 for client errors, 500 for server errors)

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

**Common Patterns**:
- **Resource not found**: "Resource '{name}' not found or could not be accessed."
- **Validation errors**: "Invalid {field} format. Please check the input and try again."
- **Operation failures**: "Failed to {action}. Please {suggestion} and try again."
- **Permission errors**: "Access denied. {explanation}."

**Impact**: Improved user experience, better debugging capability, enhanced security by not exposing internals, and consistent API behavior.

**Files**:
- `src/nodetool/api/users.py` - User management error handling
- `src/nodetool/api/file.py` - File operation error handling
- `src/nodetool/api/openai.py` - OpenAI-compatible endpoint error handling
- `src/nodetool/api/workflow.py` - Workflow operation error handling
- `src/nodetool/api/collection.py` - Collection operation error handling

**Date**: 2026-03-17
