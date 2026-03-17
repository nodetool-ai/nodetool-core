# API Exceptions Expose Internal Details

**Problem**: Multiple API endpoints used raw exception strings (`str(e)`) in HTTPException responses, exposing internal implementation details, stack traces, and potentially sensitive information to end users.

**Solution**: Replace raw exception strings with user-friendly, actionable messages while logging detailed errors server-side for debugging:
- Use specific error messages that explain what went wrong and suggest next steps
- Log full exception details server-side with context (user IDs, request details, stack traces)
- Maintain consistent error response format across all endpoints
- In production, sanitize error details while keeping actionable information

**Example**:
```python
# Before:
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e)) from e

# After:
except Exception as e:
    log.error("Error listing files in %s: %s", path, e)
    raise HTTPException(
        status_code=500,
        detail="Failed to list files. Please check the path and try again.",
    ) from e
```

**Files**:
- `src/nodetool/api/users.py:154,194,227` - User management error responses
- `src/nodetool/api/file.py:145,165,276,302` - File operation error responses
- `src/nodetool/api/openai.py:79,100` - OpenAI-compatible endpoint error responses
- `src/nodetool/api/workflow.py:192,216,503,656,1162,1209` - Workflow operation error responses
- `src/nodetool/api/collection.py:200,235` - Collection operation error responses

**Date**: 2026-03-17
