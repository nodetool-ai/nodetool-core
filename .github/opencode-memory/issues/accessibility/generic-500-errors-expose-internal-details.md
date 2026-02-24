# Generic 500 Errors Expose Internal Details

**Problem**: API endpoints used raw exception strings in 500 error responses, potentially exposing internal implementation details and providing poor user experience.

**Solution**: Replace raw exception strings with user-friendly messages while logging detailed errors server-side:

**Example**:
```python
# Before:
except Exception as e:
    log.error(f"Error listing workspaces: {e}")
    raise HTTPException(status_code=500, detail=str(e)) from e

# After:
except Exception as e:
    log.error("Error listing workspaces for user %s: %s", user, e)
    raise HTTPException(
        status_code=500,
        detail="Failed to list workspaces. Please try again later.",
    ) from e
```

**Files**:
- `src/nodetool/api/workspace.py:108,161,240,269,396,489,540`

**Date**: 2026-02-24
