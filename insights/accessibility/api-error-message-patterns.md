# API Error Message Patterns

**Insight**: When designing API error messages for developer accessibility, include three key elements: context (what resource), specifics (the ID/value), and guidance (what to do next).

**Rationale**: Developers consuming APIs need immediate clarity on:
1. What went wrong (the error itself)
2. Which resource was affected (ID or identifier)
3. How to fix it (actionable guidance or at least enough detail to debug)

Generic error messages like "Resource not found" force developers to add extensive logging or debug through the API client code to understand which request failed.

**Example Pattern**:
```python
# Bad - Generic error
raise HTTPException(status_code=404, detail="Workflow not found")

# Good - Specific with context
raise HTTPException(
    status_code=404,
    detail=f"Workflow with id '{id}' not found"
)

# Better - Specific with context and guidance
raise HTTPException(
    status_code=404,
    detail=f"Workflow '{id}' has no HTML app configured. "
           f"Use the workflow editor to create and save an HTML app interface."
)
```

**Validation Errors Pattern**:
```python
# Show actual vs expected values for validation failures
raise HTTPException(
    status_code=400,
    detail=f"Search query must be at least {MIN_LENGTH} characters long "
           f"(got {len(query.strip())} character(s)). Please provide a more specific search term."
)
```

**Impact**: Reduces debugging time for API consumers, improves developer experience, decreases support burden.

**Files**:
- `src/nodetool/api/workflow.py`
- `src/nodetool/api/asset.py`
- `src/nodetool/api/job.py`
- `src/nodetool/api/collection.py`

**Date**: 2026-02-10
