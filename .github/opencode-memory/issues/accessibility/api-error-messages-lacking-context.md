# API Error Messages Lacking Context

**Problem**: Many API error messages in nodetool-core were generic (e.g., "Workflow not found", "Asset not found") without providing the specific ID or actionable guidance for users. This makes debugging difficult for developers integrating with the API.

**Solution**: Enhanced error messages throughout the codebase to:
- Include the specific resource ID that was not found (e.g., `f"Workflow with id '{id}' not found"`)
- Provide actionable guidance when applicable (e.g., "Use the workflow editor to create and save an HTML app interface")
- Show the actual value received in validation errors (e.g., search query length showing actual vs expected)

**Files Modified**:
- `src/nodetool/api/workflow.py` - Enhanced workflow and version not found errors, HTML app errors
- `src/nodetool/api/asset.py` - Enhanced asset not found errors, video content type validation, search query validation
- `src/nodetool/api/job.py` - Enhanced job not found errors, trigger workflow validation
- `src/nodetool/api/collection.py` - Enhanced collection and workflow not found errors, input node validation

**Example Improvements**:
- Before: `raise HTTPException(status_code=404, detail="Workflow not found")`
- After: `raise HTTPException(status_code=404, detail=f"Workflow with id '{id}' not found")`

- Before: `raise HTTPException(status_code=400, detail="Search query must be at least 2 characters long")`
- After: `raise HTTPException(status_code=400, detail=f"Search query must be at least {MIN_SEARCH_QUERY_LENGTH} characters long (got {len(query.strip())} character(s)). Please provide a more specific search term.")`

**Date**: 2026-02-10
