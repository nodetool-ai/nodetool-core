# API Error Message Clarity

**Problem**: Some API error messages lack actionable guidance. For example, validation errors return raw Pydantic errors without field context.

**Solution**: Wrap validation errors with user-friendly messages that include field names and expected types. Add suggested fixes in error details.

**Files**:
- `src/nodetool/api/utils.py`
- `src/nodetool/api/server.py`

**Date**: 2026-01-20
