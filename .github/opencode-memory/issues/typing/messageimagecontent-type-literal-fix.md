# MessageImageContent Type Literal Fix

**Problem**: The `MessageImageContent` class in `src/nodetool/metadata/types.py` has a type literal `type: Literal["image_url"] = "image_url"`, but the provider handler was incorrectly passing `type="image"` when constructing instances.

**Solution**: Changed `MessageImageContent(type="image", ...)` to `MessageImageContent(type="image_url", ...)` in `src/nodetool/worker/provider_handler.py`.

**Why**: The type mismatch caused a basedpyright error `invalid-argument-type` which prevented the typecheck from passing. The correct literal value is `"image_url"` as defined in the class definition.

**Files**:
- `src/nodetool/worker/provider_handler.py`
- `src/nodetool/metadata/types.py`

**Date**: 2026-03-18
