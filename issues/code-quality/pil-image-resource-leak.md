# PIL.Image.open Resource Leak Fix

**Problem**: The `create_image_thumbnail` function in `media_utils.py` opened a PIL Image without using a context manager, potentially causing resource leaks for large images.

**Solution**: Wrapped `PIL.Image.open()` call in a `with` statement to ensure proper resource cleanup.

**Files Modified**:
- `src/nodetool/media/common/media_utils.py`

**Impact**:
- Ensures PIL Image resources are properly released after thumbnail generation
- Prevents potential memory leaks when processing large images
- All existing tests continue to pass

**Date**: 2026-01-14
