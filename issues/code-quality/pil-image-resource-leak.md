# PIL.Image.open Resource Leak Fix

**Problem**: Several functions in provider code and media utilities opened PIL Images without using context managers, potentially causing resource leaks when processing images.

**Solution**: Wrapped `PIL.Image.open()` calls in `with` statements to ensure proper resource cleanup.

**Files Modified**:
- `src/nodetool/media/common/media_utils.py` (2026-01-14)
- `src/nodetool/providers/gemini_provider.py` (2026-02-10) - Two instances fixed
- `src/nodetool/providers/huggingface_provider.py` (2026-02-10) - One instance fixed
- `src/nodetool/workflows/processing_offload.py` (2026-02-18) - `_open_image_as_rgb()` function fixed

**Impact**:
- Ensures PIL Image resources are properly released after operations complete
- Prevents potential memory leaks when processing large images
- All existing tests continue to pass

**Date**: 2026-01-14, updated 2026-02-10, updated 2026-02-18
