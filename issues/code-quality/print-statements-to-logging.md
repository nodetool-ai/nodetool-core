# Print Statement Cleanup in Provider Modules

**Problem**: Multiple print() statements were used for warnings in production code instead of the logging framework. This creates noise in production environments and bypasses log level filtering.

**Solution**: Replaced all print() statements with appropriate log.warning() or log.debug() calls in:

1. `src/nodetool/providers/openai_prediction.py`:
   - Added `import logging` and `log = logging.getLogger(__name__)`
   - Converted ~20 print() statements to log.warning() for pricing warnings
   - Removed duplicate print statements that were redundant with log.warning()

2. `src/nodetool/providers/openai_provider.py`:
   - Removed 4 redundant print() statements that duplicated log.warning() calls

3. `src/nodetool/providers/openrouter_provider.py`:
   - Removed 1 redundant print() statement that duplicated log.warning() call

**Why**: Using the logging framework instead of print() provides:
- Proper log level filtering (DEBUG, INFO, WARNING, ERROR)
- Structured output with timestamps and levels
- Configurable handlers (file, console, remote)
- Better production debugging capabilities

**Files Modified**:
- `src/nodetool/providers/openai_prediction.py`
- `src/nodetool/providers/openai_provider.py`
- `src/nodetool/providers/openrouter_provider.py`

**Date**: 2026-01-20
