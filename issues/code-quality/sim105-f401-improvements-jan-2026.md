# Code Quality Improvements - January 2026

**Problem**: Multiple code quality issues found during routine scanning:
- Unused imports (F401)
- try-except-pass anti-patterns (SIM105)
- Line length issues in examples
- f-string without placeholders

**Solution**: Fixed 4 unused imports and 28 try-except-pass patterns across 15 files:

**Files Modified**:
- `src/nodetool/api/collection.py` - Removed unused `typing.List`
- `src/nodetool/api/file.py` - Removed unused `typing.Any`
- `src/nodetool/api/font.py` - Removed unused `typing.Any`
- `src/nodetool/api/job.py` - Removed unused `asyncio` and `typing.List`
- `src/nodetool/cli.py` - Added `contextlib.suppress` for PackageNotFoundError
- `src/nodetool/integrations/huggingface/hf_websocket.py` - 3 suppress() calls
- `src/nodetool/media/video/video_utils.py` - Added OSError suppression
- `src/nodetool/migrations/versions/20260104_000001_add_autosave_fields_v2.py` - 3 suppress() calls
- `src/nodetool/models/run_state_writer.py` - 6 suppress() calls
- `src/nodetool/runtime/db_sqlite.py` - 3 suppress() calls
- `src/nodetool/workflows/event_logger.py` - Replaced CancelledError handler
- `src/nodetool/workflows/inbox.py` - Added Exception suppression
- `src/nodetool/workflows/state_manager.py` - Replaced CancelledError handler
- `src/nodetool/workflows/threaded_event_loop.py` - Added InvalidStateError suppression
- `src/nodetool/workflows/trigger_workflow_manager.py` - Replaced CancelledError handler
- `src/nodetool/metadata/types.py` - 2 suppress() calls for JSON parsing
- `src/nodetool/workflows/workflow_runner.py` - Fixed f-string without placeholders

**Impact**:
- 4 unused imports removed
- 28 try-except-pass patterns replaced with contextlib Cleaner.suppress()
-, more idiomatic Python code
- All linting passes
- All type checking passes (with expected warnings)
- Tests pass

**Date**: 2026-01-22
