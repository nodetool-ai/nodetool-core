# Project Context

## Recent Changes

- **2026-01-12**: Initialized structured OpenCode memory layout.
- **2026-01-12**: Fixed type safety in `chat_sse_runner.py` - Changed `_create_openai_error_chunk` to use typed `Choice` and `ChoiceDelta` objects instead of raw dict unpacking.
- **2026-01-12**: Fixed type safety in `threaded_event_loop.py` - Added `assert` for loop non-null check and `type: ignore` for dynamic future.task attribute.
- **2026-01-12**: Removed unused import `Optional` from `build.py`.
