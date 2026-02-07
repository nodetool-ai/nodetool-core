# Silent Exception Handlers in chat_cli.py

**Problem**: Empty `except Exception: pass` blocks in `src/nodetool/chat/chat_cli.py` silently swallow errors, making debugging difficult and hiding potential issues.

**Solution**: Replaced silent `pass` with `log.debug()` calls that log the exception with context:
- Lines 305-310: Token usage retrieval now logs debug message if attribute access fails
- Lines 327-332: History file reading now logs debug message if file cannot be read
- Lines 376-383: History file writing now logs debug message if write fails

**Why**: Silent exception handling makes it nearly impossible to diagnose issues when they occur. The debug logging preserves the non-disruptive behavior (defaults are still used) while providing visibility into what went wrong for debugging purposes.

**Files**:
- `src/nodetool/chat/chat_cli.py`

**Date**: 2026-02-07
