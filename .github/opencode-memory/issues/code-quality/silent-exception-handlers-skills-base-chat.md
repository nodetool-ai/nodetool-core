# Silent Exception Handlers in skills.py and base_chat_runner.py

**Problem**: Silent exception handlers that catch all exceptions and return None or continue without logging, making debugging difficult and hiding potential issues.

**Locations**:
- `src/nodetool/api/skills.py:84-85` - `except Exception: return None` when reading skill files
- `src/nodetool/chat/base_chat_runner.py:206-208` - `except Exception:` that drops tools on any error

**Solution**:
1. Added logging import to `api/skills.py`
2. Changed `except Exception: return None` to `except OSError as e:` with `log.debug()` for file I/O errors
3. Changed broad `except Exception:` to specific `except (TypeError, AttributeError) as e:` with debug logging in tools normalization

**Why**: Silent exception handling makes it nearly impossible to diagnose issues when they occur. The debug logging preserves the non-disruptive behavior (defaults are still used) while providing visibility into what went wrong for debugging purposes. Catching specific exception types also prevents accidentally hiding unexpected errors.

**Files**:
- `src/nodetool/api/skills.py`
- `src/nodetool/chat/base_chat_runner.py`

**Date**: 2026-02-20
