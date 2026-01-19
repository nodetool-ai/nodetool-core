# Blocking Calls in Async Code - Analysis

**Insight**: Found multiple `time.sleep()` calls in functions that are called from async contexts. However, these are typically called via `asyncio.to_thread()`, which means they block threads rather than the event loop.

**Rationale**: While `asyncio.sleep()` is preferred for async code, using `time.sleep()` in functions that are explicitly run in threads via `asyncio.to_thread()` is acceptable and doesn't block the main event loop.

**Examples**:
- `comfy_api.py:97, 158` - Called via `asyncio.to_thread()` from `comfy_local_provider.py`
- `comfy_runpod_provider.py:77` - Called via `self._to_thread()` 
- `docker_manager.py:277, 283, 287` - In sync function called from async code
- `server.py:646, 677` - In daemon threads, not blocking event loop

**Impact**: No immediate performance issue, but consider making these functions truly async for consistency.

**Date**: 2026-01-19
