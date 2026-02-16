# time.sleep() in Sync vs Async Code

**Insight**: `time.sleep()` is appropriate in synchronous functions, but should be replaced with `asyncio.sleep()` in async functions to avoid blocking the event loop.

**Rationale**:
- Automated tools may flag `time.sleep()` as problematic, but context matters
- In sync functions, `time.sleep()` is the correct choice
- In async functions, `asyncio.sleep()` prevents blocking the event loop
- Always check if the containing function is `async def` before flagging

**Example**:
```python
# Correct - sync function uses time.sleep()
def _check_health(self, ssh, results):
    time.sleep(2)  # OK - this is a sync function
    # ...

# Incorrect - async function should use asyncio.sleep()
async def _check_health_async(self, ssh, results):
    time.sleep(2)  # BAD - blocks the event loop!
    # ...

# Correct - async function uses asyncio.sleep()
async def _check_health_async(self, ssh, results):
    await asyncio.sleep(2)  # OK - yields to event loop
    # ...
```

**Impact**: Prevents false positives when scanning for blocking I/O issues.

**Files**: `src/nodetool/deploy/self_hosted.py`, `src/nodetool/proxy/docker_manager.py`

**Date**: 2026-02-16
