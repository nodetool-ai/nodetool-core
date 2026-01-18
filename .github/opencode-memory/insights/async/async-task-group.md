# Async Task Group Management

**Insight**: Using `AsyncTaskGroup` for managing related async tasks provides better error handling and resource cleanup compared to raw `asyncio.gather()`.

**Rationale**: Raw `asyncio.gather()` doesn't provide:
- Individual task tracking and identification
- Automatic cancellation on failure
- Result collection with exception handling
- Context manager support for cleanup

`AsyncTaskGroup` addresses all these gaps with a clean, composable API.

**Example**:
```python
group = AsyncTaskGroup()

async def fetch_data(url: str) -> dict[str, Any]:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

group.spawn("api1", fetch_data("https://api1.example.com/data"))
group.spawn("api2", fetch_data("https://api2.example.com/data"))
group.spawn("api3", fetch_data("https://api3.example.com/data"))

results = await group.run(raise_on_error=False)

for result in results:
    if result.success:
        print(f"{result.task_id}: {result.result}")
    else:
        print(f"{result.task_id} failed: {result.exception}")
```

**Impact**:
- Reduced boilerplate for task management
- Built-in cancellation prevents resource leaks
- Clear task identification simplifies debugging
- Context manager ensures cleanup even on exceptions

**Files**: `src/nodetool/concurrency/async_task_group.py`

**Date**: 2026-01-18
