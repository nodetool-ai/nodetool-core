# Async Fixture Cleanup with Threaded Event Loops

**Insight**: pytest-asyncio's async finalizer can hang if tests create nested/threaded event loops that aren't properly cleaned up before teardown.

**Impact**: Tests that use `ThreadedJobExecution` or similar threaded async patterns will timeout during fixture teardown when run in parallel.

**Solution**: Ensure `JobExecutionManager.shutdown()` is called in cleanup fixtures, and consider running job execution tests sequentially with `-n0`.

**Date**: 2026-01-12
