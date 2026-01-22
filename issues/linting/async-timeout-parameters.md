# ASYNC109: Async Timeout Parameter Modernization

**Problem**: 15 async functions across the codebase use a `timeout` parameter with `asyncio.wait_for()`, triggering ruff's ASYNC109 linting violation which suggests using `asyncio.timeout()` context manager instead.

**Locations**:
- `src/nodetool/api/server.py:186` - `check_ollama_availability`
- `src/nodetool/concurrency/async_lock.py:37` - `AsyncLock.acquire`
- `src/nodetool/concurrency/async_priority_queue.py:118` - `AsyncPriorityQueue.get`
- `src/nodetool/concurrency/async_task_group.py:197` - `AsyncTaskGroup.run_until_first`
- `src/nodetool/concurrency/async_utils.py:57` - `AsyncSemaphore.acquire`
- `src/nodetool/concurrency/rate_limit.py:81,128,247` - `AsyncTokenBucket.acquire`, `_wait_for_tokens_with_timeout`, `AsyncRateLimiter.acquire`
- `src/nodetool/integrations/huggingface/async_downloader.py:204,284` - `download_file`, `stream_download`
- `src/nodetool/migrations/runner.py:292` - `_acquire_lock`
- `src/nodetool/models/sqlite_adapter.py:359` - `_execute_with_timeout`
- `src/nodetool/providers/openai_provider.py:1853,1868,2044` - Video generation methods
- `src/nodetool/workflows/state_manager.py:141` - `StateManager.stop`

**Why Not Fixed**: These functions have different timeout semantics:
- Some return `False` on timeout (e.g., `AsyncLock.acquire`)
- Some raise `TimeoutError` on timeout (e.g., `AsyncPriorityQueue.get`)
- Using `asyncio.timeout()` always raises `TimeoutError`, changing the API semantics

Changing these would be a breaking change for the public API. Consider for a future major version bump.

**Date**: 2026-01-22
