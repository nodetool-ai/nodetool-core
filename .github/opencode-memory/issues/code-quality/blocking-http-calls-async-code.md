### Blocking HTTP Calls in Async Code

**Date Discovered**: 2026-01-12

**Context**: Several files use synchronous `requests` library for HTTP calls in modules that otherwise use async patterns. This blocks the event loop during network I/O.

**Solution**: Convert blocking `requests.get/post` calls to async `httpx` calls:
- `src/nodetool/providers/huggingface_provider.py:95-157` - Converted `get_remote_context_window()` from sync `requests.get` to async `httpx.client.AsyncClient.get`

**Related Files**:
- `src/nodetool/providers/huggingface_provider.py`
- `src/nodetool/providers/comfy_api.py` (still uses requests, could benefit from similar fix)
- `src/nodetool/packages/registry.py` (still uses requests)
- `src/nodetool/deploy/runpod_api.py` (still uses requests)

**Prevention**:
- Use `httpx` for all HTTP operations in async modules
- Run `ruff check` to verify no blocking patterns
- Consider wrapping sync I/O with `asyncio.to_thread()` if async conversion is not feasible
