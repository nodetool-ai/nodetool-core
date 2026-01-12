# Code Quality Issues - 2026-01-12

## Type Safety Issues

### chat_sse_runner.py - Dict unpacking to Pydantic models

**Problem**: The `_create_openai_error_chunk` method was passing a raw dict to `ChatCompletionChunk(**chunk_data)`, which caused type checker warnings about mismatched types.

**Solution**: Changed to pass typed objects directly to the constructor instead of unpacking a dict.

**Files**: `src/nodetool/chat/chat_sse_runner.py:401-424`

## Unused Imports

### build.py - Unused Optional import

**Problem**: `from typing import Optional` was imported but not used.

**Solution**: Removed the unused import.

**Files**: `build.py:40`

## Async/Await Patterns

### asyncio.run() usage

**Observation**: Many files use `asyncio.run()` in module-level code or CLI entry points. This is generally acceptable for CLI tools but should be avoided in library code.

**Files with patterns**:
- `src/nodetool/cli.py` - Multiple uses in CLI command handlers (acceptable)
- `src/nodetool/workflows/base_node.py:756` - Used in `fetch_all_models()`
- `src/nodetool/storage/supabase_storage.py:117` - Used in upload method

**Recommendation**: For library code that needs to run async operations from sync contexts, consider using `ThreadedEventLoop` or `asyncio.run()` should be documented clearly.

## time.sleep() in Async Code

**Observation**: Found several `time.sleep()` calls in async contexts. Most are acceptable:
- In threads (e.g., `api/server.py:627` - in a background thread)
- In sync functions (e.g., `proxy/docker_manager.py:277` - in `_wait_for_container_start`)

**No issues found** where `time.sleep()` blocks the main async event loop inappropriately.

## Exception Handling Patterns

**Observation**: The codebase uses `except Exception as e:` patterns extensively with variable `e` that is sometimes not used.

**Recommendation**: Consider logging exceptions or using `logging.debug()` for debugging purposes when catching generic exceptions.

## TODOs

**Count**: 22 TODO comments found across the codebase.

**Notable ones**:
- `workflows/actor.py` - Multiple tracking TODOs
- `metadata/typecheck.py:148` - Type checking for comfy types
- `deploy/runpod.py` - API integration TODOs

**Recommendation**: Review and prioritize TODOs or convert to issue tracker items.

## Lint Issues in Examples

**Files with unused imports**:
- `examples/chromadb_research_agent.py` - Unused imports
- `examples/graph_planner_integration.py` - Unused import

**Recommendation**: Clean up example files for consistency.
