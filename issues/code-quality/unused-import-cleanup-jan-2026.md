# Unused Import Cleanup

**Problem**: Four unused imports were causing F401 linting violations across the codebase.

**Solution**: Removed unused imports from three files:

1. `src/nodetool/concurrency/async_task_group.py`:
   - Removed `field` from `from dataclasses import dataclass, field` (only `dataclass` was used)

2. `src/nodetool/integrations/vectorstores/chroma/provider_embedding_function.py`:
   - Removed `List` from `from typing import List, cast` (only `cast` was used)

3. `src/nodetool/integrations/websocket/unified_websocket_runner.py`:
   - Removed `is_tracing_enabled` and `trace_workflow` from tracing imports (only `trace_websocket_message` was used)

**Why**: Unused imports add noise to the codebase, increase module load time marginally, and can mislead developers about actual dependencies.

**Impact**: 
- All F401 violations from ruff check are now resolved
- Code is cleaner and more maintainable

**Files Modified**:
- `src/nodetool/concurrency/async_task_group.py`
- `src/nodetool/integrations/vectorstores/chroma/provider_embedding_function.py`
- `src/nodetool/integrations/websocket/unified_websocket_runner.py`

**Date**: 2026-01-19
