### Error Type Attribute Bug

**Date Discovered**: 2026-01-10

**Context**: The `Error` class in `src/nodetool/workflows/types.py` has a `message` attribute, but several files were incorrectly using `.error`

**Solution**: Changed `msg.error` to `msg.message` in:
  - `src/nodetool/api/workflow.py:618`
  - `src/nodetool/api/mcp_server.py:349,429`
  - `src/nodetool/integrations/websocket/unified_websocket_runner.py:207`
  - `src/nodetool/workflows/workflow_node.py:63`

**Prevention**: Use consistent attribute names; consider using TypedDicts or type guards
