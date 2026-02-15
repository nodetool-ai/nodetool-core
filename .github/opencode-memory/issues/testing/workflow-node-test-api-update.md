# WorkflowNode Tests Out of Sync with API Changes

**Problem**: The WorkflowNode class was refactored to reference workflows by ID instead of embedding JSON directly, but the tests were not updated to match the new API.

**Solution**: 
1. Updated `test_read_workflow` to create a workflow in the database and use async/await
2. Updated `test_process` to use workflow_id instead of workflow_json
3. Fixed the iteration pattern from `handle, value` tuple unpacking to dict iteration

**Why**: The `WorkflowNode` class changed from accepting `workflow_json` parameter to requiring `workflow_id`, and the `gen_process` method now yields dicts instead of tuples.

**Files**: 
- tests/workflows/test_workflow_node.py
- src/nodetool/workflows/workflow_node.py

**Date**: 2026-02-15
