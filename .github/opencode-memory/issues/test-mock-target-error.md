### Test Mock Target Error

**Date Discovered**: 2026-01-10

**Context**: Test `test_graph_result_allows_asset_mode` patched `run_graph` but function called `run_graph_async`

**Solution**: Changed patch target from `nodetool.dsl.graph.run_graph` to `nodetool.dsl.graph.run_graph_async`

**Related Files**: `tests/dsl/test_graph_process.py`

**Prevention**: Verify mock targets match actual function calls
