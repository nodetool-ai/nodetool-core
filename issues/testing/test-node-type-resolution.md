# Test Node Type Resolution for tests.workflows Paths

**Problem**: Tests using node types from `tests.workflows.test_graph_module` (like `InNode`, `OutNode`) failed with `ValueError: Invalid node type` because `get_node_class` in `base_node.py` only handled `nodetool.workflows.test_helper` as a special case, not `tests.workflows` paths.

**Solution**: Extended the special case handling in `get_node_class` at `src/nodetool/workflows/base_node.py:2005-2010` to also recognize `tests.workflows` module paths and import them directly instead of prepending `nodetool.nodes.`.

**Files**:
- `src/nodetool/workflows/base_node.py`
- `tests/workflows/test_backpressure.py`
- `tests/workflows/test_graph_module.py`

**Date**: 2026-01-21
