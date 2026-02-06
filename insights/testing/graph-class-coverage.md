# Graph Class Test Coverage

**Insight**: Added comprehensive test coverage for `src/nodetool/workflows/graph.py` module (447 lines), which is critical for workflow management and execution but was previously untested.

**Rationale**: The Graph class is central to the workflow system, handling node and edge management, topological sorting (critical for execution order), type validation, and schema generation. Having tests ensures the workflow execution engine works correctly.

**Coverage Added**:
- Graph creation and initialization (empty and with nodes/edges)
- Node lookup functionality (`find_node`, `find_edges`)
- Topological sorting with various scenarios:
  - Simple linear chains
  - Parallel independent nodes
  - Parent ID filtering (for subgraphs)
  - Group node handling
  - Empty graphs
- Edge type validation
- Streaming upstream detection
- `Graph.from_dict()` factory method with error handling

**Test Count**: 19 new tests covering all public APIs and edge cases

**Files**: `tests/workflows/test_graph.py`

**Date**: 2026-02-06
