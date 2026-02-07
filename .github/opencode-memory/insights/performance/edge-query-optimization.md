# Edge Query Optimization in Workflow Validation

**Insight**: The workflow validation code had an O(n*m) algorithm where it iterated over all nodes and for each node scanned all edges to find matching input edges.

**File**: `src/nodetool/workflows/workflow_runner.py:1086-1088`

**Original Code**:
```python
for node in graph.nodes:
    input_edges = [edge for edge in graph.edges if edge.target == node.id]
```

**Solution**: Pre-build a dictionary mapping target node IDs to their input edges, reducing complexity from O(n*m) to O(n+m):
```python
# Build a lookup map for input edges by target node (O(m) instead of O(n*m))
input_edges_by_target: dict[str, list[Edge]] = {}
for edge in graph.edges:
    if edge.target not in input_edges_by_target:
        input_edges_by_target[edge.target] = []
    input_edges_by_target[edge.target].append(edge)

# First validate node inputs
for node in graph.nodes:
    input_edges = input_edges_by_target.get(node.id, [])
```

**Impact**: For graphs with n nodes and m edges, this reduces algorithmic complexity from O(n*m) to O(n+m). For a workflow with 100 nodes and 200 edges, this reduces operations from ~20,000 to ~300.

**Date**: 2026-02-06
