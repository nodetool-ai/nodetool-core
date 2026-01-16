# O(n²) Topological Sort Algorithm

**Problem**: The `topological_sort()` method in `Graph` class used `edges.remove()` inside a loop with `edges[:]` copy, creating O(n²) complexity.

**File**: `src/nodetool/workflows/graph.py:271-285`

**Original Code**:
```python
for edge in edges[:]:  # O(n) copy
    if edge.source == n:
        edges.remove(edge)  # O(n) remove - nested in loop = O(n²)
```

**Solution**: Use a dictionary to track outgoing edges by source node, reducing lookups to O(1):
```python
outgoing_edges: dict[str, list[Edge]] = {}
for edge in edges:
    if edge.source not in outgoing_edges:
        outgoing_edges[edge.source] = []
    outgoing_edges[edge.source].append(edge)

# Then iterate with O(1) lookups
for edge in outgoing_edges.get(n, []):
```

**Impact**: Improved algorithm from O(n²) to O(n) for edge processing, significant for graphs with many edges.

**Date**: 2026-01-16
