## 2025-02-12 - Video Export Optimization and Correctness
**Learning:** Upfront list comprehension for processing large sequences (like video frames) consumes O(N) memory and can lead to OOM. Lazy processing inside the writing loop reduces memory usage to O(1). Additionally, assuming input data types (e.g. float vs uint8) without checking can lead to critical bugs like integer overflow when scaling `uint8` arrays by 255.
**Action:** Always prefer lazy iteration/generators for large data processing pipelines. Explicitly check `numpy.dtype` before performing arithmetic scaling to ensure correctness and avoid unnecessary operations.

## 2025-02-28 - Fast O(1) Edge lookups in Graph
**Learning:** Graph structures often process many edge queries sequentially or in loops (`find_edges`). Relying on list comprehensions across all edges gives O(E) complexity which becomes O(N*E) during iterative graph traversals. Also, directly exposing state array variables (`graph.edges = new_edges`) requires external components to remember cache invalidation, leading to brittle code.
**Action:** When creating high-performance lookup capabilities on existing data classes, use cached mappings at initialization (e.g., `_outgoing_edges_cache`) and encapsulate mutation with setter methods (e.g., `update_edges(new_edges)`) to force cache rebuilding securely, achieving O(1) performance reliably.
