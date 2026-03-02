## 2025-02-12 - Video Export Optimization and Correctness
**Learning:** Upfront list comprehension for processing large sequences (like video frames) consumes O(N) memory and can lead to OOM. Lazy processing inside the writing loop reduces memory usage to O(1). Additionally, assuming input data types (e.g. float vs uint8) without checking can lead to critical bugs like integer overflow when scaling `uint8` arrays by 255.
**Action:** Always prefer lazy iteration/generators for large data processing pipelines. Explicitly check `numpy.dtype` before performing arithmetic scaling to ensure correctness and avoid unnecessary operations.

## 2023-10-27 - Graph Edge Lookup Cache Invalidation
**Learning:** When adding a cache to a Pydantic model (e.g., `_outgoing_edges_cache` for O(1) graph edge lookup based on `edges` field), external modifications to the list (like `graph.edges = new_edges`) can lead to subtle state de-synchronization. Storing the length (`_cached_edges_len = len(self.edges)`) and verifying it during cache lookup provides a robust way to lazily trigger invalidation without breaking standard Pydantic field assignment patterns.
**Action:** Always include a mechanism for invalidation when caching data derived from class properties or mutable fields to avoid stale cache bugs.
