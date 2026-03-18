## 2025-02-12 - Video Export Optimization and Correctness
**Learning:** Upfront list comprehension for processing large sequences (like video frames) consumes O(N) memory and can lead to OOM. Lazy processing inside the writing loop reduces memory usage to O(1). Additionally, assuming input data types (e.g. float vs uint8) without checking can lead to critical bugs like integer overflow when scaling `uint8` arrays by 255.
**Action:** Always prefer lazy iteration/generators for large data processing pipelines. Explicitly check `numpy.dtype` before performing arithmetic scaling to ensure correctness and avoid unnecessary operations.

## 2025-02-13 - StateManager DB Batch Saves Optimization
**Learning:** The state manager queue-based writer pattern used a simple for-loop iterating over the queued states to invoke a single DB record update using `.save()`. This lead to N+1 operation performance penalty. Batch inserting a `list[DBModel]` with a single driver call using a method like `cursor.executemany` eliminates this latency issue and increases database bandwidth efficiency.
**Action:** Always implement a dedicated `save_many` method in adapters instead of individual updates when working with bulk DB data, replacing `for state in list: state.save()` with `DBModel.save_many(list)`.
