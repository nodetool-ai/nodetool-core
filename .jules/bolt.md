## 2025-02-12 - Video Export Optimization and Correctness
**Learning:** Upfront list comprehension for processing large sequences (like video frames) consumes O(N) memory and can lead to OOM. Lazy processing inside the writing loop reduces memory usage to O(1). Additionally, assuming input data types (e.g. float vs uint8) without checking can lead to critical bugs like integer overflow when scaling `uint8` arrays by 255.
**Action:** Always prefer lazy iteration/generators for large data processing pipelines. Explicitly check `numpy.dtype` before performing arithmetic scaling to ensure correctness and avoid unnecessary operations.

## 2025-02-12 - State Manager Batch Operations
**Learning:** Batch operations (`save_many` with executemany/upsert) significantly reduce database overhead compared to looping through individual `save` commands, especially inside parallelized operations like the `StateManager`. Also, when implementing raw database operations in a provider (like PostgreSQL), do not forget explicit `await conn.commit()` calls when `executemany` is used inside a connection context block, otherwise transactions will be discarded.
**Action:** Always seek to replace loop-based database inserts/updates with batch operations where supported, and verify transaction boundaries (`commit()`) for relational database adapters.
