## 2025-02-12 - Video Export Optimization and Correctness
**Learning:** Upfront list comprehension for processing large sequences (like video frames) consumes O(N) memory and can lead to OOM. Lazy processing inside the writing loop reduces memory usage to O(1). Additionally, assuming input data types (e.g. float vs uint8) without checking can lead to critical bugs like integer overflow when scaling `uint8` arrays by 255.
**Action:** Always prefer lazy iteration/generators for large data processing pipelines. Explicitly check `numpy.dtype` before performing arithmetic scaling to ensure correctness and avoid unnecessary operations.
## 2025-02-13 - ConditionBuilder OR Chaining vs in_list()
**Learning:** Using `for condition in conditions: combined = combined.or_(condition)` to match multiple IDs in `ConditionBuilder` creates heavily nested O(N) condition trees, which can crash recursion depth or exceed database plan limits.
**Action:** Always use `Field("id").in_list(id_list)` when matching against a list of variables in Nodetool DB adapter models to ensure O(1) query complexity and native DB IN-clause optimizations.
