## 2025-02-12 - Video Export Optimization and Correctness
**Learning:** Upfront list comprehension for processing large sequences (like video frames) consumes O(N) memory and can lead to OOM. Lazy processing inside the writing loop reduces memory usage to O(1). Additionally, assuming input data types (e.g. float vs uint8) without checking can lead to critical bugs like integer overflow when scaling `uint8` arrays by 255.
**Action:** Always prefer lazy iteration/generators for large data processing pipelines. Explicitly check `numpy.dtype` before performing arithmetic scaling to ensure correctness and avoid unnecessary operations.

## 2024-05-18 - Optimize File Aggregation in get_files
**Learning:** Found an $O(n^2)$ performance bottleneck in recursive list aggregation and string concatenation in `nodetool.io.get_files`. The recursive list was aggregated using `files += ...` which creates intermediate lists and scales poorly. The string content was joined iteratively with `content += ...` which has to copy the entire string every time.
**Action:** Always use `list.extend()` instead of `+=` for extending lists, and use `list.append()` inside loops followed by `''.join()` for efficient $O(N)$ string building.
