## 2025-02-12 - Video Export Optimization and Correctness
**Learning:** Upfront list comprehension for processing large sequences (like video frames) consumes O(N) memory and can lead to OOM. Lazy processing inside the writing loop reduces memory usage to O(1). Additionally, assuming input data types (e.g. float vs uint8) without checking can lead to critical bugs like integer overflow when scaling `uint8` arrays by 255.
**Action:** Always prefer lazy iteration/generators for large data processing pipelines. Explicitly check `numpy.dtype` before performing arithmetic scaling to ensure correctness and avoid unnecessary operations.

## 2025-02-13 - String Concatenation Optimization in get_content
**Learning:** In heavily used file operations like `get_content`, performing string concatenation via `+=` inside loops leads to $O(N^2)$ time complexity due to constant reallocation and copying in Python's immutable strings.
**Action:** Use list appends (`content_parts.append(...)`) and `"".join(content_parts)` for $O(N)$ string building performance. Always identify and remove nested string concatenations in file/content aggregating utilities.
