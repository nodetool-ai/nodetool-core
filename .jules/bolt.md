## 2025-02-12 - Video Export Optimization and Correctness
**Learning:** Upfront list comprehension for processing large sequences (like video frames) consumes O(N) memory and can lead to OOM. Lazy processing inside the writing loop reduces memory usage to O(1). Additionally, assuming input data types (e.g. float vs uint8) without checking can lead to critical bugs like integer overflow when scaling `uint8` arrays by 255.
**Action:** Always prefer lazy iteration/generators for large data processing pipelines. Explicitly check `numpy.dtype` before performing arithmetic scaling to ensure correctness and avoid unnecessary operations.

## 2025-02-13 - Avoid redundant system calls in async file operations
**Learning:** In async file operations, using `asyncio.to_thread(os.path.isdir)` to check if a path is a directory requires a new thread spawn and a redundant system call. When a `stat` system call (e.g., via `aiofiles.os.stat()`) has already been performed, it returns a `stat_result` which already contains this information.
**Action:** Extract the directory status via `stat.S_ISDIR(st.st_mode)` from the existing `stat_result` object rather than spawning new threads for redundant `os.path.isdir()` checks.
