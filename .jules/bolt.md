## 2025-02-12 - Video Export Optimization and Correctness
**Learning:** Upfront list comprehension for processing large sequences (like video frames) consumes O(N) memory and can lead to OOM. Lazy processing inside the writing loop reduces memory usage to O(1). Additionally, assuming input data types (e.g. float vs uint8) without checking can lead to critical bugs like integer overflow when scaling `uint8` arrays by 255.
**Action:** Always prefer lazy iteration/generators for large data processing pipelines. Explicitly check `numpy.dtype` before performing arithmetic scaling to ensure correctness and avoid unnecessary operations.

## 2025-02-13 - Async Event Loop Blocking via os.scandir Iteration
**Learning:** `asyncio.to_thread(os.scandir, path)` successfully offloads the iterator creation to a background thread, but returning the generator and iterating over it (e.g., `for entry in entries: entry.stat()`) back in the main async task runs synchronously. Since `entry.stat()` triggers blocking disk I/O, this stalls the event loop, destroying server concurrency, and leaves the iterator unclosed risking `ResourceWarning`s.
**Action:** Always encapsulate the entire `os.scandir` iteration, context management (`with os.scandir(...) as entries:`), and data extraction inside a synchronous helper function, then execute that helper fully within `asyncio.to_thread`.
