## 2025-02-12 - Video Export Optimization and Correctness
**Learning:** Upfront list comprehension for processing large sequences (like video frames) consumes O(N) memory and can lead to OOM. Lazy processing inside the writing loop reduces memory usage to O(1). Additionally, assuming input data types (e.g. float vs uint8) without checking can lead to critical bugs like integer overflow when scaling `uint8` arrays by 255.
**Action:** Always prefer lazy iteration/generators for large data processing pipelines. Explicitly check `numpy.dtype` before performing arithmetic scaling to ensure correctness and avoid unnecessary operations.

## 2025-02-12 - AudioSegment Concatenation Optimization
**Learning:** In pydub, concatenating many `AudioSegment` objects in a loop using `+=` causes O(N^2) byte-copying performance penalties.
**Action:** Always verify if the audio segments share the same `sample_width`, `frame_rate`, and `channels`. If they do, join their raw bytes directly (`b"".join([a._data for a in audios])`) and initialize a new segment via `first_audio._spawn(raw_data)` to reduce concatenation time from O(N^2) to O(N).

## 2025-04-25 - Optimize file discovery with os.walk
**Learning:** `os.walk` is drastically faster than custom recursive directory traversal using `os.listdir` and `os.path.isdir`. `os.walk` utilizes `os.scandir` internally to avoid the performance overhead of repeated `stat` system calls.
**Action:** When writing recursive file discovery utilities, always use `os.walk` instead of custom recursive functions with `os.listdir` and `os.path.isdir`.
## 2025-04-26 - Asyncio Directory Traversal Bottleneck
**Learning:** Deep recursive directory traversal using individual `aiofiles` calls (like `aiofiles.os.listdir` or `aiofiles.os.path.is_file`) causes massive context-switching overhead in `asyncio` applications by dispatching thousands of tasks to the default thread pool.
**Action:** Optimize deep recursive file system traversals by running a single synchronous C-optimized `os.walk` or `os.scandir` inside a back-referenced `loop.run_in_executor()`. This reduces the async task dispatch count from O(N) down to O(1) while keeping the event loop unblocked.

## 2024-05-18 - [Optimize Hugging Face fast cache with os.scandir]
**Learning:** [In asyncio apps, reading directories with `aiofiles.os.listdir` and then performing individual `aiofiles.os.stat` or `aiofiles.os.path.is_dir` checks for every entry causes massive context-switching overhead because it dispatches thousands of tiny I/O tasks to the default thread pool.]
**Action:** [Use a single `await asyncio.get_running_loop().run_in_executor()` call wrapping a fast synchronous helper utilizing `os.scandir` to retrieve the directory entries and their stats in a single system call sequence.]
## 2025-05-03 - Avoid duplicate stat calls using os.scandir
**Learning:** Functions like `os.listdir` simply return strings (filenames), which forces subsequent code to repeatedly call `os.path.isfile`, `os.path.isdir`, or `os.path.getsize` on those paths. This results in duplicate `stat` system calls per file, which is a significant performance bottleneck on large directories or network mounts.
**Action:** Replace `os.listdir` with `os.scandir` when iterating over directories and querying file attributes. `os.scandir` returns `os.DirEntry` objects which inherently cache `stat` results for `is_dir()`, `is_file()`, and `stat()`, drastically reducing I/O overhead.
