## 2025-02-12 - Video Export Optimization and Correctness
**Learning:** Upfront list comprehension for processing large sequences (like video frames) consumes O(N) memory and can lead to OOM. Lazy processing inside the writing loop reduces memory usage to O(1). Additionally, assuming input data types (e.g. float vs uint8) without checking can lead to critical bugs like integer overflow when scaling `uint8` arrays by 255.
**Action:** Always prefer lazy iteration/generators for large data processing pipelines. Explicitly check `numpy.dtype` before performing arithmetic scaling to ensure correctness and avoid unnecessary operations.

## 2025-02-12 - AudioSegment Concatenation Optimization
**Learning:** In pydub, concatenating many `AudioSegment` objects in a loop using `+=` causes O(N^2) byte-copying performance penalties.
**Action:** Always verify if the audio segments share the same `sample_width`, `frame_rate`, and `channels`. If they do, join their raw bytes directly (`b"".join([a._data for a in audios])`) and initialize a new segment via `first_audio._spawn(raw_data)` to reduce concatenation time from O(N^2) to O(N).
## 2026-04-24 - Recursive directory iteration
**Learning:** Using `os.walk` is significantly faster than using custom recursive directory iteration with `os.listdir` and `os.path.isdir`, because `os.walk` uses `os.scandir` under the hood which returns directory entries containing cached file attributes, thus avoiding expensive `stat()` system calls per file.
**Action:** Whenever implementing tools or recursive file fetchers, default to `os.walk()` (or `pathlib.Path.rglob` / `os.scandir` directly) instead of recursive `os.listdir`.
