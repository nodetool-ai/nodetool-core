## 2025-02-12 - Video Export Optimization and Correctness
**Learning:** Upfront list comprehension for processing large sequences (like video frames) consumes O(N) memory and can lead to OOM. Lazy processing inside the writing loop reduces memory usage to O(1). Additionally, assuming input data types (e.g. float vs uint8) without checking can lead to critical bugs like integer overflow when scaling `uint8` arrays by 255.
**Action:** Always prefer lazy iteration/generators for large data processing pipelines. Explicitly check `numpy.dtype` before performing arithmetic scaling to ensure correctness and avoid unnecessary operations.

## 2025-02-12 - AudioSegment Concatenation Optimization
**Learning:** In pydub, concatenating many `AudioSegment` objects in a loop using `+=` causes O(N^2) byte-copying performance penalties.
**Action:** Always verify if the audio segments share the same `sample_width`, `frame_rate`, and `channels`. If they do, join their raw bytes directly (`b"".join([a._data for a in audios])`) and initialize a new segment via `first_audio._spawn(raw_data)` to reduce concatenation time from O(N^2) to O(N).

## 2025-02-13 - String Concatenation vs List Append
**Learning:** In Python (specifically CPython), iterative string concatenation using `+=` is highly optimized when there are no other references to the string. It often performs slightly better than `.append()` + `"".join()` for simple loop structures (e.g., 0.400s vs 0.414s in our benchmark).
**Action:** Do not blindly replace `+=` with `.append()` and `"".join()` unless performance profiling explicitly shows a bottleneck in that specific scenario.

## 2025-02-13 - Pydub AudioSegment Concatenation with Crossfade
**Learning:** While concatenating raw audio bytes without crossfade is O(N) and fast, using a divide-and-conquer approach to apply crossfade during concatenation actually performs worse than iterative `+=` (8.596s vs 8.358s) due to the overhead of recursive calls and intermediate AudioSegment creation.
**Action:** Do not attempt divide-and-conquer optimizations for AudioSegment crossfade without a proven, specialized blending implementation.
