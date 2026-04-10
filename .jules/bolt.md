## 2025-02-12 - Video Export Optimization and Correctness
**Learning:** Upfront list comprehension for processing large sequences (like video frames) consumes O(N) memory and can lead to OOM. Lazy processing inside the writing loop reduces memory usage to O(1). Additionally, assuming input data types (e.g. float vs uint8) without checking can lead to critical bugs like integer overflow when scaling `uint8` arrays by 255.
**Action:** Always prefer lazy iteration/generators for large data processing pipelines. Explicitly check `numpy.dtype` before performing arithmetic scaling to ensure correctness and avoid unnecessary operations.

## 2025-02-13 - Audio Segment Concatenation Optimization
**Learning:** Using `+=` iteratively on `pydub.AudioSegment` objects incurs an O(N^2) performance penalty due to continuous memory allocations and byte copying. For large numbers of segments, this is extremely slow.
**Action:** Optimize audio concatenation by joining the raw byte data of all segments simultaneously using `b''.join(a.raw_data for a in audios)` and then creating a single new `AudioSegment`, ensuring the components' basic attributes (sample width, frame rate, channels) are homogeneous.
