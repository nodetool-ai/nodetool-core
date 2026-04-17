## 2025-02-12 - Video Export Optimization and Correctness
**Learning:** Upfront list comprehension for processing large sequences (like video frames) consumes O(N) memory and can lead to OOM. Lazy processing inside the writing loop reduces memory usage to O(1). Additionally, assuming input data types (e.g. float vs uint8) without checking can lead to critical bugs like integer overflow when scaling `uint8` arrays by 255.
**Action:** Always prefer lazy iteration/generators for large data processing pipelines. Explicitly check `numpy.dtype` before performing arithmetic scaling to ensure correctness and avoid unnecessary operations.
## 2025-04-17 - PyDub Audio Concatenation O(n^2) Bottleneck
**Learning:** Concatenating multiple `AudioSegment` objects in a loop using `audio_a += audio_b` incurs an O(n^2) byte-copying penalty, causing significant CPU and memory spikes for large inputs, as each addition spawns a new audio segment internally.
**Action:** Always verify if segments share properties (sample width, frame rate, channels) and use an O(n) raw bytes join (`b"".join(a.raw_data for a in audios)`) passed directly to `first_audio._spawn(raw_data)`.
