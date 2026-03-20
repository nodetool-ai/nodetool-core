## 2025-02-12 - Video Export Optimization and Correctness
**Learning:** Upfront list comprehension for processing large sequences (like video frames) consumes O(N) memory and can lead to OOM. Lazy processing inside the writing loop reduces memory usage to O(1). Additionally, assuming input data types (e.g. float vs uint8) without checking can lead to critical bugs like integer overflow when scaling `uint8` arrays by 255.
**Action:** Always prefer lazy iteration/generators for large data processing pipelines. Explicitly check `numpy.dtype` before performing arithmetic scaling to ensure correctness and avoid unnecessary operations.

## 2025-02-13 - imageio ffmpeg BytesIO crashes
**Learning:** `imageio`'s `ffmpeg` plugin crashes with "`FFMPEG` can not handle the given uri." when given an `io.BytesIO` memory stream. Using `format="mp4"` instead allows `imageio` to successfully process the in-memory bytes and is significantly faster than writing to a temporary file for OpenCV fallback processing.
**Action:** When using `imageio.get_reader` on `io.BytesIO` byte streams, conditionally pass `format="mp4"` rather than hardcoding `format="ffmpeg"`.
