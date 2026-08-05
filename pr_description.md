## What
Optimizes the intermediate scaling operations (`clip` and multiplication by `32767.0`) in audio processing pipelines by explicitly casting `float64` and `float16` inputs to `float32` first via `np.asarray`.

## Why
When scaling a `float32` array by a python float constant (like `32767.0`), NumPy implicitly upcasts the whole array to `float64`, effectively doubling memory consumption during the operation. Alternatively, if the incoming array is `float64`, doing intermediate math in `float64` uses twice the memory bandwidth unnecessarily because we downcast to `int16` anyway. Finally, if the input is `float16`, doing math is extremely slow because modern CPUs lack native support and emulate it in software.

## Impact
- **Halves intermediate memory overhead** when scaling large `float64` or `float32` audio chunks by doing the math strictly in `float32`.
- **Significantly speeds up** `float16` audio processing by performing hardware-accelerated `float32` math instead of software-emulated `float16` math.
- Does not change correctness or audio quality, since `float32`'s 24 bits of mantissa are vastly sufficient to safely target the 16 bits of precision required by `int16`.

## Verification
- Ran project unit tests (`make test-verbose`, `uv run pytest tests/workflows/test_processing_offload_audio_scaling.py`).
- Ran linting via `uv run ruff check src` ensuring no style regressions.
- Verified on a local micro-benchmark that scaling a large array with this optimization reduces intermediate memory and time overhead while returning the identical byte payload.
