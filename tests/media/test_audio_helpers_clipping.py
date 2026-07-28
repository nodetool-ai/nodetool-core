"""Regression test: float->int16 conversion must clip instead of wrapping around."""

import numpy as np

from nodetool.media.audio.audio_helpers import numpy_to_audio_segment


def test_numpy_to_audio_segment_clips_out_of_range_samples() -> None:
    arr = np.array([1.2, -1.2, 0.5, 0.0], dtype=np.float32)

    segment = numpy_to_audio_segment(arr)
    samples = list(segment.get_array_of_samples())

    assert samples[0] == 32767
    assert samples[1] == -32767
    assert samples[2] == 16383
    assert samples[3] == 0
