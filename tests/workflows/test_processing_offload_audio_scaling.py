from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest

pytest.importorskip("pydub")

from pydub import AudioSegment  # noqa: E402

from nodetool.workflows.processing_offload import (  # noqa: E402
    _audio_segment_to_numpy,
    _numpy_audio_to_wav_bytes,
)

SAMPLE_RATE = 44100


def _roundtrip(arr: np.ndarray) -> np.ndarray:
    wav_bytes = _numpy_audio_to_wav_bytes(arr, SAMPLE_RATE)
    segment = AudioSegment.from_file(BytesIO(wav_bytes), format="wav")
    samples, _, _ = _audio_segment_to_numpy(segment, sample_rate=SAMPLE_RATE, mono=True)
    return samples


def test_float_roundtrip_preserves_full_scale_amplitude():
    arr = np.array([1.0, -1.0, 0.5, -0.5, 0.0], dtype=np.float32)
    out = _roundtrip(arr)
    assert out.shape == arr.shape
    # Previously this halved amplitude (scaled by 2**14, normalized by 2**15).
    assert np.allclose(out, arr, atol=1e-3)


def test_float_roundtrip_clips_instead_of_wrapping():
    arr = np.array([2.5, -2.5, 1.5], dtype=np.float32)
    out = _roundtrip(arr)
    # Out-of-range values must saturate, never flip sign.
    assert np.allclose(out, np.array([1.0, -1.0, 1.0]), atol=1e-3)


def test_int16_roundtrip_is_unchanged():
    arr = np.array([32767, -32768, 16384, 0], dtype=np.int16)
    out = _roundtrip(arr)
    assert np.allclose(out, arr.astype(np.float32) / 32768.0, atol=1e-4)


class _FakeSegment:
    """Minimal AudioSegment stand-in (pydub itself widens 24-bit to 32-bit)."""

    def __init__(self, raw_data: bytes, sample_width: int):
        self.raw_data = raw_data
        self.sample_width = sample_width
        self.frame_rate = SAMPLE_RATE
        self.channels = 1

    def set_frame_rate(self, frame_rate: int):
        assert frame_rate == self.frame_rate
        return self

    def set_channels(self, channels: int):  # pragma: no cover - mono already
        return self


def _pack_24bit(values: list[int]) -> bytes:
    return b"".join(int(v & 0xFFFFFF).to_bytes(3, "little") for v in values)


def test_24bit_audio_is_decoded_correctly():
    values = [0, 2**22, -(2**22), 2**23 - 1, -(2**23)]
    segment = _FakeSegment(_pack_24bit(values), sample_width=3)

    samples, rate, channels = _audio_segment_to_numpy(
        segment, sample_rate=SAMPLE_RATE, mono=True
    )

    assert rate == SAMPLE_RATE
    assert channels == 1
    # One sample per 3-byte frame (previously np.int16 gave 1.5x the samples).
    assert len(samples) == len(values)
    expected = np.array(values, dtype=np.float32) / np.float32(2**23)
    assert np.allclose(samples, expected, atol=1e-6)


def test_24bit_audio_with_truncated_frame_raises():
    segment = _FakeSegment(_pack_24bit([1, 2]) + b"\x00", sample_width=3)
    with pytest.raises(ValueError, match="multiple of 3"):
        _audio_segment_to_numpy(segment, sample_rate=SAMPLE_RATE, mono=True)


def test_unsupported_sample_width_raises():
    segment = _FakeSegment(b"\x00" * 10, sample_width=5)
    with pytest.raises(ValueError, match="Unsupported audio sample width"):
        _audio_segment_to_numpy(segment, sample_rate=SAMPLE_RATE, mono=True)
