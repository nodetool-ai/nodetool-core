"""Regression test: float audio must be converted to int16 at full scale.

``audio_from_numpy`` used to scale floats by ``2**14`` while the decode path
divides by ``2**15``, halving amplitude on every round trip.
"""

import numpy as np
import pytest

from nodetool.workflows.processing_context import ProcessingContext


@pytest.mark.asyncio
async def test_float_audio_roundtrip_preserves_amplitude(user_id: str):
    context = ProcessingContext(user_id=user_id, auth_token="t")
    sample_rate = 8000
    t = np.linspace(0, 1, sample_rate, endpoint=False, dtype=np.float32)
    original = (0.8 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    audio_ref = await context.audio_from_numpy(original, sample_rate)
    decoded, frame_rate, _channels = await context.audio_to_numpy(audio_ref, sample_rate=sample_rate)

    assert frame_rate == sample_rate
    # Amplitude must survive the round trip (previously it was halved to ~0.4)
    assert np.abs(decoded).max() == pytest.approx(0.8, abs=0.02)


@pytest.mark.asyncio
async def test_float_audio_is_clipped_not_wrapped(user_id: str):
    """Out-of-range floats must clip to full scale instead of wrapping around."""
    context = ProcessingContext(user_id=user_id, auth_token="t")
    sample_rate = 8000
    original = np.array([2.0, -2.0, 0.0, 1.0, -1.0] * 100, dtype=np.float32)

    audio_ref = await context.audio_from_numpy(original, sample_rate)
    decoded, _frame_rate, _channels = await context.audio_to_numpy(audio_ref, sample_rate=sample_rate)

    assert decoded.max() == pytest.approx(1.0, abs=0.01)
    assert decoded.min() == pytest.approx(-1.0, abs=0.01)
