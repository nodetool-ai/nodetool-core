"""Tests that WorkerContext can read back the blob:// refs its own helpers return."""

import io
import wave
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from nodetool.metadata.types import AudioRef
from nodetool.worker.context_stub import WorkerContext


def _blob_key(uri: str) -> str:
    assert uri.startswith("blob://")
    return uri[len("blob://") :]


@pytest.mark.asyncio
async def test_audio_from_numpy_ref_reads_back_through_asset_to_bytes():
    ctx = WorkerContext(secrets={})
    ref = await ctx.audio_from_numpy(np.zeros(2400, dtype=np.int16), 24000)

    data = await ctx.asset_to_bytes(ref)

    assert data == ctx.get_output_blobs()[_blob_key(ref.uri)]
    with wave.open(io.BytesIO(data)) as wav:
        assert wav.getframerate() == 24000
        assert wav.getnframes() == 2400


@pytest.mark.asyncio
async def test_image_ref_reads_back_through_asset_to_bytes():
    ctx = WorkerContext(secrets={})
    ref = await ctx.image_from_pil(Image.new("RGB", (4, 4), "red"))

    data = await ctx.asset_to_bytes(ref)

    assert data == ctx.get_output_blobs()[_blob_key(ref.uri)]
    assert Image.open(BytesIO(data)).size == (4, 4)


@pytest.mark.asyncio
async def test_missing_blob_key_raises_clear_value_error():
    ctx = WorkerContext(secrets={})

    with pytest.raises(ValueError, match=r"audio_output_missing.*not in this worker context"):
        await ctx.asset_to_bytes(AudioRef(uri="blob://audio_output_missing"))


@pytest.mark.asyncio
async def test_asset_to_io_returns_fresh_stream_each_call():
    ctx = WorkerContext(secrets={})
    ref = await ctx.image_from_bytes(b"payload", name="x")

    first = await ctx.asset_to_io(ref)
    assert first.read() == b"payload"
    second = await ctx.asset_to_io(ref)
    assert second.read() == b"payload"
