"""Regression tests: encoded audio bytes must survive the worker bridge.

``audio_from_numpy`` encodes its own WAV and registers a blob. A node handed
already encoded audio — a provider's mp3, a file read from disk — calls
``audio_from_bytes`` / ``audio_from_base64`` instead, which funnel through
``audio_from_io``. That path has to hand back a ``blob://`` ref too, because
the executor extracts blobs only from that uri scheme and strips raw ``data``
fields when it serializes a ref.
"""

import base64
from io import BytesIO

import pytest

from nodetool.metadata.types import AudioRef
from nodetool.worker.context_stub import WorkerContext
from nodetool.worker.executor import _extract_outputs

MP3 = b"ID3\x03\x00\x00\x00" + b"fake mp3 payload"


def _blob_of(ctx: WorkerContext, ref: AudioRef) -> bytes:
    return ctx.get_output_blobs()[ref.uri[len("blob://") :]]


@pytest.mark.asyncio
async def test_audio_from_bytes_registers_a_blob():
    ctx = WorkerContext()
    ref = await ctx.audio_from_bytes(MP3)

    assert ref.uri.startswith("blob://"), f"expected a blob uri, got {ref.uri!r}"
    assert _blob_of(ctx, ref) == MP3


@pytest.mark.asyncio
async def test_audio_from_io_registers_a_blob():
    ctx = WorkerContext()
    ref = await ctx.audio_from_io(BytesIO(MP3))

    assert ref.uri.startswith("blob://"), f"expected a blob uri, got {ref.uri!r}"
    assert _blob_of(ctx, ref) == MP3


@pytest.mark.asyncio
async def test_audio_from_base64_registers_a_blob():
    ctx = WorkerContext()
    ref = await ctx.audio_from_base64(base64.b64encode(MP3).decode())

    assert ref.uri.startswith("blob://"), f"expected a blob uri, got {ref.uri!r}"
    assert _blob_of(ctx, ref) == MP3


@pytest.mark.asyncio
async def test_audio_bytes_reach_the_executor_output_split():
    """The end-to-end half: bytes in the blobs map, nothing raw on the ref."""
    ctx = WorkerContext()
    ref = await ctx.audio_from_bytes(MP3)

    outputs, blobs = _extract_outputs(ref, ctx)

    assert blobs.get("output") == MP3
    assert outputs["output"]["type"] == "audio"
    # The serializer strips raw bytes at any depth; the blobs map is the only
    # place the audio may travel.
    assert not outputs["output"].get("data")


@pytest.mark.asyncio
async def test_audio_output_in_a_multi_output_dict():
    ctx = WorkerContext()
    ref = await ctx.audio_from_bytes(MP3)

    outputs, blobs = _extract_outputs({"audio": ref, "sample_rate": 44100}, ctx)

    assert blobs.get("audio") == MP3
    assert outputs["sample_rate"] == 44100


@pytest.mark.asyncio
async def test_audio_from_numpy_still_registers_its_own_blob():
    """The existing WAV path is untouched by the new override."""
    import numpy as np

    ctx = WorkerContext()
    ref = await ctx.audio_from_numpy(np.zeros(100, dtype=np.int16), 44100)

    assert ref.uri.startswith("blob://")
    assert _blob_of(ctx, ref)[:4] == b"RIFF"
