"""Regression tests: video output bytes must survive the worker bridge.

A video node calls ``context.video_from_bytes(mp4, metadata=...)`` with no
name. The stub context has to hand back a ``blob://`` ref, because the
executor extracts blobs only from that uri scheme and strips raw ``data``
fields when it serializes a ref.
"""

from io import BytesIO

import pytest

from nodetool.metadata.types import VideoRef
from nodetool.worker.context_stub import WorkerContext
from nodetool.worker.executor import _extract_outputs

MP4 = b"\x00\x00\x00\x18ftypmp42" + b"fake mp4 payload"
METADATA = {
    "fps": 25,
    "frame_count": 9,
    "width": 320,
    "height": 320,
    "format": "mp4",
    "duration_seconds": 0.36,
}


def _blob_of(ctx: WorkerContext, ref: VideoRef) -> bytes:
    return ctx.get_output_blobs()[ref.uri[len("blob://") :]]


@pytest.mark.asyncio
async def test_video_from_bytes_registers_a_blob():
    ctx = WorkerContext()
    ref = await ctx.video_from_bytes(MP4, metadata=METADATA)

    assert ref.uri.startswith("blob://"), f"expected a blob uri, got {ref.uri!r}"
    assert _blob_of(ctx, ref) == MP4
    assert ref.metadata == METADATA


@pytest.mark.asyncio
async def test_video_from_io_registers_a_blob():
    ctx = WorkerContext()
    ref = await ctx.video_from_io(BytesIO(MP4), metadata=METADATA)

    assert ref.uri.startswith("blob://"), f"expected a blob uri, got {ref.uri!r}"
    assert _blob_of(ctx, ref) == MP4
    assert ref.metadata == METADATA


@pytest.mark.asyncio
async def test_video_bytes_reach_the_executor_output_split():
    """The end-to-end half: bytes in the blobs map, metadata on the ref."""
    ctx = WorkerContext()
    ref = await ctx.video_from_bytes(MP4, metadata=METADATA)

    outputs, blobs = _extract_outputs(ref, ctx)

    assert blobs.get("output") == MP4
    assert outputs["output"]["type"] == "video"
    assert outputs["output"]["metadata"] == METADATA
    # The serializer strips raw bytes at any depth; the blobs map is the only
    # place the mp4 may travel.
    assert not outputs["output"].get("data")


@pytest.mark.asyncio
async def test_video_output_in_a_multi_output_dict():
    ctx = WorkerContext()
    ref = await ctx.video_from_bytes(MP4, metadata=METADATA)

    outputs, blobs = _extract_outputs({"video": ref, "count": 9}, ctx)

    assert blobs.get("video") == MP4
    assert outputs["count"] == 9
