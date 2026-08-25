"""Regression tests for the entry points that still dropped their bytes.

``image_from_io`` (and ``image_from_url``, which downloads and funnels
through it), ``model3d_from_io``, and ``audio_from_segment``'s no-name path —
which returned a ``memory://`` ref pointing at a store that lives and dies
inside the worker.
"""

from io import BytesIO
from typing import Any

import pytest

from nodetool.worker.context_stub import WorkerContext
from nodetool.worker.executor import _extract_outputs

PNG = b"\x89PNG\r\n\x1a\n" + b"fake png payload"
GLB = b"glTF\x02\x00\x00\x00" + b"fake glb payload"


def _blob_of(ctx: WorkerContext, ref: Any) -> bytes:
    return ctx.get_output_blobs()[ref.uri[len("blob://") :]]


@pytest.mark.asyncio
async def test_image_from_io_registers_a_blob():
    ctx = WorkerContext()
    ref = await ctx.image_from_io(BytesIO(PNG), metadata={"width": 8})

    assert ref.uri.startswith("blob://"), f"expected a blob uri, got {ref.uri!r}"
    assert _blob_of(ctx, ref) == PNG
    assert ref.metadata == {"width": 8}


@pytest.mark.asyncio
async def test_image_from_url_funnels_through_image_from_io(monkeypatch: pytest.MonkeyPatch):
    """No network: the download is stubbed, the funnel is the point."""
    ctx = WorkerContext()

    async def fake_download(url: str) -> BytesIO:
        assert url == "https://example.invalid/cat.png"
        return BytesIO(PNG)

    monkeypatch.setattr(ctx, "download_file", fake_download)
    ref = await ctx.image_from_url("https://example.invalid/cat.png")

    assert ref.uri.startswith("blob://"), f"expected a blob uri, got {ref.uri!r}"
    assert _blob_of(ctx, ref) == PNG


@pytest.mark.asyncio
async def test_model3d_from_io_registers_a_blob():
    ctx = WorkerContext()
    ref = await ctx.model3d_from_io(BytesIO(GLB), format="glb", metadata={"vertices": 3})

    assert ref.uri.startswith("blob://"), f"expected a blob uri, got {ref.uri!r}"
    assert _blob_of(ctx, ref) == GLB
    assert ref.format == "glb"
    assert ref.metadata == {"vertices": 3}


@pytest.mark.asyncio
async def test_audio_from_segment_without_a_name_registers_a_blob():
    """The base takes a memory:// shortcut here; the host cannot follow it."""
    from pydub import AudioSegment

    ctx = WorkerContext()
    ref = await ctx.audio_from_segment(AudioSegment.silent(duration=20, frame_rate=8000))

    assert ref.uri.startswith("blob://"), f"expected a blob uri, got {ref.uri!r}"
    blob = _blob_of(ctx, ref)
    assert blob[:4] == b"RIFF"

    outputs, blobs = _extract_outputs(ref, ctx)
    assert blobs.get("output") == blob
    # The encoding and the metadata stay the base implementation's.
    assert outputs["output"]["metadata"]["sample_rate"] == 8000
    assert not outputs["output"].get("data")
