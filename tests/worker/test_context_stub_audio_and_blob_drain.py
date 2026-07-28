"""Regression tests for WAV encoding, blob draining and pump cleanup."""

import asyncio
import struct
from typing import Any, AsyncGenerator

import numpy as np
import pytest

from nodetool.worker import executor as executor_module
from nodetool.worker.context_stub import WorkerContext
from nodetool.worker.executor import execute_node
from nodetool.workflows.base_node import NODE_BY_TYPE, BaseNode
from nodetool.workflows.processing_context import ProcessingContext


def _wav_header(blob: bytes) -> dict[str, int]:
    channels, sample_rate, byte_rate = struct.unpack("<HII", blob[22:32])
    return {"channels": channels, "sample_rate": sample_rate, "byte_rate": byte_rate}


def _pcm(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob[44:], dtype=np.int16)


def _blob_of(ctx: WorkerContext, ref: Any) -> bytes:
    return ctx.get_output_blobs()[ref.uri[len("blob://") :]]


@pytest.mark.asyncio
async def test_audio_from_numpy_honours_num_channels_for_2d_arrays():
    ctx = WorkerContext()
    ref = await ctx.audio_from_numpy(
        np.zeros((2, 48000), dtype=np.float32), 44100, num_channels=2
    )
    header = _wav_header(_blob_of(ctx, ref))
    assert header["channels"] == 2
    assert header["sample_rate"] == 44100
    assert header["byte_rate"] == 44100 * 2 * 2


@pytest.mark.asyncio
async def test_audio_from_numpy_defaults_to_mono_for_2d_arrays():
    ctx = WorkerContext()
    ref = await ctx.audio_from_numpy(np.zeros((2, 1000), dtype=np.float32), 44100)
    assert _wav_header(_blob_of(ctx, ref))["channels"] == 1


@pytest.mark.asyncio
async def test_audio_from_numpy_clips_floats_instead_of_wrapping():
    ctx = WorkerContext()
    ref = await ctx.audio_from_numpy(
        np.array([1.5, -1.5, 1.0, -1.0, 0.0], dtype=np.float32), 44100
    )
    assert _pcm(_blob_of(ctx, ref)).tolist() == [32767, -32767, 32767, -32767, 0]


@pytest.mark.asyncio
async def test_audio_from_numpy_passes_int16_through():
    ctx = WorkerContext()
    samples = np.array([100, -200, 32767], dtype=np.int16)
    ref = await ctx.audio_from_numpy(samples, 44100)
    assert _pcm(_blob_of(ctx, ref)).tolist() == samples.tolist()


@pytest.mark.asyncio
@pytest.mark.parametrize("dtype", [np.int32, np.uint8, np.int64])
async def test_audio_from_numpy_rejects_unsupported_dtypes(dtype):
    ctx = WorkerContext()
    with pytest.raises(ValueError, match="Unsupported dtype"):
        await ctx.audio_from_numpy(np.array([100, 200], dtype=dtype), 44100)


@pytest.mark.asyncio
async def test_take_output_blobs_returns_and_clears():
    ctx = WorkerContext()
    await ctx.image_from_bytes(b"abc", name="a")
    assert len(ctx.take_output_blobs()) == 1
    assert ctx.get_output_blobs() == {}
    assert ctx.take_output_blobs() == {}


class BlobStreamingNode(BaseNode):
    chunks: int = 3

    @classmethod
    def get_node_type(cls) -> str:
        return "test.BlobDrainStreamingNode"

    @classmethod
    def is_streaming_output(cls) -> bool:
        return True

    async def gen_process(self, context: ProcessingContext) -> AsyncGenerator[dict, None]:
        for i in range(self.chunks):
            audio = await context.audio_from_numpy(
                np.zeros(8, dtype=np.int16), 44100, name=f"c{i}"
            )
            yield {"audio": audio}


@pytest.fixture
def blob_streaming_node():
    NODE_BY_TYPE["test.BlobDrainStreamingNode"] = BlobStreamingNode
    yield
    NODE_BY_TYPE.pop("test.BlobDrainStreamingNode", None)


@pytest.mark.asyncio
async def test_streaming_path_releases_blobs_per_chunk(blob_streaming_node, monkeypatch):
    chunks: list[dict[str, Any]] = []
    retained: list[int] = []

    real_extract = executor_module._extract_named_outputs

    def spy(result, ctx, drain=False):
        out = real_extract(result, ctx, drain=drain)
        retained.append(len(ctx.get_output_blobs()))
        return out

    monkeypatch.setattr(executor_module, "_extract_named_outputs", spy)

    async def on_chunk(chunk: dict[str, Any]) -> None:
        chunks.append(chunk)

    await execute_node(
        node_type="test.BlobDrainStreamingNode",
        fields={"chunks": 3},
        secrets={},
        input_blobs={},
        emit_chunk=on_chunk,
    )

    assert len(chunks) == 3
    assert all(c["blobs"]["audio"].startswith(b"RIFF") for c in chunks)
    # Nothing is retained on the context after each chunk is emitted.
    assert retained == [0, 0, 0, 0]


@pytest.mark.asyncio
async def test_non_streaming_collect_path_keeps_blobs(blob_streaming_node):
    result = await execute_node(
        node_type="test.BlobDrainStreamingNode",
        fields={"chunks": 2},
        secrets={},
        input_blobs={},
    )
    assert result["blobs"]["audio"].startswith(b"RIFF")


class TrivialNode(BaseNode):
    @classmethod
    def get_node_type(cls) -> str:
        return "test.PumpCleanupNode"

    async def process(self, context: ProcessingContext) -> str:
        return "ok"


@pytest.mark.asyncio
async def test_progress_pump_not_leaked_when_tempdir_fails(monkeypatch):
    NODE_BY_TYPE["test.PumpCleanupNode"] = TrivialNode
    try:
        def boom(*args, **kwargs):
            raise OSError("read-only file system")

        monkeypatch.setattr(executor_module.tempfile, "mkdtemp", boom)

        before = len(asyncio.all_tasks())
        with pytest.raises(OSError, match="read-only"):
            await execute_node(
                node_type="test.PumpCleanupNode",
                fields={},
                secrets={},
                input_blobs={},
                emit_progress=lambda _p: asyncio.sleep(0),
            )
        await asyncio.sleep(0.15)
        assert len(asyncio.all_tasks()) <= before
    finally:
        NODE_BY_TYPE.pop("test.PumpCleanupNode", None)
