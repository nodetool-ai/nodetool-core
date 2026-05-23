import asyncio
from typing import Any, AsyncGenerator

import numpy as np
import pytest
from pydantic import Field

from nodetool.metadata.types import AudioRef, Chunk, ImageRef
from nodetool.worker.executor import execute_node, execute_node_stream
from nodetool.workflows.base_node import NODE_BY_TYPE, BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import NodeProgress


class EchoNode(BaseNode):
    """Echo the input text."""
    text: str = Field(default="")

    @classmethod
    def get_node_type(cls) -> str:
        return "test.EchoNode"

    async def process(self, context: ProcessingContext) -> str:
        return self.text


class ImageListNode(BaseNode):
    """Expose image list coercion for worker tests."""

    images: list[ImageRef] = Field(default_factory=list)

    @classmethod
    def get_node_type(cls) -> str:
        return "test.ImageListNode"

    async def process(self, context: ProcessingContext) -> dict[str, str | int | bool]:
        first_image = self.images[0]
        return {
            "count": len(self.images),
            "first_uri": first_image.uri,
            "first_is_empty": first_image.is_empty(),
        }


class StreamingOnlyNode(BaseNode):
    text: str = Field(default="")

    @classmethod
    def get_node_type(cls) -> str:
        return "test.StreamingOnlyNode"

    async def process(self, context: ProcessingContext) -> str:
        raise AssertionError("execute_node should use gen_process for streaming nodes")

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[dict[str, str | Chunk], None]:
        yield {
            "output": self.text,
            "chunk": Chunk(content=self.text, done=True, content_type="text"),
        }


class StreamingAudioNode(BaseNode):
    @classmethod
    def get_node_type(cls) -> str:
        return "test.StreamingAudioNode"

    async def process(self, context: ProcessingContext) -> str:
        raise AssertionError("execute_node should use gen_process for streaming nodes")

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[dict[str, AudioRef | Chunk], None]:
        audio = await context.audio_from_numpy(
            np.array([0.0, 0.25, -0.25], dtype=np.float32),
            24_000,
            name="preview",
        )
        yield {
            "audio": audio,
            "chunk": Chunk(content="", done=True, content_type="audio"),
        }


class MultiChunkStreamingNode(BaseNode):
    @classmethod
    def get_node_type(cls) -> str:
        return "test.MultiChunkStreamingNode"

    async def process(self, context: ProcessingContext) -> str:
        raise AssertionError("execute_node should use gen_process for streaming nodes")

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[dict[str, str | None | Chunk], None]:
        accumulated = ""
        for index in range(3):
            token = f"tok{index}"
            accumulated += token
            yield {
                "text": None,
                "chunk": Chunk(content=token, done=False, content_type="text"),
            }
        yield {
            "text": accumulated,
            "chunk": Chunk(content="", done=True, content_type="text"),
        }


class ProgressEmittingNode(BaseNode):
    """Posts NodeProgress mid-execution and after a small yield to the loop.

    Used to verify the background pump forwards progress in near-real-time
    rather than waiting for ``process`` to return.
    """

    @classmethod
    def get_node_type(cls) -> str:
        return "test.ProgressEmittingNode"

    async def process(self, context: ProcessingContext) -> str:
        context.post_message(
            NodeProgress(node_id="n1", progress=10, total=100, chunk="early")
        )
        # Yield to the loop so the background pump has a chance to drain.
        await asyncio.sleep(0.12)
        context.post_message(
            NodeProgress(node_id="n1", progress=90, total=100, chunk="late")
        )
        return "done"


class ProgressBeforeRaiseNode(BaseNode):
    """Posts progress then raises — verifies the final drain still ships it."""

    @classmethod
    def get_node_type(cls) -> str:
        return "test.ProgressBeforeRaiseNode"

    async def process(self, context: ProcessingContext) -> str:
        context.post_message(
            NodeProgress(node_id="n1", progress=50, total=100, chunk="mid")
        )
        raise RuntimeError("boom")


@pytest.fixture(autouse=True)
def register_echo():
    NODE_BY_TYPE["test.EchoNode"] = EchoNode
    NODE_BY_TYPE["test.ImageListNode"] = ImageListNode
    NODE_BY_TYPE["test.StreamingOnlyNode"] = StreamingOnlyNode
    NODE_BY_TYPE["test.StreamingAudioNode"] = StreamingAudioNode
    NODE_BY_TYPE["test.MultiChunkStreamingNode"] = MultiChunkStreamingNode
    NODE_BY_TYPE["test.ProgressEmittingNode"] = ProgressEmittingNode
    NODE_BY_TYPE["test.ProgressBeforeRaiseNode"] = ProgressBeforeRaiseNode
    yield
    NODE_BY_TYPE.pop("test.EchoNode", None)
    NODE_BY_TYPE.pop("test.ImageListNode", None)
    NODE_BY_TYPE.pop("test.StreamingOnlyNode", None)
    NODE_BY_TYPE.pop("test.StreamingAudioNode", None)
    NODE_BY_TYPE.pop("test.MultiChunkStreamingNode", None)
    NODE_BY_TYPE.pop("test.ProgressEmittingNode", None)
    NODE_BY_TYPE.pop("test.ProgressBeforeRaiseNode", None)


@pytest.mark.asyncio
async def test_execute_echo_node():
    result = await execute_node(
        node_type="test.EchoNode",
        fields={"text": "hello world"},
        secrets={},
        input_blobs={},
    )
    assert result["outputs"]["output"] == "hello world"
    assert result["blobs"] == {}


@pytest.mark.asyncio
async def test_execute_wraps_single_asset_blob_into_image_list():
    result = await execute_node(
        node_type="test.ImageListNode",
        fields={},
        secrets={},
        input_blobs={"images": b"fake-image-bytes"},
    )
    output = result["outputs"]
    assert output["count"] == 1
    assert output["first_is_empty"] is False
    assert output["first_uri"].startswith("file://")
    assert result["blobs"] == {}


@pytest.mark.asyncio
async def test_execute_preserves_multiple_asset_blobs_for_image_list():
    result = await execute_node(
        node_type="test.ImageListNode",
        fields={},
        secrets={},
        input_blobs={"images": [b"first-image", b"second-image"]},
    )
    output = result["outputs"]
    assert output["count"] == 2
    assert output["first_is_empty"] is False
    assert output["first_uri"].startswith("file://")
    assert result["blobs"] == {}


@pytest.mark.asyncio
async def test_execute_wraps_single_serialized_image_into_image_list():
    result = await execute_node(
        node_type="test.ImageListNode",
        fields={"images": {"type": "image", "uri": "file://example.png"}},
        secrets={},
        input_blobs={},
    )
    output = result["outputs"]
    assert output["count"] == 1
    assert output["first_is_empty"] is False
    assert output["first_uri"] == "file://example.png"
    assert result["blobs"] == {}


@pytest.mark.asyncio
async def test_execute_uses_gen_process_for_streaming_nodes():
    result = await execute_node(
        node_type="test.StreamingOnlyNode",
        fields={"text": "streamed"},
        secrets={},
        input_blobs={},
    )

    assert result["outputs"]["output"] == "streamed"
    assert result["outputs"]["chunk"]["content"] == "streamed"
    assert result["outputs"]["chunk"]["done"] is True
    assert result["outputs"]["chunk"]["content_type"] == "text"
    assert result["blobs"] == {}


@pytest.mark.asyncio
async def test_execute_serializes_streaming_audio_refs_with_protocol_type():
    result = await execute_node(
        node_type="test.StreamingAudioNode",
        fields={},
        secrets={},
        input_blobs={},
    )

    assert "audio" not in result["outputs"]
    assert result["blobs"]["audio"].startswith(b"RIFF")
    assert result["outputs"]["chunk"]["content_type"] == "audio"
    assert result["outputs"].get("chunk", {}).get("done") is True


@pytest.mark.asyncio
async def test_execute_node_stream_yields_each_chunk():
    items = [
        item
        async for item in execute_node_stream(
            node_type="test.MultiChunkStreamingNode",
            fields={},
            secrets={},
            input_blobs={},
        )
    ]

    # 3 token chunks + 1 final aggregate chunk
    assert len(items) == 4
    assert all(item["blobs"] == {} for item in items)
    assert items[0]["outputs"]["chunk"]["content"] == "tok0"
    assert items[0]["outputs"]["chunk"]["done"] is False
    assert items[0]["outputs"]["text"] is None
    assert items[-1]["outputs"]["text"] == "tok0tok1tok2"
    assert items[-1]["outputs"]["chunk"]["done"] is True


@pytest.mark.asyncio
async def test_execute_node_stream_extracts_blobs_per_chunk():
    items = [
        item
        async for item in execute_node_stream(
            node_type="test.StreamingAudioNode",
            fields={},
            secrets={},
            input_blobs={},
        )
    ]

    assert len(items) == 1
    assert "audio" not in items[0]["outputs"]
    assert items[0]["blobs"]["audio"].startswith(b"RIFF")
    assert items[0]["outputs"]["chunk"]["content_type"] == "audio"


@pytest.mark.asyncio
async def test_execute_node_emits_progress_in_real_time():
    """Progress posted mid-process must arrive before process() returns.

    The first progress should land while the node is still sleeping (well
    before the ``done`` result), not be buffered until completion.
    """
    progress_arrivals: list[tuple[float, int]] = []
    loop = asyncio.get_event_loop()
    start = loop.time()

    async def on_progress(p: dict[str, Any]) -> None:
        progress_arrivals.append((loop.time() - start, p["progress"]))

    result = await execute_node(
        node_type="test.ProgressEmittingNode",
        fields={},
        secrets={},
        input_blobs={},
        emit_progress=on_progress,
    )

    assert result["outputs"]["output"] == "done"
    # Both progress messages should have been emitted.
    progress_values = [p for _, p in progress_arrivals]
    assert 10 in progress_values
    assert 90 in progress_values
    # The early progress (10) must arrive well before the late one (90),
    # confirming the pump didn't buffer everything until process() returned.
    early_time = next(t for t, p in progress_arrivals if p == 10)
    late_time = next(t for t, p in progress_arrivals if p == 90)
    assert late_time - early_time >= 0.05  # the deliberate sleep gap


@pytest.mark.asyncio
async def test_execute_node_drains_progress_on_exception():
    """Progress queued before a node raises must still be forwarded."""
    progress: list[dict[str, Any]] = []

    async def on_progress(p: dict[str, Any]) -> None:
        progress.append(p)

    with pytest.raises(RuntimeError, match="boom"):
        await execute_node(
            node_type="test.ProgressBeforeRaiseNode",
            fields={},
            secrets={},
            input_blobs={},
            emit_progress=on_progress,
        )

    assert any(p["progress"] == 50 for p in progress)


@pytest.mark.asyncio
async def test_execute_node_emit_chunk_preserves_none_values():
    """The emit_chunk callback must see None slots, matching execute_node_stream."""
    chunks: list[dict[str, Any]] = []

    async def on_chunk(chunk: dict[str, Any]) -> None:
        chunks.append(chunk)

    result = await execute_node(
        node_type="test.MultiChunkStreamingNode",
        fields={},
        secrets={},
        input_blobs={},
        emit_chunk=on_chunk,
    )

    assert len(chunks) == 4
    # First chunk: token only, text slot present but None.
    assert chunks[0]["outputs"]["chunk"]["content"] == "tok0"
    assert chunks[0]["outputs"]["chunk"]["done"] is False
    assert chunks[0]["outputs"]["text"] is None
    # Final chunk: accumulated text + done flag.
    assert chunks[-1]["outputs"]["text"] == "tok0tok1tok2"
    assert chunks[-1]["outputs"]["chunk"]["done"] is True
    # Aggregated final result keeps the last non-None value per slot.
    assert result["outputs"]["text"] == "tok0tok1tok2"
    assert result["outputs"]["chunk"]["done"] is True


@pytest.mark.asyncio
async def test_execute_node_stream_matches_emit_chunk():
    """execute_node_stream must yield exactly what emit_chunk would receive."""
    chunks_from_callback: list[dict[str, Any]] = []

    async def on_chunk(chunk: dict[str, Any]) -> None:
        chunks_from_callback.append(chunk)

    await execute_node(
        node_type="test.MultiChunkStreamingNode",
        fields={},
        secrets={},
        input_blobs={},
        emit_chunk=on_chunk,
    )

    chunks_from_stream: list[dict[str, Any]] = []
    async for item in execute_node_stream(
        node_type="test.MultiChunkStreamingNode",
        fields={},
        secrets={},
        input_blobs={},
    ):
        chunks_from_stream.append(item)

    assert chunks_from_callback == chunks_from_stream


@pytest.mark.asyncio
async def test_execute_node_stream_propagates_errors():
    """An error raised inside execute_node must surface from execute_node_stream."""
    with pytest.raises(RuntimeError, match="boom"):
        async for _ in execute_node_stream(
            node_type="test.ProgressBeforeRaiseNode",
            fields={},
            secrets={},
            input_blobs={},
        ):
            pass
