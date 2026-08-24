"""A blob-backed asset ref must reach the host alongside its bytes.

`_extract_outputs` and `_extract_named_outputs` moved a `blob://`-backed ref's
bytes into the frame's blobs map and then dropped the ref itself. The bytes
arrived; everything else about the asset did not. `VideoRef.duration`/`format`
and `Model3DRef.format`/`material_file`/`texture_files` exist only on the ref,
so they never left the worker.

This is the seam that hid the defect from the earlier serialization fix: a
`DataframeRef` carries no blob, so it takes the `_serialize_value` branch and
was covered end to end, while every blob-backed ref took the branch that
returned before serializing anything.

The bytes must still travel once, in the blobs map only. `_serialize_asset_ref`
strips every raw-bytes `data` field, which is what makes emitting both safe.

Host half: nodetool-ai/nodetool#5189. Neither half works alone.
"""

import asyncio
from typing import Any, AsyncGenerator

import pytest
from pydantic import Field

from nodetool.metadata.types import Model3DRef, VideoRef
from nodetool.worker.context_stub import WorkerContext
from nodetool.worker.executor import (
    _extract_named_outputs,
    _extract_outputs,
    execute_node,
    execute_node_stream,
)
from nodetool.workflows.base_node import NODE_BY_TYPE, BaseNode
from nodetool.workflows.processing_context import ProcessingContext

GLB = b"glTF\x02\x00\x00\x00fake-model-bytes"


class Model3DNode(BaseNode):
    """Return a model built the canonical way: through the context factory."""

    @classmethod
    def get_node_type(cls) -> str:
        return "test.Model3DNode"

    async def process(self, context: ProcessingContext) -> Model3DRef:
        return await context.model3d_from_bytes(GLB, name="mesh", format="glb")


class Model3DNamedNode(BaseNode):
    """Return a named-output dict mixing a blob-backed ref with a scalar."""

    @classmethod
    def get_node_type(cls) -> str:
        return "test.Model3DNamedNode"

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        model = await context.model3d_from_bytes(GLB, name="mesh", format="glb")
        return {"model": model, "note": "built"}


class Model3DStreamingNode(BaseNode):
    """Yield a blob-backed ref, so the streaming path is exercised too."""

    @classmethod
    def get_node_type(cls) -> str:
        return "test.Model3DStreamingNode"

    async def process(self, context: ProcessingContext) -> str:
        raise AssertionError("execute_node should use gen_process for streaming nodes")

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[dict[str, Model3DRef], None]:
        yield {"model": await context.model3d_from_bytes(GLB, name="mesh", format="glb")}


@pytest.fixture
def model3d_nodes():
    NODE_BY_TYPE["test.Model3DNode"] = Model3DNode
    NODE_BY_TYPE["test.Model3DNamedNode"] = Model3DNamedNode
    NODE_BY_TYPE["test.Model3DStreamingNode"] = Model3DStreamingNode
    yield
    for t in ("test.Model3DNode", "test.Model3DNamedNode", "test.Model3DStreamingNode"):
        NODE_BY_TYPE.pop(t, None)


def _contains_bytes(obj: Any, needle: bytes) -> bool:
    """True if the raw payload appears anywhere in the serialized outputs."""
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj) == needle or needle in bytes(obj)
    if isinstance(obj, dict):
        return any(_contains_bytes(v, needle) for v in obj.values())
    if isinstance(obj, list):
        return any(_contains_bytes(v, needle) for v in obj)
    return False


@pytest.mark.asyncio
async def test_single_blob_backed_ref_reaches_outputs(model3d_nodes):
    """End to end through execute_node: the ref and its bytes both arrive."""
    result = await execute_node(
        node_type="test.Model3DNode", fields={}, secrets={}, input_blobs={}
    )

    output = result["outputs"]["output"]
    assert output["type"] == "model_3d"
    assert output["uri"].startswith("blob://")
    # The whole reason for the fix: this field exists only on the ref.
    assert output["format"] == "glb"
    assert result["blobs"]["output"] == GLB


@pytest.mark.asyncio
async def test_named_outputs_emit_ref_beside_blob(model3d_nodes):
    """The batch named-output path in _extract_outputs, end to end."""
    result = await execute_node(
        node_type="test.Model3DNamedNode", fields={}, secrets={}, input_blobs={}
    )

    assert result["outputs"]["model"]["format"] == "glb"
    assert result["outputs"]["model"]["type"] == "model_3d"
    assert result["outputs"]["note"] == "built"
    assert result["blobs"]["model"] == GLB


@pytest.mark.asyncio
async def test_streaming_named_outputs_emit_ref_beside_blob(model3d_nodes):
    """The streaming path in _extract_named_outputs, end to end.

    Fixing only the batch function is how this class of defect propagates:
    the two are the same bug written twice.
    """
    items = [
        item
        async for item in execute_node_stream(
            node_type="test.Model3DStreamingNode", fields={}, secrets={}, input_blobs={}
        )
    ]

    assert len(items) == 1
    assert items[0]["outputs"]["model"]["format"] == "glb"
    assert items[0]["blobs"]["model"] == GLB


@pytest.mark.asyncio
async def test_payload_is_not_duplicated(model3d_nodes):
    """The bytes belong in the blobs map only, never inline in the ref."""
    result = await execute_node(
        node_type="test.Model3DNode", fields={}, secrets={}, input_blobs={}
    )

    assert result["outputs"]["output"]["data"] is None
    assert not _contains_bytes(result["outputs"], GLB), "payload sent twice"
    assert result["blobs"]["output"] == GLB


@pytest.mark.asyncio
async def test_video_ref_metadata_survives_the_blob_path():
    """duration/format are exactly what a blob-backed VideoRef used to lose."""
    ctx = WorkerContext()
    # Register real bytes through the public factory, then point a VideoRef at
    # the same blob — the extract functions key on the uri, not the blob name.
    stored = await ctx.image_from_bytes(b"mp4-bytes", name="clip")
    video = VideoRef(uri=stored.uri, duration=12.5, format="mp4")

    outputs, blobs = _extract_outputs(video, ctx)
    assert outputs["output"]["duration"] == 12.5
    assert outputs["output"]["format"] == "mp4"
    assert blobs["output"] == b"mp4-bytes"

    named_outputs, named_blobs = _extract_named_outputs({"clip": video}, ctx)
    assert named_outputs["clip"]["duration"] == 12.5
    assert named_outputs["clip"]["format"] == "mp4"
    assert named_blobs["clip"] == b"mp4-bytes"


@pytest.mark.asyncio
async def test_ref_without_its_blob_is_still_emitted():
    """A missing blob must not make the whole output disappear from the frame.

    The old code only wrote an output when the blob was found, so a ref whose
    bytes were absent left no trace at all — the host saw the handle simply not
    fire.
    """
    ctx = WorkerContext()
    orphan = Model3DRef(uri="blob://never-registered", format="glb")

    outputs, blobs = _extract_named_outputs({"model": orphan}, ctx)
    assert outputs["model"]["format"] == "glb"
    assert "model" not in blobs

    batch_outputs, _ = _extract_outputs({"model": orphan, "note": "x"}, ctx)
    assert batch_outputs["model"]["format"] == "glb"


@pytest.mark.asyncio
async def test_model3d_from_bytes_carries_format_and_metadata():
    """The factory accepted `format` and `metadata` and built the ref without them.

    Even with the plumbing fixed, a node using the canonical factory had no
    format to carry.
    """
    ctx = WorkerContext()
    ref = await ctx.model3d_from_bytes(GLB, name="mesh", format="glb", metadata={"units": "m"})

    assert ref.format == "glb"
    assert ref.metadata == {"units": "m"}
    assert ref.uri.startswith("blob://")
    assert ctx.get_output_blobs()[ref.uri[len("blob://") :]] == GLB
