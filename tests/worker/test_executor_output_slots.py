"""Output slots and blob addressing on the worker's execute path.

B1: a node declaring the single slot ``output`` with ``-> dict[str, float]``
(every HF classifier) had a multi-key result split into undeclared slots
``{cat, dog}``, so the ``output`` edge never fired.

B2: the host pairs blobs with slots by name. Refs inside a list, a dict, or a
TypedDict field were sent with a ``blob://`` uri and their bytes keyed by the
internal blob id, which the host cannot pair with anything.
"""

from typing import AsyncGenerator, TypedDict

import pytest

from nodetool.metadata.types import AudioRef, ImageRef
from nodetool.worker.executor import execute_node
from nodetool.workflows.base_node import NODE_BY_TYPE, BaseNode
from nodetool.workflows.processing_context import ProcessingContext


class ClassifierNode(BaseNode):
    @classmethod
    def get_node_type(cls) -> str:
        return "test.slots.Classifier"

    async def process(self, context: ProcessingContext) -> dict[str, float]:
        return {"cat": 0.9, "dog": 0.1}


class AudioListNode(BaseNode):
    @classmethod
    def get_node_type(cls) -> str:
        return "test.slots.AudioList"

    async def process(self, context: ProcessingContext) -> list[AudioRef]:
        return [
            await context.audio_from_bytes(b"one", name="a"),
            await context.audio_from_bytes(b"two", name="b"),
        ]


class NestedTypedDictNode(BaseNode):
    class OutputType(TypedDict):
        image: ImageRef
        masks: list[ImageRef]
        info: dict[str, ImageRef]

    @classmethod
    def get_node_type(cls) -> str:
        return "test.slots.Nested"

    async def process(self, context: ProcessingContext) -> OutputType:
        return {
            "image": await context.image_from_bytes(b"top"),
            "masks": [await context.image_from_bytes(b"m1")],
            "info": {"preview": await context.image_from_bytes(b"pv")},
        }


class StreamingListNode(BaseNode):
    class OutputType(TypedDict):
        frames: list[ImageRef]

    @classmethod
    def get_node_type(cls) -> str:
        return "test.slots.StreamingList"

    async def gen_process(self, context: ProcessingContext) -> AsyncGenerator[OutputType, None]:
        yield {"frames": [await context.image_from_bytes(b"f1")]}


NODES = (ClassifierNode, AudioListNode, NestedTypedDictNode, StreamingListNode)


@pytest.fixture(autouse=True)
def _register():
    for cls in NODES:
        NODE_BY_TYPE[cls.get_node_type()] = cls
    yield
    for cls in NODES:
        NODE_BY_TYPE.pop(cls.get_node_type(), None)


async def _run(node_type: str, **kwargs):
    return await execute_node(node_type=node_type, fields={}, secrets={}, input_blobs={}, **kwargs)


@pytest.mark.asyncio
async def test_dict_result_of_single_output_node_is_not_split():
    result = await _run("test.slots.Classifier")
    assert result["outputs"] == {"output": {"cat": 0.9, "dog": 0.1}}
    assert result["blobs"] == {}


@pytest.mark.asyncio
async def test_list_of_refs_carries_bytes_inline():
    result = await _run("test.slots.AudioList")
    assert set(result["outputs"]) == {"output"}
    assert result["blobs"] == {}
    items = result["outputs"]["output"]
    assert [item["data"] for item in items] == [b"one", b"two"]
    assert all(item["uri"] == "" and item["type"] == "audio" for item in items)


@pytest.mark.asyncio
async def test_typed_dict_slots_keep_top_level_blobs_and_inline_nested_ones():
    result = await _run("test.slots.Nested")
    assert set(result["outputs"]) == {"image", "masks", "info"}
    assert result["blobs"] == {"image": b"top"}
    assert result["outputs"]["image"]["uri"].startswith("blob://")
    assert result["outputs"]["masks"][0]["data"] == b"m1"
    assert result["outputs"]["info"]["preview"]["data"] == b"pv"


@pytest.mark.asyncio
async def test_streaming_chunk_inlines_nested_refs():
    chunks = []

    async def emit_chunk(chunk):
        chunks.append(chunk)

    await _run("test.slots.StreamingList", emit_chunk=emit_chunk)
    assert len(chunks) == 1
    assert chunks[0]["blobs"] == {}
    assert chunks[0]["outputs"]["frames"][0]["data"] == b"f1"
    assert chunks[0]["outputs"]["frames"][0]["uri"] == ""
