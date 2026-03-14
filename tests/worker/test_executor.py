import pytest
from pydantic import Field

from nodetool.metadata.types import ImageRef
from nodetool.worker.executor import execute_node
from nodetool.workflows.base_node import NODE_BY_TYPE, BaseNode
from nodetool.workflows.processing_context import ProcessingContext


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


@pytest.fixture(autouse=True)
def register_echo():
    NODE_BY_TYPE["test.EchoNode"] = EchoNode
    NODE_BY_TYPE["test.ImageListNode"] = ImageListNode
    yield
    NODE_BY_TYPE.pop("test.EchoNode", None)
    NODE_BY_TYPE.pop("test.ImageListNode", None)


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
    output = result["outputs"]["output"]
    assert output["count"] == 1
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
    output = result["outputs"]["output"]
    assert output["count"] == 1
    assert output["first_is_empty"] is False
    assert output["first_uri"] == "file://example.png"
    assert result["blobs"] == {}
