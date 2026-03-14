import asyncio
import pytest
from pydantic import Field

from nodetool.workflows.base_node import BaseNode, NODE_BY_TYPE
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import ImageRef
from nodetool.worker.executor import execute_node


class EchoNode(BaseNode):
    """Echo the input text."""
    text: str = Field(default="")

    @classmethod
    def get_node_type(cls) -> str:
        return "test.EchoNode"

    async def process(self, context: ProcessingContext) -> str:
        return self.text


@pytest.fixture(autouse=True)
def register_echo():
    NODE_BY_TYPE["test.EchoNode"] = EchoNode
    yield
    NODE_BY_TYPE.pop("test.EchoNode", None)


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
