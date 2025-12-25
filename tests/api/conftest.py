"""Fixtures for API tests."""

from unittest.mock import patch

import pytest

from nodetool.metadata.node_metadata import NodeMetadata, PackageModel
from nodetool.metadata.type_metadata import TypeMetadata
from nodetool.metadata.types import OutputSlot
from nodetool.packages.registry import Registry
from nodetool.workflows.base_node import BaseNode, InputNode, add_node_type
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.property import Property


# Define test node classes
class IntegerInput(InputNode):
    """Test integer input node."""
    value: int = 0

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.input.IntegerInput"

    async def process(self, context: ProcessingContext) -> int:
        return self.value


class IntegerOutput(BaseNode):
    """Test integer output node."""
    value: int = 0

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.output.IntegerOutput"

    async def process(self, context: ProcessingContext) -> None:
        # Output nodes don't return values
        pass


class Concat(BaseNode):
    """Test text concatenation node."""
    a: str = ""
    b: str = ""

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.text.Concat"

    async def process(self, context: ProcessingContext) -> str:
        return self.a + self.b


# Register test node classes
add_node_type(IntegerInput)
add_node_type(IntegerOutput)
add_node_type(Concat)


@pytest.fixture(autouse=True)
def mock_registry_with_test_nodes():
    """
    Mock the Registry to return test node metadata for MCP server tests.

    This fixture automatically patches the Registry to include test nodes
    that are referenced in the MCP server tests.
    """
    # Create test node metadata
    test_nodes = [
        NodeMetadata(
            title="Integer Input",
            description="An input node that provides an integer value",
            namespace="nodetool.input",
            node_type="nodetool.input.IntegerInput",
            properties=[
                Property(
                    name="name",
                    type=TypeMetadata(type="str"),
                    description="Name of the input",
                    default="value",
                ),
                Property(
                    name="value",
                    type=TypeMetadata(type="int"),
                    description="Integer value",
                    default=0,
                ),
            ],
            outputs=[
                OutputSlot(
                    name="output",
                    type=TypeMetadata(type="int"),
                )
            ],
        ),
        NodeMetadata(
            title="Integer Output",
            description="An output node that displays an integer value",
            namespace="nodetool.output",
            node_type="nodetool.output.IntegerOutput",
            properties=[
                Property(
                    name="name",
                    type=TypeMetadata(type="str"),
                    description="Name of the output",
                    default="result",
                ),
                Property(
                    name="value",
                    type=TypeMetadata(type="int"),
                    description="Integer value to output",
                    default=0,
                ),
            ],
            outputs=[],
        ),
        NodeMetadata(
            title="Concat",
            description="Concatenate two text strings",
            namespace="nodetool.text",
            node_type="nodetool.text.Concat",
            properties=[
                Property(
                    name="a",
                    type=TypeMetadata(type="str"),
                    description="First string",
                    default="",
                ),
                Property(
                    name="b",
                    type=TypeMetadata(type="str"),
                    description="Second string",
                    default="",
                ),
            ],
            outputs=[
                OutputSlot(
                    name="output",
                    type=TypeMetadata(type="str"),
                )
            ],
        ),
    ]

    # Create a test package with these nodes
    test_package = PackageModel(
        name="nodetool-test-nodes",
        description="Test nodes for unit testing",
        version="0.0.1",
        authors=["Test Author"],
        repo_id="test/test-nodes",
        nodes=test_nodes,
        namespaces=["nodetool.input", "nodetool.output", "nodetool.text"],
    )

    # Patch the Registry methods to return our test data
    def mock_get_all_installed_nodes(self):
        # Return test nodes
        return test_nodes

    def mock_find_node_by_type(self, node_type: str):
        # Find node by type from test nodes
        for node in test_nodes:
            if node.node_type == node_type:
                node_dict = node.model_dump()
                node_dict["package"] = "nodetool-test-nodes"
                node_dict["installed"] = True
                return node_dict
        return None

    def mock_list_installed_packages(self):
        # Return test package
        return [test_package]

    with patch.object(
        Registry, "get_all_installed_nodes", mock_get_all_installed_nodes
    ), patch.object(
        Registry, "find_node_by_type", mock_find_node_by_type
    ), patch.object(
        Registry, "list_installed_packages", mock_list_installed_packages
    ):
        yield
