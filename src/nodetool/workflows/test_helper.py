from typing import Any, AsyncGenerator, TypedDict

from nodetool.workflows.base_node import InputNode, OutputNode


class StringInput(InputNode):
    """Test-only string input node for workflow testing."""

    default: str = ""

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.workflows.test_helper.StringInput"

    async def process(self, context: Any) -> str:
        return self.value


class FloatInput(InputNode):
    """Test-only float input node for workflow testing."""

    default: float = 0.0

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.workflows.test_helper.FloatInput"

    async def process(self, context: Any) -> float:
        return self.value


class IntInput(InputNode):
    """Test-only int input node for workflow testing."""

    default: int = 0

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.workflows.test_helper.IntInput"

    async def process(self, context: Any) -> int:
        return self.value


class StringOutput(OutputNode):
    """Test-only string output node for workflow testing."""

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.workflows.test_helper.StringOutput"
