from typing import Any

from pydantic import Field

from nodetool.dsl.graph import GraphNode, SingleOutputGraphNode
from nodetool.dsl.handles import Connect, OutputHandle, connect_field
from nodetool.workflows.base_node import BaseNode, InputNode, OutputNode


class StringInput(InputNode):
    """Test-only string input node for workflow testing."""

    value: str = ""

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
        return self.default


class IntInput(InputNode):
    """Test-only int input node for workflow testing."""

    default: int = 0

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.workflows.test_helper.IntInput"

    async def process(self, context: Any) -> int:
        return self.default


class StringOutput(OutputNode):
    """Test-only string output node for workflow testing."""

    value: str = ""

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.workflows.test_helper.StringOutput"


class FormatText(BaseNode):
    """Test-only format text node for workflow testing."""

    template: str = Field(default="Hello, {{ text }}", description="The template string with placeholders")
    text: str = Field(default="", description="The text value to insert into the template")

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.workflows.test_helper.FormatText"

    async def process(self, context: Any) -> str:
        result = self.template
        result = result.replace("{{ text }}", str(self.text))
        return result


# ------------------------------------------------------------------
# DSL Graph Node Wrappers for Testing
# ------------------------------------------------------------------


class StringInputDSL(SingleOutputGraphNode[str], GraphNode[str]):
    """DSL wrapper for StringInput node."""

    name: Connect[str] = connect_field(default="input", description="The parameter name for the workflow")
    value: Connect[str] = connect_field(default="", description="The input string value")

    @classmethod
    def get_node_class(cls) -> type[BaseNode]:
        return StringInput


class FloatInputDSL(SingleOutputGraphNode[float], GraphNode[float]):
    """DSL wrapper for FloatInput node."""

    name: Connect[str] = connect_field(default="float_input", description="The parameter name for the workflow")
    default: Connect[float] = connect_field(default=0.0, description="The input float value")

    @classmethod
    def get_node_class(cls) -> type[BaseNode]:
        return FloatInput


class IntInputDSL(SingleOutputGraphNode[int], GraphNode[int]):
    """DSL wrapper for IntInput node."""

    name: Connect[str] = connect_field(default="int_input", description="The parameter name for the workflow")
    default: Connect[int] = connect_field(default=0, description="The input int value")

    @classmethod
    def get_node_class(cls) -> type[BaseNode]:
        return IntInput


class FormatTextDSL(SingleOutputGraphNode[str], GraphNode[str]):
    """DSL wrapper for FormatText node."""

    template: Connect[str] = connect_field(
        default="Hello, {{ text }}", description="The template string with placeholders"
    )
    text: Connect[str] = connect_field(default="", description="The text value to insert into the template")

    @classmethod
    def get_node_class(cls) -> type[BaseNode]:
        return FormatText
