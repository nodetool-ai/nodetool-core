from typing import Any

from pydantic import Field

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
