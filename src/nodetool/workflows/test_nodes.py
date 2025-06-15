from nodetool.workflows.base_node import BaseNode, InputNode, OutputNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_workflow import log
from pydantic import Field


class NumberInput(InputNode):
    """Input node that provides a number."""

    value: float = Field(default=10.0, description="The input number")

    async def process(self, context: ProcessingContext) -> float:
        return self.value


class Multiply(BaseNode):
    """Multiplies two numbers."""

    a: float = Field(default=1.0, description="First number")
    b: float = Field(default=2.0, description="Second number")

    async def process(self, context: ProcessingContext) -> float:
        result = self.a * self.b
        log.info(f"Multiplying {self.a} * {self.b} = {result}")
        return result


class Add(BaseNode):
    """Adds two numbers."""

    a: float = Field(default=0.0, description="First number")
    b: float = Field(default=0.0, description="Second number")

    async def process(self, context: ProcessingContext) -> float:
        result = self.a + self.b
        log.info(f"Adding {self.a} + {self.b} = {result}")
        return result


class NumberOutput(OutputNode):
    """Output node that displays a number."""

    value: float = Field(default=0.0, description="The output number")

    async def process(self, context: ProcessingContext) -> float:
        log.info(f"Output value: {self.value}")
        return self.value
