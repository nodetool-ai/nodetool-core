"""Direct node execution: define nodes and call ``process`` with a ProcessingContext."""

import asyncio
import tempfile

from nodetool.runtime.resources import ResourceScope
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext


class Add(BaseNode):
    """
    Add two numbers.
    math, add, sum
    """

    a: float = 0.0
    b: float = 0.0

    async def process(self, context: ProcessingContext) -> float:
        return self.a + self.b


class Multiply(BaseNode):
    """
    Multiply two numbers.
    math, multiply, product
    """

    a: float = 0.0
    b: float = 0.0

    async def process(self, context: ProcessingContext) -> float:
        return self.a * self.b


async def main() -> None:
    async with ResourceScope():
        with tempfile.TemporaryDirectory() as workspace:
            context = ProcessingContext(workspace_dir=workspace, user_id="example_user")

            total = await Add(a=5.0, b=3.0).process(context)
            print(f"5 + 3 = {total}")

            product = await Multiply(a=total, b=2.0).process(context)
            print(f"(5 + 3) * 2 = {product}")


if __name__ == "__main__":
    asyncio.run(main())
