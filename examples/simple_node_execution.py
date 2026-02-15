"""Direct node execution examples - the simplest way to use NodeTool nodes"""

import asyncio
import tempfile

from nodetool.config.logging_config import get_logger
from nodetool.runtime.resources import ResourceScope
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.test_nodes import Add, Multiply

logger = get_logger(__name__)


async def simple_math_example():
    """Direct execution of math nodes"""

    with tempfile.TemporaryDirectory() as workspace:
        context = ProcessingContext(workspace_dir=workspace, user_id="test_user", auth_token="test_token")

        # Create an Add node
        add_node = Add(a=5.0, b=3.0)
        result = await add_node.process(context)
        logger.info(f"5 + 3 = {result}")

        # Create a Multiply node
        multiply_node = Multiply(a=result, b=2.0)
        final_result = await multiply_node.process(context)
        logger.info(f"(5 + 3) * 2 = {final_result}")


async def calculation_chain_example():
    """Chain multiple calculations together"""

    with tempfile.TemporaryDirectory() as workspace:
        context = ProcessingContext(workspace_dir=workspace, user_id="test_user", auth_token="test_token")

        # (10 + 5) * 3
        step1 = Add(a=10.0, b=5.0)
        sum_result = await step1.process(context)
        logger.info(f"Step 1: 10 + 5 = {sum_result}")

        step2 = Multiply(a=sum_result, b=3.0)
        final_result = await step2.process(context)
        logger.info(f"Step 2: {sum_result} * 3 = {final_result}")


async def run_examples():
    """Run all examples"""

    print("\n=== Simple Math Example ===")
    await simple_math_example()

    print("\n=== Calculation Chain Example ===")
    await calculation_chain_example()

    print("\n\nThese examples show direct node execution.")
    print("For workflow-based execution using GraphPlanner, see:")
    print("- graph_planner_simple_tests.py (simple workflows)")
    print("- graph_planner_integration.py (complex workflows)")


async def main():
    async with ResourceScope():
        await run_examples()


if __name__ == "__main__":
    asyncio.run(main())
