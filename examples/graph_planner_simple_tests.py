"""Simple integration examples for GraphPlanner with the Agent system"""

import asyncio
from typing import Any

from nodetool.agents.graph_planner import (
    GraphPlanner,
    print_visual_graph,
)
from nodetool.providers.huggingface_provider import HuggingFaceProvider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.types import Chunk, PlanningUpdate

# Set up logging
from nodetool.config.logging_config import get_logger

logger = get_logger(__name__)

provider = HuggingFaceProvider("cerebras")
model = "openai/gpt-oss-120b"


async def create_and_execute_workflow(
    objective: str,
    inputs: dict[str, Any],
):
    """Create and execute a workflow graph for the given objective"""

    # Create GraphPlanner
    graph_planner = GraphPlanner(
        provider=provider,
        model=model,
        objective=objective,
        verbose=True,
    )

    # Plan the graph
    logger.info(f"Planning workflow for: {objective}")
    context = ProcessingContext()

    async for update in graph_planner.create_graph(context):
        if isinstance(update, PlanningUpdate):
            logger.info(f"Planning: {update.phase} - {update.status}")
        elif isinstance(update, Chunk):
            logger.debug(f"Received chunk: {update.content}")

    if not graph_planner.graph:
        raise ValueError("Failed to create workflow graph")

    print_visual_graph(graph_planner.graph)
    graph = graph_planner.graph
    logger.info(f"Generated workflow has {len(graph.nodes)} nodes")

    req = RunJobRequest(
        graph=graph,
        params=inputs,
    )

    logger.info("Executing workflow")
    async for msg in run_workflow(req, context=context):
        logger.info(f"Workflow message: {msg}")


async def simple_arithmetic_workflow():
    """Example: Create a simple arithmetic workflow"""

    # Plan a simple addition workflow
    objective = """
    Create a workflow to add two numbers:
    - Take two numeric inputs: a and b
    - Add them together
    - Output the result
    """

    try:
        await create_and_execute_workflow(
            objective=objective,
            inputs={"a": 1, "b": 2},
        )
    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)


async def text_concatenation_workflow():
    """Example: Create a text concatenation workflow"""

    objective = """
    Create a workflow to concatenate two text strings:
    - Take two text inputs: first_name and last_name
    - Concatenate them with a space in between
    - Output the full name
    """

    try:
        await create_and_execute_workflow(
            objective=objective,
            inputs={"first_name": "John", "last_name": "Doe"},
        )
    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)


async def calculation_pipeline_workflow():
    """Example: Create a multi-step calculation workflow"""

    objective = """
    Create a workflow to perform the calculation: (a + b) * c
    - Take three numeric inputs: a, b, and c
    - First add a and b together
    - Then multiply the result by c
    - Output the final result
    """

    try:
        await create_and_execute_workflow(
            objective=objective,
            inputs={"a": 1, "b": 2, "c": 3},
        )
    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)


async def text_formatting_workflow():
    """Example: Create a text formatting workflow"""

    objective = """
    Create a workflow to format a greeting message:
    - Take a name input
    - Create a greeting message using the format: "Hello, {name}! Welcome to NodeTool."
    - Output the formatted message
    """

    try:
        await create_and_execute_workflow(
            objective=objective,
            inputs={"name": "John"},
        )
    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)


async def run_all_simple_tests():
    """Run all simple test workflows"""

    print("\n=== Running Simple Arithmetic Workflow ===")
    await simple_arithmetic_workflow()

    # print("\n=== Running Text Concatenation Workflow ===")
    # await text_concatenation_workflow()

    # print("\n=== Running Calculation Pipeline Workflow ===")
    # await calculation_pipeline_workflow()

    # print("\n=== Running Text Formatting Workflow ===")
    # await text_formatting_workflow()


if __name__ == "__main__":
    # You can run a specific test or all tests
    # asyncio.run(simple_arithmetic_workflow())
    asyncio.run(run_all_simple_tests())
