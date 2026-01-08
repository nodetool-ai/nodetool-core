"""Integration example for GraphPlanner with a Data Transformation pattern.

This example demonstrates how GraphPlanner can design a workflow that imports data,
applies transformations (filtering, grouping), and outputs the result.
"""

import asyncio
import tempfile
from typing import Any

from nodetool.workflows.types import Chunk, PlanningUpdate

from nodetool.agents.graph_planner import GraphInput, GraphOutput, GraphPlanner
from nodetool.config.logging_config import get_logger
from nodetool.metadata.type_metadata import TypeMetadata
from nodetool.metadata.types import Provider
from nodetool.providers import get_provider
from nodetool.runtime.resources import ResourceScope
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow

logger = get_logger(__name__)


async def create_and_execute_workflow(
    provider,
    model: str,
    workspace: str,
    objective: str,
    input_schema: list[GraphInput],
    output_schema: list[GraphOutput],
    inputs: dict[str, Any],
):
    """Create and execute a workflow graph for the given objective"""

    # Create GraphPlanner
    graph_planner = GraphPlanner(
        provider=provider,
        model=model,
        objective=objective,
        verbose=True,
        input_schema=input_schema,
        output_schema=output_schema,
    )

    # Plan the graph
    logger.info(f"Planning workflow for: {objective}")
    context = ProcessingContext(workspace_dir=workspace, user_id="data_pattern_test", auth_token="local_token")

    async for update in graph_planner.create_graph(context):
        if isinstance(update, PlanningUpdate):
            logger.info(f"Planning: {update.phase} - {update.status}")
        elif isinstance(update, Chunk):
            logger.debug(f"Received chunk: {update.content}")

    if not graph_planner.graph:
        raise ValueError("Failed to create workflow graph")

    graph = graph_planner.graph
    logger.info(f"Generated workflow has {len(graph.nodes)} nodes")

    # Simple validation
    node_types = [n.get_node_type() for n in graph.nodes]
    logger.info(f"Node types found: {node_types}")

    req = RunJobRequest(
        graph=graph,
        params=inputs,
    )

    logger.info("Executing workflow")
    async for msg in run_workflow(req, context=context, use_thread=False):
        logger.info(f"Workflow message: {msg}")


async def example_data_workflow():
    """Example: Create a data transformation workflow."""

    with tempfile.TemporaryDirectory() as workspace:
        # Use OpenAI as the provider
        provider = await get_provider(Provider.HuggingFaceCerebras)
        model = "openai/gpt-oss-120b"

        # Plan a data transformation workflow
        objective = """
        I have a CSV file containing sales data.
        1. Parse the CSV file into a dataframe.
        2. Filter the dataframe to only include rows where 'region' is 'North'.
        3. Output the filtered dataframe.
        """

        try:
            # Note: We simulate a CSV input by passing a file path string or dummy content
            await create_and_execute_workflow(
                provider=provider,
                model=model,
                workspace=workspace,
                objective=objective,
                inputs={
                    "csv_file": "sales_data.csv",
                },
                input_schema=[
                    GraphInput(
                        name="csv_file",
                        type=TypeMetadata(type="string"),  # DocumentInput/StringInput usually takes a path or content
                        description="Path to the CSV file",
                    )
                ],
                output_schema=[
                    GraphOutput(
                        name="filtered_data",
                        type=TypeMetadata(
                            type="string"
                        ),  # DataframeOutput usually serializes to JSON/string for display
                        description="The filtered sales data for North region",
                    )
                ],
            )
        except Exception as e:
            logger.error(f"Workflow failed: {e}", exc_info=True)


async def main():
    async with ResourceScope():
        await example_data_workflow()


if __name__ == "__main__":
    asyncio.run(main())
