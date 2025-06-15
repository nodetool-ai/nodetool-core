"""Integration example for GraphPlanner with the Agent system.

For simpler test examples, see graph_planner_simple_tests.py
"""

import asyncio
from io import StringIO
import tempfile
from typing import Any

from nodetool.agents.graph_planner import GraphInput, GraphOutput, GraphPlanner
from nodetool.chat.providers.openai_provider import OpenAIProvider
from nodetool.metadata.type_metadata import TypeMetadata
from nodetool.metadata.types import ColumnDef, DataframeRef
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.types import Chunk, PlanningUpdate

# Set up logging
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    context = ProcessingContext(
        workspace_dir=workspace, user_id="workflow_test_user", auth_token="local_token"
    )

    async for update in graph_planner.create_graph(context):
        if isinstance(update, PlanningUpdate):
            logger.info(f"Planning: {update.phase} - {update.status}")
        elif isinstance(update, Chunk):
            logger.debug(f"Received chunk: {update.content}")

    if not graph_planner.graph:
        raise ValueError("Failed to create workflow graph")

    graph = graph_planner.graph
    logger.info(f"Generated workflow has {len(graph.nodes)} nodes")

    req = RunJobRequest(
        graph=graph,
        params=inputs,
    )

    logger.info(f"Executing workflow")
    async for msg in run_workflow(req, context=context, use_thread=False):
        logger.info(f"Workflow message: {msg}")


async def example_data_processing_workflow():
    """Example: Create a data processing workflow.

    This is a more complex example using CSV data analysis.
    For simpler examples, see graph_planner_simple_tests.py
    """

    with tempfile.TemporaryDirectory() as workspace:
        sales_data = DataframeRef(
            columns=[
                ColumnDef(name="date", data_type="datetime"),
                ColumnDef(name="product", data_type="string"),
                ColumnDef(name="sales", data_type="int"),
                ColumnDef(name="region", data_type="string"),
            ],
            data=[
                ["2024-01-01", "Widget A", 100, "North"],
                ["2024-01-01", "Widget B", 150, "South"],
                ["2024-01-02", "Widget A", 120, "North"],
                ["2024-01-02", "Widget B", 180, "South"],
                ["2024-01-03", "Widget A", 90, "East"],
                ["2024-01-03", "Widget B", 200, "West"],
            ],
        )

        provider = OpenAIProvider()

        # Plan a data analysis workflow
        objective = """
        Create a workflow to analyze the sales data:
        - Read the sales data from the CSV input
        - Calculate the total sales by region
        - Output a structured report 
        """

        try:
            await create_and_execute_workflow(
                provider=provider,
                model="gpt-4o-mini",
                workspace=workspace,
                objective=objective,
                inputs={
                    "sales_data": sales_data,
                },
                input_schema=[
                    GraphInput(
                        name="sales_data",
                        type=TypeMetadata(type="dataframe"),
                        description="The sales data to analyze",
                    )
                ],
                output_schema=[
                    GraphOutput(
                        name="sales_report",
                        type=TypeMetadata(type="dataframe"),
                        description="The sales report",
                    )
                ],
            )
        except Exception as e:
            logger.error(f"Workflow failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(example_data_processing_workflow())
