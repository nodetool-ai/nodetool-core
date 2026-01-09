"""Integration example for GraphPlanner with a Preview Node pattern.

This example demonstrates how GraphPlanner can designed to include intermediate Preview nodes
to improve workflow visibility for the user.
"""

import asyncio
import tempfile
from typing import Any

from nodetool.agents.graph_planner import GraphInput, GraphOutput, GraphPlanner
from nodetool.config.logging_config import get_logger
from nodetool.metadata.type_metadata import TypeMetadata
from nodetool.metadata.types import Provider
from nodetool.providers import get_provider
from nodetool.runtime.resources import ResourceScope
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.types import Chunk, PlanningUpdate

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
    context = ProcessingContext(workspace_dir=workspace, user_id="preview_pattern_test", auth_token="local_token")

    async for update in graph_planner.create_graph(context):
        if isinstance(update, PlanningUpdate):
            logger.info(f"Planning: {update.phase} - {update.status}")
        elif isinstance(update, Chunk):
            logger.debug(f"Received chunk: {update.content}")

    if not graph_planner.graph:
        raise ValueError("Failed to create workflow graph")

    graph = graph_planner.graph
    logger.info(f"Generated workflow has {len(graph.nodes)} nodes")

    # Validation: Check for Preview nodes
    preview_nodes = [n for n in graph.nodes if "Preview" in n.get_node_type()]
    logger.info(f"Preview nodes found: {len(preview_nodes)}")
    for node in preview_nodes:
        logger.info(f"Preview node: {node.node_id} ({node.get_node_type()})")

    req = RunJobRequest(
        graph=graph,
        params=inputs,
    )

    logger.info("Executing workflow")
    async for msg in run_workflow(req, context=context, use_thread=False):
        logger.info(f"Workflow message: {msg}")


async def example_preview_workflow():
    """Example: Create a workflow with explicit visibility requirements."""

    with tempfile.TemporaryDirectory() as workspace:
        # Use OpenAI as the provider
        provider = await get_provider(Provider.HuggingFaceCerebras)
        model = "openai/gpt-oss-120b"

        # Plan a workflow that explicitly asks for visibility/previews
        objective = """
        Generate a workflow that fetches a webpage and summarizes it.
        Crucially, I want to SEE the intermediate HTML content before it is summarized.
        Please add a Preview node to show the fetched HTML.
        """

        try:
            await create_and_execute_workflow(
                provider=provider,
                model=model,
                workspace=workspace,
                objective=objective,
                inputs={
                    "url": "https://example.com",
                },
                input_schema=[
                    GraphInput(
                        name="url",
                        type=TypeMetadata(type="string"),
                        description="The URL to fetch",
                    )
                ],
                output_schema=[
                    GraphOutput(
                        name="summary",
                        type=TypeMetadata(type="string"),
                        description="The summary of the webpage",
                    )
                ],
            )
        except Exception as e:
            logger.error(f"Workflow failed: {e}", exc_info=True)


async def main():
    async with ResourceScope():
        await example_preview_workflow()


if __name__ == "__main__":
    asyncio.run(main())
