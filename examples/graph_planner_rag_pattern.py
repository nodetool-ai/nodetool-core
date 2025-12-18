"""Integration example for GraphPlanner with a RAG (Search-Format-Agent) pattern.

This example demonstrates how GraphPlanner can design a Retrieval-Augmented Generation
workflow that searches for information, formats it, and uses an agent to answer a question.
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
    context = ProcessingContext(
        workspace_dir=workspace, user_id="rag_pattern_test", auth_token="local_token"
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
    
    # Simple validation that RAG components are likely present
    node_types = [n.get_node_type() for n in graph.nodes]
    logger.info(f"Node types found: {node_types}")
    
    req = RunJobRequest(
        graph=graph,
        params=inputs,
    )

    logger.info("Executing workflow")
    async for msg in run_workflow(req, context=context, use_thread=False):
        logger.info(f"Workflow message: {msg}")


async def example_rag_workflow():
    """Example: Create a specific RAG workflow."""

    with tempfile.TemporaryDirectory() as workspace:
        # Use OpenAI as the provider
        provider = await get_provider(Provider.HuggingFaceCerebras)
        model = "openai/gpt-oss-120b"

        # Plan a standard RAG workflow
        # The prompt explicitly asks for the pattern: Search -> Format -> Agent
        objective = """
        I need to answer a question about a specific topic using external information.
        1. Search the web for information about the query.
        2. Format the search results into a clean text block.
        3. Use an AI agent to answer the original question using the formatted text as context.
        """

        try:
            await create_and_execute_workflow(
                provider=provider,
                model=model,
                workspace=workspace,
                objective=objective,
                inputs={
                    "query": "What are the latest features in Python 3.13?",
                },
                input_schema=[
                    GraphInput(
                        name="query",
                        type=TypeMetadata(type="string"),
                        description="The question to answer",
                    )
                ],
                output_schema=[
                    GraphOutput(
                        name="answer",
                        type=TypeMetadata(type="string"),
                        description="The answer based on retrieved context",
                    )
                ],
            )
        except Exception as e:
            logger.error(f"Workflow failed: {e}", exc_info=True)


async def main():
    async with ResourceScope():
        await example_rag_workflow()


if __name__ == "__main__":
    asyncio.run(main())
