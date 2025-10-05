"""Test script for the GraphPlanner"""

import asyncio
import tempfile
import os
from pathlib import Path
import json

from nodetool.agents.graph_planner import GraphPlanner, GraphInput, GraphOutput
from nodetool.providers.openai_provider import OpenAIProvider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import PlanningUpdate
from nodetool.types.graph import Graph as APIGraph
from nodetool.metadata.types import TypeMetadata

# Set up logging
from nodetool.config.logging_config import get_logger

logger = get_logger(__name__)


def save_graph_to_file(
    graph: APIGraph, workspace_dir: str, filename: str = "test_graph.json"
):
    """Save the generated graph to a JSON file for inspection"""
    graph_dict = {
        "nodes": [node.model_dump() for node in graph.nodes],
        "edges": [edge.model_dump() for edge in graph.edges],
    }

    output_path = Path(workspace_dir) / filename
    with open(output_path, "w") as f:
        json.dump(graph_dict, f, indent=2)

    logger.info(f"Graph saved to: {output_path}")
    return output_path


async def test_graph_planner():
    """Test the GraphPlanner with a file analysis objective"""

    # Create a temporary workspace
    with tempfile.TemporaryDirectory() as workspace_dir:
        logger.info(f"Using workspace: {workspace_dir}")

        # Create test input file
        input_file = Path(workspace_dir) / "input.txt"
        input_file.write_text(
            "This is a test document about artificial intelligence and machine learning."
        )

        # Initialize provider (requires OPENAI_API_KEY environment variable)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("Please set OPENAI_API_KEY environment variable")
            return

        provider = OpenAIProvider(api_key=api_key)

        # Create GraphPlanner
        planner = GraphPlanner(
            provider=provider,
            model="gpt-4o-mini",
            objective="Analyze the input text file to extract key topics, then generate a summary and a list of questions about the content. The workflow should take 'input_file' as input and output a JSON object with 'summary' and 'questions' fields.",
            inputs={"input_file": str(input_file)},  # Provide actual input values
            input_schema=[
                GraphInput(
                    name="input_file",
                    type=TypeMetadata(type="str"),
                    description="Path to input text file",
                )
            ],
            output_schema=[
                GraphOutput(
                    name="analysis_result",
                    type=TypeMetadata(type="dict"),
                    description="JSON object with 'summary' and 'questions' fields",
                )
            ],
            verbose=True,
        )

        # Create processing context
        context = ProcessingContext(
            user_id="test_user", auth_token="test_token", workspace_dir=workspace_dir
        )

        # Plan the graph
        logger.info("Creating workflow graph...")
        updates = []
        analysis_complete = False
        async for update in planner.create_graph(context):
            if isinstance(update, PlanningUpdate):
                logger.info(
                    f"Planning update - Phase: {update.phase}, Status: {update.status}"
                )
                updates.append(update)
                # Log when analysis phase completes
                if update.phase == "Analysis" and update.status == "Success":
                    analysis_complete = True
                    logger.info("Analysis phase completed - workflow design is ready")

        if planner.graph:
            logger.info(f"Graph created successfully!")
            logger.info(f"Nodes: {len(planner.graph.nodes)}")
            for node in planner.graph.nodes:
                logger.info(f"  - {node.type} ({node.id})")
            logger.info(f"Edges: {len(planner.graph.edges)}")
            for edge in planner.graph.edges:
                logger.info(f"  - {edge.source} -> {edge.target}")

            # Show the graph structure visually
            from nodetool.agents.graph_planner import print_visual_graph

            print_visual_graph(planner.graph)

            # Save the graph for inspection
            save_graph_to_file(planner.graph, workspace_dir, "file_analysis_graph.json")

            # Note: Running the workflow would require proper node implementations
            # which may not be available in this test environment
            logger.info(
                "\nNote: To run this workflow, you would need the proper node implementations."
            )
        else:
            logger.error("Failed to create graph")


async def test_simple_graph():
    """Test with a simpler objective"""

    with tempfile.TemporaryDirectory() as workspace_dir:
        logger.info(f"Using workspace: {workspace_dir}")

        # Initialize provider
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("Please set OPENAI_API_KEY environment variable")
            return

        provider = OpenAIProvider(api_key=api_key)

        # Create GraphPlanner for a simple task
        planner = GraphPlanner(
            provider=provider,
            model="gpt-4o-mini",
            objective="Generate a haiku about the seasons. The workflow should output a string containing the haiku poem.",
            inputs={},  # No inputs needed for this task
            input_schema=[],  # No inputs needed
            output_schema=[
                GraphOutput(
                    name="haiku",
                    type=TypeMetadata(type="str"),
                    description="A haiku poem about the seasons",
                )
            ],
            verbose=True,
        )

        # Create processing context
        context = ProcessingContext(
            user_id="test_user", auth_token="test_token", workspace_dir=workspace_dir
        )

        # Plan the graph
        logger.info("Creating simple workflow graph...")
        async for update in planner.create_graph(context):
            if isinstance(update, PlanningUpdate):
                logger.info(
                    f"Planning update - Phase: {update.phase}, Status: {update.status}"
                )
                if update.phase == "Analysis" and update.status == "Success":
                    logger.info("Analysis phase completed - workflow design is ready")

        if planner.graph:
            logger.info(f"\nGraph structure:")
            logger.info(f"Nodes: {[(n.type, n.id) for n in planner.graph.nodes]}")
            logger.info(f"Edges: {[(e.source, e.target) for e in planner.graph.edges]}")

            # Show the graph structure visually
            from nodetool.agents.graph_planner import print_visual_graph

            print_visual_graph(planner.graph)
        else:
            logger.error("Failed to create graph")


async def test_math_workflow():
    """Test with a math calculation workflow to show structured analysis"""

    with tempfile.TemporaryDirectory() as workspace_dir:
        logger.info(f"Using workspace: {workspace_dir}")

        # Initialize provider
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("Please set OPENAI_API_KEY environment variable")
            return

        provider = OpenAIProvider(api_key=api_key)

        # Create GraphPlanner for a math task
        planner = GraphPlanner(
            provider=provider,
            model="gpt-4o-mini",
            objective="Create a workflow that takes two numbers as input, adds them together, and outputs the result with a descriptive message.",
            inputs={"number1": 5, "number2": 3},  # Sample values
            input_schema=[
                GraphInput(
                    name="number1",
                    type=TypeMetadata(type="float"),
                    description="First number to add",
                ),
                GraphInput(
                    name="number2",
                    type=TypeMetadata(type="float"),
                    description="Second number to add",
                ),
            ],
            output_schema=[
                GraphOutput(
                    name="result",
                    type=TypeMetadata(type="str"),
                    description="Result message with the sum",
                )
            ],
            verbose=True,
        )

        # Create processing context
        context = ProcessingContext(
            user_id="test_user", auth_token="test_token", workspace_dir=workspace_dir
        )

        # Plan the graph
        logger.info("Creating math workflow graph...")
        async for update in planner.create_graph(context):
            if isinstance(update, PlanningUpdate):
                logger.info(
                    f"Planning update - Phase: {update.phase}, Status: {update.status}"
                )
                if update.phase == "Analysis" and update.status == "Success":
                    logger.info("Analysis phase completed with structured output")

        if planner.graph:
            logger.info(f"\nGraph created successfully!")
            logger.info(f"Total nodes: {len(planner.graph.nodes)}")
            logger.info(f"Total edges: {len(planner.graph.edges)}")

            logger.info("\nDetailed graph structure:")
            for node in planner.graph.nodes:
                logger.info(f"  Node: {node.id}")
                logger.info(f"    Type: {node.type}")
                if hasattr(node, "data") and node.data:
                    logger.info(f"    Data: {node.data}")

            for edge in planner.graph.edges:
                logger.info(
                    f"  Edge: {edge.source} ({edge.sourceHandle}) -> {edge.target} ({edge.targetHandle})"
                )

            # Show visual graph
            from nodetool.agents.graph_planner import print_visual_graph

            print_visual_graph(planner.graph)

            # Save the graph for inspection
            save_graph_to_file(planner.graph, workspace_dir, "math_workflow_graph.json")
        else:
            logger.error("Failed to create graph")


async def test_greeting_graph():
    """Test with a personalized greeting workflow"""

    with tempfile.TemporaryDirectory() as workspace_dir:
        logger.info(f"Using workspace: {workspace_dir}")

        # Initialize provider
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("Please set OPENAI_API_KEY environment variable")
            return

        provider = OpenAIProvider(api_key=api_key)

        # Create GraphPlanner for a greeting task
        planner = GraphPlanner(
            provider=provider,
            model="gpt-4o-mini",
            objective="Generate a personalized greeting. The workflow should take a 'name' as input and use a template to create a message like 'Hello, [name]! Welcome to the Nodetool demo.'",
            inputs={"name": "Alice"},  # Sample value
            input_schema=[
                GraphInput(
                    name="name",
                    type=TypeMetadata(type="str"),
                    description="Name of the person to greet",
                )
            ],
            output_schema=[
                GraphOutput(
                    name="greeting",
                    type=TypeMetadata(type="str"),
                    description="Personalized greeting message",
                )
            ],
            verbose=True,
        )

        # Create processing context
        context = ProcessingContext(
            user_id="test_user", auth_token="test_token", workspace_dir=workspace_dir
        )

        # Plan the graph
        logger.info("Creating greeting workflow graph...")
        async for update in planner.create_graph(context):
            if isinstance(update, PlanningUpdate):
                logger.info(
                    f"Planning update - Phase: {update.phase}, Status: {update.status}"
                )
                if update.phase == "Analysis" and update.status == "Success":
                    logger.info("Analysis phase completed - workflow design is ready")

        if planner.graph:
            logger.info(f"\nGraph structure:")
            logger.info(f"Nodes: {[(n.type, n.id) for n in planner.graph.nodes]}")
            logger.info(f"Edges: {[(e.source, e.target) for e in planner.graph.edges]}")

            # Show the graph structure visually
            from nodetool.agents.graph_planner import print_visual_graph

            print_visual_graph(planner.graph)
        else:
            logger.error("Failed to create graph")


async def test_data_processing_workflow():
    """Test with a data processing workflow to demonstrate improved edge connections"""

    with tempfile.TemporaryDirectory() as workspace_dir:
        logger.info(f"Using workspace: {workspace_dir}")

        # Initialize provider
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("Please set OPENAI_API_KEY environment variable")
            return

        provider = OpenAIProvider(api_key=api_key)

        # Create GraphPlanner for a data processing task
        planner = GraphPlanner(
            provider=provider,
            model="gpt-4o-mini",
            objective="Process CSV sales data by calculating monthly totals, identifying top products, and generating a summary report with charts",
            inputs={},  # No specific input values needed for planning
            input_schema=[
                GraphInput(
                    name="sales_data",
                    type=TypeMetadata(type="dataframe"),
                    description="CSV sales data containing transaction records",
                )
            ],
            output_schema=[
                GraphOutput(
                    name="monthly_summary",
                    type=TypeMetadata(type="dataframe"),
                    description="Monthly sales totals",
                ),
                GraphOutput(
                    name="summary_report",
                    type=TypeMetadata(type="str"),
                    description="Comprehensive analysis report",
                ),
            ],
            verbose=True,
        )

        # Create processing context
        context = ProcessingContext(
            user_id="test_user", auth_token="test_token", workspace_dir=workspace_dir
        )

        # Plan the graph
        logger.info("Creating data processing workflow graph...")
        async for update in planner.create_graph(context):
            if isinstance(update, PlanningUpdate):
                logger.info(
                    f"Planning update - Phase: {update.phase}, Status: {update.status}"
                )
                if update.phase == "Analysis" and update.status == "Success":
                    logger.info("Analysis phase completed - workflow design is ready")

        if planner.graph:
            logger.info(f"\nGraph created successfully!")
            logger.info(f"Total nodes: {len(planner.graph.nodes)}")
            logger.info(f"Total edges: {len(planner.graph.edges)}")

            # Check edge connectivity - this should be much better with the improved prompt
            logger.info("\n🔍 Edge Connectivity Analysis:")
            edge_count_by_target = {}
            for edge in planner.graph.edges:
                if edge.target not in edge_count_by_target:
                    edge_count_by_target[edge.target] = 0
                edge_count_by_target[edge.target] += 1

            nodes_without_inputs = []
            for node in planner.graph.nodes:
                if (
                    node.type
                    not in [
                        "nodetool.input.DataframeInput",
                        "nodetool.input.StringInput",
                    ]
                    and node.id not in edge_count_by_target
                ):
                    nodes_without_inputs.append(node.id)

            if nodes_without_inputs:
                logger.warning(
                    f"⚠️  Nodes without input connections: {nodes_without_inputs}"
                )
            else:
                logger.info("✅ All processing nodes have input connections!")

            # Show visual graph
            from nodetool.agents.graph_planner import print_visual_graph

            print_visual_graph(planner.graph)

            # Save the graph for inspection
            save_graph_to_file(
                planner.graph, workspace_dir, "data_processing_graph.json"
            )
        else:
            logger.error("Failed to create graph")


if __name__ == "__main__":
    print("GraphPlanner Test Script")
    print("=" * 50)
    print("\nSelect test to run:")
    print("1. Complex workflow with file analysis")
    print("2. Simple haiku generation")
    print("3. Math calculation workflow")
    print("4. Personalized greeting workflow")
    print("5. Data processing workflow (tests edge connections)")

    choice = input("\nEnter choice (1-5): ")

    if choice == "1":
        asyncio.run(test_graph_planner())
    elif choice == "2":
        asyncio.run(test_simple_graph())
    elif choice == "3":
        asyncio.run(test_math_workflow())
    elif choice == "4":
        asyncio.run(test_greeting_graph())
    elif choice == "5":
        asyncio.run(test_data_processing_workflow())
    else:
        print("Invalid choice")
