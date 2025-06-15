"""Test script for the GraphPlanner"""

import asyncio
import tempfile
import os
from pathlib import Path
import json

from nodetool.agents.graph_planner import GraphPlanner
from nodetool.chat.providers.openai_provider import OpenAIProvider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.workflow_runner import WorkflowRunner
from nodetool.workflows.types import PlanningUpdate
from nodetool.types.graph import Graph as APIGraph

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_graph_to_file(graph: APIGraph, workspace_dir: str, filename: str = "test_graph.json"):
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
        input_file.write_text("This is a test document about artificial intelligence and machine learning.")
        
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
            workspace_dir=workspace_dir,
            inputs=["input_file"],  # Changed from input_files
            verbose=True,
        )
        
        # Create processing context
        context = ProcessingContext(
            workspace_dir=workspace_dir,
            user_id="test_user",
            auth_token="test_token"
        )
        
        # Plan the graph
        logger.info("Creating workflow graph...")
        updates = []
        analysis_complete = False
        async for update in planner.create_graph(context):
            if isinstance(update, PlanningUpdate):
                logger.info(f"Planning update - Phase: {update.phase}, Status: {update.status}")
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
            logger.info("\nNote: To run this workflow, you would need the proper node implementations.")
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
            workspace_dir=workspace_dir,
            inputs=[],  # No inputs needed for this task
            verbose=True,
        )
        
        # Create processing context
        context = ProcessingContext(
            workspace_dir=workspace_dir,
            user_id="test_user",
            auth_token="test_token"
        )
        
        # Plan the graph
        logger.info("Creating simple workflow graph...")
        async for update in planner.create_graph(context):
            if isinstance(update, PlanningUpdate):
                logger.info(f"Planning update - Phase: {update.phase}, Status: {update.status}")
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
            workspace_dir=workspace_dir,
            inputs=["number1", "number2"],
            verbose=True,
        )
        
        # Create processing context
        context = ProcessingContext(
            workspace_dir=workspace_dir,
            user_id="test_user",
            auth_token="test_token"
        )
        
        # Plan the graph
        logger.info("Creating math workflow graph...")
        async for update in planner.create_graph(context):
            if isinstance(update, PlanningUpdate):
                logger.info(f"Planning update - Phase: {update.phase}, Status: {update.status}")
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
                if hasattr(node, 'data') and node.data:
                    logger.info(f"    Data: {node.data}")
            
            for edge in planner.graph.edges:
                logger.info(f"  Edge: {edge.source} ({edge.sourceHandle}) -> {edge.target} ({edge.targetHandle})")
            
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
            workspace_dir=workspace_dir,
            inputs=["name"],
            verbose=True,
        )
        
        # Create processing context
        context = ProcessingContext(
            workspace_dir=workspace_dir,
            user_id="test_user",
            auth_token="test_token"
        )
        
        # Plan the graph
        logger.info("Creating greeting workflow graph...")
        async for update in planner.create_graph(context):
            if isinstance(update, PlanningUpdate):
                logger.info(f"Planning update - Phase: {update.phase}, Status: {update.status}")
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


if __name__ == "__main__":
    print("GraphPlanner Test Script")
    print("=" * 50)
    print("\nSelect test to run:")
    print("1. Complex workflow with file analysis")
    print("2. Simple haiku generation")
    print("3. Math calculation workflow")
    print("4. Personalized greeting workflow")
    
    choice = input("\nEnter choice (1-4): ")
    
    if choice == "1":
        asyncio.run(test_graph_planner())
    elif choice == "2":
        asyncio.run(test_simple_graph())
    elif choice == "3":
        asyncio.run(test_math_workflow())
    elif choice == "4":
        asyncio.run(test_greeting_graph())
    else:
        print("Invalid choice")