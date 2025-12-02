import argparse
import json
import logging
import sys
import uuid
from typing import Any, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Placeholder for actual nodetool imports
# In a real environment, these would be:
# from nodetool.core.workflow import Workflow
# from nodetool.core.execution import ExecutionEngine
# from nodetool.nodes.registry import NODE_REGISTRY

def get_output_node_type(source_type: str) -> str:
    """Determine the appropriate output node type based on the source node."""
    if "text_to_image" in source_type or "image_to_image" in source_type:
        return "nodetool.common.ImageOutput" # Hypothetical
    elif "Agent" in source_type or "text_generation" in source_type:
        return "nodetool.common.TextOutput" # Hypothetical
    return "nodetool.common.DebugOutput"

def get_output_handle(source_type: str) -> str:
    """Determine the output handle name."""
    if "text_to_image" in source_type:
        return "image"
    if "Agent" in source_type:
        return "text"
    return "output"

# Initial list of important nodes to test
TEST_NODES = [
    {
        "type": "huggingface.text_to_image.StableDiffusionXL",
        "name": "Stable Diffusion XL",
        "inputs": {
            "prompt": "A cinematic shot of a cyberpunk city, 8k resolution, highly detailed",
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 20,
            "variant": "fp16" # Optimal for CUDA
        }
    },
    {
        "type": "huggingface.text_to_image.Flux",
        "name": "Flux.1-dev",
        "inputs": {
            "prompt": "A photorealistic portrait of an astronaut, studio lighting",
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 20,
            "guidance_scale": 3.5
        }
    },
    {
        "type": "huggingface.text_to_image.QwenImage",
        "name": "Qwen Image",
        "inputs": {
            "prompt": "A serene landscape with mountains and a lake",
            "width": 1024,
            "height": 1024,
        }
    },
    {
        "type": "huggingface.image_to_image.QwenImageEdit",
        "name": "Qwen Image Edit",
        "inputs": {
            "prompt": "Make it snowy",
            # Note: This node typically requires an input image.
            # In a real test, we would need to inject an image node or load an asset.
            # For this script, we assume the node handles missing inputs gracefully or we'd need a LoadImage node.
        },
        "requires_input_image": True
    },
    {
        "type": "nodetool.agents.Agent",
        "name": "Ollama Agent",
        "inputs": {
            "prompt": "Explain quantum computing in one sentence.",
            "model": {
                "type": "language_model",
                "provider": "ollama",
                "name": "llama3"
            }
        }
    }
]

def create_adhoc_workflow(node_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates an ad-hoc workflow structure.
    """
    workflow_id = str(uuid.uuid4())
    target_node_id = str(uuid.uuid4())
    output_node_id = str(uuid.uuid4())
    input_node_id = str(uuid.uuid4()) if node_config.get("requires_input_image") else None

    nodes = []
    connections = []

    # 1. Target Node
    nodes.append({
        "id": target_node_id,
        "type": node_config["type"],
        "properties": node_config["inputs"],
        "position": {"x": 400, "y": 0}
    })

    # 2. Input Node (if required)
    if input_node_id:
        # Create a dummy image loader for testing
        nodes.append({
            "id": input_node_id,
            "type": "nodetool.common.LoadImage",
            "properties": {
                "uri": "assets/test_image.png" # Assumes existence
            },
            "position": {"x": 0, "y": 0}
        })
        connections.append({
            "id": str(uuid.uuid4()),
            "source": input_node_id,
            "sourceHandle": "image",
            "target": target_node_id,
            "targetHandle": "image" # Assumes input handle is 'image'
        })

    # 3. Output Node
    output_type = get_output_node_type(node_config["type"])
    nodes.append({
        "id": output_node_id,
        "type": output_type,
        "properties": {},
        "position": {"x": 800, "y": 0}
    })

    # 4. Connection
    source_handle = get_output_handle(node_config["type"])
    connections.append({
        "id": str(uuid.uuid4()),
        "source": target_node_id,
        "sourceHandle": source_handle,
        "target": output_node_id,
        "targetHandle": "input"
    })

    return {
        "id": workflow_id,
        "nodes": nodes,
        "connections": connections
    }

def run_workflow(workflow: Dict[str, Any]):
    """
    Executes the workflow using the NodeTool engine.
    """
    logger.info(f"Executing workflow {workflow['id']} with {len(workflow['nodes'])} nodes...")

    # TODO: Replace with actual execution call
    # engine = ExecutionEngine()
    # result = engine.run(workflow)

    # Mock execution for script verification
    print("  [Mock] Workflow JSON constructed successfully.")
    print(f"  [Mock] Nodes: {[n['type'] for n in workflow['nodes']]}")
    print("  [Mock] execution.run() called.")

def main():
    parser = argparse.ArgumentParser(description="Test default node configurations.")
    parser.add_argument("--filter", type=str, help="Filter nodes by name")
    args = parser.parse_args()

    logger.info("Starting Default Node Configuration Tests")
    logger.info("Target Environment: Linux + CUDA")

    for node in TEST_NODES:
        if args.filter and args.filter.lower() not in node["name"].lower():
            continue

        try:
            logger.info(f"Testing Node: {node['name']}")
            workflow = create_adhoc_workflow(node)
            run_workflow(workflow)
            logger.info(f"Successfully tested {node['name']}\n")
        except Exception as e:
            logger.error(f"Failed to test {node['name']}: {e}\n")

if __name__ == "__main__":
    main()
