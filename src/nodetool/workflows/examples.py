import json
import os
import logging
from datetime import datetime
from typing import List
from nodetool.types.workflow import Workflow
from nodetool.types.graph import Graph

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

examples_folder = os.path.join(os.path.dirname(__file__), "examples")
examples = None


def load_example(name: str) -> Workflow:
    """
    Load a single example workflow from a JSON file.

    Args:
        name (str): The filename of the example workflow to load

    Returns:
        Workflow: The loaded workflow object, or an error workflow if loading fails
    """
    example_path = os.path.join(examples_folder, name)
    with open(example_path, "r") as f:
        try:
            props = json.load(f)
            return Workflow(**props)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON for example workflow {name}: {e}")
            # Return an empty Workflow with the name indicating it is broken
            now_str = datetime.now().isoformat()
            return Workflow(
                id="",
                name=f"[ERROR] {name}",
                tags=[],
                graph=Graph(nodes=[], edges=[]),
                access="",
                created_at=now_str,
                updated_at=now_str,
                description="Error loading this workflow",
            )


def load_examples() -> List[Workflow]:
    """
    Load all example workflows from the examples folder.

    Files starting with underscore (_) are ignored.
    Results are cached for subsequent calls.

    Returns:
        List[Workflow]: A list of all loaded example workflows
    """
    global examples
    # no commit
    examples = None
    if examples is None:
        examples = [
            load_example(name) for name in os.listdir(examples_folder) if name[0] != "_"
        ]
    return examples


def find_example(id: str) -> Workflow | None:
    """
    Find an example workflow by its ID.

    Args:
        id (str): The ID of the workflow to find

    Returns:
        Workflow | None: The found workflow or None if not found
    """
    examples = load_examples()
    return next((example for example in examples if example.id == id), None)


def save_example(id: str, workflow: Workflow) -> Workflow:
    """
    Save a workflow as an example.

    Args:
        id (str): The ID of the workflow to save
        workflow (Workflow): The workflow object to save

    Returns:
        Workflow: The saved workflow

    Note:
        This function removes the user_id field before saving and
        invalidates the cached examples.
    """
    workflow_dict = workflow.model_dump()

    # Remove unnecessary fields
    workflow_dict.pop("user_id", None)

    example_path = os.path.join(examples_folder, f"{workflow.name}.json")
    with open(example_path, "w") as f:
        json.dump(workflow_dict, f, indent=2)

    # Invalidate the cached examples
    global examples
    examples = None

    return workflow
