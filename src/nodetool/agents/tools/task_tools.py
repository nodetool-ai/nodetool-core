"""
Task management tools for dynamic subtask creation and modification.

These tools enable agents to dynamically add, modify, or remove subtasks
during execution, making the agent system more flexible and adaptive.
"""

import uuid
import json
from typing import Any, Dict, List, Optional
from nodetool.agents.tools.base import Tool
from nodetool.metadata.types import SubTask, Task
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


class AddSubtaskTool(Tool):
    """
    Dynamically add a new subtask to the current task plan.

    This tool allows agents to create additional subtasks during execution
    when they realize more work is needed to complete the overall objective.
    The new subtask will be automatically picked up by the TaskExecutor.
    """

    name: str = "add_subtask"
    description: str = """
    Add a new subtask to the current task plan. Use this when you realize additional
    work is needed to complete the overall objective. The new subtask will be executed
    after its dependencies are satisfied.

    Parameters:
    - content: Clear instructions describing what the subtask should accomplish
    - input_tasks: List of subtask IDs that must complete before this subtask runs (optional)
    - input_files: List of file paths that must exist before this subtask runs (optional)
    - output_schema: JSON schema describing the expected output structure (optional)
    - max_tool_calls: Maximum number of tool calls allowed for this subtask (default: 10)

    The subtask will be added to the task plan and executed when its dependencies are met.
    """

    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Clear instructions describing what the subtask should accomplish",
            },
            "input_tasks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of subtask IDs that must complete before this subtask runs",
                "default": [],
            },
            "input_files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of file paths that must exist before this subtask runs",
                "default": [],
            },
            "output_schema": {
                "type": "string",
                "description": "JSON schema (as string) describing the expected output structure",
                "default": '{"type": "object", "description": "Subtask result"}',
            },
            "max_tool_calls": {
                "type": "integer",
                "description": "Maximum number of tool calls allowed for this subtask",
                "default": 10,
            },
        },
        "required": ["content"],
        "additionalProperties": False,
    }

    def __init__(self, task: Task):
        """
        Initialize the AddSubtaskTool with a reference to the current task.

        Args:
            task: The Task object to which subtasks will be added
        """
        super().__init__()
        self.task = task

    def user_message(self, params: Dict[str, Any]) -> str:
        """Generate a user-friendly message describing the tool call."""
        content = params.get("content", "")
        return f"Adding new subtask: {content[:100]}..."

    async def process(self, context: ProcessingContext, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a new subtask to the task plan.

        Args:
            context: The processing context
            params: Parameters for the new subtask

        Returns:
            Dictionary with the new subtask ID and confirmation message
        """
        # Generate unique ID for the new subtask
        subtask_id = f"subtask_{uuid.uuid4().hex[:8]}"

        # Parse output schema if provided as string
        output_schema = params.get("output_schema", '{"type": "object", "description": "Subtask result"}')
        if isinstance(output_schema, str):
            try:
                # Validate it's valid JSON
                json.loads(output_schema)
            except json.JSONDecodeError as e:
                log.warning(f"Invalid output_schema JSON, using default: {e}")
                output_schema = '{"type": "object", "description": "Subtask result"}'

        # Create the new subtask
        new_subtask = SubTask(
            id=subtask_id,
            content=params.get("content", ""),
            input_tasks=params.get("input_tasks", []),
            input_files=params.get("input_files", []),
            output_schema=output_schema,
            max_tool_calls=params.get("max_tool_calls", 10),
            completed=False,
            logs=[],
        )

        # Add to the task's subtask list
        # This is thread-safe because we're just appending to a list
        # The TaskExecutor will pick it up in the next iteration
        self.task.subtasks.append(new_subtask)

        log.info(f"Added new subtask {subtask_id}: {new_subtask.content}")

        return {
            "subtask_id": subtask_id,
            "message": f"Successfully added subtask '{subtask_id}'",
            "content": new_subtask.content,
            "dependencies": {
                "input_tasks": new_subtask.input_tasks,
                "input_files": new_subtask.input_files,
            },
        }


class ListSubtasksTool(Tool):
    """
    List all subtasks in the current task plan with their status.

    This tool helps agents understand what subtasks exist, which are complete,
    and which are pending, enabling better planning and decision-making.
    """

    name: str = "list_subtasks"
    description: str = """
    List all subtasks in the current task plan with their completion status.
    Use this to understand what work has been done and what remains.
    """

    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def __init__(self, task: Task):
        """
        Initialize the ListSubtasksTool with a reference to the current task.

        Args:
            task: The Task object containing the subtasks
        """
        super().__init__()
        self.task = task

    def user_message(self, params: Dict[str, Any]) -> str:
        """Generate a user-friendly message describing the tool call."""
        return "Listing all subtasks in the current task plan"

    async def process(self, context: ProcessingContext, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        List all subtasks with their status.

        Args:
            context: The processing context
            params: Tool parameters (none required)

        Returns:
            Dictionary with subtask information
        """
        subtasks_info = []

        for subtask in self.task.subtasks:
            subtasks_info.append({
                "id": subtask.id,
                "content": subtask.content,
                "completed": subtask.completed,
                "input_tasks": subtask.input_tasks,
                "input_files": subtask.input_files,
                "max_tool_calls": subtask.max_tool_calls,
            })

        completed_count = sum(1 for st in self.task.subtasks if st.completed)
        total_count = len(self.task.subtasks)

        return {
            "subtasks": subtasks_info,
            "summary": {
                "total": total_count,
                "completed": completed_count,
                "pending": total_count - completed_count,
            },
        }
