"""
Task management tools module.

This module provides simple tools for managing tasks in a markdown format:
- AddTaskTool: Add a new task to the list
- FinishTaskTool: Mark a task as complete

Tasks are stored as markdown headings with subtasks as checkbox lists:
- Tasks are markdown headings (e.g., "# Task Title")
- Subtasks are markdown list items under headings
- Unfinished subtasks are marked with [ ]
- Completed subtasks are marked with [x]
"""

from typing import Any, Dict, List, Optional

from nodetool.workflows.processing_context import ProcessingContext
from .base import Tool
from pydantic import BaseModel, Field


class SubTask(BaseModel):
    """A subtask item with completion status, dependencies, and tools."""

    id: str = Field(default="", description="The ID of the subtask")
    content: str = Field(default="", description="The content of the subtask")
    tool: str = Field(
        default="", description="The tool that can be used to complete the subtask"
    )
    completed: bool = Field(
        default=False, description="Whether the subtask is completed"
    )
    dependencies: List[str] = Field(
        default=[], description="The dependencies of the subtask, a list of subtask IDs"
    )

    def to_markdown(self) -> str:
        """Convert the subtask to markdown format."""
        checkbox = "[x]" if self.completed else "[ ]"
        deps_str = (
            f" (depends on #{', #'.join(self.dependencies)})"
            if self.dependencies
            else ""
        )
        return f"- {checkbox} #{self.id} {self.content}{deps_str}"


class Task(BaseModel):
    """A task containing a title, description, and list of subtasks."""

    title: str = Field(default="", description="The title of the task")
    description: str = Field(
        default="", description="A description of the task, not used for execution"
    )
    subtasks: List[SubTask] = Field(
        default=[], description="The subtasks of the task, a list of subtask IDs"
    )

    def _generate_id(self) -> str:
        """Creates a unique 8-character UUID for task identification."""
        import uuid

        return str(uuid.uuid4())[:8]

    def is_completed(self) -> bool:
        """Returns True if all subtasks are marked as completed."""
        return all(subtask.completed for subtask in self.subtasks)

    def to_markdown(self) -> str:
        """Converts task and subtasks to markdown format with headings and checkboxes."""
        lines = f"# {self.title}\n"
        if self.subtasks:
            for subtask in self.subtasks:
                lines += f"{subtask.to_markdown()}\n"
        return lines

    def add_subtask(
        self,
        subtask_id: str,
        content: str,
        dependencies: List[str] = [],
    ) -> SubTask:
        """Creates and adds a new subtask to the task."""
        subtask = SubTask(
            id=subtask_id,
            content=content,
            dependencies=dependencies,
        )
        self.subtasks.append(subtask)
        return subtask

    def find_subtask_by_content(self, content: str) -> Optional[SubTask]:
        """Searches for a subtask by its content text, case-insensitive."""
        content = content.strip()
        for subtask in self.subtasks:
            if subtask.content.lower() == content.lower():
                return subtask
        return None

    def find_subtask_by_id(self, subtask_id: str) -> Optional[SubTask]:
        """Retrieves a subtask by its unique identifier."""
        for subtask in self.subtasks:
            if subtask.id == subtask_id:
                return subtask
        return None


class TaskList(BaseModel):
    """Manager for organizing and manipulating a collection of tasks."""

    title: str = Field(default="", description="The title of the task list")
    tasks: List[Task] = Field(
        default=[], description="The tasks of the task list, a list of task IDs"
    )

    def to_markdown(self) -> str:
        """Convert all tasks to a markdown string."""
        lines = []
        for task in self.tasks:
            lines += task.to_markdown()

        return "\n".join(lines)

    def find_task_by_title(self, title: str) -> Optional[Task]:
        """Find a task by its title."""
        title = title.strip()
        for task in self.tasks:
            if task.title.lower() == title.lower():
                return task
        return None

    def find_task_by_id(self, task_id: str) -> tuple[Optional[Task], Optional[SubTask]]:
        """Find a task by its ID."""
        for task in self.tasks:
            for subtask in task.subtasks:
                if subtask.id == task_id:
                    return task, subtask
        return None, None

    def find_subtask(
        self, task_title: str, subtask_content: str
    ) -> tuple[Optional[Task], Optional[SubTask]]:
        """Find a subtask by its task title and content."""
        task = self.find_task_by_title(task_title)
        if not task:
            return None, None

        subtask = task.find_subtask_by_content(subtask_content)
        return task, subtask

    def add_task(self, title: str, description: str = "") -> Dict[str, Any]:
        """Creates a new task with the given title if it doesn't already exist."""
        # Check if task with this title already exists
        existing_task = self.find_task_by_title(title)
        if existing_task:
            return {
                "success": False,
                "error": f"Task with title '{title}' already exists",
            }

        # Create the task
        task = Task(title=title, description=description)
        self.tasks.append(task)

        return {
            "success": True,
            "title": task.title,
        }

    def add_subtask(
        self,
        task_title: str,
        subtask_id: str,
        content: str,
        dependencies: List[str] = [],
    ) -> Dict[str, Any]:
        """Adds a subtask to an existing task, with optional completion status and dependencies."""
        task = self.find_task_by_title(task_title)
        if not task:
            return {
                "success": False,
                "error": f"Task with title '{task_title}' not found",
            }

        # Check if subtask with this content already exists
        if task.find_subtask_by_content(content):
            return {
                "success": False,
                "error": f"Subtask with content '{content}' already exists in task '{task_title}'",
            }

        # Create the subtask
        subtask = task.add_subtask(subtask_id, content, dependencies=dependencies)

        return {
            "success": True,
            "task_title": task.title,
            "subtask_content": subtask.content,
            "subtask_id": subtask.id,
            "dependencies": subtask.dependencies,
        }

    def finish_subtask(self, task_title: str, content: str) -> Dict[str, Any]:
        """Marks a specific subtask as complete within its parent task."""
        task, subtask = self.find_subtask(task_title, content)

        if not task:
            return {
                "success": False,
                "error": f"Task with title '{task_title}' not found",
            }

        if not subtask:
            return {
                "success": False,
                "error": f"Subtask with content '{content}' not found in task '{task_title}'",
            }

        subtask.completed = True

        return {
            "success": True,
            "task_title": task.title,
            "content": subtask.content,
            "completed": subtask.completed,
        }


class TaskBaseTool(Tool):
    """Base tool class for task management operations."""

    def __init__(self, name: str, description: str, task_list: TaskList):
        super().__init__(name=name, description=description)
        # Shared in-memory task list for all tools
        self._task_list = task_list

    def get_task_list(self) -> TaskList:
        """Get the in-memory TaskList instance."""
        return self._task_list


class AddTaskTool(TaskBaseTool):
    """Tool for creating new tasks or adding subtasks to existing tasks."""

    def __init__(self, task_list: TaskList):
        super().__init__(
            name="add_task",
            description="Add a new task or subtask to the task list",
            task_list=task_list,
        )
        self.input_schema = {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The title of the task (will be a markdown heading)",
                },
                "subtask_content": {
                    "type": "string",
                    "description": "Content of the subtask (if adding a subtask to an existing task)",
                },
                "completed": {
                    "type": "boolean",
                    "description": "Whether the subtask is already completed (only for subtasks)",
                    "default": False,
                },
            },
            "required": ["title"],
        }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        try:
            # Get parameters
            title = params["title"]
            subtask_content = params.get("subtask_content")
            completed = params.get("completed", False)

            # Load task list
            task_list = self.get_task_list()

            # Add task or subtask
            if subtask_content:
                # Adding a subtask to an existing task
                result = task_list.add_subtask(title, subtask_content, completed)
            else:
                # Adding a new task
                result = task_list.add_task(title)

            return result

        except Exception as e:
            return {"success": False, "error": str(e)}


class FinishTaskTool(TaskBaseTool):
    """Tool for marking subtasks as complete."""

    def __init__(self, task_list: TaskList):
        super().__init__(
            name="finish_subtask",
            description="Mark a subtask as complete",
            task_list=task_list,
        )
        self.input_schema = {
            "type": "object",
            "properties": {
                "task_title": {
                    "type": "string",
                    "description": "The title of the parent task",
                },
                "subtask_content": {
                    "type": "string",
                    "description": "The content of the subtask to mark as complete",
                },
            },
            "required": ["task_title", "subtask_content"],
        }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        try:
            # Get parameters
            task_title = params["task_title"]
            subtask_content = params["subtask_content"]

            # Load task list
            task_list = self.get_task_list()

            # Mark subtask as complete
            result = task_list.finish_subtask(task_title, subtask_content)

            return result

        except Exception as e:
            return {"success": False, "error": str(e)}
