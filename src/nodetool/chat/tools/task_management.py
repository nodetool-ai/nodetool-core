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

import os
import re
from typing import Any, Dict, List, Optional

from nodetool.workflows.processing_context import ProcessingContext
from .base import Tool


class SubTask:
    """Represents a subtask in a task list."""

    def __init__(
        self,
        content: str,
        completed: bool = False,
        subtask_id: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
    ):
        self.content = content.strip()
        self.completed = completed
        self.subtask_id = subtask_id or self._generate_id()
        self.dependencies = dependencies or []

    def _generate_id(self) -> str:
        """Generate a unique ID for this subtask."""
        import uuid

        return str(uuid.uuid4())[:3]  # Using first 3 chars of a UUID for brevity

    def to_markdown(self) -> str:
        """Convert the subtask to markdown format."""
        checkbox = "[x]" if self.completed else "[ ]"
        deps_str = (
            f" (depends on #{', #'.join(self.dependencies)})"
            if self.dependencies
            else ""
        )
        return f"- {checkbox} #{self.subtask_id} {self.content}{deps_str}"

    def add_dependency(self, dependency_id: str) -> None:
        """Add a dependency to this subtask."""
        if dependency_id not in self.dependencies:
            self.dependencies.append(dependency_id)


class Task:
    """Represents a task with a title (heading) and subtasks."""

    def __init__(self, title: str, task_id: Optional[str] = None):
        self.title = title.strip()
        self.subtasks: List[SubTask] = []

    def _generate_id(self) -> str:
        """Generate a unique ID for this task."""
        import uuid

        return str(uuid.uuid4())[:8]

    def is_completed(self) -> bool:
        """Check if all subtasks are completed."""
        return all(subtask.completed for subtask in self.subtasks)

    def to_markdown(self) -> str:
        """Convert the task to markdown format with heading and subtasks."""
        lines = f"# {self.title}\n"
        if self.subtasks:
            for subtask in self.subtasks:
                lines += f"{subtask.to_markdown()}\n"
        return lines

    def add_subtask(
        self,
        content: str,
        completed: bool = False,
        subtask_id: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
    ) -> SubTask:
        """Add a subtask to this task."""
        subtask = SubTask(content, completed, subtask_id, dependencies)
        self.subtasks.append(subtask)
        return subtask

    def find_subtask_by_content(self, content: str) -> Optional[SubTask]:
        """Find a subtask by its content."""
        content = content.strip()
        for subtask in self.subtasks:
            if subtask.content.lower() == content.lower():
                return subtask
        return None

    def find_subtask_by_id(self, subtask_id: str) -> Optional[SubTask]:
        """Find a subtask by its ID."""
        for subtask in self.subtasks:
            if subtask.subtask_id == subtask_id:
                return subtask
        return None


class TaskList:
    """Manager for a task list stored in markdown format."""

    def __init__(self):
        """
        Initialize a task list from either a file path or markdown content.
        """
        self.tasks: List[Task] = []

    def from_markdown(self, markdown_content: str) -> None:
        """
        Load tasks from a markdown string.

        Args:
            markdown_content: The markdown content to parse
        """
        self.tasks = []  # Clear existing tasks

        if not markdown_content.strip():
            return

        lines = markdown_content.splitlines()
        current_task = None

        for line in lines:
            line = line.rstrip()

            # Skip empty lines
            if not line.strip():
                continue

            heading_match = re.match(r"^#+\s+(.+)$", line)
            if heading_match:
                # Found a new task heading
                task_title = heading_match.group(1).strip()
                current_task = Task(task_title)
                self.tasks.append(current_task)
                continue

            if current_task:
                # Check for subtask with dependencies in the new format: - [ ] #id content (depends on #id1, #id2)
                new_format_match = re.match(
                    r"^- \[([ xX])\] #([a-zA-Z0-9]+) (.+?)(?:\s+\(depends on #([a-zA-Z0-9,\s#]+)\))?$",
                    line,
                )
                if new_format_match:
                    completed = new_format_match.group(1).lower() == "x"
                    subtask_id = new_format_match.group(2)
                    content = new_format_match.group(3).strip()
                    dependencies = []
                    if new_format_match.group(4):
                        deps_str = new_format_match.group(4)
                        # Handle both comma-separated or space-separated formats
                        deps_list = re.findall(r"#([a-zA-Z0-9]+)", deps_str)
                        dependencies = deps_list
                    current_task.add_subtask(
                        content, completed, subtask_id, dependencies
                    )
                    continue

                # Check for subtask with ID
                subtask_with_id_match = re.match(
                    r"^- \[([ xX])\] (.+) \(([a-zA-Z0-9]+)\)$", line
                )
                if subtask_with_id_match:
                    completed = subtask_with_id_match.group(1).lower() == "x"
                    content = subtask_with_id_match.group(2).strip()
                    subtask_id = subtask_with_id_match.group(3)
                    current_task.add_subtask(content, completed, subtask_id)
                    continue

                # Regular subtask without ID (older format)
                subtask_match = re.match(r"^- \[([ xX])\] (.+)$", line)
                if subtask_match:
                    completed = subtask_match.group(1).lower() == "x"
                    content = subtask_match.group(2)
                    current_task.add_subtask(content, completed)
                    continue

                # Check for alternative subtask formats
                # 1. No space between dash and bracket: -[x] or -[ ]
                alt_subtask_match = re.match(r"^-\[([ xX])\] (.+)$", line)
                if alt_subtask_match:
                    completed = alt_subtask_match.group(1).lower() == "x"
                    content = alt_subtask_match.group(2)
                    current_task.add_subtask(content, completed)
                    continue

                # 2. Simple list item without checkbox: "- Task"
                simple_list_match = re.match(r"^- (.+)$", line)
                if simple_list_match:
                    content = simple_list_match.group(1)
                    current_task.add_subtask(content, False)  # Assume uncompleted

    def to_markdown(self) -> str:
        """Convert all tasks to a markdown string."""
        lines = []
        for task in self.tasks:
            lines += task.to_markdown()

        return "\n".join(lines)

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get a list of all tasks with their metadata."""
        result = []
        for task in self.tasks:
            task_info = {
                "title": task.title,
                "completed": task.is_completed(),
                "subtasks": [
                    {
                        "id": subtask.subtask_id,
                        "content": subtask.content,
                        "completed": subtask.completed,
                        "dependencies": subtask.dependencies,
                    }
                    for subtask in task.subtasks
                ],
            }
            result.append(task_info)
        return result

    def find_task_by_title(self, title: str) -> Optional[Task]:
        """Find a task by its title."""
        title = title.strip()
        for task in self.tasks:
            if task.title.lower() == title.lower():
                return task
        return None

    def find_task_by_id(self, task_id: str) -> Optional[Task]:
        """Find a task by its ID."""
        return None

    def find_subtask(
        self, task_title: str, subtask_content: str
    ) -> tuple[Optional[Task], Optional[SubTask]]:
        """Find a subtask by its task title and content."""
        task = self.find_task_by_title(task_title)
        if not task:
            return None, None

        subtask = task.find_subtask_by_content(subtask_content)
        return task, subtask

    def add_task(self, title: str) -> Dict[str, Any]:
        """
        Add a new task with the given title.

        Args:
            title: The title of the task (will be used as heading)

        Returns:
            Dictionary with the new task's information
        """
        # Check if task with this title already exists
        existing_task = self.find_task_by_title(title)
        if existing_task:
            return {
                "success": False,
                "error": f"Task with title '{title}' already exists",
            }

        # Create the task
        task = Task(title)
        self.tasks.append(task)

        return {
            "success": True,
            "title": task.title,
        }

    def add_subtask(
        self,
        task_title: str,
        content: str,
        completed: bool = False,
        dependencies: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Add a subtask to an existing task.

        Args:
            task_title: The title of the parent task
            content: The text of the subtask
            completed: Whether the subtask is already completed
            dependencies: List of task IDs this subtask depends on

        Returns:
            Dictionary with the result
        """
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
        subtask = task.add_subtask(content, completed, dependencies=dependencies)

        return {
            "success": True,
            "task_title": task.title,
            "subtask_content": subtask.content,
            "subtask_id": subtask.subtask_id,
            "dependencies": subtask.dependencies,
        }

    def finish_subtask(self, task_title: str, content: str) -> Dict[str, Any]:
        """
        Mark a subtask as complete.

        Args:
            task_title: The title of the parent task
            content: The content of the subtask to mark as complete

        Returns:
            Dictionary with the result
        """
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
    """Base class for task management tools."""

    def __init__(self, name: str, description: str, task_list: TaskList):
        super().__init__(name=name, description=description)
        # Shared in-memory task list for all tools
        self._task_list = task_list

    def get_task_list(self) -> TaskList:
        """Get the in-memory TaskList instance."""
        return self._task_list


class AddTaskTool(TaskBaseTool):
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
