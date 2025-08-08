from __future__ import annotations

from typing import Optional, Dict, Any

from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TaskProgressColumn,
    SpinnerColumn,
)


class ProgressManager:
    """Manages persistent progress bars for different operations."""

    def __init__(self, console: Optional[Console] = None):
        self.console: Console = console or Console()
        self.progress: Optional[Progress] = None
        self.tasks: Dict[str, int] = {}  # Maps operation IDs to task IDs
        self.current_operations: Dict[str, Dict[str, Any]] = {}  # operation info

    def start(self) -> None:
        """Start the progress display."""
        if self.progress is None:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=self.console,
                transient=False,
            )
            self.progress.start()

    def stop(self) -> None:
        """Stop the progress display."""
        if self.progress:
            self.progress.stop()
            self.progress = None
            self.tasks.clear()
            self.current_operations.clear()

    def add_task(
        self, operation_id: str, description: str, total: Optional[float] = None
    ) -> int:
        """Add a new progress task."""
        self.start()
        assert self.progress is not None
        if operation_id not in self.tasks:
            task_id = self.progress.add_task(description, total=total)
            self.tasks[operation_id] = task_id
            self.current_operations[operation_id] = {
                "description": description,
                "total": total,
                "completed": 0,
            }
        return self.tasks[operation_id]

    def update_task(
        self,
        operation_id: str,
        completed: Optional[float] = None,
        description: Optional[str] = None,
    ) -> None:
        """Update a progress task."""
        if operation_id in self.tasks and self.progress:
            assert self.progress is not None
            task_id = self.tasks[operation_id]
            update_kwargs: Dict[str, Any] = {}

            if completed is not None:
                # Calculate advance amount
                current_completed = self.current_operations[operation_id][
                    "completed"
                ]
                advance = completed - current_completed
                if advance > 0:
                    update_kwargs["advance"] = advance
                self.current_operations[operation_id]["completed"] = completed

            if description is not None:
                update_kwargs["description"] = description
                self.current_operations[operation_id]["description"] = description

            if update_kwargs:
                self.progress.update(task_id, **update_kwargs)

    def complete_task(self, operation_id: str) -> None:
        """Mark a task as completed."""
        if operation_id in self.tasks and self.progress:
            assert self.progress is not None
            task_id = self.tasks[operation_id]
            op_info = self.current_operations[operation_id]
            if op_info["total"]:
                # Complete the progress bar
                self.progress.update(task_id, completed=op_info["total"])
            # Remove the task after a brief moment
            self.progress.remove_task(task_id)
            del self.tasks[operation_id]
            del self.current_operations[operation_id]

            # If no more tasks, stop the progress display
            if not self.tasks:
                self.stop()

    def remove_task(self, operation_id: str) -> None:
        """Remove a task without completing it."""
        if operation_id in self.tasks and self.progress:
            assert self.progress is not None
            task_id = self.tasks[operation_id]
            self.progress.remove_task(task_id)
            del self.tasks[operation_id]
            del self.current_operations[operation_id]

            # If no more tasks, stop the progress display
            if not self.tasks:
                self.stop()


