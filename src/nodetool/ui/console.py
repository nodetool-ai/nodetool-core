"""
UI Console for displaying Agent progress using Rich.
"""

import os
from typing import List, Optional, Union, TYPE_CHECKING

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from nodetool.metadata.types import Task, ToolCall


class AgentConsole:
    """
    Manages Rich library components for displaying Agent planning and execution status.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the AgentConsole.

        Args:
            verbose (bool): Enable/disable rich output.
        """
        self.verbose: bool = verbose
        self.console: Optional[Console] = Console() if verbose else None
        self.live: Optional[Live] = None
        self.current_table: Optional[Table] = None

    def start_live(self, initial_table: Table) -> None:
        """
        Start the Rich Live display with an initial table.

        Args:
            initial_table (Table): The table to display initially.
        """
        if not self.verbose or self.live or not self.console:
            return
        self.current_table = initial_table
        self.live = Live(
            self.current_table,
            console=self.console,
            refresh_per_second=4,
            vertical_overflow="visible",
        )
        self.live.start()

    def stop_live(self) -> None:
        """Stop the Rich Live display if it is active."""
        if self.live and self.live.is_started:
            self.live.stop()
        self.live = None
        self.current_table = None

    def update_live(self, new_table: Table) -> None:
        """
        Update the Rich Live display with a new table.

        Args:
            new_table (Table): The new table to display.
        """
        if self.live and self.live.is_started:
            self.current_table = new_table  # Update the reference
            self.live.update(self.current_table)

    def create_planning_table(self, title: str) -> Table:
        """
        Create a Rich Table configured for displaying planning phases.

        Args:
            title (str): The title for the table.

        Returns:
            Table: The configured Rich table.
        """
        table = Table(
            title=title,
            show_header=True,
            header_style="bold magenta",
            show_lines=True,
            title_justify="left",
        )
        table.add_column("Phase", style="cyan", ratio=1)
        table.add_column("Status", style="white", ratio=1)
        table.add_column("Details", style="green", ratio=8)
        return table

    def update_planning_display(
        self,
        phase_name: str,
        status: str,
        content: Union[str, Text],
        is_error: bool = False,
    ) -> None:
        """
        Add a row to the current planning table in the live display.

        Args:
            phase_name (str): Name of the planning phase.
            status (str): Status of the phase (e.g., "Running", "Success", "Failed").
            content (Union[str, Text]): Details or output for the phase.
            is_error (bool): Flag indicating if the status represents an error.
        """
        if self.live and self.current_table and self.live.is_started:
            status_style = (
                "bold red"
                if is_error
                else "bold green" if status == "Success" else "bold yellow"
            )
            # Truncate long content for better table display
            content_str = str(content)
            if len(content_str) > 1000:
                content_str = content_str[:1000] + "..."
            status_text = Text(status, style=status_style)
            # Use Text object if already Text, otherwise create one
            content_text = content if isinstance(content, Text) else Text(content_str)
            # Add row to the table object that Live is referencing
            self.current_table.add_row(phase_name, status_text, content_text)

    def create_execution_table(
        self, title: str, task: "Task", tool_calls: List["ToolCall"]
    ) -> Table:
        """Create a rich table for displaying subtasks and their tool calls."""
        table = Table(title=title, title_justify="left")
        table.add_column("Status", style="cyan", no_wrap=True, ratio=1)
        table.add_column("Content", style="green", ratio=5)
        table.add_column("Files", style="white", ratio=4)  # Adjusted ratio

        # Guard against task or subtasks being None
        if not task or not task.subtasks:
            return table

        for subtask in task.subtasks:
            status_symbol = (
                "✓" if subtask.completed else "▶" if subtask.is_running() else "⏳"
            )
            status_style = (
                "green"
                if subtask.completed
                else "yellow" if subtask.is_running() else "white"
            )
            # Ensure tool_calls is not None before filtering
            subtask_tool_calls = [
                call
                for call in (tool_calls or [])
                if call.subtask_id == subtask.content
            ]

            # Combine inputs and outputs with color coding
            files_str_parts = []
            if subtask.input_files:
                input_str = ", ".join(subtask.input_files)
                files_str_parts.append(f"[blue]Inputs:[/] {input_str}")
            if subtask.output_file:
                # Display only the basename for cleaner output
                output_basename = os.path.basename(subtask.output_file)
                files_str_parts.append(f"[yellow]Output:[/] {output_basename}")

            files_str = "\n".join(files_str_parts) if files_str_parts else "none"

            # Prepare status display
            status_display = f"[{status_style}]{status_symbol}[/]"
            if status_symbol == "▶":  # If running
                relevant_tool_calls = [
                    call
                    for call in subtask_tool_calls
                    if call.name not in ["finish_task", "finish_subtask"]
                ]
                if relevant_tool_calls:
                    last_tool_message = str(relevant_tool_calls[-1].message)
                    # Truncate long messages
                    if len(last_tool_message) > 50:
                        last_tool_message = last_tool_message[:47] + "..."
                    status_display += f" [dim]({last_tool_message})[/]"

            table.add_row(
                status_display,  # Use the combined status display
                subtask.content,
                files_str,
            )

        return table

    def print(self, message: object, style: Optional[str] = None) -> None:
        """
        Print a message to the console if verbose mode is enabled.

        Args:
            message (object): The message or object to print.
            style (Optional[str]): Rich style to apply.
        """
        if self.verbose and self.console:
            self.console.print(message, style=style)

    def print_exception(self, *args, **kwargs) -> None:
        """
        Print an exception traceback to the console if verbose mode is enabled.
        """
        if self.verbose and self.console:
            self.console.print_exception(*args, **kwargs)
