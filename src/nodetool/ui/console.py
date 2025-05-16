"""
UI Console for displaying Agent progress using Rich.
"""

import os
from typing import List, Optional, Union, TYPE_CHECKING, Dict, Any

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from rich.style import Style

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
        self.current_tree: Optional[Tree] = None
        self.phase_nodes: Dict[str, Any] = {}
        self.subtask_nodes: Dict[str, Any] = {}

    def start_live(self, initial_content: Union[Table, Tree]) -> None:
        """
        Start the Rich Live display with initial content (table or tree).

        Args:
            initial_content (Union[Table, Tree]): The content to display initially.
        """
        if not self.verbose or self.live or not self.console:
            return

        if isinstance(initial_content, Table):
            self.current_table = initial_content
            self.current_tree = None
        else:  # Tree
            self.current_tree = initial_content
            self.current_table = None

        self.live = Live(
            initial_content,
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
        self.current_tree = None
        self.phase_nodes = {}
        self.subtask_nodes = {}

    def update_live(self, new_content: Union[Table, Tree]) -> None:
        """
        Update the Rich Live display with new content.

        Args:
            new_content (Union[Table, Tree]): The new content to display.
        """
        if self.live and self.live.is_started:
            if isinstance(new_content, Table):
                self.current_table = new_content
                self.current_tree = None
            else:  # Tree
                self.current_tree = new_content
                self.current_table = None

            self.live.update(new_content)

    def create_planning_table(self, title: str) -> Tree:
        """
        Create a Rich Tree configured for displaying planning phases.

        Args:
            title (str): The title for the tree.

        Returns:
            Tree: The configured Rich tree.
        """
        tree = Tree(f"[bold magenta]{title}[/]", guide_style="dim")
        self.phase_nodes = {}
        return tree

    def update_planning_display(
        self,
        phase_name: str,
        status: str,
        content: Union[str, Text],
        is_error: bool = False,
    ) -> None:
        """
        Add a node to the current planning tree in the live display.

        Args:
            phase_name (str): Name of the planning phase.
            status (str): Status of the phase (e.g., "Running", "Success", "Failed").
            content (Union[str, Text]): Details or output for the phase.
            is_error (bool): Flag indicating if the status represents an error.
        """
        if self.live and self.current_tree and self.live.is_started:
            status_style = (
                "bold red"
                if is_error
                else "bold green" if status == "Success" else "bold yellow"
            )
            # Truncate long content for better display
            content_str = str(content)
            if len(content_str) > 1000:
                content_str = content_str[:1000] + "..."

            # Create the node label with phase and status
            status_icon = "❌" if is_error else "✓" if status == "Success" else "⏳"
            node_label = (
                f"[cyan]{phase_name}[/] [{status_style}]{status_icon} {status}[/]"
            )

            # Check if we already have a node for this phase
            if phase_name in self.phase_nodes:
                # Update existing node
                node = self.phase_nodes[phase_name]
                node.label = node_label

                # Replace or add the content as a child node
                if len(node.children) > 0:
                    # Remove the old content child node
                    node.children.clear()

                # Add the new content
                node.add(f"[dim]{content_str}[/]")
            else:
                # Create a new node
                node = self.current_tree.add(node_label)
                node.add(f"[dim]{content_str}[/]")
                self.phase_nodes[phase_name] = node

    def create_execution_table(
        self, title: str, task: "Task", tool_calls: List["ToolCall"]
    ) -> Tree:
        """Create a rich tree for displaying subtasks and their tool calls."""
        tree = Tree(f"[bold magenta]{title}[/]", guide_style="dim")
        self.subtask_nodes = {}

        # Guard against task or subtasks being None
        if not task or not task.subtasks:
            return tree

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
                call for call in (tool_calls or []) if call.subtask_id == subtask.id
            ]

            # Prepare status display
            status_text = f"[{status_style}]{status_symbol}[/]"
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
                    status_text += f" [dim]({last_tool_message})[/]"

            # Create subtask node with status and content
            node_label = f"{status_text} [green]{subtask.content}[/]"
            subtask_node = tree.add(node_label)
            self.subtask_nodes[subtask.id] = subtask_node

            # Add files information as child nodes
            if subtask.input_files:
                input_str = ", ".join(subtask.input_files)
                subtask_node.add(f"[blue]📁 Inputs:[/] {input_str}")

            if subtask.output_file:
                # Display only the basename for cleaner output
                output_basename = os.path.basename(subtask.output_file)
                subtask_node.add(f"[yellow]📂 Output:[/] {output_basename}")

            # Add tool calls as child nodes if there are any relevant ones
            relevant_tool_calls = [
                call
                for call in subtask_tool_calls
                if call.name not in ["finish_task", "finish_subtask"]
            ]

            if relevant_tool_calls:
                tool_node = subtask_node.add("[cyan]🔧 Tools[/]")
                for i, call in enumerate(
                    relevant_tool_calls[-3:]
                ):  # Show last 3 tool calls
                    tool_name = call.name
                    message = str(call.message)
                    if len(message) > 70:
                        message = message[:67] + "..."
                    tool_node.add(f"[dim]{tool_name}: {message}[/]")

                if len(relevant_tool_calls) > 3:
                    tool_node.add(
                        f"[dim]+ {len(relevant_tool_calls) - 3} more tool calls[/]"
                    )

        return tree

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
