"""
UI Console for displaying Agent progress using Rich.
"""

import json
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from nodetool.metadata.types import LogEntry, SubTask, Task, ToolCall


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
        self.current_subtask: Optional[SubTask] = None
        self.task: Optional[Task] = None

        # Phase-specific logging storage
        self.phase_logs: Dict[str, List[Dict[str, Any]]] = {}
        self.current_phase: Optional[str] = None

    def start_live(self, initial_content: Table | Tree) -> None:
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
        self.current_phase = None
        self.task = None

    def update_live(self, new_content: Table | Tree) -> None:
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

    def create_planning_tree(self, title: str) -> Tree:
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
        content: str | Text,
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
                else "bold green"
                if status == "Success"
                else "bold yellow"
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

    def create_execution_tree(
        self, title: str, task: "Task", tool_calls: List["ToolCall"]
    ) -> Tree:
        """Create a rich tree for displaying subtasks and their tool calls."""
        tree = Tree(f"[bold magenta]{title}[/]", guide_style="dim")
        self.subtask_nodes = {}
        self.task = task

        # Guard against task or subtasks being None
        if not task or not task.subtasks:
            return tree

        for subtask in task.subtasks:
            status_symbol = (
                "✅" if subtask.completed else "🔄" if subtask.is_running() else "⏳"
            )
            (
                "green"
                if subtask.completed
                else "yellow"
                if subtask.is_running()
                else "white"
            )

            # Create subtask node with status and content
            node_label = f"{status_symbol} [green]{subtask.content}[/]"
            subtask_node = tree.add(node_label)
            self.subtask_nodes[subtask.id] = subtask_node

            # Add files information as child nodes
            if subtask.input_tasks:
                input_str = ", ".join(subtask.input_tasks)
                subtask_node.add(f"[blue]📁 Inputs:[/] {input_str}")

            # Show logs for all subtasks
            if subtask.logs:
                log_limit = 3
                for log in reversed(subtask.logs[-log_limit:]):
                    formatted_log = self._format_log_entry(log)
                    subtask_node.add(formatted_log)
                if len(subtask.logs) > log_limit:
                    subtask_node.add(
                        f"[dim]+ {len(subtask.logs) - log_limit} more logs[/]"
                    )

            # Add tool calls as child nodes if there are any relevant ones (for non-current subtasks)
            if subtask.completed:
                # Ensure tool_calls is not None before filtering
                subtask_tool_calls = [
                    call for call in (tool_calls or []) if call.subtask_id == subtask.id
                ]

                if subtask_tool_calls:
                    tool_node = subtask_node.add("[cyan]🔧 Tools[/]")
                    for i, call in enumerate(
                        subtask_tool_calls[-3:]
                    ):  # Show last 3 tool calls
                        tool_name = call.name
                        message = str(call.message)
                        if len(message) > 70:
                            message = message[:67] + "..."
                        tool_node.add(f"[dim]{tool_name}: {message}[/]")

                    if len(subtask_tool_calls) > 3:
                        tool_node.add(
                            f"[dim]+ {len(subtask_tool_calls) - 3} more tool calls[/]"
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

    def set_current_phase(self, phase_name: str) -> None:
        """
        Set the current phase for logging purposes.

        Args:
            phase_name (str): The name of the current phase.
        """
        self.current_phase = phase_name
        if phase_name not in self.phase_logs:
            self.phase_logs[phase_name] = []

    def log_to_phase(
        self, level: str, message: str, phase_name: Optional[str] = None
    ) -> None:
        """
        Add a log entry to a specific phase.

        Args:
            level (str): The log level (debug, info, warning, error).
            message (str): The log message.
            phase_name (Optional[str]): The phase to log to. If None, uses current_phase.
        """
        target_phase = phase_name or self.current_phase
        if target_phase is None:
            # Fall back to a default phase if no current phase is set
            target_phase = "General"

        if target_phase not in self.phase_logs:
            self.phase_logs[target_phase] = []

        log_entry = {"level": level, "message": message, "timestamp": int(time.time())}
        self.phase_logs[target_phase].append(log_entry)

    def get_phase_logs(self, phase_name: str) -> List[Dict[str, Any]]:
        """
        Get all log entries for a specific phase.

        Args:
            phase_name (str): The name of the phase.

        Returns:
            List[Dict[str, Any]]: List of log entries for the phase.
        """
        return self.phase_logs.get(phase_name, [])

    def get_all_phase_logs(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all log entries organized by phase.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Dictionary mapping phase names to log entries.
        """
        return self.phase_logs.copy()

    def clear_phase_logs(self, phase_name: Optional[str] = None) -> None:
        """
        Clear log entries for a specific phase or all phases.

        Args:
            phase_name (Optional[str]): The phase to clear. If None, clears all phases.
        """
        if phase_name is None:
            self.phase_logs.clear()
        else:
            self.phase_logs.pop(phase_name, None)

    def set_current_subtask(self, subtask: "SubTask") -> None:
        """
        Set the current subtask that is being executed.

        Args:
            subtask (SubTask): The subtask currently being executed.
        """
        self.current_subtask = subtask
        # Refresh the tree display to show the new current subtask's logs
        if self.live and self.current_tree and self.live.is_started:
            self.update_execution_display()

    def log_to_subtask(self, level: str, message: str) -> None:
        """
        Add a log entry to the current subtask.

        Args:
            level (str): The log level (debug, info, warning, error).
            message (str): The log message.
        """
        if self.current_subtask is not None:
            from nodetool.metadata.types import LogEntry

            log_entry = LogEntry(
                message=message,
                level=level,
                timestamp=int(time.time()),  # type: ignore
            )
            self.current_subtask.logs.append(log_entry)
            self.update_execution_display()

    def debug_subtask_only(self, message: object) -> None:
        """
        Add a debug log entry ONLY to the current subtask (not phase).

        Args:
            message (object): The debug message.
        """
        msg_str = str(message)
        self.log_to_subtask("debug", msg_str)

    def info_subtask_only(self, message: object) -> None:
        """
        Add an info log entry ONLY to the current subtask (not phase).

        Args:
            message (object): The info message.
        """
        msg_str = str(message)
        self.log_to_subtask("info", msg_str)

    def warning_subtask_only(self, message: object) -> None:
        """
        Add a warning log entry ONLY to the current subtask (not phase).

        Args:
            message (object): The warning message.
        """
        msg_str = str(message)
        self.log_to_subtask("warning", msg_str)

    def error_subtask_only(self, message: object) -> None:
        """
        Add an error log entry ONLY to the current subtask (not phase).

        Args:
            message (object): The error message.
        """
        msg_str = str(message)
        self.log_to_subtask("error", msg_str)

    def update_execution_display(self) -> None:
        """
        Refresh the execution tree with logs for all subtasks.
        """
        if (
            self.live
            and self.current_tree
            and self.live.is_started
            and self.task
            and self.task.subtasks
        ):
            # Iterate through all subtasks and update their logs
            for subtask in self.task.subtasks:
                if subtask.id in self.subtask_nodes:
                    subtask_node = self.subtask_nodes[subtask.id]

                    # Remove existing log children (keep other children like inputs and tools)
                    children_to_keep = []
                    for child in subtask_node.children:
                        # Keep non-log children (inputs and tools sections)
                        child_label = str(child.label)
                        if child_label.startswith(
                            "[blue]📁 Inputs:[/]"
                        ) or child_label.startswith("[cyan]🔧 Tools[/]"):
                            children_to_keep.append(child)

                    subtask_node.children = children_to_keep

                    # Add current logs
                    if subtask.logs:
                        log_limit = 3
                        for log in reversed(subtask.logs[-log_limit:]):
                            formatted_log = self._format_log_entry(log)
                            subtask_node.add(formatted_log)
                        if len(subtask.logs) > log_limit:
                            subtask_node.add(
                                f"[dim]+ {len(subtask.logs) - log_limit} more logs[/]"
                            )

            self.live.update(self.current_tree)

    def _format_log_entry(self, log: "LogEntry") -> str:
        """
        Format a log entry with appropriate colors.

        Args:
            log (LogEntry): The log entry to format.

        Returns:
            str: The formatted log entry string.
        """
        # Color coding for log levels
        color_map = {
            "debug": "dim",
            "info": "blue",
            "warning": "yellow",
            "error": "bold red",
        }

        # Icon mapping for log levels
        icon_map = {"debug": "🐛", "info": "🔧", "warning": "⚠️", "error": "❌"}

        color = color_map.get(log.level, "white")
        icon = icon_map.get(log.level, "📝")

        return f"[{color}]{icon} {log.message}[/]"

    def debug(self, message: object, style: Optional[str] = None) -> None:
        """
        Add a debug log entry to the current subtask and current phase.

        Args:
            message (object): The debug message.
            style (Optional[str]): Rich style to apply (ignored, kept for compatibility).
        """
        msg_str = str(message)
        self.log_to_subtask("debug", msg_str)
        self.log_to_phase("debug", msg_str)

    def info(self, message: object, style: Optional[str] = None) -> None:
        """
        Add an info log entry to the current subtask and current phase.

        Args:
            message (object): The info message.
            style (Optional[str]): Rich style to apply (ignored, kept for compatibility).
        """
        msg_str = str(message)
        self.log_to_subtask("info", msg_str)
        self.log_to_phase("info", msg_str)

    def warning(self, message: object, style: Optional[str] = None) -> None:
        """
        Add a warning log entry to the current subtask and current phase.

        Args:
            message (object): The warning message.
            style (Optional[str]): Rich style to apply (ignored, kept for compatibility).
        """
        msg_str = str(message)
        self.log_to_subtask("warning", msg_str)
        self.log_to_phase("warning", msg_str)

    def error(
        self, message: object, style: Optional[str] = None, exc_info: bool = False
    ) -> None:
        """
        Add an error log entry to the current subtask and current phase.

        Args:
            message (object): The error message.
            style (Optional[str]): Rich style to apply (ignored, kept for compatibility).
            exc_info (bool): Whether to print exception information.
        """
        msg_str = str(message)
        self.log_to_subtask("error", msg_str)
        self.log_to_phase("error", msg_str)
        if exc_info and self.console:
            self.print_exception()

    def display_subtask_start(self, subtask: "SubTask") -> None:
        """
        Display the beginning of a subtask with beautiful formatting.

        Args:
            subtask (SubTask): The subtask being started.
        """
        if self.console:
            panel = Panel(
                f"[bold cyan]Content:[/] {subtask.content}\n"
                f"[bold cyan]ID:[/] {subtask.id}\n"
                f"[bold cyan]Input Tasks:[/] {', '.join(subtask.input_tasks) if subtask.input_tasks else 'None'}",
                title="[bold green]🚀 Starting SubTask[/]",
                border_style="green",
                expand=False,
            )
            self.console.print(panel)

    def display_iteration_status(
        self, iteration: int, max_iterations: int, token_count: int, max_tokens: int
    ) -> None:
        """
        Display current iteration and token status with progress bars.

        Args:
            iteration (int): Current iteration number.
            max_iterations (int): Maximum allowed iterations.
            token_count (int): Current token count.
            max_tokens (int): Maximum allowed tokens.
        """
        if self.console:
            # Create progress bars for iterations and tokens
            iteration_progress = f"[cyan]Iteration:[/] {iteration}/{max_iterations}"
            token_progress = f"[cyan]Tokens:[/] {token_count}/{max_tokens}"

            # Calculate percentages
            iter_percent = (iteration / max_iterations) * 100
            token_percent = (token_count / max_tokens) * 100

            # Create visual progress bars
            iter_bar = self._create_progress_bar(iter_percent, "blue")
            token_bar = self._create_progress_bar(
                token_percent, "yellow" if token_percent < 80 else "red"
            )

            # Display in columns
            columns = Columns(
                [f"{iteration_progress}\n{iter_bar}", f"{token_progress}\n{token_bar}"],
                equal=True,
                expand=True,
            )

            self.console.print(
                Panel(columns, title="[bold]📊 Status[/]", border_style="dim")
            )

    def display_tool_execution(self, tool_name: str, args: dict) -> None:
        """
        Display tool execution with beautiful formatting.

        Args:
            tool_name (str): Name of the tool being executed.
            args (dict): Arguments passed to the tool.
        """
        if self.console:
            # Format arguments as JSON with syntax highlighting
            args_json = Syntax(
                json.dumps(args, indent=2, ensure_ascii=False),
                "json",
                theme="monokai",
                line_numbers=False,
            )

            panel = Panel(
                args_json,
                title=f"[bold yellow]🔧 Executing Tool: {tool_name}[/]",
                border_style="yellow",
                expand=False,
            )
            self.console.print(panel)

    def display_tool_result(
        self, tool_name: str, result: Any, compressed: bool = False
    ) -> None:
        """
        Display tool results with beautiful formatting.

        Args:
            tool_name (str): Name of the tool that was executed.
            result (Any): The result returned by the tool.
            compressed (bool): Whether the result was compressed.
        """
        if self.console:
            # Format result based on type
            if isinstance(result, dict):
                # Pretty format JSON results
                result_str = json.dumps(result, indent=2, ensure_ascii=False)
                if len(result_str) > 1000:
                    result_str = result_str[:997] + "..."
                result_display = Syntax(
                    result_str, "json", theme="monokai", line_numbers=False
                )
            elif isinstance(result, str):
                # Truncate long strings
                result_display = result[:1000] + "..." if len(result) > 1000 else result
            else:
                result_display = str(result)[:1000]

            title = f"[bold green]✅ Tool Result: {tool_name}[/]"
            if compressed:
                title += " [dim yellow](compressed)[/]"

            panel = Panel(
                result_display, title=title, border_style="green", expand=False
            )
            self.console.print(panel)

    def display_completion_event(
        self, subtask: "SubTask", success: bool, result: Any = None
    ) -> None:
        """
        Display subtask completion with nice formatting.

        Args:
            subtask (SubTask): The completed subtask.
            success (bool): Whether the subtask completed successfully.
            result (Any): The final result of the subtask.
        """
        if self.console:
            status = (
                "[bold green]✅ Success[/]" if success else "[bold red]❌ Failed[/]"
            )
            border_style = "green" if success else "red"

            content = [
                f"[bold cyan]SubTask:[/] {subtask.content}",
                f"[bold cyan]Status:[/] {status}",
                f"[bold cyan]Duration:[/] {subtask.end_time - subtask.start_time if subtask.end_time and subtask.start_time else 'N/A'} seconds",
            ]

            if result:
                result_str = (
                    json.dumps(result, indent=2, ensure_ascii=False)
                    if isinstance(result, dict)
                    else str(result)
                )
                if len(result_str) > 500:
                    result_str = result_str[:497] + "..."
                content.append(f"\n[bold cyan]Result:[/]\n{result_str}")

            panel = Panel(
                "\n".join(content),
                title="[bold]🏁 SubTask Completed[/]",
                border_style=border_style,
                expand=False,
            )
            self.console.print(panel)

    def display_token_warning(self, current: int, limit: int) -> None:
        """
        Display token limit warning with visual indicator.

        Args:
            current (int): Current token count.
            limit (int): Token limit.
        """
        if self.console:
            percentage = (current / limit) * 100
            warning_text = "[bold yellow]⚠️  Token Usage Warning[/]\n\n"
            warning_text += f"Current tokens: [bold]{current:,}[/] / {limit:,} ([bold red]{percentage:.1f}%[/])\n"
            warning_text += self._create_progress_bar(percentage, "red")

            panel = Panel(warning_text, border_style="yellow", expand=False)
            self.console.print(panel)

    def display_conclusion_stage(self) -> None:
        """Display entering conclusion stage with special formatting."""
        if self.console:
            self.console.print(
                Rule("[bold magenta]🎯 Entering Conclusion Stage[/]", style="magenta")
            )
            panel = Panel(
                "[yellow]The conversation history is approaching the token limit.\n"
                "The agent will now synthesize all gathered information and finalize the subtask.\n"
                "Only the finish tool is available in this stage.[/]",
                title="[bold yellow]⚡ Conclusion Stage[/]",
                border_style="yellow",
                expand=False,
            )
            self.console.print(panel)

    def _create_progress_bar(self, percentage: float, color: str = "blue") -> str:
        """
        Create a simple text-based progress bar.

        Args:
            percentage (float): Progress percentage (0-100).
            color (str): Color for the filled portion.

        Returns:
            str: A formatted progress bar string.
        """
        bar_width = 20
        filled = int((percentage / 100) * bar_width)
        empty = bar_width - filled

        bar = f"[{color}]{'█' * filled}[/][dim]{'░' * empty}[/]"
        return f"{bar} {percentage:.1f}%"

    def display_task_update(self, event: str, details: Optional[str] = None) -> None:
        """
        Display a task update event with appropriate styling.

        Args:
            event (str): The event type.
            details (Optional[str]): Additional details about the event.
        """
        if self.console:
            event_styles = {
                "SUBTASK_STARTED": ("🚀", "green"),
                "SUBTASK_COMPLETED": ("✅", "green"),
                "ENTERED_CONCLUSION_STAGE": ("🎯", "magenta"),
                "ERROR": ("❌", "red"),
            }

            icon, color = event_styles.get(event, ("📌", "white"))
            message = f"[{color}]{icon} {event}[/]"
            if details:
                message += f" - {details}"

            self.console.print(message)

    def display_message_history_info(self, num_messages: int) -> None:
        """
        Display information about message history in a compact format.

        Args:
            num_messages (int): Number of messages in the history.
        """
        if self.console:
            self.console.print(f"[dim]📝 Message history: {num_messages} messages[/]")
