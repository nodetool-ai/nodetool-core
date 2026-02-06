"""
UI Console for displaying Agent progress using Rich.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, Optional

from rich.columns import Columns
from rich.console import Console, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from nodetool.metadata.types import LogEntry, Step, Task, ToolCall

# Display constants
_SPINNER_NAME = "dots"
_MAX_INSTRUCTION_LEN = 100
_REFRESH_PER_SECOND = 8


def _truncate(text: str, max_len: int = _MAX_INSTRUCTION_LEN) -> str:
    """Truncate text with ellipsis if it exceeds max_len."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


class AgentConsole:
    """
    Manages Rich library components for displaying Agent planning and execution status.
    """

    def __init__(self, verbose: bool = True, console: Optional[Console] = None):
        """
        Initialize the AgentConsole.

        Args:
            verbose (bool): Enable/disable rich output.
            console (Optional[Console]): Existing Rich console to use.
        """
        self.verbose: bool = verbose
        self.console: Optional[Console] = console or (Console() if verbose else None)
        self.live: Optional[Live] = None
        self.current_table: Optional[Table] = None
        self.current_tree: Optional[Tree] = None
        self.phase_nodes: dict[str, Any] = {}
        self.step_nodes: dict[str, Any] = {}
        self.current_step: Optional[Step] = None
        self.task: Optional[Task] = None
        self.tool_calls: list[ToolCall] = []

        # Phase-specific logging storage
        self.phase_logs: dict[str, list[dict[str, Any]]] = {}
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
            refresh_per_second=_REFRESH_PER_SECOND,
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
        self.step_nodes = {}
        self.current_phase = None
        self.task = None
        self.tool_calls = []

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
            content_str = _truncate(str(content), 200)

            # Build label based on status
            label: RenderableType
            if is_error:
                label = Text(f"✗ {phase_name}", style="bold red")
            elif status == "Success":
                label = Text(f"✓ {phase_name}", style="green")
            elif status == "Running":
                label = Spinner(
                    _SPINNER_NAME,
                    text=Text(f" {phase_name}", style="bold"),
                )
            else:
                label = Text(f"○ {phase_name}", style="dim")

            if phase_name in self.phase_nodes:
                node = self.phase_nodes[phase_name]
                node.label = label
                node.children.clear()
                # Only show detail for running/error phases
                if status == "Running" or is_error:
                    node.add(Text(content_str, style="dim"))
            else:
                node = self.current_tree.add(label)
                if status == "Running" or is_error:
                    node.add(Text(content_str, style="dim"))
                self.phase_nodes[phase_name] = node

    def create_execution_tree(self, title: str, task: Task, tool_calls: list[ToolCall]) -> Tree:
        """Create a compact rich tree for displaying steps with animated spinners."""
        tree = Tree(f"[bold]{title}[/]", guide_style="dim")
        self.step_nodes = {}
        self.task = task
        self.tool_calls = tool_calls or []

        if not task or not task.steps:
            return tree

        for step in task.steps:
            instr = _truncate(step.instructions)

            # Build label with animated spinner for running steps
            label: RenderableType
            if step.completed:
                label = Text(f"✓ {instr}", style="dim green")
            elif step.is_running():
                label = Spinner(_SPINNER_NAME, text=Text(f" {instr}", style="bold"))
            else:
                label = Text(f"○ {instr}", style="dim")

            step_node = tree.add(label)
            self.step_nodes[step.id] = step_node

            # Show details only for the running step
            if step.is_running():
                if step.depends_on:
                    step_node.add(Text(f"← {', '.join(step.depends_on)}", style="dim"))

                # Show latest log entry only
                if step.logs:
                    step_node.add(self._format_log_entry(step.logs[-1]))

                # Show latest tool call compactly
                step_tool_calls = [c for c in self.tool_calls if c.step_id == step.id]
                if step_tool_calls:
                    last = step_tool_calls[-1]
                    msg = _truncate(str(last.message or ""), 60)
                    count_info = f" (+{len(step_tool_calls) - 1})" if len(step_tool_calls) > 1 else ""
                    step_node.add(
                        Text(f"↳ {last.name}: {msg}{count_info}", style="dim cyan")
                    )
            elif step.completed:
                # Compact summary for completed steps
                step_tool_calls = [c for c in self.tool_calls if c.step_id == step.id]
                if step_tool_calls:
                    step_node.add(
                        Text(f"{len(step_tool_calls)} tool calls", style="dim")
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

    def log_to_phase(self, level: str, message: str, phase_name: Optional[str] = None) -> None:
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

    def get_phase_logs(self, phase_name: str) -> list[dict[str, Any]]:
        """
        Get all log entries for a specific phase.

        Args:
            phase_name (str): The name of the phase.

        Returns:
            List[Dict[str, Any]]: List of log entries for the phase.
        """
        return self.phase_logs.get(phase_name, [])

    def get_all_phase_logs(self) -> dict[str, list[dict[str, Any]]]:
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

    def set_current_step(self, step: Step) -> None:
        """
        Set the current step that is being executed.

        Args:
            step (Step): The step currently being executed.
        """
        self.current_step = step
        # Refresh the tree display to show the new current step's logs
        if self.live and self.current_tree and self.live.is_started:
            self.update_execution_display()

    def log_to_step(self, level: str, message: str) -> None:
        """
        Add a log entry to the current step.

        Args:
            level (str): The log level (debug, info, warning, error).
            message (str): The log message.
        """
        if self.current_step is not None:
            from nodetool.metadata.types import LogEntry

            log_entry = LogEntry(
                message=message,
                level=level,  # type: ignore[arg-type]
                timestamp=int(time.time()),  # type: ignore
            )
            self.current_step.logs.append(log_entry)
            self.update_execution_display()

    def debug_step_only(self, message: object) -> None:
        """
        Add a debug log entry ONLY to the current step (not phase).

        Args:
            message (object): The debug message.
        """
        msg_str = str(message)
        self.log_to_step("debug", msg_str)

    def info_step_only(self, message: object) -> None:
        """
        Add an info log entry ONLY to the current step (not phase).

        Args:
            message (object): The info message.
        """
        msg_str = str(message)
        self.log_to_step("info", msg_str)

    def warning_step_only(self, message: object) -> None:
        """
        Add a warning log entry ONLY to the current step (not phase).

        Args:
            message (object): The warning message.
        """
        msg_str = str(message)
        self.log_to_step("warning", msg_str)

    def error_step_only(self, message: object) -> None:
        """
        Add an error log entry ONLY to the current step (not phase).

        Args:
            message (object): The error message.
        """
        msg_str = str(message)
        self.log_to_step("error", msg_str)

    def update_execution_display(self) -> None:
        """
        Refresh the execution tree with logs for the running step only.
        """
        if self.live and self.current_tree and self.live.is_started and self.task and self.task.steps:
            for step in self.task.steps:
                if step.id in self.step_nodes and step.is_running():
                    step_node = self.step_nodes[step.id]
                    instr = _truncate(step.instructions)

                    # Keep spinner label fresh
                    step_node.label = Spinner(
                        _SPINNER_NAME, text=Text(f" {instr}", style="bold")
                    )

                    # Rebuild children: deps + latest log + latest tool
                    step_node.children.clear()

                    if step.depends_on:
                        step_node.add(Text(f"← {', '.join(step.depends_on)}", style="dim"))

                    if step.logs:
                        step_node.add(self._format_log_entry(step.logs[-1]))

                    step_tool_calls = [c for c in self.tool_calls if c.step_id == step.id]
                    if step_tool_calls:
                        last = step_tool_calls[-1]
                        msg = _truncate(str(last.message or ""), 60)
                        count_info = f" (+{len(step_tool_calls) - 1})" if len(step_tool_calls) > 1 else ""
                        step_node.add(
                            Text(f"↳ {last.name}: {msg}{count_info}", style="dim cyan")
                        )

            self.live.update(self.current_tree)

    def _format_log_entry(self, log: LogEntry) -> Text:
        """
        Format a log entry with minimal styling.

        Args:
            log (LogEntry): The log entry to format.

        Returns:
            Text: The formatted log entry.
        """
        style_map = {
            "debug": "dim",
            "info": "dim cyan",
            "warning": "yellow",
            "error": "bold red",
        }
        icon_map = {"debug": "·", "info": "›", "warning": "⚠", "error": "✗"}  # noqa: RUF001

        style = style_map.get(log.level, "white")
        icon = icon_map.get(log.level, "·")
        msg = _truncate(log.message, 120)

        return Text(f"{icon} {msg}", style=style)

    def debug(self, message: object, style: Optional[str] = None) -> None:
        """
        Add a debug log entry to the current step and current phase.

        Args:
            message (object): The debug message.
            style (Optional[str]): Rich style to apply (ignored, kept for compatibility).
        """
        msg_str = str(message)
        self.log_to_step("debug", msg_str)
        self.log_to_phase("debug", msg_str)

    def info(self, message: object, style: Optional[str] = None) -> None:
        """
        Add an info log entry to the current step and current phase.

        Args:
            message (object): The info message.
            style (Optional[str]): Rich style to apply (ignored, kept for compatibility).
        """
        msg_str = str(message)
        self.log_to_step("info", msg_str)
        self.log_to_phase("info", msg_str)

    def warning(self, message: object, style: Optional[str] = None) -> None:
        """
        Add a warning log entry to the current step and current phase.

        Args:
            message (object): The warning message.
            style (Optional[str]): Rich style to apply (ignored, kept for compatibility).
        """
        msg_str = str(message)
        self.log_to_step("warning", msg_str)
        self.log_to_phase("warning", msg_str)

    def error(self, message: object, style: Optional[str] = None, exc_info: bool = False) -> None:
        """
        Add an error log entry to the current step and current phase.

        Args:
            message (object): The error message.
            style (Optional[str]): Rich style to apply (ignored, kept for compatibility).
            exc_info (bool): Whether to print exception information.
        """
        msg_str = str(message)
        self.log_to_step("error", msg_str)
        self.log_to_phase("error", msg_str)
        if exc_info and self.console:
            self.print_exception()

    def display_step_start(self, step: Step) -> None:
        """
        Display the beginning of a step with beautiful formatting.

        Args:
            step (Step): The step being started.
        """
        if self.console and not (self.live and self.live.is_started):
            panel = Panel(
                f"[bold cyan]Content:[/] {step.instructions}\n"
                f"[bold cyan]ID:[/] {step.id}\n"
                f"[bold cyan]Input Tasks:[/] {', '.join(step.depends_on) if step.depends_on else 'None'}",
                title="[bold green]🚀 Starting Step[/]",
                border_style="green",
                expand=False,
            )
            self.console.print(panel)

    def display_iteration_status(self, iteration: int, max_iterations: int, token_count: int, max_tokens: int) -> None:
        """
        Display current iteration and token status with progress bars.

        Args:
            iteration (int): Current iteration number.
            max_iterations (int): Maximum allowed iterations.
            token_count (int): Current token count.
            max_tokens (int): Maximum allowed tokens.
        """
        if self.console and not (self.live and self.live.is_started):
            # Create progress bars for iterations and tokens
            iteration_progress = f"[cyan]Iteration:[/] {iteration}/{max_iterations}"
            token_progress = f"[cyan]Tokens:[/] {token_count}/{max_tokens}"

            # Calculate percentages
            iter_percent = (iteration / max_iterations) * 100
            token_percent = (token_count / max_tokens) * 100

            # Create visual progress bars
            iter_bar = self._create_progress_bar(iter_percent, "blue")
            token_bar = self._create_progress_bar(token_percent, "yellow" if token_percent < 80 else "red")

            # Display in columns
            columns = Columns(
                [f"{iteration_progress}\n{iter_bar}", f"{token_progress}\n{token_bar}"],
                equal=True,
                expand=True,
            )

            self.console.print(Panel(columns, title="[bold]📊 Status[/]", border_style="dim"))

    def display_tool_execution(self, tool_name: str, args: dict) -> None:
        """
        Display tool execution with beautiful formatting.

        Args:
            tool_name (str): Name of the tool being executed.
            args (dict): Arguments passed to the tool.
        """
        if self.console and not (self.live and self.live.is_started):
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

    def display_tool_result(self, tool_name: str, result: Any, compressed: bool = False) -> None:
        """
        Display tool results with beautiful formatting.

        Args:
            tool_name (str): Name of the tool that was executed.
            result (Any): The result returned by the tool.
            compressed (bool): Whether the result was compressed.
        """
        if self.console and not (self.live and self.live.is_started):
            # Format result based on type
            if isinstance(result, dict):
                # Pretty format JSON results
                result_str = json.dumps(result, indent=2, ensure_ascii=False)
                if len(result_str) > 1000:
                    result_str = result_str[:997] + "..."
                result_display = Syntax(result_str, "json", theme="monokai", line_numbers=False)
            elif isinstance(result, str):
                # Truncate long strings
                result_display = result[:1000] + "..." if len(result) > 1000 else result
            else:
                result_display = str(result)[:1000]

            title = f"[bold green]✅ Tool Result: {tool_name}[/]"
            if compressed:
                title += " [dim yellow](compressed)[/]"

            panel = Panel(result_display, title=title, border_style="green", expand=False)
            self.console.print(panel)

    def display_completion_event(self, step: Step, success: bool, result: Any = None) -> None:
        """
        Display step completion with nice formatting.

        Args:
            step (Step): The completed step.
            success (bool): Whether the step completed successfully.
            result (Any): The final result of the step.
        """
        if self.console and not (self.live and self.live.is_started):
            status = "[bold green]✅ Success[/]" if success else "[bold red]❌ Failed[/]"
            border_style = "green" if success else "red"

            content = [
                f"[bold cyan]Step:[/] {step.instructions}",
                f"[bold cyan]Status:[/] {status}",
                f"[bold cyan]Duration:[/] {step.end_time - step.start_time if step.end_time and step.start_time else 'N/A'} seconds",
            ]

            if result:
                result_str = (
                    json.dumps(result, indent=2, ensure_ascii=False) if isinstance(result, dict) else str(result)
                )
                if len(result_str) > 500:
                    result_str = result_str[:497] + "..."
                content.append(f"\n[bold cyan]Result:[/]\n{result_str}")

            panel = Panel(
                "\n".join(content),
                title="[bold]🏁 Step Completed[/]",
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
        if self.console and not (self.live and self.live.is_started):
            percentage = (current / limit) * 100
            warning_text = "[bold yellow]⚠️  Token Usage Warning[/]\n\n"
            warning_text += f"Current tokens: [bold]{current:,}[/] / {limit:,} ([bold red]{percentage:.1f}%[/])\n"
            warning_text += self._create_progress_bar(percentage, "red")

            panel = Panel(warning_text, border_style="yellow", expand=False)
            self.console.print(panel)

    def display_conclusion_stage(self) -> None:
        """Display entering conclusion stage with special formatting."""
        if self.console and not (self.live and self.live.is_started):
            self.console.print(Rule("[bold magenta]🎯 Entering Conclusion Stage[/]", style="magenta"))
            panel = Panel(
                "[yellow]The conversation history is approaching the token limit.\n"
                "The agent will now synthesize all gathered information and finalize the step.\n"
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
        if self.console and not (self.live and self.live.is_started):
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
