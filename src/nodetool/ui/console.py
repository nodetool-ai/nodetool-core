"""
UI Console for displaying Agent progress using Rich.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional

from rich.columns import Columns
from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text

# Try to import textual - it's an optional dependency
try:
    from textual.app import App, ComposeResult
    from textual.containers import Vertical
    from textual.widgets import RichLog, Static

    TEXTUAL_AVAILABLE = True
except ImportError:
    # textual is not installed, create stub types
    TEXTUAL_AVAILABLE = False

    # Create stub types for runtime when textual is not available
    class App:  # type: ignore[no-redef]
        pass

    class Vertical:  # type: ignore[no-redef]
        @contextmanager
        def __call__(self) -> Iterator[None]:
            yield

    class ComposeResult:  # type: ignore[no-redef]
        pass

    class RichLog:  # type: ignore[no-redef]
        pass

    class Static:  # type: ignore[no-redef]
        pass

# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from nodetool.metadata.types import LogEntry, Step, Task, ToolCall

# Display constants
_MAX_INSTRUCTION_LEN = 100
_MAX_AGENT_OUTPUT_CHARS = 12000
_SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


@dataclass
class _LiveContent:
    """Internal representation of live screen content."""

    kind: str
    title: str
    body: str = ""


if TEXTUAL_AVAILABLE:

    class _AgentLiveApp(App[None]):  # type: ignore[misc]
        """Textual app for rendering agent status and output panes."""

        CSS = """
        Screen {
            layout: vertical;
        }
        #status {
            height: 1fr;
            border: round $primary;
            padding: 0 1;
        }
        #output_log {
            height: 3fr;
            border: round cyan;
        }
        .hidden {
            display: none;
        }
        """

        def __init__(self, manager: AgentConsole):
            super().__init__()
            self.manager = manager
            self._last_status = ""
            self._last_output = ""

        def compose(self) -> ComposeResult:
            with Vertical():  # type: ignore[misc]
                yield Static("", id="status")
                yield RichLog(id="output_log", auto_scroll=True, wrap=True, highlight=False, markup=False)

        def on_mount(self) -> None:
            self.set_interval(0.15, self._refresh_from_state)
            self._refresh_from_state()

        def _refresh_from_state(self) -> None:
            status_widget = self.query_one("#status", Static)
            output_widget = self.query_one("#output_log", RichLog)

            status_title = self.manager.live_title or "Agent"
            status_text = self.manager._render_live_status_text()
            full_status = f"{status_title}\n\n{status_text}" if status_text else status_title

            if full_status != self._last_status:
                status_widget.update(full_status)
                self._last_status = full_status

            if self.manager.show_agent_output_widget:
                output_widget.remove_class("hidden")
            else:
                output_widget.add_class("hidden")

            output_text = self.manager.agent_output_buffer
            if output_text == self._last_output:
                return

            if not output_text:
                output_widget.clear()
                self._last_output = ""
                return

            if not output_text.startswith(self._last_output):
                output_widget.clear()
                output_widget.write(output_text)
                self._last_output = output_text
                return

            delta = output_text[len(self._last_output) :]
            if not delta:
                return

            output_widget.write(delta)
            self._last_output = output_text


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
        self.live_title: str = ""
        self.live: Optional[Any] = None  # _AgentLiveApp if TEXTUAL_AVAILABLE
        self.live_task: Optional[asyncio.Task[Any]] = None
        self.live_active: bool = False
        self._external_update_callback: Optional[Callable[[], None]] = None
        self.current_step: Optional[Step] = None
        self.task: Optional[Task] = None
        self.tool_calls: list[ToolCall] = []
        self.current_live_content: Optional[_LiveContent] = None
        self._planning_nodes: dict[str, tuple[str, str, str, bool]] = {}
        self._planning_order: list[str] = []
        self.show_agent_output_widget: bool = False
        self.agent_output_buffer: str = ""
        self.agent_output_segments: list[tuple[bool, str]] = []
        self.agent_output_title: str = "Agent Output"
        self._spinner_index: int = 0

        # Phase-specific logging storage
        self.phase_logs: dict[str, list[dict[str, Any]]] = {}
        self.current_phase: Optional[str] = None

    def set_external_update_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Attach a callback for embedded Textual hosts that render AgentConsole state."""
        self._external_update_callback = callback

    def _notify_external_update(self) -> None:
        if self._external_update_callback:
            self._external_update_callback()

    def start_live(self, initial_content: Any, show_agent_output_widget: bool = False, agent_output_title: str = "Agent Output") -> None:
        """
        Start the Rich Live display with initial content (table or tree).

        Args:
            initial_content: Internal live content object.
        """
        if not self.verbose or self.live_active:
            return

        if isinstance(initial_content, _LiveContent):
            self.current_live_content = initial_content
            self.live_title = initial_content.title
        else:
            self.current_live_content = _LiveContent(kind="execution", title="Agent", body=str(initial_content))
            self.live_title = "Agent"

        self.show_agent_output_widget = show_agent_output_widget
        self.agent_output_title = agent_output_title
        self.agent_output_buffer = ""
        self.agent_output_segments = []
        self.live_active = True
        self._notify_external_update()

        # Embedded mode: outer app owns the screen and renders our state.
        if self._external_update_callback:
            return

        # Textual is required for live display
        if not TEXTUAL_AVAILABLE:
            self.live_active = False
            return

        self.live = _AgentLiveApp(self)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.live = None
            self.live_active = False
            return
        self.live_task = loop.create_task(self.live.run_async(inline=True, inline_no_clear=True))

    def stop_live(self) -> None:
        """Stop the Rich Live display if it is active."""
        if self.live:
            self.live.exit()
        self.live = None
        self.live_task = None
        self.live_active = False
        self.live_title = ""
        self.current_live_content = None
        self._planning_nodes = {}
        self._planning_order = []
        self.current_phase = None
        self.task = None
        self.tool_calls = []
        self.show_agent_output_widget = False
        self.agent_output_buffer = ""
        self.agent_output_segments = []
        self.agent_output_title = "Agent Output"
        self._notify_external_update()

    def update_live(self, new_content: Any) -> None:
        """
        Update the Rich Live display with new content.

        Args:
            new_content: Internal live content object.
        """
        if isinstance(new_content, _LiveContent):
            self.current_live_content = new_content
            self.live_title = new_content.title
            self._notify_external_update()

    def is_live_active(self) -> bool:
        """Return whether the live renderer is currently active."""
        return self.live_active

    def append_agent_output(self, content: str, *, thinking: bool = False) -> None:
        """Append text to the reserved live output panel used during execution."""
        if not content:
            return

        if self.agent_output_segments and self.agent_output_segments[-1][0] == thinking:
            prev_thinking, prev_text = self.agent_output_segments[-1]
            self.agent_output_segments[-1] = (prev_thinking, prev_text + content)
        else:
            self.agent_output_segments.append((thinking, content))

        self.agent_output_buffer = "".join(text for _, text in self.agent_output_segments)
        if len(self.agent_output_buffer) > _MAX_AGENT_OUTPUT_CHARS:
            self.agent_output_buffer = self.agent_output_buffer[-_MAX_AGENT_OUTPUT_CHARS :]
            # Keep segment list consistent with truncated buffer.
            rebuilt: list[tuple[bool, str]] = []
            remaining = self.agent_output_buffer
            for is_thinking, text in reversed(self.agent_output_segments):
                if not remaining:
                    break
                keep_len = min(len(text), len(remaining))
                kept = text[-keep_len:]
                rebuilt.append((is_thinking, kept))
                remaining = remaining[:-keep_len]
            rebuilt.reverse()
            self.agent_output_segments = rebuilt
        self._notify_external_update()

    def get_agent_output_segments(self) -> list[tuple[bool, str]]:
        """Return agent output segmented by thinking/non-thinking chunks."""
        return list(self.agent_output_segments)

    def create_planning_tree(self, title: str) -> _LiveContent:
        """
        Create content for displaying planning phases.

        Returns:
            _LiveContent: Internal live content object.
        """
        self._planning_nodes = {}
        self._planning_order = []
        content = _LiveContent(kind="planning", title=title)
        self.current_live_content = content
        return content

    def update_planning_display(
        self,
        phase_name: str,
        status: str,
        content: str | Text,
        is_error: bool = False,
        show_phase: bool = True,
    ) -> None:
        """
        Add a node to the current planning tree in the live display.

        Args:
            phase_name (str): Name of the planning phase.
            status (str): Status of the phase (e.g., "Running", "Success", "Failed").
            content (Union[str, Text]): Details or output for the phase.
            is_error (bool): Flag indicating if the status represents an error.
        """
        content_str = _truncate(str(content), 200)
        node_key = phase_name if show_phase else "__planner_status__"
        if node_key not in self._planning_nodes:
            self._planning_order.append(node_key)
        self._planning_nodes[node_key] = (status, phase_name if show_phase else "", content_str, is_error)
        self.current_live_content = _LiveContent(kind="planning", title=self.live_title or "Task Planner")
        self._notify_external_update()

    def create_execution_tree(self, title: str, task: Task, tool_calls: list[ToolCall]) -> _LiveContent:
        """Create textual execution status content."""
        self.task = task
        self.tool_calls = tool_calls or []
        body = self._render_execution_body()
        content = _LiveContent(kind="execution", title=title, body=body)
        self.current_live_content = content
        self._notify_external_update()
        return content

    def _render_execution_body(self) -> str:
        """Render execution state as text for Textual status pane."""
        if not self.task or not self.task.steps:
            return ""
        spinner = _SPINNER_FRAMES[self._spinner_index % len(_SPINNER_FRAMES)]
        self._spinner_index += 1
        lines: list[str] = []
        for step in self.task.steps:
            instr = _truncate(step.instructions)
            if step.completed:
                prefix = "✓"
            elif step.is_running():
                prefix = spinner
            else:
                prefix = "○"
            lines.append(f"{prefix} {instr}")
            if step.is_running():
                if step.depends_on:
                    lines.append(f"  ← {', '.join(step.depends_on)}")
                if step.logs:
                    lines.append(f"  {self._format_log_entry(step.logs[-1]).plain}")
                step_tool_calls = [c for c in self.tool_calls if c.step_id == step.id]
                if step_tool_calls:
                    last = step_tool_calls[-1]
                    msg = _truncate(str(last.message or ""), 60)
                    count_info = f" (+{len(step_tool_calls) - 1})" if len(step_tool_calls) > 1 else ""
                    lines.append(f"  ↳ {last.name}: {msg}{count_info}")
            elif step.completed:
                step_tool_calls = [c for c in self.tool_calls if c.step_id == step.id]
                if step_tool_calls:
                    lines.append(f"  {len(step_tool_calls)} tool calls")
        return "\n".join(lines)

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
        # Refresh live state to show the new current step's logs.
        if self.is_live_active():
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
        if self.task and self.current_live_content and self.current_live_content.kind == "execution":
            self.current_live_content.body = self._render_execution_body()
            self._notify_external_update()

    def _render_live_status_text(self) -> str:
        """Render current live status text for Textual status pane."""
        if not self.current_live_content:
            return ""
        if self.current_live_content.kind == "planning":
            lines: list[str] = []
            for key in self._planning_order:
                status, label_text, content, is_error = self._planning_nodes[key]
                if is_error:
                    prefix = "✗"
                elif status == "Success":
                    prefix = "✓"
                elif status == "Running":
                    prefix = _SPINNER_FRAMES[self._spinner_index % len(_SPINNER_FRAMES)]
                    self._spinner_index += 1
                else:
                    prefix = "○"
                header = f"{prefix} {label_text}".strip()
                lines.append(header)
                if status == "Running" or is_error or key == "__planner_status__":
                    lines.append(f"  {content}")
            return "\n".join(lines)
        return self.current_live_content.body

    def has_running_activity(self) -> bool:
        """Return True when planner/task execution has active running items."""
        if self.current_live_content and self.current_live_content.kind == "planning":
            return any(status == "Running" for status, _, _, _ in self._planning_nodes.values())
        if self.task and self.task.steps:
            return any(step.is_running() for step in self.task.steps)
        return False

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
        if self.console and not self.is_live_active():
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
        if self.console and not self.is_live_active():
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
        if self.console and not self.is_live_active():
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
        if self.console and not self.is_live_active():
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
        if self.console and not self.is_live_active():
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
        if self.console and not self.is_live_active():
            percentage = (current / limit) * 100
            warning_text = "[bold yellow]⚠️  Token Usage Warning[/]\n\n"
            warning_text += f"Current tokens: [bold]{current:,}[/] / {limit:,} ([bold red]{percentage:.1f}%[/])\n"
            warning_text += self._create_progress_bar(percentage, "red")

            panel = Panel(warning_text, border_style="yellow", expand=False)
            self.console.print(panel)

    def display_conclusion_stage(self) -> None:
        """Display entering conclusion stage with special formatting."""
        if self.console and not self.is_live_active():
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
        if self.console and not self.is_live_active():
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
