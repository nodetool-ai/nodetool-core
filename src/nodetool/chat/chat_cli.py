"""CLI interface for the Chain of Thought (CoT) agent.

This module provides a command-line interface for interacting with the CoT agent.
It supports multiple LLM providers, model selection, and tool management.

Features:
- Interactive chat with command history and tab completion
- Support for multiple LLM providers (OpenAI, Anthropic, Ollama)
- Model selection and provider management
- Chain of Thought reasoning with step-by-step output
- Tool usage and execution
- Workspace management with file system commands
"""

import asyncio
import json
import os
import re  # Add re for grep
import shutil  # Add shutil for cp and mv
import shlex
import subprocess
import sys
import traceback
from datetime import datetime
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import NestedCompleter, PathCompleter, WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from pydantic import ValidationError
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.suggester import SuggestFromList
from textual.widgets import Input, RichLog, Static

from nodetool.config.logging_config import configure_logging, get_logger
from nodetool.config.settings import get_log_path
from nodetool.messaging.agent_message_processor import AgentMessageProcessor
from nodetool.messaging.message_processor import MessageProcessor
from nodetool.messaging.regular_chat_processor import RegularChatProcessor
from nodetool.metadata.types import LanguageModel, Message, MessageTextContent, Provider
from nodetool.providers import get_provider
from nodetool.runtime.resources import ResourceScope
from nodetool.ui.console import AgentConsole
from nodetool.workflows.processing_context import ProcessingContext

log = get_logger(__name__)


# Helper --------------------------------------------------------------------


def _determine_chat_log_level(explicit_level: Optional[str] = None) -> str:
    """Resolve the logging level for the chat CLI.

    Precedence ordering (highest first):
    1. Explicit `--log-level` argument passed into the CLI entrypoint
    2. `NODETOOL_CHAT_LOG_LEVEL` environment variable
    3. Generic `LOG_LEVEL` / `NODETOOL_LOG_LEVEL` environment variables
    """

    def _normalize(value: str) -> str:
        return value.upper().strip()

    if explicit_level:
        return _normalize(explicit_level)

    chat_specific = os.getenv("NODETOOL_CHAT_LOG_LEVEL")
    if chat_specific:
        return _normalize(chat_specific)

    generic_level = os.getenv("LOG_LEVEL")
    if generic_level:
        return _normalize(generic_level)

    nodetool_level = os.getenv("NODETOOL_LOG_LEVEL")
    if nodetool_level:
        return _normalize(nodetool_level)

    debug_flag = os.getenv("DEBUG")
    if debug_flag and debug_flag.lower() not in {"", "0", "false", "no", "off"}:
        return "DEBUG"

    return "ERROR"


class _ConsoleProxy:
    """Proxy Rich Console output into a callback (for Textual embedding)."""

    def __init__(self, base_console: Console):
        self._base = base_console
        self._writer: Optional[Callable[[str], None]] = None
        self._clear_cb: Optional[Callable[[], None]] = None

    def set_writer(self, writer: Optional[Callable[[str], None]], clear_cb: Optional[Callable[[], None]] = None) -> None:
        self._writer = writer
        self._clear_cb = clear_cb

    def print(self, *args: Any, **kwargs: Any) -> None:
        if not self._writer:
            self._base.print(*args, **kwargs)
            return
        with self._base.capture() as capture:
            self._base.print(*args, **kwargs)
        rendered = capture.get().rstrip("\n")
        if rendered:
            self._writer(rendered)

    def clear(self, *args: Any, **kwargs: Any) -> None:
        if self._writer and self._clear_cb:
            self._clear_cb()
            return
        self._base.clear(*args, **kwargs)

    def print_exception(self, *args: Any, **kwargs: Any) -> None:
        if not self._writer:
            self._base.print_exception(*args, **kwargs)
            return
        with self._base.capture() as capture:
            self._base.print_exception(*args, **kwargs)
        rendered = capture.get().rstrip("\n")
        if rendered:
            self._writer(rendered)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


class ChatTextualApp(App[None]):
    """Textual host for the interactive NodeTool chat UI."""

    CSS = """
    Screen {
        background: #0a0b0d;
        color: #d7d7d7;
        layout: vertical;
        padding: 1;
    }
    #topbar {
        height: auto;
        border: round #1d1f23;
        color: #f2f2f2;
        background: #0d0f12;
        padding: 0 1 0 1;
        margin: 0 0 1 0;
    }
    #main {
        height: 1fr;
        width: 100%;
    }
    #left_pane {
        width: 3fr;
        border: round #1a1d22;
        background: #090a0c;
        margin: 0 1 0 0;
    }
    #agent_output {
        height: 1fr;
        border: none;
        padding: 0 1 1 1;
        color: #e6e6e6;
    }
    #input_row {
        height: 3;
        border-top: solid #1a1d22;
        padding: 0 1;
    }
    #input {
        border: none;
        background: #111317;
    }
    #hint_row {
        height: 1;
        color: #7b7f86;
        padding: 0 1;
        margin: 0 0 1 0;
    }
    #sidebar {
        width: 1fr;
        min-width: 34;
        border: round #1a1d22;
        background: #0d0f12;
        padding: 0 1;
    }
    #sidebar_title {
        height: auto;
        color: #f1f1f1;
        text-style: bold;
        margin: 0 0 1 0;
    }
    .sidebar_section {
        height: auto;
        margin: 0 0 1 0;
        color: #c7c7c7;
    }
    #task_plan {
        height: 1fr;
        border: round #20242b;
        padding: 0 1;
        background: #0a0c0f;
        color: #dfdfdf;
    }
    #sidebar_footer {
        height: auto;
        color: #8a8f98;
        margin: 1 0 0 0;
    }
    """

    def __init__(self, cli: "ChatCLI"):
        super().__init__()
        self.cli = cli
        self._busy = False
        self._processing_task: asyncio.Task[None] | None = None
        self._last_output_key: Any = None
        self._manual_output_buffer = ""
        self._ansi_re = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
        self._rich_tag_re = re.compile(r"\[/?[a-zA-Z0-9 _=#.-]+\]")
        self._input_history: list[str] = []
        self._history_index: int | None = None
        self._history_draft: str = ""
        self._refresh_dirty: bool = True

    def compose(self) -> ComposeResult:
        yield Static("", id="topbar")
        with Horizontal(id="main"):
            with Vertical(id="left_pane"):
                yield RichLog(id="agent_output", wrap=True, auto_scroll=False, markup=False, highlight=False)
                with Vertical(id="input_row"):
                    yield Input(placeholder="Type message or command", id="input")
                yield Static("tab autocomplete   up/down history   /help commands", id="hint_row")
            with Vertical(id="sidebar"):
                yield Static("Session", id="sidebar_title")
                yield Static("", classes="sidebar_section", id="sidebar_context")
                yield Static("Task Plan\n\nidle", id="task_plan")
                yield Static("nodetool chat", id="sidebar_footer")

    async def on_mount(self) -> None:
        self.cli.console.set_writer(self._write_chat_log, self._clear_chat_log)
        self.cli.display_manager.set_external_update_callback(self._mark_refresh_dirty)
        await self.cli.ensure_selected_model()
        self._load_input_history()
        self._refresh_input_suggester()
        self._refresh_topbar()
        self.set_interval(0.15, self._refresh_if_dirty)

    def on_unmount(self) -> None:
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
        self.cli.console.set_writer(None, None)
        self.cli.display_manager.set_external_update_callback(None)

    def _mark_refresh_dirty(self) -> None:
        self._refresh_dirty = True

    def _refresh_if_dirty(self) -> None:
        manager = self.cli.display_manager
        if not self._refresh_dirty and not manager.has_running_activity():
            return
        self._refresh_dirty = False
        self._refresh_agent_panel()

    def _write_chat_log(self, text: str) -> None:
        if not text:
            return
        cleaned = self._ansi_re.sub("", text).replace("\r", "")
        cleaned = self._rich_tag_re.sub("", cleaned)
        self._manual_output_buffer += (cleaned + "\n")
        self._mark_refresh_dirty()

    def _clear_chat_log(self) -> None:
        self._manual_output_buffer = ""
        self._mark_refresh_dirty()

    def _refresh_topbar(self) -> None:
        provider_display = (
            self.cli.selected_model.provider.value
            if self.cli.selected_model
            else self.cli.selected_provider.value
            if self.cli.selected_provider
            else "None"
        )
        model_display = self.cli.selected_model.id if self.cli.selected_model else self.cli.model_id_from_settings or "None"
        agent_display = "ON" if self.cli.agent_mode else "OFF"
        topbar = self.query_one("#topbar", Static)
        topbar.update(f"Provider: {provider_display}    Model: {model_display}    Agent: {agent_display} (/agent)")

        sidebar_context = self.query_one("#sidebar_context", Static)
        token_line = "tokens: n/a"
        try:
            usage = getattr(getattr(self.cli, "selected_model", None), "usage", None)
            if usage:
                token_line = f"tokens: {usage}"
        except Exception:
            pass
        sidebar_context.update(
            "Context\n"
            f"{token_line}\n\n"
            "MCP\n"
            "• tools available"
        )

    def _load_input_history(self) -> None:
        """Load history from prompt_toolkit FileHistory format or plain lines."""
        history_path = Path(self.cli.history_file)
        if not history_path.exists():
            self._input_history = []
            return

        entries: list[str] = []
        try:
            text = history_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            self._input_history = []
            return

        # prompt_toolkit FileHistory stores entries as blocks of +prefixed lines.
        current: list[str] = []
        saw_plus = False
        for raw in text.splitlines():
            if raw.startswith("+"):
                saw_plus = True
                current.append(raw[1:])
                continue
            if current:
                entry = "\n".join(current).strip()
                if entry:
                    entries.append(entry)
                current = []
            if raw and not raw.startswith("#") and not saw_plus:
                entries.append(raw.strip())
        if current:
            entry = "\n".join(current).strip()
            if entry:
                entries.append(entry)

        deduped: list[str] = []
        seen: set[str] = set()
        for item in entries:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        self._input_history = deduped[-500:]

    def _append_input_history(self, value: str) -> None:
        """Persist input history in prompt_toolkit-compatible format."""
        value = value.strip()
        if not value:
            return
        if self._input_history and self._input_history[-1] == value:
            return
        self._input_history.append(value)
        self._input_history = self._input_history[-500:]
        self._refresh_input_suggester()

        history_path = Path(self.cli.history_file)
        try:
            history_path.parent.mkdir(parents=True, exist_ok=True)
            with history_path.open("a", encoding="utf-8") as f:
                f.write(f"\n# {datetime.now().isoformat()}\n")
                for line in value.splitlines():
                    f.write(f"+{line}\n")
        except Exception:
            pass

    def _refresh_input_suggester(self) -> None:
        """Update inline suggestion candidates for the input box."""
        commands = sorted({f"/{name}" for name in self.cli.commands.keys()})
        workspace_cmds = ["pwd", "ls", "cd", "mkdir", "rm", "open", "cat", "cp", "mv", "grep", "cdw"]
        candidates = commands + workspace_cmds + self._input_history[-100:]
        input_widget = self.query_one("#input", Input)
        input_widget.suggester = SuggestFromList(candidates, case_sensitive=False)

    def _path_completion_candidates(self, current: str) -> list[str]:
        try:
            parts = shlex.split(current)
        except ValueError:
            parts = current.split()
        if not parts:
            return []

        command = parts[0]
        if command not in {"ls", "cd", "mkdir", "rm", "open", "cat", "cp", "mv", "grep"}:
            return []

        tokens = current.split()
        last = tokens[-1] if len(tokens) > 1 else ""
        if command in {"cp", "mv"} and len(tokens) > 2:
            last = tokens[-1]
        if command in {"grep"} and len(tokens) < 2:
            return []

        base = Path.cwd()
        prefix = last or "."
        expanded = Path(prefix).expanduser()
        parent = expanded.parent if expanded.parent != Path("") else base
        stem = expanded.name if expanded.name else ""

        if not parent.is_absolute():
            parent = (base / parent).resolve()
        if not parent.exists():
            return []

        matches: list[str] = []
        for child in sorted(parent.iterdir(), key=lambda p: p.name):
            if stem and not child.name.startswith(stem):
                continue
            display = str(child)
            if child.is_dir():
                display += "/"
            replacement = current[: len(current) - len(last)] + display if last else f"{current} {display}".strip()
            matches.append(replacement)
        return matches[:50]

    def _apply_tab_completion(self) -> bool:
        input_widget = self.query_one("#input", Input)
        current = input_widget.value
        if not current:
            return False

        prefix_matches = [h for h in self._input_history if h.startswith(current)]
        command_matches = [f"/{name}" for name in self.cli.commands.keys() if f"/{name}".startswith(current)]
        workspace_matches = [c for c in ["pwd", "ls", "cd", "mkdir", "rm", "open", "cat", "cp", "mv", "grep", "cdw"] if c.startswith(current)]
        path_matches = self._path_completion_candidates(current)

        candidates = []
        for collection in (command_matches, workspace_matches, prefix_matches, path_matches):
            for candidate in collection:
                if candidate not in candidates:
                    candidates.append(candidate)

        if len(candidates) == 1:
            completed = candidates[0]
            if completed in {f"/{name}" for name in self.cli.commands.keys()}:
                completed += " "
            input_widget.value = completed
            input_widget.cursor_position = len(completed)
            return True

        if len(candidates) > 1:
            self._write_chat_log("[dim]Suggestions:[/dim] " + ", ".join(candidates[:8]))
            return True
        return False

    def _refresh_agent_panel(self) -> None:
        manager = self.cli.display_manager
        status_widget = self.query_one("#task_plan", Static)
        status_title = manager.live_title or "Task Plan"
        status_body = manager._render_live_status_text() if manager.is_live_active() else "idle"
        status_widget.update(f"{status_title}\n\n{status_body}")

        if manager.is_live_active() and manager.show_agent_output_widget:
            output_segments = manager.get_agent_output_segments()
            output_key: Any = ("live", manager.agent_output_buffer)
        else:
            output_segments = [(False, self._manual_output_buffer)]
            output_key = ("manual", self._manual_output_buffer)

        if output_key == self._last_output_key:
            return

        output_widget = self.query_one("#agent_output", RichLog)
        output_widget.clear()
        for renderable in self._build_output_renderables(output_segments):
            output_widget.write(renderable)
        self._last_output_key = output_key

    def _build_output_renderables(self, segments: list[tuple[bool, str]]) -> list[Any]:
        """Build renderables: markdown for output, grey text for thinking chunks."""
        renderables: list[Any] = []
        for is_thinking, text in segments:
            if not text:
                continue
            if is_thinking:
                renderables.append(Text(text, style="grey62"))
            else:
                renderables.append(Markdown(text))
        if not renderables:
            return [Text("")]
        return renderables

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.rstrip("\n")
        event.input.value = ""
        self._history_index = None
        normalized = value.strip()
        if not normalized:
            return
        self._append_input_history(normalized)
        if self._busy:
            self._write_chat_log("[yellow]Processing previous request...[/yellow]")
            return
        self._busy = True
        self._write_chat_log(f"[bold cyan]>[/bold cyan] {escape(normalized)}")
        self._processing_task = asyncio.create_task(self._process_submitted_input(normalized, value))

    async def _process_submitted_input(self, normalized: str, value: str) -> None:
        """Run chat/agent processing in background to keep UI responsive."""
        try:
            if normalized.startswith(
                ("pwd", "ls", "cd", "mkdir", "rm", "open", "cat", "cp", "mv", "grep", "cdw")
            ):
                parts = normalized.split()
                self.cli.handle_workspace_command(parts[0], parts[1:] if len(parts) > 1 else [])
                return

            if normalized.startswith("/"):
                parts = normalized.lower().split()
                should_exit = await self.cli.handle_command(parts[0], parts[1:] if len(parts) > 1 else [])
                self._refresh_topbar()
                if should_exit:
                    self.exit()
                return

            if not self.cli.selected_model:
                self._write_chat_log("[bold red]Error:[/bold red] No model selected")
                return

            if self.cli.agent_mode:
                await self.cli.process_agent_response(value)
            else:
                await self.cli.process_regular_message(value)
        except Exception as exc:
            self._write_chat_log(f"[bold red]Error:[/bold red] {escape(str(exc))}")
            self._write_chat_log(
                Syntax(traceback.format_exc(), "python", theme="monokai", line_numbers=True).code
            )
        finally:
            self._busy = False

    async def on_key(self, event) -> None:
        """Restore shell-like input history and tab completion in Textual input."""
        input_widget = self.query_one("#input", Input)
        if self.focused is not input_widget:
            return

        if event.key == "up":
            if not self._input_history:
                return
            if self._history_index is None:
                self._history_draft = input_widget.value
                self._history_index = len(self._input_history) - 1
            else:
                self._history_index = max(0, self._history_index - 1)
            value = self._input_history[self._history_index]
            input_widget.value = value
            input_widget.cursor_position = len(value)
            event.prevent_default()
            event.stop()
            return

        if event.key == "down":
            if self._history_index is None:
                return
            if self._history_index < len(self._input_history) - 1:
                self._history_index += 1
                value = self._input_history[self._history_index]
            else:
                self._history_index = None
                value = self._history_draft
            input_widget.value = value
            input_widget.cursor_position = len(value)
            event.prevent_default()
            event.stop()
            return

        if event.key == "tab":
            if self._apply_tab_completion():
                event.prevent_default()
                event.stop()


class ChatCLI:
    """Modern interactive command-line chat interface with rich UI and advanced features.

    This class implements a command-line interface for interacting with various AI models
    through different providers (OpenAI, Anthropic, Ollama) with a modern UI.

    Features:
    1. Rich, colorful terminal UI with tables and styled text
    2. Advanced command completion with context-aware suggestions
    3. Structured command system with help documentation
    4. Progress indicators for long-running operations
    5. Multi-line input support
    6. Enhanced history management
    7. All features from the original CLI (providers, models, agent, tools, etc.)

    The CLI supports commands through a command registry system that's easy to extend.
    """

    def __init__(self):
        """Initialize the ChatCLI with default settings and a rich console for output."""
        # Suppress warnings
        import warnings

        warnings.filterwarnings("ignore", category=UserWarning)

        # Console proxy to route output either to terminal or Textual log widget.
        self.console = _ConsoleProxy(Console())
        self.display_manager = AgentConsole(console=self.console)

        # Configure logging: all logs to file, nothing to console
        log_level = _determine_chat_log_level()
        configure_logging(
            level=log_level,
            log_file=get_log_path("chat.log"),
            console_output=False,
        )

        # Initialize state
        self.context = ProcessingContext(user_id="1", auth_token="local_token")
        self.messages: list[Message] = []
        self.agent_mode = True  # Default to agent mode ON - omnipotent agent

        # Store selected LanguageModel object and model ID preference
        self.language_models: list[LanguageModel] = []
        self.selected_model: Optional[LanguageModel] = None
        self.model_id_from_settings: Optional[str] = None
        self.selected_provider: Optional[Provider] = None
        self._models_by_provider: dict[Provider, list[LanguageModel]] = {}
        self._completer = None

        # Model attributes for agent config (can be overridden)
        # These might eventually also become LanguageModel selections
        # For now, we'll default them based on the primary selected model
        self.planner_model_id: Optional[str] = None
        self.summarization_model_id: Optional[str] = None
        self.retrieval_model_id: Optional[str] = None

        self.settings_file = os.path.join(os.path.expanduser("~"), ".nodetool_settings")
        self.history_file = os.path.join(os.path.expanduser("~"), ".nodetool_history")

        # Register commands
        self.commands = {}
        self.register_commands()

        # Load settings if they exist (loads model_id_from_settings)
        self.load_settings()

    def register_commands(self):
        """Register all available commands."""
        from nodetool.chat.commands.agent import AgentCommand
        from nodetool.chat.commands.clear import ClearCommand
        from nodetool.chat.commands.debug import DebugCommand
        from nodetool.chat.commands.exit import ExitCommand
        from nodetool.chat.commands.help import HelpCommand
        from nodetool.chat.commands.model import ModelCommand, ModelsCommand
        from nodetool.chat.commands.provider import ProviderCommand
        from nodetool.chat.commands.providers import ProvidersCommand
        from nodetool.chat.commands.usage import UsageCommand
        from nodetool.chat.commands.workflow import RunWorkflowCommand

        commands = [
            HelpCommand(),
            ExitCommand(),
            ModelCommand(),
            ModelsCommand(),
            ClearCommand(),
            AgentCommand(),
            DebugCommand(),
            UsageCommand(),
            ProviderCommand(),
            ProvidersCommand(),
            RunWorkflowCommand(),
        ]

        for command in commands:
            self.commands[command.name] = command
            # Register aliases
            for alias in command.aliases:
                self.commands[alias] = command

    def _build_command_completer(self) -> NestedCompleter:
        """Construct the nested completer with provider/model/tool suggestions."""
        command_completer: dict[str, Optional[WordCompleter]] = dict.fromkeys(self.commands.keys())
        provider_names = [provider.value for provider in Provider]
        model_ids = [model.id for model in self.language_models]

        command_completer["agent"] = WordCompleter(["on", "off"])
        command_completer["a"] = command_completer["agent"]
        command_completer["debug"] = WordCompleter(["on", "off"])
        command_completer["d"] = command_completer["debug"]
        command_completer["provider"] = WordCompleter(provider_names, ignore_case=True)
        command_completer["pr"] = command_completer["provider"]
        command_completer["model"] = WordCompleter(model_ids, ignore_case=True)
        command_completer["m"] = command_completer["model"]

        prefixed_command_completer = {f"/{cmd}": completer for cmd, completer in command_completer.items()}
        return NestedCompleter(
            {
                **prefixed_command_completer,
                "pwd": None,
                "ls": PathCompleter(only_directories=False),
                "cd": PathCompleter(only_directories=True),
                "mkdir": PathCompleter(only_directories=True),
                "rm": PathCompleter(only_directories=False),
                "open": PathCompleter(only_directories=False),
                "cat": PathCompleter(only_directories=False),
                "cp": PathCompleter(only_directories=False),
                "mv": PathCompleter(only_directories=False),
                "grep": PathCompleter(only_directories=False),
            }
        )

    def _refresh_completer(self) -> None:
        """Rebuild and apply the completer (used when providers/models change)."""
        self._completer = self._build_command_completer()
        if hasattr(self, "session"):
            self.session.completer = self._completer

    async def load_models_for_provider(self, provider: Provider) -> list[LanguageModel]:
        """Fetch and cache models for a single provider, updating current state."""
        if provider in self._models_by_provider:
            models = self._models_by_provider[provider]
        else:
            try:
                provider_instance = await get_provider(provider, user_id="1")
                models = await provider_instance.get_available_language_models()
            except Exception as e:
                self.console.print(
                    f"[bold yellow]Warning:[/bold yellow] Provider '{provider.value}' is unavailable: {e}"
                )
                models = []
            self._models_by_provider[provider] = models

        if self.selected_provider == provider:
            self.language_models = models
            self._refresh_completer()
        return models

    def set_selected_model(self, model: LanguageModel) -> None:
        """Select a model and align related configuration."""
        self.selected_model = model
        self.selected_provider = model.provider
        self.model_id_from_settings = model.id
        self.planner_model_id = model.id
        self.summarization_model_id = model.id
        self.retrieval_model_id = model.id
        self._refresh_completer()

    def set_selected_provider(self, provider: Provider) -> None:
        """Switch providers without loading them immediately."""
        if self.selected_provider != provider:
            self.selected_provider = provider
            self.selected_model = None
            self.language_models = self._models_by_provider.get(provider, [])
            self._refresh_completer()

    async def ensure_selected_model(self) -> None:
        """Ensure a selected model is available, loading provider models lazily."""
        if self.selected_model:
            return

        if not self.selected_provider:
            raise ValueError("No provider selected")

        models = await self.load_models_for_provider(self.selected_provider)
        if not models:
            # Try other providers if the selected one is unavailable/misconfigured.
            for fallback_provider in Provider:
                if fallback_provider == self.selected_provider:
                    continue
                fallback_models = await self.load_models_for_provider(fallback_provider)
                if fallback_models:
                    self.console.print(
                        f"[bold yellow]Warning:[/bold yellow] Falling back to provider '{fallback_provider.value}'."
                    )
                    self.set_selected_model(fallback_models[0])
                    return
            raise ValueError(f"No models available for provider {self.selected_provider.value}")

        target_id = self.model_id_from_settings
        selected = None
        if target_id:
            for model in models:
                if model.id == target_id:
                    selected = model
                    break

        if not selected:
            selected = models[0]
            self.console.print(
                "[bold yellow]Warning:[/bold yellow] Defaulting to first available "
                f"model '{selected.id}' for provider '{self.selected_provider.value}'."
            )

        self.set_selected_model(selected)

    async def setup_prompt_session(self):
        """Set up prompt_toolkit session with completers and styling."""
        if self.selected_provider:
            await self.ensure_selected_model()
        self._refresh_completer()
        completer = self._completer
        if completer is None:
            completer = self._build_command_completer()
            self._completer = completer

        # Create prompt style
        style = Style.from_dict(
            {
                "prompt": "ansicyan bold",
                "toolbar": "ansiblack bg:ansiwhite",
            }
        )

        key_bindings = KeyBindings()

        @key_bindings.add("enter")
        def submit_on_enter(event):
            event.current_buffer.validate_and_handle()

        @key_bindings.add("escape", "enter")
        def newline_on_alt_enter(event):
            event.current_buffer.insert_text("\n")

        @key_bindings.add("c-j")
        def submit_buffer(event):
            event.current_buffer.validate_and_handle()

        # Create session with history and auto-suggest
        self.session = PromptSession(
            history=FileHistory(self.history_file),
            auto_suggest=AutoSuggestFromHistory(),
            completer=completer,
            style=style,
            complete_in_thread=True,
            key_bindings=key_bindings,
        )

    def _create_user_message(self, content: str, *, agent_mode: bool) -> Message:
        """Create a Message instance for user input with current configuration."""
        if not self.selected_model:
            raise ValueError("No model selected")
        return Message(
            role="user",
            content=content,
            provider=self.selected_model.provider,
            model=self.selected_model.id,
            agent_mode=agent_mode,
            tools=None,
        )

    def _should_store_message(self, message: Message) -> bool:
        """Determine whether a processor message should be added to history."""
        return message.role in {"assistant", "tool", "system", "user"}

    def _extract_display_text(self, message: Message) -> str:
        """Extract text content from a message for terminal rendering."""
        if isinstance(message.content, str):
            return message.content

        if isinstance(message.content, list):
            parts = [item.text for item in message.content if isinstance(item, MessageTextContent) and item.text]
            return "".join(parts)

        return ""

    def _render_message_content(self, message: Message, stream_buffer: list[str]) -> None:
        """Render assistant content unless it was already streamed via chunk events."""
        if message.role != "assistant":
            return

        text_content = self._extract_display_text(message)
        if not text_content.strip():
            return

        streamed_content = "".join(stream_buffer).strip()
        if streamed_content and streamed_content == text_content.strip():
            return

        self.console.print(Markdown(text_content))

    async def _run_message_processor(
        self,
        processor: MessageProcessor,
        chat_history: list[Message],
        *,
        processing_context: ProcessingContext | None = None,
        **kwargs: Any,
    ) -> None:
        """Run a message processor and stream its output to the CLI."""

        context = processing_context or self.context

        async def process() -> None:
            try:
                await processor.process(
                    chat_history=chat_history,
                    processing_context=context,
                    **kwargs,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                traceback.print_exc()
                await processor.send_message({"type": "error", "message": str(exc)})
                processor.is_processing = False

        processor_task = asyncio.create_task(process())
        stream_buffer: list[str] = []
        display_manager = kwargs.get("display_manager")

        try:
            while processor.has_messages() or processor.is_processing:
                message = await processor.get_message()
                if message:
                    message_type = message.get("type")

                    if message_type == "chunk":
                        content = message.get("content") or ""
                        is_thinking = bool(message.get("thinking", False))
                        if content:
                            if display_manager and display_manager.is_live_active():
                                display_manager.append_agent_output(content, thinking=is_thinking)
                            else:
                                self.console.print(content, end="")
                            stream_buffer.append(content)
                        if message.get("done"):
                            if not (display_manager and display_manager.is_live_active()):
                                self.console.print()  # Final newline

                    elif message_type == "tool_call_update":
                        pass

                    elif message_type == "message":
                        try:
                            parsed = Message.model_validate(message)
                        except ValidationError as validation_error:
                            self.console.print(f"[bold red]Failed to parse message:[/bold red] {validation_error}")
                            continue

                        if self._should_store_message(parsed):
                            self.messages.append(parsed)

                        self._render_message_content(parsed, stream_buffer)
                        stream_buffer.clear()

                    elif message_type == "error":
                        error_msg = message.get("message", "Unknown error")
                        self.console.print(f"[bold red]Error:[/bold red] {error_msg}")
                else:
                    await asyncio.sleep(0.01)

            await processor_task
        finally:
            if not processor_task.done():
                processor_task.cancel()
                with suppress(asyncio.CancelledError):
                    await processor_task

    async def process_agent_response(self, problem: str):
        """Process input in agent mode using the messaging processors."""
        await self.ensure_selected_model()
        if not self.selected_model:
            raise ValueError("Cannot process agent message without a selected model")

        user_message = self._create_user_message(problem, agent_mode=True)
        self.messages.append(user_message)

        provider = await get_provider(self.selected_model.provider, user_id="1")
        processor = AgentMessageProcessor(provider)

        await self._run_message_processor(
            processor=processor,
            chat_history=list(self.messages),
            processing_context=self.context,
            display_manager=self.display_manager,
        )

    async def process_regular_message(self, user_input: str) -> None:
        """Process standard chat input using the regular message processor."""
        await self.ensure_selected_model()
        if not self.selected_model:
            raise ValueError("Cannot process chat message without a selected model")

        user_message = self._create_user_message(user_input, agent_mode=False)
        self.messages.append(user_message)

        provider = await get_provider(self.selected_model.provider, user_id="1")
        processor = RegularChatProcessor(provider)

        await self._run_message_processor(
            processor=processor,
            chat_history=list(self.messages),
            processing_context=self.context,
            collections=user_message.collections,
            graph=user_message.graph,
        )

    def handle_workspace_command(self, cmd: str, args: list[str]) -> None:
        """Handle workspace-related commands using the process's CWD."""
        try:
            current_dir = Path(os.getcwd())  # Use actual CWD

            if cmd == "pwd":
                self.console.print(str(current_dir), style="bold green")
            elif cmd == "ls":
                target_path_str = args[0] if args else "."
                target_path = (current_dir / target_path_str).resolve()

                # Check if target exists before listing
                if not target_path.exists():
                    self.console.print(f"[bold red]Error:[/bold red] Path '{target_path_str}' does not exist.")
                    return

                # If target is a file, just show the file
                if target_path.is_file():
                    table = Table(show_header=True)
                    table.add_column("Type", style="cyan", width=2)
                    table.add_column("Name", style="green")
                    table.add_row("f", target_path.name)
                    self.console.print(table)
                    return

                # If target is a directory, list contents
                if target_path.is_dir():
                    table = Table(show_header=True)
                    table.add_column("Type", style="cyan", width=2)
                    table.add_column("Name", style="green")

                    items = list(target_path.iterdir())
                    items.sort(key=lambda x: (not x.is_dir(), x.name))

                    for item in items:
                        type_char = "d" if item.is_dir() else "f"
                        table.add_row(type_char, item.name)
                    self.console.print(table)
                else:
                    self.console.print(
                        f"[bold red]Error:[/bold red] '{target_path_str}' is not a valid file or directory."
                    )

            elif cmd == "cd":
                new_dir_path = Path.home() if not args else (current_dir / args[0]).resolve()

                if not new_dir_path.is_dir():
                    self.console.print(
                        f"[bold red]Error:[/bold red] '{args[0] if args else 'Home'}' is not a directory or not accessible."
                    )
                    return
                try:
                    os.chdir(new_dir_path)
                    # Update context.workspace_dir to reflect the change, although it's less central now
                    self.context.workspace_dir = str(new_dir_path)
                    self.console.print(f"Changed to: [bold green]{os.getcwd()}[/bold green]")
                except Exception as e:
                    self.console.print(f"[bold red]Error changing directory:[/bold red] {e}")

            elif cmd == "mkdir":
                if not args:
                    self.console.print("[bold red]Error:[/bold red] Directory name required")
                    return
                new_dir_path = (current_dir / args[0]).resolve()
                try:
                    new_dir_path.mkdir(parents=True, exist_ok=True)
                    self.console.print(f"Created directory: [bold green]{new_dir_path}[/bold green]")
                except Exception as e:
                    self.console.print(f"[bold red]Error creating directory:[/bold red] {e}")

            elif cmd == "rm":
                if not args:
                    self.console.print("[bold red]Error:[/bold red] Path required")
                    return
                target_path = (current_dir / args[0]).resolve()
                if not target_path.exists():
                    self.console.print(f"[bold red]Error:[/bold red] Path '{args[0]}' does not exist.")
                    return

                try:
                    if target_path.is_dir():
                        shutil.rmtree(target_path)
                        self.console.print(f"Removed directory: [bold green]{target_path}[/bold green]")
                    else:
                        target_path.unlink()
                        self.console.print(f"Removed file: [bold green]{target_path}[/bold green]")
                except Exception as e:
                    self.console.print(f"[bold red]Error removing:[/bold red] {e}")

            elif cmd == "open":
                if not args:
                    self.console.print("[bold red]Error:[/bold red] File or directory path required")
                    return
                target_path_str = args[0]
                target_path = (current_dir / target_path_str).resolve()

                if not target_path.exists():
                    self.console.print(f"[bold red]Error:[/bold red] Path '{target_path_str}' does not exist.")
                    return

                try:
                    if sys.platform == "win32":
                        os.startfile(target_path)
                    elif sys.platform == "darwin":  # macOS
                        subprocess.run(["open", str(target_path)], check=True)
                    else:  # Linux and other POSIX
                        subprocess.run(["xdg-open", str(target_path)], check=True)
                    self.console.print(f"Opened: [bold green]{target_path}[/bold green]")
                except Exception as e:
                    self.console.print(f"[bold red]Error opening:[/bold red] {e}")

            elif cmd == "cat":
                if not args:
                    self.console.print("[bold red]Error:[/bold red] File path required")
                    return
                target_path = (current_dir / args[0]).resolve()

                if not target_path.is_file():
                    self.console.print(f"[bold red]Error:[/bold red] '{args[0]}' is not a file or does not exist")
                    return

                try:
                    # Determine lexer based on file extension
                    lexer = Syntax.guess_lexer(str(target_path), code="")
                    syntax = Syntax.from_path(
                        str(target_path),
                        lexer=lexer,
                        theme="monokai",
                        line_numbers=True,
                    )
                    self.console.print(syntax)
                except Exception as e:
                    self.console.print(f"[bold red]Error reading file:[/bold red] {e}")
            elif cmd == "cp":
                if len(args) != 2:
                    self.console.print("[bold red]Usage: cp <source> <destination>[/bold red]")
                    return
                src_path = (current_dir / args[0]).resolve()
                dest_path = (current_dir / args[1]).resolve()

                if not src_path.exists():
                    self.console.print(f"[bold red]Error:[/bold red] Source '{args[0]}' does not exist.")
                    return

                # Prevent copying directory onto itself or file onto itself
                if src_path == dest_path:
                    self.console.print("[bold red]Error:[/bold red] Source and destination are the same.")
                    return

                try:
                    if src_path.is_dir():
                        # Ensure destination is a directory or doesn't exist
                        if dest_path.exists() and not dest_path.is_dir():
                            self.console.print(
                                f"[bold red]Error:[/bold red] Cannot overwrite non-directory '{args[1]}' with directory '{args[0]}'."
                            )
                            return
                        # If dest is an existing dir, copy *into* it
                        dest_final = dest_path if not dest_path.is_dir() else dest_path / src_path.name
                        shutil.copytree(src_path, dest_final, dirs_exist_ok=True)
                    else:  # Source is a file
                        # If destination exists and is a directory, copy into it
                        dest_final = dest_path
                        if dest_path.is_dir():
                            dest_final = dest_path / src_path.name
                        shutil.copy2(src_path, dest_final)
                    self.console.print(
                        f"Copied [bold green]{args[0]}[/bold green] to [bold green]{args[1]}[/bold green]"
                    )
                except Exception as e:
                    self.console.print(f"[bold red]Error copying:[/bold red] {e}")

            elif cmd == "mv":
                if len(args) != 2:
                    self.console.print("[bold red]Usage: mv <source> <destination>[/bold red]")
                    return
                src_path = (current_dir / args[0]).resolve()
                dest_path = (current_dir / args[1]).resolve()

                if not src_path.exists():
                    self.console.print(f"[bold red]Error:[/bold red] Source '{args[0]}' does not exist.")
                    return

                # Prevent moving onto itself
                if src_path == dest_path:
                    self.console.print("[bold red]Error:[/bold red] Source and destination are the same.")
                    return

                try:
                    # If destination is an existing directory, move source *into* it
                    final_dest_path = dest_path
                    if dest_path.is_dir():
                        final_dest_path = dest_path / src_path.name

                    shutil.move(str(src_path), str(final_dest_path))
                    self.console.print(
                        f"Moved [bold green]{args[0]}[/bold green] to [bold green]{args[1]}[/bold green]"
                    )
                except Exception as e:
                    self.console.print(f"[bold red]Error moving:[/bold red] {e}")

            elif cmd == "grep":
                if not args:
                    self.console.print("[bold red]Usage: grep <pattern> [path][/bold red]")
                    return

                pattern = args[0]
                search_path_str = args[1] if len(args) > 1 else "."
                # Resolve search path relative to current CWD
                search_path = (current_dir / search_path_str).resolve()

                if not search_path.exists():
                    self.console.print(f"[bold red]Error:[/bold red] Path '{search_path_str}' does not exist.")
                    return

                try:
                    regex = re.compile(pattern)
                except re.error as e:
                    self.console.print(f"[bold red]Invalid regex pattern:[/bold red] {e}")
                    return

                files_to_search = []
                if search_path.is_file():
                    files_to_search.append(search_path)
                elif search_path.is_dir():
                    # Recursively find files, skipping hidden files/dirs and potentially large/binary files
                    for item in search_path.rglob("*"):
                        # Check if item is within the search_path tree (resolve symlinks etc)
                        try:
                            resolved_item = item.resolve()
                            if not str(resolved_item).startswith(str(search_path)):
                                continue  # Skip items outside the resolved search path
                        except Exception:
                            continue  # Skip if resolve fails

                        if item.is_file() and not item.name.startswith("."):
                            # Basic check to avoid huge files or likely binaries (can be improved)
                            try:
                                if item.stat().st_size < 10 * 1024 * 1024:  # Skip files > 10MB
                                    files_to_search.append(item)
                            except OSError:
                                continue  # Skip files we can't access stats for

                match_found = False
                for file_path in files_to_search:
                    try:
                        # Attempt to read as text, skip if decoding fails
                        # Use '.resolve()' to get the absolute path before making it relative
                        abs_file_path = file_path.resolve()
                        content = abs_file_path.read_text(encoding="utf-8")
                        lines = content.splitlines()
                        file_match_found = False
                        for i, line in enumerate(lines):
                            if regex.search(line):
                                if not file_match_found:
                                    # Print filename relative to CWD only once
                                    try:
                                        rel_file_path = abs_file_path.relative_to(current_dir)
                                    except ValueError:  # Handle case where file is not under CWD (e.g., symlink)
                                        rel_file_path = abs_file_path  # Show absolute path if not relative

                                    self.console.print(f"\n[bold magenta]{rel_file_path}[/bold magenta]:")
                                    file_match_found = True
                                    match_found = True

                                # Highlight the match within the line
                                highlighted_line = line.strip()  # Process stripped line first
                                with suppress(Exception):
                                    highlighted_line = regex.sub(
                                        lambda m: f"[bold yellow]{m.group(0)}[/bold yellow]",
                                        highlighted_line,
                                    )

                                self.console.print(f"  [cyan]{i + 1}:[/cyan] {highlighted_line}")

                    except UnicodeDecodeError:
                        # Skip files that are not valid UTF-8 text
                        continue
                    except Exception as e:
                        # Use absolute path in error message if relative path fails
                        try:
                            rel_file_path = file_path.resolve().relative_to(current_dir)
                        except Exception:
                            rel_file_path = file_path.resolve()

                        self.console.print(f"[yellow]Warning:[/yellow] Could not read or process {rel_file_path}: {e}")

                if not match_found:
                    self.console.print(f"No matches found for pattern '{pattern}' in '{search_path_str}'")

        except Exception as e:
            self.console.print(f"[bold red]Error executing workspace command:[/bold red] {e}")
            # Optionally print traceback if needed for debugging
            # self.console.print(Syntax(traceback.format_exc(), "python", theme="monokai"))

    def save_settings(self) -> None:
        """Save current settings to the dotfile."""
        if not self.selected_model:
            # Avoid saving if no model is selected (e.g., during init failure)
            return

        settings = {
            # Save the selected model's ID
            "model_id": self.selected_model.id,
            "provider": self.selected_model.provider.value,
            "agent_mode": self.agent_mode,
        }

        try:
            with open(self.settings_file, "w") as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            self.console.print(f"[bold yellow]Warning:[/bold yellow] Failed to save settings: {e}")

    def load_settings(self) -> None:
        """Load settings from the dotfile."""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file) as f:
                    settings = json.load(f)

                # Load the model ID preference to be used in initialize()
                self.model_id_from_settings = settings.get("model_id", self.model_id_from_settings)
                provider_value = settings.get("provider")
                if provider_value:
                    with suppress(Exception):
                        self.selected_provider = Provider(provider_value)

                # Load other settings
                self.agent_mode = settings.get("agent_mode", False)

        except Exception as e:
            self.console.print(f"[bold yellow]Warning:[/bold yellow] Failed to load settings: {e}")
            # Keep default settings
            pass

    async def handle_command(self, cmd: str, args: list[str]) -> bool:
        """Handle CLI commands (starting with /)."""
        # Strip the leading slash
        if cmd.startswith("/"):
            cmd = cmd[1:]

        if cmd in self.commands:
            return await self.commands[cmd].execute(self, args)
        else:
            self.console.print(f"[bold red]Unknown command:[/bold red] {cmd}")
            self.console.print("Type [bold]/help[/bold] for available commands")
        return False

    async def run(self):
        """Run the chat CLI as a Textual application."""
        app = ChatTextualApp(self)
        await app.run_async()


async def chat_cli():
    """Entry point for the chat CLI."""
    cli = ChatCLI()
    # Bind a ResourceScope so providers (HTTP clients, DB) have required context
    async with ResourceScope():
        await cli.run()


if __name__ == "__main__":
    asyncio.run(chat_cli())
