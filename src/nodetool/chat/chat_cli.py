"""CLI interface for the Chain of Thought (CoT) agent.

This module provides a command-line interface for interacting with the CoT agent.
It supports multiple LLM providers, model selection, and tool management.

Features:
- Interactive chat with command history and tab completion
- Support for multiple LLM providers (OpenAI, Anthropic, Ollama)
- Model selection and provider management
- Chain of Thought reasoning with step-by-step output
- Tool usage and execution
- Debug mode to display tool calls and results
- Workspace management with file system commands
"""

import asyncio
import json
import os
import re  # Add re for grep
import shutil  # Add shutil for cp and mv
import subprocess
import sys
import traceback
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, List, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import NestedCompleter, PathCompleter, WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from pydantic import ValidationError
from rich.align import Align
from rich.columns import Columns  # Add Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from nodetool.agents.tools.browser_tools import BrowserTool, ScreenshotTool
from nodetool.agents.tools.email_tools import (
    AddLabelToEmailTool,
    ArchiveEmailTool,
    SearchEmailTool,
)
from nodetool.agents.tools.filesystem_tools import (
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
)
from nodetool.agents.tools.google_tools import (
    GoogleGroundedSearchTool,
    GoogleImageGenerationTool,
)
from nodetool.agents.tools.http_tools import DownloadFileTool
from nodetool.agents.tools.openai_tools import (
    OpenAIImageGenerationTool,
    OpenAITextToSpeechTool,
    OpenAIWebSearchTool,
)
from nodetool.agents.tools.pdf_tools import (
    ConvertPDFToMarkdownTool,
    ExtractPDFTablesTool,
    ExtractPDFTextTool,
)
from nodetool.agents.tools.serp_tools import (
    GoogleImagesTool,
    GoogleNewsTool,
    GoogleSearchTool,
)
from nodetool.config.logging_config import configure_logging, get_logger
from nodetool.config.settings import get_log_path
from nodetool.messaging.agent_message_processor import AgentMessageProcessor
from nodetool.messaging.message_processor import MessageProcessor
from nodetool.messaging.regular_chat_processor import RegularChatProcessor
from nodetool.metadata.types import LanguageModel, Message, Provider
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
    4. `DEBUG` environment toggle (truthy -> DEBUG)
    5. Default to "ERROR" to keep interactive output tidy
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

        # Rich console for beautiful output
        self.console = Console()
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
        self.agent_mode = False
        self.debug_mode = False

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

        # Tool management
        self.enabled_tools: dict[str, bool] = {}  # Track enabled/disabled tools
        self.all_tools: list = []  # Store all available tools
        self.tools: list = []

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
        from nodetool.chat.commands.tools import (
            ToolDisableCommand,
            ToolEnableCommand,
            ToolsCommand,
        )
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
            ToolsCommand(),
            ToolEnableCommand(),
            ToolDisableCommand(),
            RunWorkflowCommand(),
        ]

        for command in commands:
            self.commands[command.name] = command
            # Register aliases
            for alias in command.aliases:
                self.commands[alias] = command

    def load_all_node_packages(self):
        """Load all available node packages to populate NODE_BY_TYPE registry."""
        try:
            import importlib

            from nodetool.metadata.node_metadata import get_node_classes_from_namespace
            from nodetool.packages.registry import Registry

            registry = Registry()
            packages = registry.list_installed_packages()

            total_loaded = 0
            total_packages = len(packages)

            for package in packages:
                if package.nodes:
                    # Collect unique namespaces from this package
                    namespaces = set()
                    for node_metadata in package.nodes:
                        node_type = node_metadata.node_type
                        # Extract namespace from node_type (e.g., "nodetool.text" from "nodetool.text.Concatenate")
                        namespace_parts = node_type.split(".")[:-1]
                        if len(namespace_parts) >= 2:  # Must have at least nodetool.something
                            namespace = ".".join(namespace_parts)
                            namespaces.add(namespace)

                    # Load each unique namespace from this package
                    for namespace in namespaces:
                        try:
                            # Try to import the module directly
                            if namespace.startswith("nodetool.nodes."):
                                module_path = namespace
                            else:
                                module_path = f"nodetool.nodes.{namespace}"

                            importlib.import_module(module_path)
                            total_loaded += 1
                        except ImportError:
                            # Try alternative approach using get_node_classes_from_namespace
                            try:
                                if namespace.startswith("nodetool."):
                                    namespace_suffix = namespace[9:]  # Remove 'nodetool.'
                                    get_node_classes_from_namespace(f"nodetool.nodes.{namespace_suffix}")
                                    total_loaded += 1
                                else:
                                    get_node_classes_from_namespace(f"nodetool.nodes.{namespace}")
                                    total_loaded += 1
                            except Exception:
                                # Silent fail for packages that can't be loaded
                                pass

            from nodetool.workflows.base_node import NODE_BY_TYPE

            total_nodes = len(NODE_BY_TYPE)

            self.console.print(
                f"[bold green]Loaded {total_packages} packages with {total_nodes} available nodes[/bold green]"
            )

        except Exception as e:
            self.console.print(f"[bold yellow]Warning:[/bold yellow] Failed to load all packages: {e}")
            # Continue anyway - some nodes may still be available

    async def initialize(self):
        """Initialize async components and workspace with visual feedback."""
        # Initialize standard tools (tool modules keep heavy deps lazy inside their methods)
        self.all_tools = [
            AddLabelToEmailTool(),
            ArchiveEmailTool(),
            BrowserTool(),
            ConvertPDFToMarkdownTool(),
            DownloadFileTool(),
            ExtractPDFTablesTool(),
            ExtractPDFTextTool(),
            GoogleGroundedSearchTool(),
            GoogleImageGenerationTool(),
            GoogleImagesTool(),
            GoogleNewsTool(),
            GoogleSearchTool(),
            ListDirectoryTool(),
            ReadFileTool(),
            WriteFileTool(),
            ScreenshotTool(),
            SearchEmailTool(),
            OpenAIImageGenerationTool(),
            OpenAITextToSpeechTool(),
            OpenAIWebSearchTool(),
        ]
        
        # Add MCP tools to make all nodetool functionality available
        try:
            from nodetool.agents.tools.mcp_tools import get_all_mcp_tools
            mcp_tools = await get_all_mcp_tools()
            self.all_tools.extend(mcp_tools)
            self.console.print(f"[dim]Loaded {len(mcp_tools)} MCP tools[/dim]")
        except Exception as e:
            self.console.print(f"[yellow]Warning: Failed to load MCP tools: {e}[/yellow]")

        # Initialize enabled_tools tracking if not already set
        for tool in self.all_tools:
            tool_name = tool.name
            if tool_name not in self.enabled_tools:
                self.enabled_tools[tool_name] = False  # Default to disabled

        # Filter tools based on enabled status
        self.refresh_tools()

    def refresh_tools(self):
        """Refresh the tools list based on current enabled status."""
        self.tools = [tool for tool in self.all_tools if self.enabled_tools.get(tool.name, False)]

    def _build_command_completer(self) -> NestedCompleter:
        """Construct the nested completer with provider/model/tool suggestions."""
        command_completer: dict[str, Optional[WordCompleter]] = dict.fromkeys(self.commands.keys())
        provider_names = [provider.value for provider in Provider]
        model_ids = [model.id for model in self.language_models]
        all_tool_names = [tool.name for tool in self.all_tools]

        command_completer["agent"] = WordCompleter(["on", "off"])
        command_completer["a"] = command_completer["agent"]
        command_completer["debug"] = WordCompleter(["on", "off"])
        command_completer["d"] = command_completer["debug"]
        command_completer["provider"] = WordCompleter(provider_names, ignore_case=True)
        command_completer["pr"] = command_completer["provider"]
        command_completer["model"] = WordCompleter(model_ids, ignore_case=True)
        command_completer["m"] = command_completer["model"]
        command_completer["tools"] = WordCompleter(all_tool_names, match_middle=True)
        command_completer["t"] = command_completer["tools"]
        enable_disable_completer = WordCompleter([*all_tool_names, "all"], match_middle=True)
        command_completer["enable"] = enable_disable_completer
        command_completer["en"] = enable_disable_completer
        command_completer["disable"] = enable_disable_completer
        command_completer["dis"] = enable_disable_completer

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
            provider_instance = await get_provider(provider, user_id="1")
            models = await provider_instance.get_available_language_models()
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
            }
        )

        # Create session with history and auto-suggest
        self.session = PromptSession(
            history=FileHistory(self.history_file),
            auto_suggest=AutoSuggestFromHistory(),
            completer=completer,
            style=style,
            complete_in_thread=True,
        )

    def _get_enabled_tool_names(self) -> list[str]:
        """Return the names of currently enabled tools."""
        return [tool.name for tool in self.tools]

    def _create_user_message(self, content: str, *, agent_mode: bool) -> Message:
        """Create a Message instance for user input with current configuration."""
        if not self.selected_model:
            raise ValueError("No model selected")

        tool_names = self._get_enabled_tool_names()

        return Message(
            role="user",
            content=content,
            provider=self.selected_model.provider,
            model=self.selected_model.id,
            agent_mode=agent_mode,
            tools=tool_names if tool_names else None,
        )

    def _should_store_message(self, message: Message) -> bool:
        """Determine whether a processor message should be added to history."""
        return message.role in {"assistant", "tool", "system", "user"}

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

        try:
            while processor.has_messages() or processor.is_processing:
                message = await processor.get_message()
                if message:
                    message_type = message.get("type")

                    if message_type == "chunk":
                        content = message.get("content") or ""
                        if content:
                            self.console.print(content, end="")
                            stream_buffer.append(content)
                        if message.get("done"):
                            self.console.print()  # Final newline
                            stream_buffer.clear()

                    elif message_type == "tool_call_update":
                        name = message.get("name")
                        msg = message.get("message")
                        if self.debug_mode:
                            args = message.get("args")
                            self.console.print(
                                f"\n[bold cyan]Tool Call ({escape(str(name))}):[/bold cyan] {escape(str(args))}"
                            )
                        elif self.display_manager:
                            # Skip if display_manager is present, as it handles tool calls in its tree view
                            pass
                        else:
                            self.console.print(f"\n[italic cyan]{escape(str(msg))}[/italic cyan]")

                    elif message_type == "message":
                        try:
                            # Check if this is an special agent execution event
                            if message.get("role") == "agent_execution":
                                event_type = message.get("execution_event_type")
                                content = message.get("content") or {}

                                if event_type == "planning_update":
                                    log.info(
                                        f"[Planning] Phase: {content.get('phase')}, Status: {content.get('status')}"
                                    )
                                    if self.display_manager and not self.debug_mode:
                                        # When display_manager is active, it handles planning updates
                                        pass
                                    else:
                                        phase = content.get("phase")
                                        status = content.get("status")
                                        inner_content = content.get("content")
                                        self.console.print(
                                            f"[bold blue]Planning Phase [{escape(str(phase))}]:[/bold blue] {escape(str(status))}"
                                        )
                                        if inner_content and self.debug_mode:
                                            self.console.print(f"  {escape(str(inner_content))}")

                                elif event_type == "task_update":
                                    log.debug(f"[Task Event] {content.get('event')}")
                                    if self.display_manager and not self.debug_mode:
                                        # When display_manager is active, it handles task and step updates
                                        pass
                                    else:
                                        event = content.get("event")
                                        step = content.get("step")
                                        if event == "SUBTASK_STARTED" and step:
                                            instructions = step.get("instructions")
                                            self.console.print(
                                                f"\n[bold green]➜ {escape(str(instructions))}[/bold green]"
                                            )
                                        elif event == "ENTERED_CONCLUSION_STAGE":
                                            self.console.print(
                                                "[bold yellow]Entering conclusion stage...[/bold yellow]"
                                            )
                                        elif self.debug_mode:
                                            self.console.print(f"[dim]Task Event: {escape(str(event))}[/dim]")

                                elif event_type == "log_update":
                                    log_content = content.get("content")
                                    severity = content.get("severity", "info")
                                    # Log to file regardless of console suppression
                                    log_func = getattr(log, severity.lower(), log.info)
                                    log_func(f"[Agent Log] {log_content}")

                                    if self.display_manager and not self.debug_mode:
                                        # Consistently skip technical logs in CLI unless in debug mode,
                                        # matching the frontend refinement.
                                        pass
                                    else:
                                        color = (
                                            "red"
                                            if severity == "error"
                                            else "yellow"
                                            if severity == "warning"
                                            else "white"
                                        )
                                        self.console.print(
                                            f"[bold {color}]Log:[/bold {color}] {escape(str(log_content))}"
                                        )

                                elif event_type == "step_result":
                                    log.info(f"[Step Result] {content.get('result')}")
                                    if self.display_manager and not self.debug_mode:
                                        # display_manager handles step results
                                        pass
                                    else:
                                        result = content.get("result")
                                        if self.debug_mode:
                                            self.console.print(
                                                f"[bold green]Step Result:[/bold green] {escape(str(result))}"
                                            )

                                continue  # Already handled this special message

                            parsed = Message.model_validate(message)
                        except ValidationError as validation_error:
                            if self.debug_mode:
                                self.console.print(f"[bold red]Failed to parse message:[/bold red] {validation_error}")
                            continue

                        if self.debug_mode and parsed.tool_calls and parsed.role == "assistant":
                            for tool_call in parsed.tool_calls:
                                args_preview = json.dumps(tool_call.args)
                                if len(args_preview) > 120:
                                    args_preview = args_preview[:120] + "..."
                                self.console.print(
                                    f"\n[bold cyan][{escape(str(tool_call.name))}]:[/bold cyan] {escape(str(args_preview))}"
                                )
                        elif self.debug_mode and parsed.role == "tool":
                            preview = parsed.content
                            if isinstance(preview, str) and len(preview) > 200:
                                preview = preview[:200] + "..."
                            self.console.print(
                                f"\n[bold magenta][Tool Result: {escape(str(parsed.name))}][/bold magenta] {escape(str(preview))}"
                            )

                        has_streaming_output = bool(stream_buffer)

                        if (
                            parsed.role == "assistant"
                            and isinstance(parsed.content, str)
                            and parsed.content
                            and not has_streaming_output
                        ):
                            self.console.print(Markdown(parsed.content))
                        elif (
                            parsed.role == "assistant" and isinstance(parsed.content, dict) and not has_streaming_output
                        ):
                            # If assistant returned structured content, pretty-print as Markdown code block
                            self.console.print(Markdown(f"```json\n{json.dumps(parsed.content, indent=2)}\n```"))

                        if self._should_store_message(parsed):
                            self.messages.append(parsed)

                        stream_buffer.clear()

                    elif message_type == "error":
                        error_msg = message.get("message", "Unknown error")
                        self.console.print(f"[bold red]Error:[/bold red] {error_msg}")
                    else:
                        if self.debug_mode:
                            self.console.print(f"[yellow]Unhandled processor message:[/yellow] {message}")
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
            verbose=not self.debug_mode,
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
            "debug_mode": self.debug_mode,
            "enabled_tools": self.enabled_tools,  # Save tool enable/disable states
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
                self.debug_mode = settings.get("debug_mode", False)
                # Load tool enable/disable states
                self.enabled_tools = settings.get("enabled_tools", {})

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
        """Run the chat CLI main loop with rich UI and improved input handling."""
        # Initialize components with progress indication
        await self.initialize()
        await self.setup_prompt_session()

        # Display welcome message and settings with rich formatting
        settings_list = []
        provider_display = (
            self.selected_model.provider.value
            if self.selected_model
            else self.selected_provider.value
            if self.selected_provider
            else "None"
        )
        model_display = self.selected_model.id if self.selected_model else self.model_id_from_settings or "None"
        settings_list.extend(
            [
                f"[bold cyan]Provider:[/bold cyan] {provider_display}",
                f"[bold cyan]Model:[/bold cyan] {model_display}",
            ]
        )

        enabled_tools_count = len([t for t in self.enabled_tools.values() if t])
        total_tools_count = len(self.all_tools)

        settings_list.extend(
            [
                f"[bold cyan]Agent:[/bold cyan] {'ON' if self.agent_mode else 'OFF'} (/agent)",
                f"[bold cyan]Debug:[/bold cyan] {'ON' if self.debug_mode else 'OFF'} (/debug)",
                f"[bold cyan]Tools:[/bold cyan] {enabled_tools_count}/{total_tools_count} enabled (/tools)",
                # Span workspace across full width potentially, or keep it separate
                # f"[bold cyan]Workspace:[/bold cyan] {str(self.context.workspace_dir)}"
            ]
        )

        settings_columns = Columns(settings_list, equal=True, expand=True)

        # Center the welcome panel
        welcome_panel = Panel(
            Columns(
                [  # Use Columns to arrange elements
                    "[bold green]Welcome to NodeTool Chat CLI[/bold green]",
                    settings_columns,  # Add the settings columns here
                    "Type [bold cyan]/help[/bold cyan] for application commands (/help)",
                ],
                equal=True,
            ),  # Make columns equal width
            title="NodeTool",
            border_style="green",
            width=100,  # Adjust width as needed
        )
        self.console.print(Align.center(welcome_panel))  # Apply centering

        while True:
            try:
                # Create prompt with current directory info
                cwd = os.getcwd()
                home_dir = str(Path.home())
                # Show ~ for home directory, otherwise show full path
                display_path = cwd.replace(home_dir, "~", 1) if cwd.startswith(home_dir) else cwd
                prompt = f"[{display_path}]> "

                # Get input with prompt_toolkit (supports multi-line input and better completion)
                user_input = await self.session.prompt_async(prompt)

                if not user_input:
                    continue

                # Handle workspace commands
                if user_input.startswith(
                    (
                        "pwd",
                        "ls",
                        "cd",
                        "mkdir",
                        "rm",
                        "open",
                        "cat",
                        "cp",
                        "mv",
                        "grep",
                        "cdw",
                    )
                ):
                    cmd_parts = user_input.split()
                    cmd = cmd_parts[0]
                    args = cmd_parts[1:] if len(cmd_parts) > 1 else []
                    self.handle_workspace_command(cmd, args)
                    continue

                # Handle CLI commands
                if user_input.startswith("/"):
                    cmd_parts = user_input.lower().split()
                    cmd = cmd_parts[0]
                    args = cmd_parts[1:] if len(cmd_parts) > 1 else []
                    if await self.handle_command(cmd, args):
                        break
                    continue

                # Process chat input
                if not self.selected_model:
                    self.console.print("[bold red]Error:[/bold red] No model selected")
                    continue

                if self.agent_mode:
                    await self.process_agent_response(user_input)
                else:
                    await self.process_regular_message(user_input)

            except KeyboardInterrupt:
                self.console.print("\n[bold yellow]Interrupted. Press Ctrl+C again to exit.[/bold yellow]")
                try:
                    # Wait for a moment to see if the user presses Ctrl+C again
                    await asyncio.sleep(1)
                except KeyboardInterrupt:
                    self.console.print("[bold yellow]Exiting...[/bold yellow]")
                return
            except EOFError:
                self.console.print("[bold yellow]Exiting...[/bold yellow]")
                return
            except Exception as e:
                self.console.print(f"[bold red]Error:[/bold red] {e}")
                self.console.print(
                    Syntax(
                        traceback.format_exc(),
                        "python",
                        theme="monokai",
                        line_numbers=True,
                    )
                )


async def chat_cli():
    """Entry point for the chat CLI."""
    cli = ChatCLI()
    # Bind a ResourceScope so providers (HTTP clients, DB) have required context
    async with ResourceScope():
        await cli.run()


if __name__ == "__main__":
    asyncio.run(chat_cli())
