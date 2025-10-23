"""
CLI interface for the Chain of Thought (CoT) agent.

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
import subprocess
import traceback
import sys
from typing import List, Optional, Dict, Any
from pathlib import Path
import shutil  # Add shutil for cp and mv
import re  # Add re for grep

from nodetool.config.logging_config import configure_logging

# New imports
from nodetool.ml.models.language_models import get_all_language_models
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter, NestedCompleter, PathCompleter
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.align import Align
from rich.columns import Columns  # Add Columns

# Existing imports
from nodetool.providers import get_provider
from nodetool.messaging.agent_message_processor import AgentMessageProcessor
from nodetool.messaging.message_processor import MessageProcessor
from nodetool.messaging.regular_chat_processor import RegularChatProcessor
from nodetool.metadata.types import Message, LanguageModel
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.agents.tools.browser_tools import BrowserTool, ScreenshotTool
from nodetool.agents.tools.email_tools import (
    AddLabelToEmailTool,
    ArchiveEmailTool,
    SearchEmailTool,
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
from nodetool.agents.tools.filesystem_tools import (
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
)
from pydantic import ValidationError


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

        # Initialize state
        self.context = ProcessingContext(user_id="1", auth_token="local_token")
        self.messages: list[Message] = []
        self.agent_mode = False
        self.debug_mode = False

        # Store selected LanguageModel object and model ID preference
        self.language_models: List[LanguageModel] = []
        self.selected_model: Optional[LanguageModel] = None
        self.model_id_from_settings: Optional[str] = "gpt-4o-mini"

        # Model attributes for agent config (can be overridden)
        # These might eventually also become LanguageModel selections
        # For now, we'll default them based on the primary selected model
        self.planner_model_id: Optional[str] = None
        self.summarization_model_id: Optional[str] = None
        self.retrieval_model_id: Optional[str] = None

        # Tool management
        self.enabled_tools: Dict[str, bool] = {}  # Track enabled/disabled tools
        self.all_tools: List = []  # Store all available tools

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
            from nodetool.packages.registry import Registry
            from nodetool.metadata.node_metadata import get_node_classes_from_namespace
            import importlib

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
                        if (
                            len(namespace_parts) >= 2
                        ):  # Must have at least nodetool.something
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
                                    namespace_suffix = namespace[
                                        9:
                                    ]  # Remove 'nodetool.'
                                    get_node_classes_from_namespace(
                                        f"nodetool.nodes.{namespace_suffix}"
                                    )
                                    total_loaded += 1
                                else:
                                    get_node_classes_from_namespace(
                                        f"nodetool.nodes.{namespace}"
                                    )
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
            self.console.print(
                f"[bold yellow]Warning:[/bold yellow] Failed to load all packages: {e}"
            )
            # Continue anyway - some nodes may still be available

    async def initialize(self):
        """Initialize async components and workspace with visual feedback."""
        # Initialize components with progress indicators
        async def load_language_models():
            self.language_models = await get_all_language_models("1")

            # Find and set the selected model based on loaded settings or default
            found_model = None
            if self.model_id_from_settings:
                for model in self.language_models:
                    if model.id == self.model_id_from_settings:
                        found_model = model
                        break
                if not found_model:
                    self.console.print(
                        f"[bold yellow]Warning:[/bold yellow] Saved model ID '{self.model_id_from_settings}' not found. ",
                        end="",
                    )

            self.selected_model = found_model

        asyncio.create_task(load_language_models())

        # Initialize standard tools
        standard_tools = [
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
        ]
        self.all_tools = standard_tools

        # Initialize enabled_tools tracking if not already set
        for tool in self.all_tools:
            tool_name = tool.name
            if tool_name not in self.enabled_tools:
                self.enabled_tools[tool_name] = False  # Default to disabled

        # Filter tools based on enabled status
        self.tools = [
            tool for tool in self.all_tools if self.enabled_tools.get(tool.name, False)
        ]

    def refresh_tools(self):
        """Refresh the tools list based on current enabled status."""
        self.tools = [
            tool for tool in self.all_tools if self.enabled_tools.get(tool.name, False)
        ]

    async def setup_prompt_session(self):
        """Set up prompt_toolkit session with completers and styling."""
        # Create nested completer for commands
        command_completer: Dict[str, Optional[WordCompleter]] = {
            cmd: None for cmd in self.commands.keys()
        }
        # Get model IDs from the loaded models
        model_ids = [model.id for model in self.language_models]
        # Get tool names for completion (use all_tools instead of just enabled tools)
        all_tool_names = [tool.name for tool in self.all_tools]

        # Add special completers for commands with arguments
        command_completer["agent"] = WordCompleter(["on", "off"])
        command_completer["a"] = command_completer["agent"]
        command_completer["debug"] = WordCompleter(["on", "off"])
        command_completer["d"] = command_completer["debug"]
        # Use model IDs for completion
        command_completer["model"] = WordCompleter(model_ids)
        command_completer["m"] = command_completer["model"]
        # Use tool names for completion
        command_completer["tools"] = WordCompleter(all_tool_names, match_middle=True)
        command_completer["t"] = command_completer["tools"]
        # Add completers for tool enable/disable commands
        enable_disable_completer = WordCompleter(
            all_tool_names + ["all"], match_middle=True
        )
        command_completer["enable"] = enable_disable_completer
        command_completer["en"] = enable_disable_completer
        command_completer["disable"] = enable_disable_completer
        command_completer["dis"] = enable_disable_completer

        # Create nested completer with commands prefixed with "/" and workspace commands
        prefixed_command_completer = {
            f"/{cmd}": completer for cmd, completer in command_completer.items()
        }
        completer = NestedCompleter(
            {
                **prefixed_command_completer,  # Use the prefixed commands
                "pwd": None,
                # Add PathCompleter for ls
                "ls": PathCompleter(only_directories=False),
                # Use PathCompleter for cd, suggesting only directories
                "cd": PathCompleter(only_directories=True),
                # Use PathCompleter for mkdir, suggesting only directories
                "mkdir": PathCompleter(only_directories=True),
                # Use PathCompleter for rm
                "rm": PathCompleter(only_directories=False),
                # Use PathCompleter for open
                "open": PathCompleter(only_directories=False),
                # Use PathCompleter for cat
                "cat": PathCompleter(only_directories=False),
                # Add completers for new commands
                "cp": PathCompleter(only_directories=False),
                "mv": PathCompleter(only_directories=False),
                # grep pattern doesn't have standard completion, but path does
                "grep": PathCompleter(only_directories=False),
            }
        )

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
        chunk_printed = False
        stream_buffer: list[str] = []

        try:
            while processor.has_messages() or processor.is_processing:
                message = await processor.get_message()
                if message:
                    message_type = message.get("type")

                    if message_type == "chunk":
                        content = message.get("content") or ""
                        done = message.get("done", False)
                        if content:
                            chunk_printed = True
                            stream_buffer.append(content)
                            self.console.print(content, end="", highlight=False)
                        if done and chunk_printed:
                            self.console.print()
                            chunk_printed = False

                    elif message_type == "message":
                        try:
                            parsed = Message.model_validate(message)
                        except ValidationError as validation_error:
                            if self.debug_mode:
                                self.console.print(
                                    f"[bold red]Failed to parse message:[/bold red] {validation_error}"
                                )
                            continue

                        if (
                            self.debug_mode
                            and parsed.tool_calls
                            and parsed.role == "assistant"
                        ):
                            for tool_call in parsed.tool_calls:
                                args_preview = json.dumps(tool_call.args)
                                if len(args_preview) > 120:
                                    args_preview = args_preview[:120] + "..."
                                self.console.print(
                                    f"\n[bold cyan][{tool_call.name}]:[/bold cyan] {args_preview}"
                                )
                        elif self.debug_mode and parsed.role == "tool":
                            preview = parsed.content
                            if isinstance(preview, str) and len(preview) > 200:
                                preview = preview[:200] + "..."
                            self.console.print(
                                f"\n[bold magenta][Tool Result: {parsed.name}][/bold magenta] {preview}"
                            )

                        has_streaming_output = bool(stream_buffer)

                        if (
                            parsed.role == "assistant"
                            and isinstance(parsed.content, str)
                            and parsed.content
                            and not has_streaming_output
                        ):
                            self.console.print(parsed.content)

                        if self._should_store_message(parsed):
                            self.messages.append(parsed)

                        stream_buffer.clear()

                    elif message_type == "error":
                        error_msg = message.get("message", "Unknown error")
                        self.console.print(
                            f"[bold red]Error:[/bold red] {error_msg}"
                        )
                    else:
                        if self.debug_mode:
                            self.console.print(
                                f"[yellow]Unhandled processor message:[/yellow] {message}"
                            )
                else:
                    await asyncio.sleep(0.01)

            await processor_task
        finally:
            if not processor_task.done():
                processor_task.cancel()
                try:
                    await processor_task
                except asyncio.CancelledError:
                    pass

    async def process_agent_response(self, problem: str):
        """Process input in agent mode using the messaging processors."""
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
        )

    async def process_regular_message(self, user_input: str) -> None:
        """Process standard chat input using the regular message processor."""
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

    def handle_workspace_command(self, cmd: str, args: List[str]) -> None:
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
                    self.console.print(
                        f"[bold red]Error:[/bold red] Path '{target_path_str}' does not exist."
                    )
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
                if not args:
                    # Go to home directory if no argument provided
                    new_dir_path = Path.home()
                else:
                    new_dir_path = (current_dir / args[0]).resolve()

                if not new_dir_path.is_dir():
                    self.console.print(
                        f"[bold red]Error:[/bold red] '{args[0] if args else 'Home'}' is not a directory or not accessible."
                    )
                    return
                try:
                    os.chdir(new_dir_path)
                    # Update context.workspace_dir to reflect the change, although it's less central now
                    self.context.workspace_dir = str(new_dir_path)
                    self.console.print(
                        f"Changed to: [bold green]{os.getcwd()}[/bold green]"
                    )
                except Exception as e:
                    self.console.print(
                        f"[bold red]Error changing directory:[/bold red] {e}"
                    )

            elif cmd == "mkdir":
                if not args:
                    self.console.print(
                        "[bold red]Error:[/bold red] Directory name required"
                    )
                    return
                new_dir_path = (current_dir / args[0]).resolve()
                try:
                    new_dir_path.mkdir(parents=True, exist_ok=True)
                    self.console.print(
                        f"Created directory: [bold green]{new_dir_path}[/bold green]"
                    )
                except Exception as e:
                    self.console.print(
                        f"[bold red]Error creating directory:[/bold red] {e}"
                    )

            elif cmd == "rm":
                if not args:
                    self.console.print("[bold red]Error:[/bold red] Path required")
                    return
                target_path = (current_dir / args[0]).resolve()
                if not target_path.exists():
                    self.console.print(
                        f"[bold red]Error:[/bold red] Path '{args[0]}' does not exist."
                    )
                    return

                try:
                    if target_path.is_dir():
                        shutil.rmtree(target_path)
                        self.console.print(
                            f"Removed directory: [bold green]{target_path}[/bold green]"
                        )
                    else:
                        target_path.unlink()
                        self.console.print(
                            f"Removed file: [bold green]{target_path}[/bold green]"
                        )
                except Exception as e:
                    self.console.print(f"[bold red]Error removing:[/bold red] {e}")

            elif cmd == "open":
                if not args:
                    self.console.print(
                        "[bold red]Error:[/bold red] File or directory path required"
                    )
                    return
                target_path_str = args[0]
                target_path = (current_dir / target_path_str).resolve()

                if not target_path.exists():
                    self.console.print(
                        f"[bold red]Error:[/bold red] Path '{target_path_str}' does not exist."
                    )
                    return

                try:
                    if sys.platform == "win32":
                        os.startfile(target_path)
                    elif sys.platform == "darwin":  # macOS
                        subprocess.run(["open", str(target_path)], check=True)
                    else:  # Linux and other POSIX
                        subprocess.run(["xdg-open", str(target_path)], check=True)
                    self.console.print(
                        f"Opened: [bold green]{target_path}[/bold green]"
                    )
                except Exception as e:
                    self.console.print(f"[bold red]Error opening:[/bold red] {e}")

            elif cmd == "cat":
                if not args:
                    self.console.print("[bold red]Error:[/bold red] File path required")
                    return
                target_path = (current_dir / args[0]).resolve()

                if not target_path.is_file():
                    self.console.print(
                        f"[bold red]Error:[/bold red] '{args[0]}' is not a file or does not exist"
                    )
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
                    self.console.print(
                        "[bold red]Usage: cp <source> <destination>[/bold red]"
                    )
                    return
                src_path = (current_dir / args[0]).resolve()
                dest_path = (current_dir / args[1]).resolve()

                if not src_path.exists():
                    self.console.print(
                        f"[bold red]Error:[/bold red] Source '{args[0]}' does not exist."
                    )
                    return

                # Prevent copying directory onto itself or file onto itself
                if src_path == dest_path:
                    self.console.print(
                        "[bold red]Error:[/bold red] Source and destination are the same."
                    )
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
                        dest_final = (
                            dest_path
                            if not dest_path.is_dir()
                            else dest_path / src_path.name
                        )
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
                    self.console.print(
                        "[bold red]Usage: mv <source> <destination>[/bold red]"
                    )
                    return
                src_path = (current_dir / args[0]).resolve()
                dest_path = (current_dir / args[1]).resolve()

                if not src_path.exists():
                    self.console.print(
                        f"[bold red]Error:[/bold red] Source '{args[0]}' does not exist."
                    )
                    return

                # Prevent moving onto itself
                if src_path == dest_path:
                    self.console.print(
                        "[bold red]Error:[/bold red] Source and destination are the same."
                    )
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
                    self.console.print(
                        "[bold red]Usage: grep <pattern> [path][/bold red]"
                    )
                    return

                pattern = args[0]
                search_path_str = args[1] if len(args) > 1 else "."
                # Resolve search path relative to current CWD
                search_path = (current_dir / search_path_str).resolve()

                if not search_path.exists():
                    self.console.print(
                        f"[bold red]Error:[/bold red] Path '{search_path_str}' does not exist."
                    )
                    return

                try:
                    regex = re.compile(pattern)
                except re.error as e:
                    self.console.print(
                        f"[bold red]Invalid regex pattern:[/bold red] {e}"
                    )
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
                                if (
                                    item.stat().st_size < 10 * 1024 * 1024
                                ):  # Skip files > 10MB
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
                                        rel_file_path = abs_file_path.relative_to(
                                            current_dir
                                        )
                                    except ValueError:  # Handle case where file is not under CWD (e.g., symlink)
                                        rel_file_path = abs_file_path  # Show absolute path if not relative

                                    self.console.print(
                                        f"\n[bold magenta]{rel_file_path}[/bold magenta]:"
                                    )
                                    file_match_found = True
                                    match_found = True

                                # Highlight the match within the line
                                highlighted_line = (
                                    line.strip()
                                )  # Process stripped line first
                                try:
                                    # Use original pattern for highlighting
                                    highlighted_line = regex.sub(
                                        lambda m: f"[bold yellow]{m.group(0)}[/bold yellow]",
                                        highlighted_line,
                                    )
                                except (
                                    Exception
                                ):  # Fallback if complex regex causes issues with sub
                                    pass  # Keep original line if highlighting fails

                                self.console.print(
                                    f"  [cyan]{i+1}:[/cyan] {highlighted_line}"
                                )

                    except UnicodeDecodeError:
                        # Skip files that are not valid UTF-8 text
                        continue
                    except Exception as e:
                        # Use absolute path in error message if relative path fails
                        try:
                            rel_file_path = file_path.resolve().relative_to(current_dir)
                        except Exception:
                            rel_file_path = file_path.resolve()

                        self.console.print(
                            f"[yellow]Warning:[/yellow] Could not read or process {rel_file_path}: {e}"
                        )

                if not match_found:
                    self.console.print(
                        f"No matches found for pattern '{pattern}' in '{search_path_str}'"
                    )

        except Exception as e:
            self.console.print(
                f"[bold red]Error executing workspace command:[/bold red] {e}"
            )
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
            "agent_mode": self.agent_mode,
            "debug_mode": self.debug_mode,
            "enabled_tools": self.enabled_tools,  # Save tool enable/disable states
        }

        try:
            with open(self.settings_file, "w") as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            self.console.print(
                f"[bold yellow]Warning:[/bold yellow] Failed to save settings: {e}"
            )

    def load_settings(self) -> None:
        """Load settings from the dotfile."""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, "r") as f:
                    settings = json.load(f)

                # Load the model ID preference to be used in initialize()
                self.model_id_from_settings = settings.get(
                    "model_id", self.model_id_from_settings
                )

                # Load other settings
                self.agent_mode = settings.get("agent_mode", False)
                self.debug_mode = settings.get("debug_mode", False)
                # Load tool enable/disable states
                self.enabled_tools = settings.get("enabled_tools", {})

        except Exception as e:
            self.console.print(
                f"[bold yellow]Warning:[/bold yellow] Failed to load settings: {e}"
            )
            # Keep default settings
            pass

    async def handle_command(self, cmd: str, args: List[str]) -> bool:
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
        if self.selected_model:
            settings_list.extend(
                [
                    f"[bold cyan]Provider:[/bold cyan] {self.selected_model.provider.value}",
                    f"[bold cyan]Model:[/bold cyan] {self.selected_model.id}",
                ]
            )
        else:
            settings_list.extend(
                [
                    "[bold cyan]Provider:[/bold cyan] None",
                    "[bold cyan]Model:[/bold cyan] None",
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
                display_path = (
                    cwd.replace(home_dir, "~", 1) if cwd.startswith(home_dir) else cwd
                )
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
                self.console.print(
                    "\n[bold yellow]Interrupted. Press Ctrl+C again to exit.[/bold yellow]"
                )
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
    await cli.run()


if __name__ == "__main__":
    asyncio.run(chat_cli())
