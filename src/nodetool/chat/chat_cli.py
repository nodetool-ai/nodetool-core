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
from typing import List, Optional, Dict
from pathlib import Path
import shutil  # Add shutil for cp and mv
import re  # Add re for grep

# New imports
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
from rich.prompt import Confirm  # Added for boolean toggles
from rich.columns import Columns  # Add Columns

# Existing imports
from nodetool.agents.agent import Agent
from nodetool.api.model import get_language_models
from nodetool.chat.providers import get_provider
from nodetool.chat.regular_chat import process_regular_chat
from nodetool.metadata.types import Provider, Message, ToolCall, LanguageModel
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.agents.tools import (
    AddLabelTool,
    ArchiveEmailTool,
    BrowserTool,
    ConvertPDFToMarkdownTool,
    DownloadFileTool,
    ExtractPDFTablesTool,
    ExtractPDFTextTool,
    GoogleGroundedSearchTool,
    GoogleImageGenerationTool,
    GoogleImagesTool,
    GoogleNewsTool,
    GoogleSearchTool,
    ListAssetsDirectoryTool,
    OpenAIImageGenerationTool,
    OpenAITextToSpeechTool,
    OpenAIWebSearchTool,
    ReadAssetTool,
    SaveAssetTool,
    ScreenshotTool,
    SearchEmailTool,
)
from nodetool.workflows.types import Chunk


class Command:
    """Base class for CLI commands with documentation and execution logic."""

    def __init__(self, name: str, description: str, aliases: List[str] = []):
        self.name = name
        self.description = description
        self.aliases = aliases or []

    async def execute(self, cli: "ChatCLI", args: List[str]) -> bool:
        """Execute the command with the given arguments.

        Returns:
            bool: True if the CLI should exit, False otherwise
        """
        raise NotImplementedError("Command must implement execute method")


class HelpCommand(Command):
    def __init__(self):
        super().__init__("help", "Display available commands", ["h"])

    async def execute(self, cli: "ChatCLI", args: List[str]) -> bool:
        cli.console.print("\n[bold]Available Commands[/bold]", style="cyan")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Command", style="cyan")
        table.add_column("Description", style="green")

        # Get unique command objects to avoid printing aliases separately
        unique_commands = sorted(set(cli.commands.values()), key=lambda cmd: cmd.name)

        for cmd in unique_commands:
            aliases = (
                f" ({', '.join(['/' + a for a in cmd.aliases])})" if cmd.aliases else ""
            )
            table.add_row(f"/{cmd.name}{aliases}", cmd.description)

        cli.console.print(table)
        cli.console.print("\n[bold]Workspace Commands[/bold]", style="cyan")

        workspace_table = Table(show_header=True, header_style="bold magenta")
        workspace_table.add_column("Command", style="cyan")
        workspace_table.add_column("Description", style="green")

        workspace_commands = [
            ("pwd", "Print current workspace directory"),
            ("ls [path]", "List contents of workspace directory"),
            ("cd [path]", "Change directory within workspace"),
            ("mkdir [dir]", "Create new directory in workspace"),
            ("rm [path]", "Remove file or directory in workspace"),
            ("open [file]", "Open file in system default application"),
            ("cat [file]", "Display the content of a file"),
            ("cp <src> <dest>", "Copy file or directory within workspace"),
            ("mv <src> <dest>", "Move/rename file or directory within workspace"),
            ("grep <pattern> [path]", "Search for pattern in files within workspace"),
            ("cdw", "Change directory to the defined workspace root"),
        ]

        for cmd, desc in workspace_commands:
            workspace_table.add_row(cmd, desc)

        cli.console.print(workspace_table)
        return False


class ExitCommand(Command):
    def __init__(self):
        super().__init__("exit", "Exit the chat interface", ["quit", "q"])

    async def execute(self, cli: "ChatCLI", args: List[str]) -> bool:
        cli.console.print("[bold yellow]Exiting chat...[/bold yellow]")
        return True


class ModelCommand(Command):
    def __init__(self):
        super().__init__("model", "Set the model for all agents by ID", ["m"])

    async def execute(self, cli: "ChatCLI", args: List[str]) -> bool:
        if not cli.selected_model:
            cli.console.print(
                "[bold red]Error:[/bold red] No models loaded. Cannot set model."
            )
            return False

        if not args:
            cli.console.print(
                f"Current model: [bold green]{cli.selected_model.name}[/bold green] (ID: {cli.selected_model.id}, Provider: {cli.selected_model.provider.value})"
            )
            return False

        model_id_to_set = args[0]
        found_model = None
        for model in cli.language_models:
            if model.id == model_id_to_set:
                found_model = model
                break

        if found_model:
            cli.selected_model = found_model
            # Update related model IDs (can be changed later via specific commands if needed)
            cli.planner_model_id = found_model.id
            cli.summarization_model_id = found_model.id
            cli.retrieval_model_id = found_model.id

            cli.console.print(
                f"Model set to [bold green]{found_model.name}[/bold green] (ID: {found_model.id})"
            )
            # Save settings after changing model
            cli.save_settings()
        else:
            cli.console.print(
                f"[bold red]Error:[/bold red] Model ID '{model_id_to_set}' not found. Use /models to list available IDs."
            )

        return False


class ModelsCommand(Command):
    def __init__(self):
        super().__init__(
            "models", "List available models for the current provider", ["ms"]
        )

    async def execute(self, cli: "ChatCLI", args: List[str]) -> bool:
        try:
            table = Table(title=f"Available Models", show_header=True)
            table.add_column("Provider", style="cyan")
            table.add_column("Model Name", style="cyan")
            table.add_column("Model ID", style="cyan")
            for model in cli.language_models:
                table.add_row(model.provider, model.name, model.id)

            cli.console.print(table)
        except Exception as e:
            cli.console.print(f"[bold red]Error listing models:[/bold red] {e}")

        return False


class ClearCommand(Command):
    def __init__(self):
        super().__init__("clear", "Clear chat history", ["cls"])

    async def execute(self, cli: "ChatCLI", args: List[str]) -> bool:
        cli.messages = []
        cli.console.print("[bold green]Chat history cleared[/bold green]")
        return False


class AgentCommand(Command):
    def __init__(self):
        super().__init__("agent", "Toggle agent mode (on/off)", ["a"])

    async def execute(self, cli: "ChatCLI", args: List[str]) -> bool:
        if not args:
            status = (
                "[bold green]ON[/bold green]"
                if cli.agent_mode
                else "[bold red]OFF[/bold red]"
            )
            cli.console.print(f"Agent mode is currently: {status}")
            return False

        if args[0].lower() == "on":
            cli.agent_mode = True
            cli.console.print("[bold green]Agent mode turned ON[/bold green]")
        elif args[0].lower() == "off":
            cli.agent_mode = False
            cli.console.print("[bold red]Agent mode turned OFF[/bold red]")
        else:
            cli.console.print("[bold yellow]Usage: /agent [on|off][/bold yellow]")

        # Save settings after changing agent mode
        cli.save_settings()
        return False


class DebugCommand(Command):
    def __init__(self):
        super().__init__("debug", "Toggle debug mode (on/off)", ["d"])

    async def execute(self, cli: "ChatCLI", args: List[str]) -> bool:
        new_state = None
        if not args:
            current_state = (
                "[bold green]ON[/bold green]"
                if cli.debug_mode
                else "[bold red]OFF[/bold red]"
            )
            cli.console.print(f"Debug mode is currently: {current_state}")
            # Ask user if they want to toggle
            if Confirm.ask(
                "Toggle debug mode?", default=cli.debug_mode, console=cli.console
            ):
                new_state = not cli.debug_mode
            else:
                return False  # No change
        elif args[0].lower() == "on":
            new_state = True
        elif args[0].lower() == "off":
            new_state = False
        else:
            cli.console.print(
                "[bold yellow]Usage: /debug [on|off] (or just /debug to toggle)[/bold yellow]"
            )
            return False

        if new_state is not None and new_state != cli.debug_mode:
            cli.debug_mode = new_state
            status = (
                "[bold green]ON[/bold green]"
                if cli.debug_mode
                else "[bold red]OFF[/bold red]"
            )
            message = (
                "Debug mode turned ON - Will display tool calls and results"
                if cli.debug_mode
                else "Debug mode turned OFF - Tool calls and results hidden"
            )
            cli.console.print(f"[bold green]{message}[/bold green]")
            cli.save_settings()  # Save settings after changing
        elif new_state is not None:
            cli.console.print(
                f"Debug mode is already {'ON' if cli.debug_mode else 'OFF'}."
            )

        return False


class AnalysisPhaseCommand(Command):
    """Command to toggle the Analysis Phase for the Agent planner."""

    def __init__(self):
        super().__init__("analysis", "Toggle agent analysis phase (on/off)", ["an"])

    async def execute(self, cli: "ChatCLI", args: List[str]) -> bool:
        new_state = None
        if not args:
            current_state = (
                "[bold green]ON[/bold green]"
                if cli.enable_analysis_phase
                else "[bold red]OFF[/bold red]"
            )
            cli.console.print(f"Agent Analysis Phase is currently: {current_state}")
            if Confirm.ask(
                "Toggle Analysis Phase?",
                default=cli.enable_analysis_phase,
                console=cli.console,
            ):
                new_state = not cli.enable_analysis_phase
            else:
                return False  # No change
        elif args[0].lower() == "on":
            new_state = True
        elif args[0].lower() == "off":
            new_state = False
        else:
            cli.console.print(
                "[bold yellow]Usage: /analysis [on|off] (or just /analysis to toggle)[/bold yellow]"
            )
            return False

        if new_state is not None and new_state != cli.enable_analysis_phase:
            cli.enable_analysis_phase = new_state
            status = (
                "[bold green]ON[/bold green]"
                if cli.enable_analysis_phase
                else "[bold red]OFF[/bold red]"
            )
            message = f"Agent Analysis Phase turned {status}"
            cli.console.print(f"[bold green]{message}[/bold green]")
            cli.save_settings()
        elif new_state is not None:
            cli.console.print(
                f"Agent Analysis Phase is already {'ON' if cli.enable_analysis_phase else 'OFF'}."
            )
        return False


class FlowAnalysisCommand(Command):
    """Command to toggle the Data Flow Analysis Phase for the Agent planner."""

    def __init__(self):
        super().__init__(
            "flow", "Toggle agent data flow analysis phase (on/off)", ["fl"]
        )

    async def execute(self, cli: "ChatCLI", args: List[str]) -> bool:
        new_state = None
        if not args:
            current_state = (
                "[bold green]ON[/bold green]"
                if cli.enable_flow_analysis
                else "[bold red]OFF[/bold red]"
            )
            cli.console.print(
                f"Agent Data Flow Analysis Phase is currently: {current_state}"
            )
            if Confirm.ask(
                "Toggle Data Flow Analysis Phase?",
                default=cli.enable_flow_analysis,
                console=cli.console,
            ):
                new_state = not cli.enable_flow_analysis
            else:
                return False  # No change
        elif args[0].lower() == "on":
            new_state = True
        elif args[0].lower() == "off":
            new_state = False
        else:
            cli.console.print(
                "[bold yellow]Usage: /flow [on|off] (or just /flow to toggle)[/bold yellow]"
            )
            return False

        if new_state is not None and new_state != cli.enable_flow_analysis:
            cli.enable_flow_analysis = new_state
            status = (
                "[bold green]ON[/bold green]"
                if cli.enable_flow_analysis
                else "[bold red]OFF[/bold red]"
            )
            message = f"Agent Data Flow Analysis Phase turned {status}"
            cli.console.print(f"[bold green]{message}[/bold green]")
            cli.save_settings()
        elif new_state is not None:
            cli.console.print(
                f"Agent Data Flow Analysis Phase is already {'ON' if cli.enable_flow_analysis else 'OFF'}."
            )
        return False


class ChangeToWorkspaceCommand(Command):
    """Command to change the current directory to the context's workspace directory."""

    def __init__(self):
        super().__init__("cdw", "Change directory to the defined workspace root")

    async def execute(self, cli: "ChatCLI", args: List[str]) -> bool:
        workspace_dir = Path(cli.context.workspace_dir).resolve()
        if not workspace_dir.is_dir():
            cli.console.print(
                f"[bold red]Error:[/bold red] Workspace directory '{cli.context.workspace_dir}' does not exist or is not a directory."
            )
            return False
        try:
            os.chdir(workspace_dir)
            # No need to update context.workspace_dir here, we are just navigating to it
            cli.console.print(
                f"Changed to workspace: [bold green]{os.getcwd()}[/bold green]"
            )
        except Exception as e:
            cli.console.print(
                f"[bold red]Error changing to workspace directory:[/bold red] {e}"
            )
        return False


class UsageCommand(Command):
    def __init__(self):
        super().__init__(
            "usage", "Display usage statistics for the selected model's provider", ["u"]
        )

    async def execute(self, cli: "ChatCLI", args: List[str]) -> bool:
        if cli.selected_model:
            # Get the provider instance for the selected model
            try:
                provider_instance = get_provider(cli.selected_model.provider)
                cli.console.print(
                    f"[bold]Usage statistics for provider: {cli.selected_model.provider.value}[/bold]"
                )
                # Assuming the provider instance has a 'usage' attribute or method
                usage_data = getattr(provider_instance, "usage", {})
                syntax = Syntax(
                    json.dumps(usage_data, indent=2),
                    "json",
                    theme="monokai",
                    line_numbers=True,
                )
                cli.console.print(syntax)
            except Exception as e:
                cli.console.print(
                    f"[bold red]Error getting usage for provider {cli.selected_model.provider.value}:[/bold red] {e}"
                )
        else:
            cli.console.print(
                "[yellow]No model selected. Cannot display usage.[/yellow]"
            )
        return False


class ToolsCommand(Command):
    def __init__(self):
        super().__init__(
            "tools", "List available tools or show details about a specific tool", ["t"]
        )

    async def execute(self, cli: "ChatCLI", args: List[str]) -> bool:
        if not args:
            table = Table(title="Available Tools", show_header=True)
            table.add_column("Tool Name", style="cyan")
            table.add_column("Description", style="green")

            for tool in cli.tools:
                table.add_row(tool.__class__.__name__, tool.description)

            cli.console.print(table)
        else:
            tool_name = args[0]
            found = False

            for tool in cli.tools:
                if tool.__class__.__name__.lower() == tool_name.lower():
                    found = True
                    panel = Panel(
                        f"[bold]Description:[/bold] {tool.description}\n\n"
                        f"[bold]Parameters:[/bold] {json.dumps(tool.parameters, indent=2)}",
                        title=f"Tool: {tool.__class__.__name__}",
                        border_style="green",
                    )
                    cli.console.print(panel)
                    break

            if not found:
                cli.console.print(f"[bold red]Tool '{tool_name}' not found[/bold red]")

        return False


class ReasoningModelCommand(Command):
    """Command to set the reasoning model used by the agent."""

    def __init__(self):
        super().__init__(
            "reasoning",
            "Set the reasoning model for the agent (use 'default' to sync with main model)",
            ["r"],
        )

    async def execute(self, cli: "ChatCLI", args: List[str]) -> bool:
        if not cli.selected_model:
            cli.console.print(
                "[bold red]Error:[/bold red] No models loaded. Cannot set reasoning model."
            )
            return False

        current_reasoning_model_id = cli.reasoning_model_id or cli.selected_model.id
        current_reasoning_model = None
        for model in cli.language_models:
            if model.id == current_reasoning_model_id:
                current_reasoning_model = model
                break

        if not args:
            if current_reasoning_model:
                cli.console.print(
                    f"Current reasoning model: [bold green]{current_reasoning_model.name}[/bold green] (ID: {current_reasoning_model.id})"
                )
            else:
                # Should ideally not happen if models are loaded and IDs are synced
                cli.console.print(
                    f"Current reasoning model ID: [bold green]{current_reasoning_model_id}[/bold green] (Model details not found, using main model default)"
                )
                cli.console.print(
                    "Use '/reasoning [model_id]' or '/reasoning default' to set."
                )
            return False

        model_id_to_set = args[0].lower()

        if model_id_to_set == "default":
            # Set reasoning model to track the main selected model
            cli.reasoning_model_id = None  # Use None to signify tracking default
            cli.console.print(
                f"Reasoning model set to track main model: [bold green]{cli.selected_model.name}[/bold green]"
            )
            cli.save_settings()
            return False

        # Find the specified model ID
        found_model = None
        for model in cli.language_models:
            if model.id == model_id_to_set:
                found_model = model
                break

        if found_model:
            cli.reasoning_model_id = found_model.id
            cli.console.print(
                f"Reasoning model set to [bold green]{found_model.name}[/bold green] (ID: {found_model.id})"
            )
            cli.save_settings()
        else:
            cli.console.print(
                f"[bold red]Error:[/bold red] Model ID '{model_id_to_set}' not found. Use /models to list available IDs."
            )

        return False


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
        self.context = ProcessingContext(user_id="test", auth_token="test")
        self.messages: list[Message] = []
        self.agent_mode = False
        self.debug_mode = False
        self.agent = None
        # Add new phase settings
        self.enable_analysis_phase = True
        self.enable_flow_analysis = True

        # Store selected LanguageModel object and model ID preference
        self.language_models: List[LanguageModel] = []
        self.selected_model: Optional[LanguageModel] = None
        self.model_id_from_settings: Optional[str] = "gpt-4o"  # Default model ID

        # Model attributes for agent config (can be overridden)
        # These might eventually also become LanguageModel selections
        # For now, we'll default them based on the primary selected model
        self.planner_model_id: Optional[str] = None
        self.summarization_model_id: Optional[str] = None
        self.retrieval_model_id: Optional[str] = None
        self.reasoning_model_id: Optional[str] = None  # Add reasoning model ID

        self.settings_file = os.path.join(os.path.expanduser("~"), ".nodetool_settings")
        self.history_file = os.path.join(os.path.expanduser("~"), ".nodetool_history")

        # Register commands
        self.commands = {}
        self.register_commands()

        # Load settings if they exist (loads model_id_from_settings)
        self.load_settings()

    def register_commands(self):
        """Register all available commands."""
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
            AnalysisPhaseCommand(),  # Add new command
            FlowAnalysisCommand(),  # Add new command
            ChangeToWorkspaceCommand(),  # Add new command
            ReasoningModelCommand(),  # Add new command
        ]

        for command in commands:
            self.commands[command.name] = command
            # Register aliases
            for alias in command.aliases:
                self.commands[alias] = command

    async def initialize(self):
        """Initialize async components and workspace with visual feedback."""
        # Initialize components with progress indicators
        try:
            self.language_models = await get_language_models()
        except Exception as e:
            self.console.print(
                f"[bold red]Error fetching language models:[/bold red] {e}"
            )
            self.language_models = []  # Ensure it's an empty list

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

        # If no model found from settings or no setting existed, try finding the default 'gpt-4o' or the first available model
        if not found_model:
            default_id_to_try = "gpt-4o"  # Fallback default
            for model in self.language_models:
                if model.id == default_id_to_try:
                    found_model = model
                    break
            if found_model:
                self.console.print(f"Using default model '{found_model.name}'.")
            elif self.language_models:
                # If default not found, pick the first available model
                found_model = self.language_models[0]
                self.console.print(f"Using first available model '{found_model.name}'.")
            else:
                self.console.print(
                    "[bold red]Error:[/bold red] No language models available. Please check configuration."
                )
                # Set to None, subsequent operations should handle this
                self.selected_model = None
                return  # Cannot proceed without models

        self.selected_model = found_model
        # Set other model defaults based on the primary selected one
        self.planner_model_id = self.selected_model.id
        self.summarization_model_id = self.selected_model.id
        self.retrieval_model_id = self.selected_model.id
        # Initialize reasoning model ID based on settings or default to selected model if not explicitly set
        # Check if reasoning_model_id from settings corresponds to an existing model
        reasoning_model_exists = False
        if self.reasoning_model_id:
            for model in self.language_models:
                if model.id == self.reasoning_model_id:
                    reasoning_model_exists = True
                    break
        # If the loaded reasoning_model_id is not valid or not set, default it to the selected model's ID
        if not self.reasoning_model_id or not reasoning_model_exists:
            if (
                self.selected_model
            ):  # Ensure selected_model exists before accessing its id
                self.reasoning_model_id = self.selected_model.id
            else:
                # Handle case where no model is selected - perhaps set to None or log an error
                self.reasoning_model_id = None  # Or handle as appropriate
                self.console.print(
                    "[bold yellow]Warning:[/bold yellow] No valid selected model, cannot default reasoning model."
                )

        # Initialize tools (keep this after model loading if tools depend on models/providers)
        self.tools = [
            AddLabelTool(),
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
            ListAssetsDirectoryTool(),
            OpenAIImageGenerationTool(),
            OpenAITextToSpeechTool(),
            OpenAIWebSearchTool(),
            ReadAssetTool(),
            SaveAssetTool(),
            ScreenshotTool(),
            SearchEmailTool(),
            ReasoningTool(),
        ]

    async def setup_prompt_session(self):
        """Set up prompt_toolkit session with completers and styling."""
        # Create nested completer for commands
        command_completer: Dict[str, Optional[WordCompleter]] = {
            cmd: None for cmd in self.commands.keys()
        }
        # Get model IDs from the loaded models
        model_ids = [model.id for model in self.language_models]
        # Get tool names for completion
        tool_names = [tool.__class__.__name__ for tool in self.tools]

        # Add special completers for commands with arguments
        command_completer["agent"] = WordCompleter(["on", "off"])
        command_completer["a"] = command_completer["agent"]
        command_completer["debug"] = WordCompleter(["on", "off"])
        command_completer["d"] = command_completer["debug"]
        # Use model IDs for completion
        command_completer["model"] = WordCompleter(model_ids)
        command_completer["m"] = command_completer["model"]
        # Use tool names for completion
        command_completer["tools"] = WordCompleter(tool_names)
        command_completer["t"] = command_completer["tools"]
        # Add completers for new phase commands
        command_completer["analysis"] = WordCompleter(["on", "off"])
        command_completer["an"] = command_completer["analysis"]
        command_completer["flow"] = WordCompleter(["on", "off"])
        command_completer["fl"] = command_completer["flow"]
        # Add completer for reasoning command
        command_completer["reasoning"] = WordCompleter(model_ids + ["default"])
        command_completer["r"] = command_completer["reasoning"]

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
                # Add new workspace command
                "cdw": None,
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

    def initialize_agent(self, objective: str):
        """Initialize or reinitialize the agent with current settings."""
        if not self.selected_model:
            raise ValueError("Cannot initialize agent: No model selected or loaded.")

        # Get the provider instance using the selected model's provider type
        provider_instance = get_provider(self.selected_model.provider)

        # Pass the selected model ID to the agent
        agent = Agent(
            name="Agent",
            objective=objective,
            provider=provider_instance,
            model=self.selected_model.id,  # Use the ID of the selected model
            tools=self.tools,
            # planner_model, summarization_model, retrieval_model are not direct Agent args
            # They might be configured differently or passed via context if needed
            enable_analysis_phase=self.enable_analysis_phase,  # Pass setting
            enable_data_contracts_phase=self.enable_flow_analysis,  # Pass setting
            # Pass the specific reasoning model ID if set, otherwise default to main model ID
            reasoning_model=self.reasoning_model_id or self.selected_model.id,
        )
        return agent

    async def process_agent_response(self, problem: str):
        """Process a problem with the Agent and display the response with rich formatting."""
        self.agent = self.initialize_agent(problem)
        output = ""
        async for item in self.agent.execute(self.context):
            if isinstance(item, Chunk):
                output += item.content
                # Print the chunk directly for real-time feedback
                self.console.print(item.content, end="", highlight=False)
            elif isinstance(item, ToolCall):
                args = json.dumps(item.args)
                if len(args) > 120:
                    args = args[:120] + "..."
                if self.debug_mode:
                    self.console.print(
                        f"\n[bold cyan][{item.name}]:[/bold cyan] {args}"
                    )

        # Print final newline to separate from prompt
        self.console.print()

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
                        f"[bold red]Error:[/bold red] Source and destination are the same."
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
                        f"[bold red]Error:[/bold red] Source and destination are the same."
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
                                    except (
                                        ValueError
                                    ):  # Handle case where file is not under CWD (e.g., symlink)
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
            "enable_analysis_phase": self.enable_analysis_phase,  # Save setting
            "enable_flow_analysis": self.enable_flow_analysis,  # Save setting
            "reasoning_model_id": self.reasoning_model_id,  # Save reasoning model setting
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
                # Load phase settings, defaulting to True if not found
                self.enable_analysis_phase = settings.get("enable_analysis_phase", True)
                # Load flow setting, defaulting to True if not found
                self.enable_flow_analysis = settings.get("enable_flow_analysis", True)
                # Handle potential legacy setting name for backward compatibility
                if (
                    "enable_data_contracts_phase" in settings
                    and "enable_flow_analysis" not in settings
                ):
                    self.enable_flow_analysis = settings.get(
                        "enable_data_contracts_phase", True
                    )
                # Load reasoning model ID
                self.reasoning_model_id = settings.get("reasoning_model_id", None)

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

        settings_list.extend(
            [
                f"[bold cyan]Agent:[/bold cyan] {'ON' if self.agent_mode else 'OFF'} (/agent)",
                f"[bold cyan]Debug:[/bold cyan] {'ON' if self.debug_mode else 'OFF'} (/debug)",
                f"[bold cyan]Analysis:[/bold cyan] {'ON' if self.enable_analysis_phase else 'OFF'} (/analysis)",
                f"[bold cyan]Flow:[/bold cyan] {'ON' if self.enable_flow_analysis else 'OFF'} (/flow)",
                # Display reasoning model setting, handle None case for selected_model
                f"[bold cyan]Reasoning:[/bold cyan] {self.reasoning_model_id or ('Default (' + self.selected_model.id + ')' if self.selected_model else 'Default (None)')} (/reasoning)",
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
                    f"[bold cyan]Workspace:[/bold cyan] {str(self.context.workspace_dir)} {os.getcwd()}",
                    "Type [bold cyan]/help[/bold cyan] for application commands (/help)",
                    "Use workspace commands: [bold]pwd, ls, cd, mkdir, rm, open, cat, cp, mv, grep, cdw[/bold]",
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
                if self.agent_mode:
                    await self.process_agent_response(user_input)
                else:
                    # Display user message in chat style - indented
                    self.console.print(
                        f"\n  [bold cyan]You:[/bold cyan] {user_input}"
                    )  # Indent user message

                    if not self.selected_model:
                        self.console.print(
                            "[bold red]Error:[/bold red] No model selected"
                        )
                        continue

                    # Show "thinking" indicator while processing
                    with self.console.status(
                        "[bold green]Thinking...", spinner="dots"
                    ) as status:
                        self.messages = await process_regular_chat(
                            user_input=user_input,
                            messages=self.messages,
                            model=self.selected_model.id,
                            provider=get_provider(self.selected_model.provider),
                            status=status,
                            context=self.context,
                            debug_mode=self.debug_mode,
                        )

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
