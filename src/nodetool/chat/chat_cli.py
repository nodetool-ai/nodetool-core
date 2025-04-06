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
from typing import List, Optional, Dict, Any, Callable, Awaitable
from pathlib import Path
from functools import wraps

# New imports
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter, NestedCompleter
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live

# Existing imports
from nodetool.chat.agent import Agent
from nodetool.chat.providers import get_provider
from nodetool.chat.providers.base import ChatProvider
from nodetool.chat.regular_chat import process_regular_chat
from nodetool.chat.task_planner import TaskPlanner
from nodetool.chat.tools.browser import DownloadFileTool
from nodetool.chat.workspace_manager import WorkspaceManager
from nodetool.metadata.types import Provider, Message, ToolCall
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.chat.ollama_service import get_ollama_models
from nodetool.chat.chat import get_openai_models
from nodetool.chat.tools import (
    GoogleSearchTool,
    BrowserTool,
    DownloadFileTool,
    WebFetchTool,
    ScreenshotTool,
    ChromaTextSearchTool,
    ChromaHybridSearchTool,
    ExtractPDFTablesTool,
    ExtractPDFTextTool,
    ConvertPDFToMarkdownTool,
    ReadWorkspaceFileTool,
    ListWorkspaceContentsTool,
    ExecuteWorkspaceCommandTool,
)
from nodetool.workflows.types import Chunk


def provider_from_model(model: str) -> ChatProvider:
    if model.startswith("claude"):
        return get_provider(Provider.Anthropic)
    elif model.startswith("gpt"):
        return get_provider(Provider.OpenAI)
    elif model.startswith("gemini"):
        return get_provider(Provider.Gemini)
    else:
        raise ValueError(f"Unsupported model: {model}")


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

        for cmd in cli.commands.values():
            aliases = f" ({', '.join(cmd.aliases)})" if cmd.aliases else ""
            table.add_row(f"{cmd.name}{aliases}", cmd.description)

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
        super().__init__("model", "Set the model for all agents", ["m"])

    async def execute(self, cli: "ChatCLI", args: List[str]) -> bool:
        if not args:
            cli.console.print(f"Current model: [bold green]{cli.model}[/bold green]")
            return False

        model_name = args[0]
        cli.model = model_name

        cli.console.print(
            f"Model set to [bold green]{model_name}[/bold green] for all agents"
        )
        # Save settings after changing model
        cli.save_settings()
        return False


class ModelsCommand(Command):
    def __init__(self):
        super().__init__(
            "models", "List available models for the current provider", ["ms"]
        )

    async def execute(self, cli: "ChatCLI", args: List[str]) -> bool:
        try:
            table = Table(title=f"Available Models", show_header=True)
            table.add_column("Model Name", style="cyan")

            for model_info in cli.ollama_models:
                table.add_row(model_info.name)
            for model_info in cli.openai_models[:10]:
                table.add_row(model_info.id)
            for model in [
                "gemini-2.5-pro-exp-03-25",
                "gemini-2.0-flash",
                "gemini-2.0-flash-lite",
                "gemini-1.5-flash-8b",
                "gemini-1.5-flash",
                "gemini-1.5-pro",
            ]:
                table.add_row(model)
            for model in [
                "claude-3-haiku-20240307",
                "claude-3-5-sonnet-20241022",
                "claude-3-7-sonnet-20250219",
            ]:
                table.add_row(model)

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
        if not args:
            status = (
                "[bold green]ON[/bold green]"
                if cli.debug_mode
                else "[bold red]OFF[/bold red]"
            )
            cli.console.print(f"Debug mode is currently: {status}")
            return False

        if args[0].lower() == "on":
            cli.debug_mode = True
            cli.console.print(
                "[bold green]Debug mode turned ON[/bold green] - Will display tool calls and results"
            )
        elif args[0].lower() == "off":
            cli.debug_mode = False
            cli.console.print(
                "[bold red]Debug mode turned OFF[/bold red] - Tool calls and results hidden"
            )
        else:
            cli.console.print("[bold yellow]Usage: /debug [on|off][/bold yellow]")

        # Save settings after changing debug mode
        cli.save_settings()
        return False


class UsageCommand(Command):
    def __init__(self):
        super().__init__("usage", "Display current provider's usage statistics", ["u"])

    async def execute(self, cli: "ChatCLI", args: List[str]) -> bool:
        if cli.agent and cli.agent.provider:
            cli.console.print("[bold]Provider usage statistics:[/bold]")
            syntax = Syntax(
                json.dumps(cli.agent.provider.usage, indent=2),
                "json",
                theme="monokai",
                line_numbers=True,
            )
            cli.console.print(syntax)
        else:
            cli.console.print("[yellow]No usage statistics available[/yellow]")
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
        self.model = "gpt-4o"
        self.planner_model = self.model
        self.summarization_model = self.model
        self.retrieval_model = self.model

        self.settings_file = os.path.join(os.path.expanduser("~"), ".nodetool_settings")
        self.history_file = os.path.join(os.path.expanduser("~"), ".nodetool_history")

        # Register commands
        self.commands = {}
        self.register_commands()

        # Load settings if they exist
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
        ]

        for command in commands:
            self.commands[command.name] = command
            # Register aliases
            for alias in command.aliases:
                self.commands[alias] = command

    async def initialize(self):
        """Initialize async components and workspace with visual feedback."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            # Initialize components with progress indicators
            task1 = progress.add_task("[cyan]Fetching Ollama models...", total=1)
            try:
                self.ollama_models = await get_ollama_models()
                progress.update(task1, completed=1)
            except Exception as e:
                pass

            task2 = progress.add_task("[cyan]Fetching OpenAI models...", total=1)
            self.openai_models = await get_openai_models()
            progress.update(task2, completed=1)

            task3 = progress.add_task("[cyan]Setting up workspace...", total=1)

            progress.update(task3, completed=1)
            workspace_dir = self.context.workspace_dir

            task4 = progress.add_task("[cyan]Initializing tools...", total=1)
            self.tools = [
                ExtractPDFTablesTool(workspace_dir),
                ExtractPDFTextTool(workspace_dir),
                ConvertPDFToMarkdownTool(workspace_dir),
                GoogleSearchTool(workspace_dir),
                WebFetchTool(workspace_dir),
                DownloadFileTool(workspace_dir),
                BrowserTool(workspace_dir),
                ScreenshotTool(workspace_dir),
                ExecuteWorkspaceCommandTool(workspace_dir),
            ]
            progress.update(task4, completed=1)

    async def setup_prompt_session(self):
        """Set up prompt_toolkit session with completers and styling."""
        # Create nested completer for commands
        command_completer: Dict[str, Optional[WordCompleter]] = {
            cmd: None for cmd in self.commands.keys()
        }

        # Add special completers for commands with arguments
        command_completer["agent"] = WordCompleter(["on", "off"])
        command_completer["debug"] = WordCompleter(["on", "off"])

        # Add model completers
        model_names = []
        if hasattr(self, "ollama_models"):
            model_names.extend([m.name for m in self.ollama_models])
        if hasattr(self, "openai_models"):
            model_names.extend([m.id for m in self.openai_models])
        model_names.extend(
            [
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
                "claude-3-5-sonnet-20241022",
                "claude-3-7-sonnet-20250219",
                "gemini-2.5-pro-exp-03-25",
                "gemini-2.0-flash",
                "gemini-2.0-flash-lite",
                "gemini-1.5-flash-8b",
                "gemini-1.5-flash",
                "gemini-1.5-pro",
            ]
        )
        command_completer["model"] = WordCompleter(model_names)

        # Create nested completer with commands and workspace commands
        completer = NestedCompleter(
            {
                **command_completer,
                "pwd": None,
                "ls": None,
                "cd": None,
                "mkdir": None,
                "rm": None,
                "open": None,
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
        # Use provider_from_model to dynamically select the provider based on model name
        provider_instance = provider_from_model(self.model)
        agent = Agent(
            name="Agent",
            objective=objective,
            provider=provider_instance,
            model=self.model,
            tools=self.tools,
        )
        return agent

    async def process_agent_response(self, problem: str):
        """Process a problem with the Agent and display the response with rich formatting."""
        self.agent = self.initialize_agent(problem)
        with self.console.status(
            "[bold green]Agent thinking...", spinner="dots"
        ) as status:
            output = ""
            async for item in self.agent.execute(self.context):
                if isinstance(item, Chunk):
                    output += item.content
                    # Update status message to show we're receiving content
                    status.update("[bold green]Receiving response...")
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
        """Handle workspace-related commands with rich output formatting."""
        try:
            if cmd == "pwd":
                self.console.print(str(self.context.workspace_dir), style="bold green")
            elif cmd == "ls":
                target = self.context.workspace_dir
                if args:
                    path = (Path(self.context.workspace_dir) / args[0]).resolve()
                    if not str(path).startswith(str(self.context.workspace_dir)):
                        self.console.print(
                            "[bold red]Error:[/bold red] Cannot access paths outside workspace"
                        )
                        return
                    target = path

                try:
                    table = Table(show_header=True)
                    table.add_column("Type", style="cyan", width=2)
                    table.add_column("Name", style="green")

                    items = list(Path(target).iterdir())
                    items.sort(key=lambda x: (not x.is_dir(), x.name))

                    for item in items:
                        type_char = "d" if item.is_dir() else "f"
                        table.add_row(type_char, item.name)

                    self.console.print(table)
                except Exception as e:
                    self.console.print(f"[bold red]Error:[/bold red] {e}")
            elif cmd == "cd":
                new_dir = (Path(self.context.workspace_dir) / args[0]).resolve()
                if not str(new_dir).startswith(str(self.context.workspace_dir)):
                    self.console.print(
                        "[bold red]Error:[/bold red] Cannot access paths outside workspace"
                    )
                    return
                if not new_dir.is_dir():
                    self.console.print(
                        f"[bold red]Error:[/bold red] {args[0]} is not a directory"
                    )
                    return
                self.context.workspace_dir = str(new_dir)
                self.console.print(
                    f"Changed to: [bold green]{self.context.workspace_dir}[/bold green]"
                )
            elif cmd == "mkdir":
                if not args:
                    self.console.print(
                        "[bold red]Error:[/bold red] Directory name required"
                    )
                    return
                new_dir = (Path(self.context.workspace_dir) / args[0]).resolve()
                if not str(new_dir).startswith(str(self.context.workspace_dir)):
                    self.console.print(
                        "[bold red]Error:[/bold red] Cannot create directory outside workspace"
                    )
                    return
                new_dir.mkdir(parents=True, exist_ok=True)
                self.console.print(
                    f"Created directory: [bold green]{new_dir}[/bold green]"
                )
            elif cmd == "rm":
                if not args:
                    self.console.print("[bold red]Error:[/bold red] Path required")
                    return
                target = (Path(self.context.workspace_dir) / args[0]).resolve()
                if not str(target).startswith(str(self.context.workspace_dir)):
                    self.console.print(
                        "[bold red]Error:[/bold red] Cannot remove paths outside workspace"
                    )
                    return
                if target.is_dir():
                    import shutil

                    shutil.rmtree(target)
                    self.console.print(
                        f"Removed directory: [bold green]{target}[/bold green]"
                    )
                else:
                    target.unlink()
                    self.console.print(
                        f"Removed file: [bold green]{target}[/bold green]"
                    )
            elif cmd == "open":
                target = self.context.workspace_dir
                if args:
                    target = (Path(self.context.workspace_dir) / args[0]).resolve()

                if not str(target).startswith(str(self.context.workspace_dir)):
                    self.console.print(
                        "[bold red]Error:[/bold red] Cannot open files outside workspace"
                    )
                    return

                if os.name == "nt":  # Windows
                    os.startfile(target)
                elif os.name == "posix":  # macOS and Linux
                    subprocess.run(
                        [
                            "open" if sys.platform == "darwin" else "xdg-open",
                            str(target),
                        ]
                    )
                self.console.print(f"Opened: [bold green]{target}[/bold green]")
        except Exception as e:
            self.console.print(f"[bold red]Error:[/bold red] {e}")

    def save_settings(self) -> None:
        """Save current settings to the dotfile."""
        settings = {
            "provider": self.provider.value,
            "model": self.model,
            "agent_mode": self.agent_mode,
            "debug_mode": self.debug_mode,
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

                # Apply loaded settings
                if "provider" in settings:
                    if settings["provider"] == "openai":
                        self.provider = Provider.OpenAI
                    elif settings["provider"] == "anthropic":
                        self.provider = Provider.Anthropic
                    elif settings["provider"] == "ollama":
                        self.provider = Provider.Ollama

                self.model = settings.get("model", self.model)
                self.agent_mode = settings.get("agent_mode", False)
                self.debug_mode = settings.get("debug_mode", False)
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
        self.console.print(
            Panel(
                "[bold green]Welcome to NodeTool Chat CLI[/bold green]\n\n"
                "Type [bold cyan]/help[/bold cyan] for available commands",
                title="NodeTool",
                border_style="green",
            )
        )

        # Display current settings
        settings_table = Table(
            title="Current Settings", show_header=True, header_style="bold"
        )
        settings_table.add_column("Setting", style="cyan")
        settings_table.add_column("Value", style="green")
        settings_table.add_row("Provider", self.provider.value)
        settings_table.add_row("Model", self.model)
        settings_table.add_row("Agent Mode", "ON" if self.agent_mode else "OFF")
        settings_table.add_row("Debug Mode", "ON" if self.debug_mode else "OFF")
        settings_table.add_row("Workspace", str(self.context.workspace_dir))

        self.console.print(settings_table)
        self.console.print(
            "Use workspace commands: [bold]pwd[/bold], [bold]ls[/bold], [bold]cd[/bold], [bold]mkdir[/bold], [bold]rm[/bold], [bold]open[/bold]"
        )

        while True:
            try:
                # Create prompt with current directory info
                rel_path = os.path.relpath(
                    self.context.workspace_dir,
                )
                rel_path = "." if rel_path == "." else rel_path
                prompt = f"[{rel_path}]> " if rel_path != "." else "> "

                # Get input with prompt_toolkit (supports multi-line input and better completion)
                user_input = await self.session.prompt_async(prompt)

                if not user_input:
                    continue

                # Handle workspace commands
                if user_input.startswith(("pwd", "ls", "cd", "mkdir", "rm", "open")):
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
                    # Display user message in chat style
                    self.console.print(f"\n[bold cyan]You:[/bold cyan] {user_input}")

                    # Show "thinking" indicator while processing
                    with self.console.status("[bold green]Thinking...", spinner="dots"):
                        self.messages = await process_regular_chat(
                            user_input=user_input,
                            messages=self.messages,
                            model=self.model,
                            provider=provider_from_model(self.model),
                            workspace_dir=str(self.context.workspace_dir),
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
