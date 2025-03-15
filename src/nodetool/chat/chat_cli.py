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
import tempfile
import traceback
import readline
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Sequence, Optional

from nodetool.chat.cot_agent import CoTAgent
from nodetool.chat.providers import get_provider, Chunk
from nodetool.chat.tools import Tool
from nodetool.chat.chat import (
    run_tool,
)
from nodetool.chat.regular_chat import process_regular_chat
from nodetool.metadata.types import Provider, Message, ToolCall, FunctionModel
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.chat.ollama_service import get_ollama_models
from nodetool.chat.chat import get_openai_models


class ChatCLI:
    """Interactive command-line chat interface with multi-provider support and CoT agent.

    This class implements a command-line interface for interacting with various AI models
    through different providers (OpenAI, Anthropic, Ollama). It features:

    1. Provider Selection: Switch between different LLM providers
    2. Model Selection: Choose specific models from each provider
    3. Command History: Preserves history between sessions using readline
    4. Tab Completion: Smart completion for commands and options
    5. Chain of Thought (CoT) Agent: Step-by-step reasoning capabilities
    6. Debug Mode: Display tool calls and their results
    7. Workspace Management: File system commands for managing a sandbox workspace

    The CLI supports the following commands:
    - /provider [openai|anthropic|ollama]: Switch between AI providers
    - /model [model_name]: Select a specific model
    - /models: List available models for the current provider
    - /clear: Clear the conversation history
    - /agent [on|off]: Toggle CoT agent mode
    - /debug [on|off]: Toggle debug mode
    - /usage: Display current provider's usage statistics
    - /quit or /exit: Exit the chat interface
    - /help: Display available commands

    Workspace commands:
    - pwd: Print current workspace directory
    - ls [path]: List contents of workspace directory
    - cd [path]: Change directory within workspace
    - mkdir [dir]: Create new directory in workspace
    - rm [path]: Remove file or directory in workspace
    - open [file]: Open file in system default application

    Attributes:
        COMMANDS (List[str]): List of available CLI commands
        AGENT_OPTIONS (List[str]): Options for agent mode command
        DEBUG_OPTIONS (List[str]): Options for debug mode command
    """

    COMMANDS = [
        "help",
        "quit",
        "exit",
        "provider",
        "model",
        "models",
        "clear",
        "agent",
        "debug",
        "usage",
        "pwd",
        "ls",
        "cd",
        "mkdir",
        "rm",
        "open",
    ]
    AGENT_OPTIONS = ["on", "off"]
    DEBUG_OPTIONS = ["on", "off"]

    def __init__(self):
        """Initialize the ChatCLI with default settings.

        This constructor sets up the basic state of the CLI, including:
        - Processing context for tool execution
        - Empty message history
        - Default agent mode (off)
        - Default debug mode (off)
        - Default provider (OpenAI)
        - Default models for each provider
        - CoT agent (initially None, initialized later)

        It does not perform any async initialization, which is handled separately
        in the initialize() method.
        """
        # Suppress warnings
        import warnings

        warnings.filterwarnings("ignore", category=UserWarning)

        # Initialize state
        self.context = ProcessingContext(user_id="test", auth_token="test")
        self.messages: list[Message] = []
        self.agent_mode = False
        self.debug_mode = False
        self.cot_agent = None

        # Set up default models and provider
        self.default_models = {
            Provider.OpenAI: "gpt-4o",
            Provider.Anthropic: "claude-3-5-sonnet-20241022",
            Provider.Ollama: "llama3.2:3b",
        }
        self.provider = Provider.OpenAI
        self.model = FunctionModel(
            name=self.default_models[self.provider], provider=self.provider
        )

    async def initialize(self):
        """Initialize async components and workspace.

        This method handles all initialization tasks that require async operations:
        1. Fetches available models from providers (Ollama, OpenAI)
        2. Sets up the workspace directory structure
        3. Initializes the CoT agent
        4. Sets up readline with history and tab completion

        The workspace is created as a unique directory in the system's temp directory.

        Returns:
            None

        Raises:
            Various exceptions may be raised during model fetching or filesystem operations.
            These exceptions are not caught here and should be handled by the caller.
        """
        # Get available models
        self.ollama_models = await get_ollama_models()
        self.openai_models = await get_openai_models()
        self.providers = [p.value.lower() for p in Provider]

        # Set up workspace
        self.workspace_root = Path(tempfile.gettempdir()) / "nodetool-workspaces"
        self.workspace_root.mkdir(exist_ok=True)
        self.workspace_name = f"workspace-{int(asyncio.get_event_loop().time())}"
        self.workspace_dir = self.workspace_root / self.workspace_name
        self.workspace_dir.mkdir(exist_ok=True)
        self.current_dir = self.workspace_dir

        # Initialize CoT agent
        self.initialize_cot_agent()

        # Set up readline
        self.setup_readline()

    def setup_readline(self):
        """Set up readline with history and tab completion.

        Configures the readline library for command history and tab completion:
        1. Sets the history file path (~/.nodetool_history)
        2. Attempts to read history from this file
        3. Sets the completer function for tab completion
        4. Binds the tab key to the completion function

        The history file persists command history between sessions.

        Returns:
            None

        Raises:
            FileNotFoundError: If the history file doesn't exist (handled internally)
        """
        self.histfile = os.path.join(os.path.expanduser("~"), ".nodetool_history")
        try:
            readline.read_history_file(self.histfile)
        except FileNotFoundError:
            pass

        readline.set_completer(self.completer)
        readline.parse_and_bind("tab: complete")

    def completer(self, text: str, state: int) -> Optional[str]:
        """Provide tab completion for commands and options.

        This method is called by the readline library to provide tab completion
        suggestions for the current input text. It handles completion for:
        1. Base commands (like /help, /model, etc.)
        2. Model names when typing '/model ...'
        3. Provider names when typing '/provider ...'
        4. Agent mode options when typing '/agent ...'
        5. Debug mode options when typing '/debug ...'

        Args:
            text (str): The text to complete (usually the current word being typed)
            state (int): The state of the completion (0 for first match, 1 for second, etc.)

        Returns:
            Optional[str]: A completion suggestion, or None if no suggestions are available
                           or the state exceeds the number of available completions
        """
        if text.startswith("model"):
            model_text = text.split()[-1]
            options = [
                f"model {m}"
                for m in self.ollama_models
                if m.name.startswith(model_text)
            ]
            options.extend(
                [
                    f"model {m}"
                    for m in self.openai_models
                    if m.id.startswith(model_text)
                ]
            )
            return options[state] if state < len(options) else None
        elif text.startswith("provider"):
            provider_text = text.split()[-1]
            options = [
                f"provider {p}" for p in self.providers if p.startswith(provider_text)
            ]
            return options[state] if state < len(options) else None
        elif text.startswith("agent"):
            agent_text = text.split()[-1]
            options = [
                f"agent {opt}"
                for opt in self.AGENT_OPTIONS
                if opt.startswith(agent_text)
            ]
            return options[state] if state < len(options) else None
        elif text.startswith("debug"):
            debug_text = text.split()[-1]
            options = [
                f"debug {opt}"
                for opt in self.DEBUG_OPTIONS
                if opt.startswith(debug_text)
            ]
            return options[state] if state < len(options) else None
        else:
            options = [f"{cmd}" for cmd in self.COMMANDS if cmd.startswith(text)]
            return options[state] if state < len(options) else None

    def initialize_cot_agent(self):
        """Initialize or reinitialize the CoT agent.

        Creates a new Chain of Thought agent instance with:
        1. The current provider (OpenAI, Anthropic, or Ollama)
        2. The current model
        3. The workspace directory for file system operations

        This method is called during initial setup and whenever
        the provider or model changes.

        Returns:
            None
        """
        provider_instance = get_provider(self.provider)
        self.cot_agent = CoTAgent(
            provider=provider_instance,
            model=self.model,
            workspace_dir=str(self.workspace_dir),
        )

    def find_tool_by_name(self, name: str) -> Optional[Tool]:
        """Find a tool by its name.

        Searches the CoT agent's available tools for a tool with the given name.

        Args:
            name (str): The name of the tool to find

        Returns:
            Optional[Tool]: The Tool object if found, None otherwise

        Note:
            Returns None if the CoT agent is not initialized.
        """
        if self.cot_agent:
            for tool in self.cot_agent.tools:
                if tool.name == name:
                    return tool
        return None

    async def process_cot_response(self, problem: str):
        """Process a problem with the CoT agent and display the step-by-step reasoning.

        This method:
        1. Sends the user's input to the CoT agent
        2. Displays the agent's step-by-step reasoning as it's generated
        3. Executes any tool calls the agent makes
        4. Optionally displays tool results (if debug mode is on)

        Args:
            problem (str): The user's input/question to be processed

        Returns:
            None

        Raises:
            Exception: If there's an error during CoT reasoning (handled internally)

        Note:
            This is only used when agent_mode is True.
        """
        if not self.cot_agent:
            print("Error: CoT agent not initialized")
            return

        try:
            async for item in self.cot_agent.solve_problem(problem, show_thinking=True):
                if isinstance(item, Chunk):
                    print(item.content, end="", flush=True)
                elif isinstance(item, ToolCall):
                    args = json.dumps(item.args)
                    if len(args) > 60:
                        args = args[:60] + "..."
                    print(f"\n[{item.name}]: {args}")
            print("\n")
        except Exception as e:
            print(f"\nError during CoT reasoning: {e}")
            print(traceback.format_exc())

    def handle_workspace_command(self, cmd: str, args: List[str]) -> None:
        """Handle workspace-related commands (pwd, ls, cd, mkdir, rm, open).

        Manages file system operations within the sandbox workspace:
        - pwd: Print current directory
        - ls [path]: List contents of a directory
        - cd [path]: Change current directory
        - mkdir [dir]: Create new directory
        - rm [path]: Remove file or directory
        - open [file]: Open file/directory in system default application

        Security measures prevent accessing paths outside the workspace.

        Args:
            cmd (str): The workspace command to execute
            args (List[str]): Arguments for the command

        Returns:
            None

        Raises:
            Various exceptions may be raised during filesystem operations, but
            they are caught and handled internally, displaying error messages.
        """
        try:
            if cmd == "pwd":
                print(self.current_dir)
            elif cmd == "ls":
                target = self.current_dir
                if args:
                    path = (self.current_dir / args[0]).resolve()
                    if not str(path).startswith(str(self.workspace_dir)):
                        print("Error: Cannot access paths outside workspace")
                        return
                    target = path
                try:
                    for item in target.iterdir():
                        print(f"{'d' if item.is_dir() else 'f'} {item.name}")
                except Exception as e:
                    print(f"Error: {e}")
            elif cmd == "cd":
                if not args:
                    self.current_dir = self.workspace_dir
                else:
                    new_dir = (self.current_dir / args[0]).resolve()
                    if not str(new_dir).startswith(str(self.workspace_dir)):
                        print("Error: Cannot access paths outside workspace")
                        return
                    if not new_dir.is_dir():
                        print(f"Error: {args[0]} is not a directory")
                        return
                    self.current_dir = new_dir
            elif cmd == "mkdir":
                if not args:
                    print("Error: Directory name required")
                    return
                new_dir = (self.current_dir / args[0]).resolve()
                if not str(new_dir).startswith(str(self.workspace_dir)):
                    print("Error: Cannot create directory outside workspace")
                    return
                new_dir.mkdir(parents=True, exist_ok=True)
            elif cmd == "rm":
                if not args:
                    print("Error: Path required")
                    return
                target = (self.current_dir / args[0]).resolve()
                if not str(target).startswith(str(self.workspace_dir)):
                    print("Error: Cannot remove paths outside workspace")
                    return
                if target.is_dir():
                    import shutil

                    shutil.rmtree(target)
                else:
                    target.unlink()
            elif cmd == "open":
                target = str(self.current_dir)
                if not str(target).startswith(str(self.workspace_dir)):
                    print("Error: Cannot open files outside workspace")
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
        except Exception as e:
            print(f"Error: {e}")

    def handle_command(self, cmd: str, args: List[str]) -> bool:
        """Handle CLI commands (starting with /).

        Processes commands for controlling the chat interface:
        - /quit, /exit: Exit the chat interface
        - /help: Display available commands
        - /models: List available models for the current provider
        - /provider [provider]: Set the AI provider
        - /model [model]: Set the model
        - /clear: Clear chat history
        - /agent [on|off]: Toggle CoT agent mode
        - /debug [on|off]: Toggle debug mode
        - /usage: Display provider's usage statistics

        Args:
            cmd (str): The command to execute (without the leading /)
            args (List[str]): Arguments for the command

        Returns:
            bool: True if the CLI should exit, False otherwise

        Raises:
            Exceptions may be raised during certain operations (e.g., listing models),
            but they are caught and handled internally.
        """
        if cmd == "quit" or cmd == "exit":
            return True
        elif cmd == "help":
            print("Commands:")
            print("  /provider [openai|anthropic|ollama] - Set the provider")
            print("  /model [model_name] - Set the model")
            print("  /models - List available models for the current provider")
            print("  /agent [on|off] - Toggle Chain of Thought agent mode")
            print(
                "  /debug [on|off] - Toggle debug mode to display tool calls and results"
            )
            print("  /usage - Display current provider's usage statistics")
            print("  /tools - List available tools")
            print("  /tools [tool_name] - Show details about a specific tool")
            print("  /clear - Clear chat history")
            print("  /quit or /exit - Exit the chat")
        elif cmd == "models":
            try:
                if self.provider == Provider.Ollama:
                    print("\nAvailable Ollama models:")
                    for model_info in self.ollama_models:
                        print(f"- {model_info.name}")
                elif self.provider == Provider.OpenAI:
                    print("\nAvailable OpenAI models (partial list):")
                    for model_info in self.openai_models[:10]:
                        print(f"- {model_info.id}")
                elif self.provider == Provider.Anthropic:
                    print("\nAvailable Anthropic models:")
                    print("- claude-3-opus-20240229")
                    print("- claude-3-sonnet-20240229")
                    print("- claude-3-haiku-20240307")
                    print("- claude-3-5-sonnet-2024102")
                    print("- claude-3-7-sonnet-20250219")
                print()
            except Exception as e:
                print(f"Error listing models: {e}")
        elif cmd == "provider":
            if not args:
                print(f"Current provider: {self.provider.value}")
                print(f"Available providers: {[p.value for p in Provider]}")
                return False

            provider_name = args[0].capitalize()
            try:
                self.provider = Provider[provider_name]
                self.model = FunctionModel(
                    name=self.default_models[self.provider], provider=self.provider
                )
                self.initialize_cot_agent()
                print(
                    f"Provider set to {self.provider.value} with default model {self.default_models[self.provider]}"
                )
            except KeyError:
                print(f"Invalid provider. Choose from: {[p.value for p in Provider]}")
        elif cmd == "model":
            if not args:
                print(f"Current model: {self.model.name}")
                return False

            model_name = args[0]
            self.model = FunctionModel(name=model_name, provider=self.provider)
            self.initialize_cot_agent()
            print(f"Model set to {model_name}")
        elif cmd == "clear":
            self.messages = []
            if self.cot_agent:
                self.cot_agent.clear_history()
            print("Chat history cleared")
        elif cmd == "agent":
            if not args:
                print(f"Agent mode is currently: {'ON' if self.agent_mode else 'OFF'}")
                return False

            if args[0].lower() == "on":
                self.agent_mode = True
                print("Agent mode turned ON - Using Chain of Thought reasoning")
            elif args[0].lower() == "off":
                self.agent_mode = False
                print("Agent mode turned OFF - Using standard chat")
            else:
                print("Usage: /agent [on|off]")
        elif cmd == "debug":
            if not args:
                print(f"Debug mode is currently: {'ON' if self.debug_mode else 'OFF'}")
                return False

            if args[0].lower() == "on":
                self.debug_mode = True
                print("Debug mode turned ON - Will display tool calls and results")
            elif args[0].lower() == "off":
                self.debug_mode = False
                print("Debug mode turned OFF - Tool calls and results hidden")
            else:
                print("Usage: /debug [on|off]")
        elif cmd == "usage":
            if self.cot_agent and self.cot_agent.provider:
                print(f"\nProvider usage statistics:")
                print(json.dumps(self.cot_agent.provider.usage, indent=2))
            else:
                print("No usage statistics available")
        else:
            print("Unknown command. Type /help for available commands")
        return False

    async def run(self):
        """Run the chat CLI main loop.

        This is the main entry point that:
        1. Initializes the CLI components (async)
        2. Displays welcome information
        3. Enters the main input loop
        4. Processes user input, commands, and generates responses
        5. Handles exceptions gracefully

        The main loop processes:
        - Empty input (skipped)
        - Workspace commands (cd, ls, etc.)
        - CLI commands (starting with /)
        - Chat messages (processed by the agent or regular chat)

        Returns:
            None

        Raises:
            KeyboardInterrupt: If Ctrl+C is pressed (handled internally)
            EOFError: If Ctrl+D is pressed (handled internally)
            Exception: Other exceptions are caught and displayed
        """
        await self.initialize()

        print("Chat CLI - Type /help for commands")
        print(f"Using {self.provider.value} with model {self.model.name}")
        print("Agent mode is OFF (use /agent on to enable Chain of Thought reasoning)")
        print("Debug mode is OFF (use /debug on to display tool calls and results)")
        print(f"\nWorkspace created at: {self.workspace_dir}")
        print("Use workspace commands: pwd, ls, cd, mkdir, rm, open")

        while True:
            try:
                # Show current directory in prompt
                rel_path = os.path.relpath(self.current_dir, self.workspace_dir)
                prompt = f"[{rel_path}]> " if rel_path != "." else "> "
                user_input = input(prompt).strip()
                readline.write_history_file(self.histfile)

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
                    cmd_parts = user_input[1:].lower().split()
                    cmd = cmd_parts[0]
                    args = cmd_parts[1:] if len(cmd_parts) > 1 else []
                    if self.handle_command(cmd, args):
                        break
                    continue

                # Process chat input
                if self.agent_mode:
                    await self.process_cot_response(user_input)
                else:
                    self.messages = await process_regular_chat(
                        user_input=user_input,
                        messages=self.messages,
                        model=self.model,
                        context=self.context,
                        debug_mode=self.debug_mode,
                    )

            except KeyboardInterrupt:
                return
            except EOFError:
                return
            except Exception as e:
                print(f"Error: {e}")
                print(f"Stacktrace: {traceback.format_exc()}")


async def chat_cli():
    """Entry point for the chat CLI.

    Creates an instance of the ChatCLI class and runs it.
    This function is the main entry point for the CLI when
    imported from other modules.

    Returns:
        None
    """
    cli = ChatCLI()
    await cli.run()


if __name__ == "__main__":
    asyncio.run(chat_cli())
