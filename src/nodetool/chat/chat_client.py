"""
NodeTool Chat Client

A command-line client for connecting to NodeTool chat servers using OpenAI Chat Completions API.
Provides an interactive chat interface that connects to a running chat server with full OpenAI compatibility.
"""

import asyncio
import sys
import os
from typing import Optional, List, Dict, Union, Any
from pathlib import Path
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionAssistantMessageParam, ChatCompletionMessageParam, ChatCompletionUserMessageParam
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.markdown import Markdown

# Prompt toolkit imports for advanced input handling
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter, NestedCompleter, Completer
from prompt_toolkit.styles import Style


console = Console()


class OpenAIChatClient:
    """
    OpenAI-compatible chat client for NodeTool chat servers.

    Provides an interactive command-line interface for chatting with AI providers
    through a NodeTool chat server using OpenAI Chat Completions API.
    """

    def __init__(
        self,
        server_url: str,
        auth_token: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        """
        Initialize the chat client.

        Args:
            server_url: Base URL of the chat server (e.g., 'http://localhost:8080')
            auth_token: Optional authentication token
            model: Optional initial model to use (e.g., 'gpt-oss:20b')
            provider: Optional provider to use (e.g., 'openai', 'anthropic', 'ollama')
        """
        self.server_url = server_url.rstrip("/")
        self.auth_token = auth_token
        self.history: List[ChatCompletionMessageParam] = []
        self.current_model = model or "gpt-oss:20b"  # Default model

        # Determine provider based on server URL if not specified
        if provider:
            self.current_provider = provider
        elif "api.openai.com" in server_url:
            self.current_provider = "openai"
        else:
            self.current_provider = "ollama"  # Default for local servers

        # Initialize OpenAI client with custom base URL
        self.client = AsyncOpenAI(
            api_key=auth_token or "nodetool-local",  # Dummy key for local server
            base_url=f"{self.server_url}/v1",
        )
        
        # Setup history file for readline functionality
        self.history_file = os.path.join(os.path.expanduser("~"), ".nodetool_chat_history")
        
        # Initialize prompt session (will be set up in setup_prompt_session)
        self.session: Optional[PromptSession] = None

    async def setup_prompt_session(self) -> None:
        """Set up prompt_toolkit session with completers and styling."""
        # Define commands available for completion
        commands = [
            "clear", "help", "history", "model", "quit", "exit"
        ]
        
        # Create nested completer for commands
        command_completer: Dict[str, Optional[Completer]] = {
            f"/{cmd}": None for cmd in commands
        }
        
        # Create the main completer
        completer = NestedCompleter(command_completer)
        
        # Create prompt style
        style = Style.from_dict({
            "prompt": "ansiblue bold",
            "completion-menu.completion": "bg:#008888 #ffffff",
            "completion-menu.completion.current": "bg:#00aaaa #000000",
            "scrollbar.background": "bg:#88aaaa",
            "scrollbar.button": "bg:#222222",
        })
        
        # Create session with history and auto-suggest
        self.session = PromptSession(
            history=FileHistory(self.history_file),
            auto_suggest=AutoSuggestFromHistory(),
            completer=completer,
            style=style,
            complete_in_thread=True,
            complete_while_typing=True,
        )

    async def send_message(self, message: str) -> None:
        """
        Send a message to the chat server and handle the streaming response.

        Args:
            message: The message content to send
        """
        # Add user message to history
        user_message = ChatCompletionUserMessageParam(role="user", content=message)
        self.history.append(user_message)

        try:
            # Create OpenAI-compatible request using the SDK
            stream = await self.client.chat.completions.create(
                model=self.current_model,
                messages=self.history,
                stream=True,
            )

            assistant_content = ""

            # Create a live display for streaming response
            with Live(console=console, refresh_per_second=10) as live:
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        assistant_content += content

                        # Update live display with current response
                        response_panel = Panel(
                            Markdown(assistant_content),
                            title="[bold green]Assistant",
                            border_style="green",
                        )
                        live.update(response_panel)

                    # Check if stream is finished
                    if chunk.choices and chunk.choices[0].finish_reason:
                        break

            # Add assistant response to history
            if assistant_content:
                assistant_message = ChatCompletionAssistantMessageParam(
                    role="assistant", content=assistant_content
                )
                self.history.append(assistant_message)

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")

    async def test_connection(self) -> bool:
        """
        Test connection to the chat server.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            models = await self.client.models.list()
            console.print(
                f"[bold green]✅ Connected to OpenAI API[/bold green]"
            )
            console.print(f"Available models: {len(models.data)} found")
            return True

        except Exception as e:
            console.print(f"[bold red]❌ Connection failed:[/bold red] {e}")
            return False

    def show_history(self) -> None:
        """Display conversation history."""
        if not self.history:
            console.print("[yellow]📭 No conversation history[/yellow]")
            return
            
        console.print("[bold cyan]📚 Conversation History[/bold cyan]")
        console.print()
        
        for i, message in enumerate(self.history, 1):
            role = message.get("role", "unknown").title()
            content = message.get("content", "")
            
            # Handle different content types (string or array)
            if isinstance(content, str):
                display_content = content
            elif content is None:
                display_content = "[No content]"
            else:
                # For complex content types, convert to string
                display_content = str(content)
            
            if role == "User":
                console.print(f"[bold blue]{i}. You:[/bold blue]")
                console.print(f"   {display_content}")
            else:
                console.print(f"[bold green]{i}. Assistant:[/bold green]")
                # Truncate long responses for history display
                if len(display_content) > 200:
                    display_content = display_content[:200] + "..."
                console.print(f"   {display_content}")
            console.print()

    async def interactive_chat(self) -> None:
        """
        Start an interactive chat session with advanced readline features.
        """
        # Test connection first
        if not await self.test_connection():
            console.print(
                "[bold red]Failed to connect to server. Please check the server URL and try again.[/bold red]"
            )
            return

        # Setup prompt session for advanced input
        await self.setup_prompt_session()

        console.print("\n[bold cyan]🤖 NodeTool Chat Client[/bold cyan]")
        console.print(
            "Type your messages below. Use 'quit', 'exit', or Ctrl+C to exit."
        )
        console.print("Use '/clear' to clear conversation history.")
        console.print("Use '/history' to view conversation history.")
        console.print("Use '/model' to change the AI model.")
        console.print("Use '/help' for more information.")
        console.print(
            f"\nCurrent model: [bold]{self.current_model}[/bold] (provider: {self.current_provider})"
        )
        console.print("[dim]💡 Tab completion and command history enabled[/dim]\n")

        try:
            while True:
                try:
                    # Get user input with advanced prompt features
                    if self.session is None:
                        console.print("[bold red]Error: Prompt session not initialized[/bold red]")
                        break
                        
                    user_input = await self.session.prompt_async(
                        "[You]> ",
                        multiline=False,
                    )
                    user_input = user_input.strip()

                    if not user_input:
                        continue

                    # Handle special commands
                    if user_input.lower() in ["quit", "exit"]:
                        break
                    elif user_input == "/clear":
                        self.history.clear()
                        console.print(
                            "[yellow]💫 Conversation history cleared[/yellow]"
                        )
                        continue
                    elif user_input == "/history":
                        self.show_history()
                        continue
                    elif user_input == "/help":
                        self.show_help()
                        continue
                    elif user_input.startswith("/model"):
                        self.change_model(user_input)
                        continue

                    # Send message to server
                    await self.send_message(user_input)
                    console.print()  # Add spacing between exchanges

                except KeyboardInterrupt:
                    console.print("\n[yellow]Chat interrupted by user[/yellow]")
                    break
                except EOFError:
                    console.print("\n[yellow]Chat session ended[/yellow]")
                    break

        finally:
            await self.client.close()
            console.print("[bold cyan]👋 Chat session ended[/bold cyan]")

    def show_help(self) -> None:
        """Display help information."""
        help_text = """[bold cyan]💡 Chat Client Help[/bold cyan]

[bold]Commands:[/bold]
• [cyan]quit[/cyan] or [cyan]exit[/cyan] - Exit the chat
• [cyan]/clear[/cyan] - Clear conversation history  
• [cyan]/history[/cyan] - Show conversation history
• [cyan]/model[/cyan] - Change the AI model
• [cyan]/help[/cyan] - Show this help message

[bold]Model Examples:[/bold]
• [cyan]/model gemma2:27b ollama[/cyan] - Use Gemma 2 27B via Ollama
• [cyan]/model claude-3-opus-20240229 anthropic[/cyan] - Use Claude 3 Opus
• [cyan]/model gpt-4o openai[/cyan] - Use GPT-4o

[bold]Readline Features:[/bold]
• [cyan]Tab completion[/cyan] - Press Tab for command and model completion
• [cyan]Command history[/cyan] - Use ↑/↓ arrows to navigate previous commands
• [cyan]Auto-suggestions[/cyan] - See suggested completions based on history
• [cyan]Multi-line editing[/cyan] - Advanced line editing with Ctrl+A/E, etc.

[bold]Tips:[/bold]
• Messages support Markdown formatting
• Use Ctrl+C to interrupt at any time
• Your conversation history is maintained during the session
• Command history is saved across sessions in ~/.nodetool_chat_history
"""
        console.print(Panel(help_text, title="Help", border_style="cyan"))

    def change_model(self, command: str) -> None:
        """
        Change the current AI model and provider.

        Args:
            command: The model command (e.g., '/model gpt-oss:20b ollama')
        """
        parts = command.split()

        if len(parts) == 1:
            # Just '/model' - show current model
            console.print(f"[bold]Current model:[/bold] {self.current_model}")
            console.print(f"[bold]Current provider:[/bold] {self.current_provider}")
            console.print("[dim]Usage: /model <model_name> <provider>[/dim]")
            return

        if len(parts) == 2:
            # Model name only - guess provider
            model_name = parts[1]

            # Guess provider based on model name
            if "gpt" in model_name.lower():
                provider = "openai"
            elif "claude" in model_name.lower():
                provider = "anthropic"
            elif "gemini" in model_name.lower():
                provider = "google"
            elif ":" in model_name:  # Ollama models often have :tag format
                provider = "ollama"
            else:
                console.print(
                    "[yellow]⚠️ Could not determine provider. Please specify provider explicitly.[/yellow]"
                )
                console.print("[dim]Usage: /model <model_name> <provider>[/dim]")
                return

        elif len(parts) >= 3:
            # Model name and provider specified
            model_name = parts[1]
            provider = parts[2]

        else:
            console.print("[red]Invalid command format[/red]")
            console.print("[dim]Usage: /model <model_name> <provider>[/dim]")
            return

        # Update model and provider
        self.current_model = model_name
        self.current_provider = provider

        console.print(
            f"[green]✅ Switched to model:[/green] [bold]{self.current_model}[/bold]"
        )
        console.print(f"[green]Provider:[/green] [bold]{self.current_provider}[/bold]")


async def run_chat_client(
    server_url: str,
    auth_token: Optional[str] = None,
    message: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> None:
    """
    Run the chat client.

    Args:
        server_url: URL of the chat server
        auth_token: Optional authentication token
        message: Optional single message to send (non-interactive mode)
        model: Optional initial model to use
        provider: Optional provider to use (for local servers)
    """
    client = OpenAIChatClient(server_url, auth_token, model, provider)

    if message:
        # Non-interactive mode: send single message
        console.print(f"[bold cyan]🤖 Sending message to {server_url}[/bold cyan]")
        if await client.test_connection():
            await client.send_message(message)
        await client.client.close()
    else:
        # Interactive mode
        await client.interactive_chat()


if __name__ == "__main__":
    # Simple test run
    import sys

    if len(sys.argv) > 1:
        server_url = sys.argv[1]
        token = sys.argv[2] if len(sys.argv) > 2 else None
        asyncio.run(run_chat_client(server_url, token))
    else:
        asyncio.run(run_chat_client("http://localhost:8080"))
