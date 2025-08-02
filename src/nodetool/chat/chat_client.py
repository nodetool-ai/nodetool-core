"""
NodeTool Chat Client

A command-line client for connecting to NodeTool chat servers using Server-Sent Events (SSE).
Provides an interactive chat interface that connects to a running chat server.
"""

import asyncio
import json
import sys
from typing import Optional, List, Dict, Any
import httpx
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.prompt import Prompt
from rich.markdown import Markdown
from nodetool.metadata.types import Message


console = Console()


class SSEChatClient:
    """
    SSE-based chat client for NodeTool chat servers.
    
    Provides an interactive command-line interface for chatting with AI providers
    through a NodeTool chat server using Server-Sent Events.
    """
    
    def __init__(self, server_url: str, auth_token: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the chat client.
        
        Args:
            server_url: Base URL of the chat server (e.g., 'http://localhost:8080')
            auth_token: Optional authentication token
            model: Optional initial model to use (e.g., 'gemma3n:latest')
        """
        self.server_url = server_url.rstrip('/')
        self.auth_token = auth_token
        self.history: List[Message] = []
        self.client = httpx.AsyncClient(timeout=30.0)
        self.current_model = model or "gemma3n:latest"  # Default model
        self.current_provider = "ollama"  # Default provider
    
    async def send_message(self, message: str) -> None:
        """
        Send a message to the chat server and handle the streaming response.
        
        Args:
            message: The message content to send
        """
        # Add user message to history
        user_message = Message(role="user", content=message)
        self.history.append(user_message)
        
        # Prepare request data
        request_data = {
            "role": "user",
            "content": message,
            "model": self.current_model,
            "provider": self.current_provider,
            "history": [msg.model_dump() for msg in self.history[:-1]]  # Exclude current message
        }
        
        if self.auth_token:
            request_data["auth_token"] = self.auth_token
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache"
        }
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        try:
            console.print(f"[bold blue]You:[/bold blue] {message}")
            
            # Stream the response
            async with self.client.stream(
                "POST",
                f"{self.server_url}/chat/sse",
                json=request_data,
                headers=headers
            ) as response:
                
                if response.status_code != 200:
                    error_text = await response.aread()
                    console.print(f"[bold red]Error {response.status_code}:[/bold red] {error_text.decode()}")
                    return
                
                assistant_content = ""
                
                # Create a live display for streaming response
                with Live(console=console, refresh_per_second=10) as live:
                    async for data in self._parse_sse_stream(response):
                        if data.get("type") == "chunk":
                            content = data.get("content", "")
                            done = data.get("done", False)
                            assistant_content += content
                            # Update live display with current response
                            response_panel = Panel(
                                Markdown(assistant_content),
                                title="[bold green]Assistant",
                                border_style="green"
                            )
                            live.update(response_panel)
                            if done:
                                break
                        
                        elif data.get("type") == "error":
                            console.print(f"[bold red]Server Error:[/bold red] {data.get('message', 'Unknown error')}")
                            return
                        
                        elif data.get("type") == "close":
                            break
                
                # Add assistant response to history
                if assistant_content:
                    assistant_message = Message(role="assistant", content=assistant_content)
                    self.history.append(assistant_message)
                
        except httpx.RequestError as e:
            console.print(f"[bold red]Connection Error:[/bold red] {e}")
        except Exception as e:
            console.print(f"[bold red]Unexpected Error:[/bold red] {e}")
    
    async def _parse_sse_stream(self, response):
        """
        Parse Server-Sent Events stream according to SSE specification.
        
        Args:
            response: The HTTP response stream
            
        Yields:
            dict: Parsed SSE event with 'event' and 'data' fields
        """
        async for line in response.aiter_lines():
            line = line.strip()
            yield json.loads(line)
    
    async def test_connection(self) -> bool:
        """
        Test connection to the chat server.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            response = await self.client.get(f"{self.server_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                console.print(f"[bold green]✅ Connected to chat server[/bold green]")
                console.print(f"Status: {health_data.get('status', 'unknown')}")
                console.print(f"Protocol: {health_data.get('protocol', 'unknown')}")
                return True
            else:
                console.print(f"[bold red]❌ Server returned status {response.status_code}[/bold red]")
                return False
        except httpx.RequestError as e:
            console.print(f"[bold red]❌ Connection failed:[/bold red] {e}")
            return False
    
    async def interactive_chat(self) -> None:
        """
        Start an interactive chat session.
        """
        # Test connection first
        if not await self.test_connection():
            console.print("[bold red]Failed to connect to server. Please check the server URL and try again.[/bold red]")
            return
        
        console.print("\n[bold cyan]🤖 NodeTool Chat Client[/bold cyan]")
        console.print("Type your messages below. Use 'quit', 'exit', or Ctrl+C to exit.")
        console.print("Use '/clear' to clear conversation history.")
        console.print("Use '/history' to view conversation history.")
        console.print("Use '/model' to change the AI model.")
        console.print(f"\nCurrent model: [bold]{self.current_model}[/bold] (provider: {self.current_provider})\n")
        
        try:
            while True:
                try:
                    # Get user input
                    user_input = Prompt.ask("[bold blue]You", console=console).strip()
                    
                    if not user_input:
                        continue
                    
                    # Handle special commands
                    if user_input.lower() in ['quit', 'exit']:
                        break
                    elif user_input == '/clear':
                        self.history.clear()
                        console.print("[yellow]💫 Conversation history cleared[/yellow]")
                        continue
                    elif user_input == '/history':
                        self.show_history()
                        continue
                    elif user_input == '/help':
                        self.show_help()
                        continue
                    elif user_input.startswith('/model'):
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
            await self.client.aclose()
            console.print("[bold cyan]👋 Chat session ended[/bold cyan]")
    
    def show_history(self) -> None:
        """Display the conversation history."""
        if not self.history:
            console.print("[yellow]No conversation history[/yellow]")
            return
        
        console.print("[bold cyan]📜 Conversation History[/bold cyan]")
        for i, msg in enumerate(self.history, 1):
            role_color = "blue" if msg.role == "user" else "green"
            role_name = "You" if msg.role == "user" else "Assistant"
            console.print(f"[bold {role_color}]{i}. {role_name}:[/bold {role_color}] {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}")
    
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
• [cyan]/model gemma3n:latest ollama[/cyan] - Use Gemma 3 via Ollama
• [cyan]/model claude-3-opus-20240229 anthropic[/cyan] - Use Claude 3 Opus
• [cyan]/model gpt-4 openai[/cyan] - Use GPT-4

[bold]Tips:[/bold]
• Messages support Markdown formatting
• Use Ctrl+C to interrupt at any time
• Your conversation history is maintained during the session
"""
        console.print(Panel(help_text, title="Help", border_style="cyan"))
    
    def change_model(self, command: str) -> None:
        """
        Change the current AI model and provider.
        
        Args:
            command: The model command (e.g., '/model gemma3n:latest ollama')
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
                console.print("[yellow]⚠️ Could not determine provider. Please specify provider explicitly.[/yellow]")
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
        
        console.print(f"[green]✅ Switched to model:[/green] [bold]{self.current_model}[/bold]")
        console.print(f"[green]Provider:[/green] [bold]{self.current_provider}[/bold]")


async def run_chat_client(
    server_url: str,
    auth_token: Optional[str] = None,
    message: Optional[str] = None,
    model: Optional[str] = None
) -> None:
    """
    Run the chat client.
    
    Args:
        server_url: URL of the chat server
        auth_token: Optional authentication token
        message: Optional single message to send (non-interactive mode)
        model: Optional initial model to use
    """
    client = SSEChatClient(server_url, auth_token, model)
    
    if message:
        # Non-interactive mode: send single message
        console.print(f"[bold cyan]🤖 Sending message to {server_url}[/bold cyan]")
        if await client.test_connection():
            await client.send_message(message)
        await client.client.aclose()
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