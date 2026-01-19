#!/usr/bin/env python3
"""
Minimal WebSocket Chat Client for testing UnifiedWebSocketRunner

### Features

- Split-screen layout with messages above and input below
- Real-time message streaming with visual feedback
- Rich formatting for different message types (user, assistant, system, tool)
- Status bar showing connection state and current model
- Support for both text (JSON) and binary (MessagePack) message formats
- Authentication support via JWT tokens

### Usage

```bash
# Basic usage (connects to localhost:7777)
python websocket_test_client.py

# With custom URL
python websocket_test_client.py --url ws://localhost:8080/chat

# With authentication
python websocket_test_client.py --url ws://localhost:7777/chat --token your_jwt_token

# With binary message format
python websocket_test_client.py --binary
```

### Commands

- `/help` - Show available commands
- `/model <name>` - Change the AI model (default: gpt-4)
- `/tools <t1,t2>` - Set tools (comma-separated)
- `/workflow <id>` - Set workflow ID
- `/clear` - Clear message history
- `/quit` - Exit the client
on websocket_test_client.py --url ws://localhost:8000/chat --token your_jwt_token
"""

import argparse
import asyncio
import json
import readline
import sys
from contextlib import suppress
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional

import msgpack
import websockets

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection


class MessageFormat(str, Enum):
    TEXT = "text"
    BINARY = "binary"


class ChatWebSocketClient:
    """Minimal WebSocket client for testing chat functionality."""

    def __init__(
        self,
        url: str,
        auth_token: Optional[str] = None,
        message_format: MessageFormat = MessageFormat.TEXT,
    ):
        self.url = url + "?api_key=" + auth_token if auth_token else url
        self.auth_token = auth_token
        self.message_format = message_format
        self.websocket: Optional[ClientConnection] = None
        self.running = False
        self.receive_task: Optional[asyncio.Task] = None
        self.current_assistant_message = ""
        self.prompt_needed = False

        # Settings
        self.current_model = "gpt-4o"
        self.current_tools = None
        self.current_workflow = None

    def print_message(self, timestamp: str, role: str, content: str):
        """Print a message to the terminal."""
        role_colors = {
            "user": "\033[34m",  # Blue
            "assistant": "\033[32m",  # Green
            "system": "\033[33m",  # Yellow
            "tool": "\033[36m",  # Cyan
        }
        reset = "\033[0m"
        dim = "\033[2m"

        color = role_colors.get(role, "")
        print(f"{dim}{timestamp}{reset} {color}{role.capitalize()}:{reset} {content}")

        # Reprint the prompt if we're in the middle of input
        if self.prompt_needed:
            print("> ", end="", flush=True)

    async def connect(self):
        """Establish WebSocket connection with optional authentication."""
        try:
            self.websocket = await websockets.connect(self.url)
            self.print_message(
                datetime.now().strftime("%H:%M:%S"),
                "system",
                f"✓ Connected to {self.url}",
            )
            self.running = True

            # Start receiving messages
            self.receive_task = asyncio.create_task(self.receive_messages())

        except websockets.exceptions.WebSocketException as e:
            self.print_message(
                datetime.now().strftime("%H:%M:%S"),
                "system",
                f"✗ Connection failed: {e}",
            )
            raise
        except Exception as e:
            self.print_message(
                datetime.now().strftime("%H:%M:%S"),
                "system",
                f"✗ Unexpected error: {e}",
            )
            raise

    async def disconnect(self):
        """Close WebSocket connection gracefully."""
        self.running = False

        if self.receive_task:
            self.receive_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.receive_task

        if self.websocket:
            await self.websocket.close()
            self.print_message(datetime.now().strftime("%H:%M:%S"), "system", "Disconnected")

    async def send_message(self, content: str):
        """Send a message to the WebSocket server."""
        if not self.websocket:
            self.print_message(datetime.now().strftime("%H:%M:%S"), "system", "Not connected")
            return

        # Construct message
        message: dict[str, Any] = {
            "role": "user",
            "content": content,
            "model": self.current_model,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        if self.current_tools:
            message["tools"] = self.current_tools

        if self.current_workflow:
            message["workflow_id"] = self.current_workflow

        # Wrap in command structure
        command_msg = {"command": "chat_message", "data": message}

        # Send message in appropriate format
        try:
            if self.message_format == MessageFormat.BINARY:
                packed = msgpack.packb(command_msg, use_bin_type=True)
                assert self.websocket, "WebSocket not connected"
                await self.websocket.send(packed or b"")
            else:
                assert self.websocket, "WebSocket not connected"
                await self.websocket.send(json.dumps(command_msg))

        except Exception as e:
            self.print_message(
                datetime.now().strftime("%H:%M:%S"),
                "system",
                f"Error sending message: {e}",
            )

    async def send_command(self, command: str, data: dict[str, Any]):
        """Send a generic command to the WebSocket server."""
        if not self.websocket:
            self.print_message(datetime.now().strftime("%H:%M:%S"), "system", "Not connected")
            return

        message = {"command": command, "data": data}

        try:
            if self.message_format == MessageFormat.BINARY:
                packed = msgpack.packb(message, use_bin_type=True)
                assert self.websocket, "WebSocket not connected"
                await self.websocket.send(packed or b"")
            else:
                assert self.websocket, "WebSocket not connected"
                await self.websocket.send(json.dumps(message))

        except Exception as e:
            self.print_message(
                datetime.now().strftime("%H:%M:%S"),
                "system",
                f"Error sending command {command}: {e}",
            )

    async def receive_messages(self):
        """Continuously receive and display messages from the server."""
        try:
            assert self.websocket, "WebSocket not connected"
            async for message in self.websocket:
                # Parse message based on format
                data = msgpack.unpackb(message, raw=False) if isinstance(message, bytes) else json.loads(message)

                # Process based on message type
                await self.process_message(data)

        except asyncio.CancelledError:
            pass
        except websockets.exceptions.ConnectionClosed:
            self.print_message(
                datetime.now().strftime("%H:%M:%S"),
                "system",
                "Connection closed by server",
            )
            self.running = False
        except Exception as e:
            self.print_message(
                datetime.now().strftime("%H:%M:%S"),
                "system",
                f"Error receiving message: {e}",
            )
            self.running = False

    async def process_message(self, data: dict[str, Any]):
        """Process received message based on type."""
        msg_type = data.get("type", "unknown")
        timestamp = datetime.now().strftime("%H:%M:%S")

        if msg_type == "chunk":
            # Streaming content chunk
            content = data.get("content", "")
            if content:
                self.current_assistant_message += content
                # Clear the prompt line and print streaming indicator
                if self.prompt_needed:
                    print("\r" + " " * 3 + "\r", end="")  # Clear prompt
                print(f"\r{self.current_assistant_message}▌", end="", flush=True)
            if data.get("done", False):
                # Finalize the message
                if self.current_assistant_message:
                    print(
                        "\r" + " " * (len(self.current_assistant_message) + 1) + "\r",
                        end="",
                    )  # Clear line
                    self.print_message(timestamp, "assistant", self.current_assistant_message)
                    self.current_assistant_message = ""

        elif msg_type == "tool_call":
            # Tool being called
            tool_call = data.get("tool_call", {})
            self.print_message(
                timestamp,
                "tool",
                f"Calling tool: {tool_call.get('name')} (id: {tool_call.get('id')})",
            )

        elif msg_type == "tool_result":
            # Tool execution result
            result = data.get("result", {})
            self.print_message(
                timestamp,
                "tool",
                f"Tool result from {result.get('name')}: {json.dumps(result.get('result', {}), indent=2)}",
            )

        elif msg_type == "job_update":
            # Workflow job update
            status = data.get("status", "unknown")
            self.print_message(timestamp, "system", f"Job status: {status}")

        elif msg_type == "error":
            # Error message
            self.print_message(timestamp, "system", f"Error: {data.get('message', 'Unknown error')}")

        else:
            # Unknown message type
            self.print_message(timestamp, "system", f"Received {msg_type}: {json.dumps(data)}")

    async def handle_command(self, command: str):
        """Handle slash commands."""
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        timestamp = datetime.now().strftime("%H:%M:%S")

        if cmd == "/help":
            print("\nCommands:")
            print("  /help              - Show this help")
            print(f"  /model <name>      - Change model (current: {self.current_model})")
            print("  /tools <t1,t2>     - Set tools (comma-separated)")
            print("  /workflow <id>     - Set workflow ID")
            print("  /resume <job_id>   - Resume a suspended/failed job")
            print("  /clear             - Clear screen")
            print("  /quit              - Exit\n")

        elif cmd == "/model":
            if args:
                self.current_model = args
                self.print_message(timestamp, "system", f"Model set to: {self.current_model}")
            else:
                self.print_message(timestamp, "system", f"Current model: {self.current_model}")

        elif cmd == "/tools":
            if args:
                self.current_tools = [t.strip() for t in args.split(",")]
                self.print_message(timestamp, "system", f"Tools set to: {self.current_tools}")
            else:
                self.print_message(timestamp, "system", f"Current tools: {self.current_tools}")

        elif cmd == "/workflow":
            if args:
                self.current_workflow = args
                self.print_message(timestamp, "system", f"Workflow ID set to: {self.current_workflow}")
            else:
                self.print_message(timestamp, "system", f"Current workflow: {self.current_workflow}")

        elif cmd == "/resume":
            if not args:
                self.print_message(timestamp, "system", "Usage: /resume <job_id>")
            else:
                self.print_message(timestamp, "system", f"Sending resume command for job: {args}")
                await self.send_command("resume_job", {"job_id": args})

        elif cmd == "/clear":
            print("\033[2J\033[H")  # Clear screen and move cursor to top
            self.print_message(timestamp, "system", "Screen cleared")

        elif cmd == "/quit":
            self.running = False

        else:
            self.print_message(timestamp, "system", f"Unknown command: {cmd}")

    async def run_interactive_session(self):
        """Run the interactive chat session."""
        print("\033[2J\033[H")  # Clear screen
        print("=== WebSocket Chat Client ===")
        print(f"Model: {self.current_model}")
        print("Type /help for commands\n")

        # Configure readline
        readline.set_history_length(1000)
        readline.parse_and_bind("tab: complete")

        # Set up input loop
        loop = asyncio.get_event_loop()

        try:
            # Connect to WebSocket
            await self.connect()

            while self.running:
                try:
                    # Set flag to indicate we're waiting for input
                    self.prompt_needed = True

                    # Get user input using readline in a thread
                    user_input = await loop.run_in_executor(None, lambda: input("> "))

                    # Clear the prompt flag
                    self.prompt_needed = False

                    if not user_input:
                        continue

                    # Handle commands
                    if user_input.startswith("/"):
                        await self.handle_command(user_input)
                    else:
                        # Display user message and send it
                        self.print_message(datetime.now().strftime("%H:%M:%S"), "user", user_input)
                        await self.send_message(user_input)

                except KeyboardInterrupt:
                    self.prompt_needed = False
                    print("\nUse /quit to exit")
                    continue
                except EOFError:
                    break

        finally:
            await self.disconnect()


async def main():
    """Main entry point for the WebSocket client."""
    parser = argparse.ArgumentParser(description="WebSocket Chat Client")
    parser.add_argument(
        "--url",
        default="ws://localhost:8000/ws",
        help="WebSocket URL (default: ws://localhost:8000/ws)",
    )
    parser.add_argument("--token", help="Authentication token (JWT)")
    parser.add_argument("--binary", action="store_true", help="Use binary message format (MessagePack)")

    args = parser.parse_args()

    # Determine message format
    message_format = MessageFormat.BINARY if args.binary else MessageFormat.TEXT

    # Create and run client
    client = ChatWebSocketClient(url=args.url, auth_token=args.token, message_format=message_format)

    try:
        await client.run_interactive_session()
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
