"""Base command class for CLI commands."""

from typing import List, Optional

from nodetool.chat.chat_cli import ChatCLI


class Command:
    """Base class for CLI commands with documentation and execution logic."""

    def __init__(self, name: str, description: str, aliases: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.aliases = aliases or []

    async def execute(self, cli: ChatCLI, args: List[str]) -> bool:
        """Execute the command with the given arguments.

        Returns:
            bool: True if the CLI should exit, False otherwise
        """
        raise NotImplementedError("Command must implement execute method")
