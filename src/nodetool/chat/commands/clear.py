"""Clear command implementation."""

from typing import List

from nodetool.chat.chat_cli import ChatCLI

from .base import Command


class ClearCommand(Command):
    def __init__(self):
        super().__init__("clear", "Clear chat history", ["cls"])

    async def execute(self, cli: ChatCLI, args: List[str]) -> bool:
        cli.messages = []
        cli.console.print("[bold green]Chat history cleared[/bold green]")
        return False
