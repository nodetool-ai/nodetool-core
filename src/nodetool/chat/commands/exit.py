"""Exit command implementation."""


from nodetool.chat.chat_cli import ChatCLI

from .base import Command


class ExitCommand(Command):
    def __init__(self):
        super().__init__("exit", "Exit the chat interface", ["quit", "q"])

    async def execute(self, cli: ChatCLI, args: list[str]) -> bool:
        cli.console.print("[bold yellow]Exiting chat...[/bold yellow]")
        return True
