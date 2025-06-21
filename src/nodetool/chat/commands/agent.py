"""Agent mode management commands."""

from typing import List
from nodetool.chat.chat_cli import ChatCLI
from rich.prompt import Confirm
from .base import Command


class AgentCommand(Command):
    def __init__(self):
        super().__init__("agent", "Toggle agent mode (on/off)", ["a"])

    async def execute(self, cli: ChatCLI, args: List[str]) -> bool:
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
