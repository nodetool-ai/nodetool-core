"""Providers listing command."""


from rich.table import Table

from nodetool.chat.chat_cli import ChatCLI
from nodetool.providers import list_providers

from .base import Command


class ProvidersCommand(Command):
    def __init__(self):
        super().__init__("providers", "List available providers", ["p"])

    async def execute(self, cli: ChatCLI, args: list[str]) -> bool:
        try:
            providers = await list_providers("1")

            if not providers:
                cli.console.print(
                    "[bold yellow]No providers available.[/bold yellow] Configure provider secrets to enable them."
                )
                return False

            table = Table(title="Available Providers", show_header=True)
            table.add_column("Provider", style="cyan")
            table.add_column("Status", style="green")

            for provider in providers:
                status = (
                    "selected"
                    if cli.selected_provider and provider.provider_name == cli.selected_provider.value
                    else "available"
                )
                table.add_row(provider.provider_name, status)

            cli.console.print(table)
        except Exception as e:
            cli.console.print(f"[bold red]Error listing providers:[/bold red] {e}")

        return False
