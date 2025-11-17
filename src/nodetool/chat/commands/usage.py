"""Usage command implementation."""

import json
from typing import List

from rich.syntax import Syntax

from nodetool.chat.chat_cli import ChatCLI
from nodetool.providers import get_provider

from .base import Command


class UsageCommand(Command):
    def __init__(self):
        super().__init__(
            "usage", "Display usage statistics for the selected model's provider", ["u"]
        )

    async def execute(self, cli: ChatCLI, args: List[str]) -> bool:
        if cli.selected_model:
            # Get the provider instance for the selected model
            try:
                provider_instance = await get_provider(cli.selected_model.provider)
                cli.console.print(
                    f"[bold]Usage statistics for provider: {cli.selected_model.provider.value}[/bold]"
                )
                # Assuming the provider instance has a 'usage' attribute or method
                usage_data = getattr(provider_instance, "usage", {})
                syntax = Syntax(
                    json.dumps(usage_data, indent=2),
                    "json",
                    theme="monokai",
                    line_numbers=True,
                )
                cli.console.print(syntax)
            except Exception as e:
                cli.console.print(
                    f"[bold red]Error getting usage for provider {cli.selected_model.provider.value}:[/bold red] {e}"
                )
        else:
            cli.console.print(
                "[yellow]No model selected. Cannot display usage.[/yellow]"
            )
        return False
