"""Provider selection command."""


from nodetool.chat.chat_cli import ChatCLI
from nodetool.metadata.types import Provider

from .base import Command


class ProviderCommand(Command):
    def __init__(self) -> None:
        super().__init__("provider", "Set the current provider", ["pr"])

    async def execute(self, cli: ChatCLI, args: list[str]) -> bool:
        if not args:
            if cli.selected_provider:
                cli.console.print(f"Current provider: [bold green]{cli.selected_provider.value}[/bold green]")
            else:
                cli.console.print("[bold yellow]No provider selected.[/bold yellow]")
            return False

        provider_name = args[0]
        provider = next(
            (p for p in Provider if p.value.lower() == provider_name.lower()),
            None,
        )
        if provider is None:
            cli.console.print(
                f"[bold red]Invalid provider:[/bold red] {provider_name}. Use /providers to list available options."
            )
            return False
        cli.set_selected_provider(provider)
        cli.console.print(
            f"Provider set to [bold green]{provider.value}[/bold green]. Use /models to list models for this provider."
        )
        cli.save_settings()
        return False
