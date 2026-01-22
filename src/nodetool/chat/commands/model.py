"""Model management commands."""


from rich.table import Table

from nodetool.chat.chat_cli import ChatCLI

from .base import Command


class ModelCommand(Command):
    def __init__(self):
        super().__init__("model", "Set the model for all agents by ID", ["m"])

    async def execute(self, cli: ChatCLI, args: list[str]) -> bool:
        if not cli.selected_provider:
            cli.console.print("[bold red]No provider selected.[/bold red] Use /provider <name> first.")
            return False

        if not args:
            if cli.selected_model:
                cli.console.print(
                    f"Current model: [bold green]{cli.selected_model.name}[/bold green] (ID: {cli.selected_model.id}, Provider: {cli.selected_model.provider.value})"
                )
            elif cli.selected_provider:
                cli.console.print(
                    f"[bold yellow]Provider selected:[/bold yellow] {cli.selected_provider.value}. Use /models to list models or /model <model_id> to select."
                )
            else:
                cli.console.print("[bold red]No provider selected.[/bold red] Use /provider <name> to select one.")
            return False

        if len(args) != 1:
            cli.console.print(
                "[bold red]Error:[/bold red] Provide exactly one model ID. Use /models to list available IDs."
            )
            return False

        if ":" in args[0]:
            cli.console.print(
                "[bold red]Error:[/bold red] Do not include a provider here. Use /provider <name> to switch providers, then /model <model_id>."
            )
            return False

        model_id_to_set = args[0]
        model_id_lower = model_id_to_set.lower()

        models = await cli.load_models_for_provider(cli.selected_provider)
        found_model = next((m for m in models if m.id.lower() == model_id_lower), None)

        if found_model:
            cli.set_selected_model(found_model)
            cli.console.print(f"Model set to [bold green]{found_model.name}[/bold green] (ID: {found_model.id})")
            cli.save_settings()
        else:
            cli.console.print(
                f"[bold red]Error:[/bold red] Model ID '{model_id_to_set}' not found for provider '{cli.selected_provider.value}'. Use /models to list available IDs."
            )

        return False


class ModelsCommand(Command):
    def __init__(self):
        super().__init__("models", "List available models for the current provider", ["ms"])

    async def execute(self, cli: "ChatCLI", args: list[str]) -> bool:
        try:
            if not cli.selected_provider:
                cli.console.print(
                    "[bold red]Error:[/bold red] No provider selected. Use /model <provider> <model_id> to select one."
                )
                return False

            models = await cli.load_models_for_provider(cli.selected_provider)
            if not models:
                cli.console.print(
                    f"[bold yellow]No models available for provider {cli.selected_provider.value}.[/bold yellow]"
                )
                return False

            table = Table(title="Available Models", show_header=True)
            table.add_column("Provider", style="cyan")
            table.add_column("Model Name", style="cyan")
            table.add_column("Model ID", style="cyan")
            for model in models:
                table.add_row(model.provider.value, model.name, model.id)

            cli.console.print(table)
        except Exception as e:
            cli.console.print(f"[bold red]Error listing models:[/bold red] {e}")

        return False
