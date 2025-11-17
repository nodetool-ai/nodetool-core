"""Model management commands."""

from typing import List

from rich.table import Table

from nodetool.chat.chat_cli import ChatCLI

from .base import Command


class ModelCommand(Command):
    def __init__(self):
        super().__init__("model", "Set the model for all agents by ID", ["m"])

    async def execute(self, cli: ChatCLI, args: List[str]) -> bool:
        if not cli.selected_model:
            cli.console.print(
                "[bold red]Error:[/bold red] No models loaded. Cannot set model."
            )
            return False

        if not args:
            cli.console.print(
                f"Current model: [bold green]{cli.selected_model.name}[/bold green] (ID: {cli.selected_model.id}, Provider: {cli.selected_model.provider.value})"
            )
            return False

        model_id_to_set = args[0]
        found_model = None
        for model in cli.language_models:
            if model.id == model_id_to_set:
                found_model = model
                break

        if found_model:
            cli.selected_model = found_model
            # Update related model IDs (can be changed later via specific commands if needed)
            cli.planner_model_id = found_model.id
            cli.summarization_model_id = found_model.id
            cli.retrieval_model_id = found_model.id

            cli.console.print(
                f"Model set to [bold green]{found_model.name}[/bold green] (ID: {found_model.id})"
            )
            # Save settings after changing model
            cli.save_settings()
        else:
            cli.console.print(
                f"[bold red]Error:[/bold red] Model ID '{model_id_to_set}' not found. Use /models to list available IDs."
            )

        return False


class ModelsCommand(Command):
    def __init__(self):
        super().__init__(
            "models", "List available models for the current provider", ["ms"]
        )

    async def execute(self, cli: "ChatCLI", args: List[str]) -> bool:
        try:
            table = Table(title="Available Models", show_header=True)
            table.add_column("Provider", style="cyan")
            table.add_column("Model Name", style="cyan")
            table.add_column("Model ID", style="cyan")
            for model in cli.language_models:
                table.add_row(model.provider, model.name, model.id)

            cli.console.print(table)
        except Exception as e:
            cli.console.print(f"[bold red]Error listing models:[/bold red] {e}")

        return False
