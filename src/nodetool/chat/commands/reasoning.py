"""Reasoning model command implementation."""

from typing import List

from nodetool.chat.chat_cli import ChatCLI
from .base import Command


class ReasoningModelCommand(Command):
    """Command to set the reasoning model used by the agent."""

    def __init__(self):
        super().__init__(
            "reasoning",
            "Set the reasoning model for the agent (use 'default' to sync with main model)",
            ["r"],
        )

    async def execute(self, cli: ChatCLI, args: List[str]) -> bool:
        if not cli.selected_model:
            cli.console.print(
                "[bold red]Error:[/bold red] No models loaded. Cannot set reasoning model."
            )
            return False

        current_reasoning_model_id = cli.reasoning_model_id or cli.selected_model.id
        current_reasoning_model = None
        for model in cli.language_models:
            if model.id == current_reasoning_model_id:
                current_reasoning_model = model
                break

        if not args:
            if current_reasoning_model:
                cli.console.print(
                    f"Current reasoning model: [bold green]{current_reasoning_model.name}[/bold green] (ID: {current_reasoning_model.id})"
                )
            else:
                # Should ideally not happen if models are loaded and IDs are synced
                cli.console.print(
                    f"Current reasoning model ID: [bold green]{current_reasoning_model_id}[/bold green] (Model details not found, using main model default)"
                )
                cli.console.print(
                    "Use '/reasoning [model_id]' or '/reasoning default' to set."
                )
            return False

        model_id_to_set = args[0].lower()

        if model_id_to_set == "default":
            # Set reasoning model to track the main selected model
            cli.reasoning_model_id = None  # Use None to signify tracking default
            cli.console.print(
                f"Reasoning model set to track main model: [bold green]{cli.selected_model.name}[/bold green]"
            )
            cli.save_settings()
            return False

        # Find the specified model ID
        found_model = None
        for model in cli.language_models:
            if model.id == model_id_to_set:
                found_model = model
                break

        if found_model:
            cli.reasoning_model_id = found_model.id
            cli.console.print(
                f"Reasoning model set to [bold green]{found_model.name}[/bold green] (ID: {found_model.id})"
            )
            cli.save_settings()
        else:
            cli.console.print(
                f"[bold red]Error:[/bold red] Model ID '{model_id_to_set}' not found. Use /models to list available IDs."
            )

        return False