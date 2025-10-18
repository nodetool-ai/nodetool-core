"""Workspace command implementation."""

import os
from pathlib import Path
from typing import List

from nodetool.chat.chat_cli import ChatCLI
from .base import Command


class ChangeToWorkspaceCommand(Command):
    """Command to change the current directory to the context's workspace directory."""

    def __init__(self):
        super().__init__("cdw", "Change directory to the defined workspace root")

    async def execute(self, cli: ChatCLI, args: List[str]) -> bool:
        workspace_dir = Path(cli.context.workspace_dir).resolve()
        if not workspace_dir.is_dir():
            cli.console.print(
                f"[bold red]Error:[/bold red] Workspace directory '{cli.context.workspace_dir}' does not exist or is not a directory."
            )
            return False
        try:
            os.chdir(workspace_dir)
            # No need to update context.workspace_dir here, we are just navigating to it
            cli.console.print(
                f"Changed to workspace: [bold green]{os.getcwd()}[/bold green]"
            )
        except Exception as e:
            cli.console.print(
                f"[bold red]Error changing to workspace directory:[/bold red] {e}"
            )
        return False
