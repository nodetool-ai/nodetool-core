"""Debug command implementation."""


from rich.prompt import Confirm

from nodetool.chat.chat_cli import ChatCLI

from .base import Command


class DebugCommand(Command):
    def __init__(self):
        super().__init__("debug", "Toggle debug mode (on/off)", ["d"])

    async def execute(self, cli: ChatCLI, args: list[str]) -> bool:
        new_state = None
        if not args:
            current_state = "[bold green]ON[/bold green]" if cli.debug_mode else "[bold red]OFF[/bold red]"
            cli.console.print(f"Debug mode is currently: {current_state}")
            # Ask user if they want to toggle
            if Confirm.ask("Toggle debug mode?", default=cli.debug_mode, console=cli.console):
                new_state = not cli.debug_mode
            else:
                return False  # No change
        elif args[0].lower() == "on":
            new_state = True
        elif args[0].lower() == "off":
            new_state = False
        else:
            cli.console.print("[bold yellow]Usage: /debug [on|off] (or just /debug to toggle)[/bold yellow]")
            return False

        if new_state is not None and new_state != cli.debug_mode:
            cli.debug_mode = new_state
            message = (
                "Debug mode turned ON - Will display tool calls and results"
                if cli.debug_mode
                else "Debug mode turned OFF - Tool calls and results hidden"
            )
            cli.console.print(f"[bold green]{message}[/bold green]")
            cli.save_settings()  # Save settings after changing
        elif new_state is not None:
            cli.console.print(f"Debug mode is already {'ON' if cli.debug_mode else 'OFF'}.")

        return False
