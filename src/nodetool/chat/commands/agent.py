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


class AnalysisPhaseCommand(Command):
    """Command to toggle the Analysis Phase for the Agent planner."""

    def __init__(self):
        super().__init__("analysis", "Toggle agent analysis phase (on/off)", ["an"])

    async def execute(self, cli: "ChatCLI", args: List[str]) -> bool:
        new_state = None
        if not args:
            current_state = (
                "[bold green]ON[/bold green]"
                if cli.enable_analysis_phase
                else "[bold red]OFF[/bold red]"
            )
            cli.console.print(f"Agent Analysis Phase is currently: {current_state}")
            if Confirm.ask(
                "Toggle Analysis Phase?",
                default=cli.enable_analysis_phase,
                console=cli.console,
            ):
                new_state = not cli.enable_analysis_phase
            else:
                return False  # No change
        elif args[0].lower() == "on":
            new_state = True
        elif args[0].lower() == "off":
            new_state = False
        else:
            cli.console.print(
                "[bold yellow]Usage: /analysis [on|off] (or just /analysis to toggle)[/bold yellow]"
            )
            return False

        if new_state is not None and new_state != cli.enable_analysis_phase:
            cli.enable_analysis_phase = new_state
            status = (
                "[bold green]ON[/bold green]"
                if cli.enable_analysis_phase
                else "[bold red]OFF[/bold red]"
            )
            message = f"Agent Analysis Phase turned {status}"
            cli.console.print(f"[bold green]{message}[/bold green]")
            cli.save_settings()
        elif new_state is not None:
            cli.console.print(
                f"Agent Analysis Phase is already {'ON' if cli.enable_analysis_phase else 'OFF'}."
            )
        return False


class FlowAnalysisCommand(Command):
    """Command to toggle the Data Flow Analysis Phase for the Agent planner."""

    def __init__(self):
        super().__init__(
            "flow", "Toggle agent data flow analysis phase (on/off)", ["fl"]
        )

    async def execute(self, cli: "ChatCLI", args: List[str]) -> bool:
        new_state = None
        if not args:
            current_state = (
                "[bold green]ON[/bold green]"
                if cli.enable_flow_analysis
                else "[bold red]OFF[/bold red]"
            )
            cli.console.print(
                f"Agent Data Flow Analysis Phase is currently: {current_state}"
            )
            if Confirm.ask(
                "Toggle Data Flow Analysis Phase?",
                default=cli.enable_flow_analysis,
                console=cli.console,
            ):
                new_state = not cli.enable_flow_analysis
            else:
                return False  # No change
        elif args[0].lower() == "on":
            new_state = True
        elif args[0].lower() == "off":
            new_state = False
        else:
            cli.console.print(
                "[bold yellow]Usage: /flow [on|off] (or just /flow to toggle)[/bold yellow]"
            )
            return False

        if new_state is not None and new_state != cli.enable_flow_analysis:
            cli.enable_flow_analysis = new_state
            status = (
                "[bold green]ON[/bold green]"
                if cli.enable_flow_analysis
                else "[bold red]OFF[/bold red]"
            )
            message = f"Agent Data Flow Analysis Phase turned {status}"
            cli.console.print(f"[bold green]{message}[/bold green]")
            cli.save_settings()
        elif new_state is not None:
            cli.console.print(
                f"Agent Data Flow Analysis Phase is already {'ON' if cli.enable_flow_analysis else 'OFF'}."
            )
        return False