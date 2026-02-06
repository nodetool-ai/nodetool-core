"""Agent mode management commands."""


from nodetool.chat.chat_cli import ChatCLI

from .base import Command


class AgentCommand(Command):
    def __init__(self):
        super().__init__("agent", "Toggle agent mode (on/off) - Omnipotent mode with full MCP tool access", ["a"])

    async def execute(self, cli: ChatCLI, args: list[str]) -> bool:
        if not args:
            if cli.agent_mode:
                from nodetool.agents.tools.mcp_tools import get_all_mcp_tools

                mcp_tools = get_all_mcp_tools()
                cli.console.print("[bold green]Agent mode: OMNIPOTENT[/bold green]")
                cli.console.print(f"[cyan]Available MCP tools ({len(mcp_tools)}):[/cyan]")
                for tool in mcp_tools:
                    cli.console.print(f"  • {tool.name}")
            else:
                cli.console.print("[bold red]Agent mode: OFF[/bold red]")
            return False

        if args[0].lower() == "on":
            cli.agent_mode = True
            from nodetool.agents.tools.mcp_tools import get_all_mcp_tools

            mcp_tools = get_all_mcp_tools()
            cli.console.print("[bold green]Agent mode: OMNIPOTENT[/bold green]")
            cli.console.print(f"[cyan]Agent now has access to {len(mcp_tools)} MCP tools for full nodetool control[/cyan]")
        elif args[0].lower() == "off":
            cli.agent_mode = False
            cli.console.print("[bold red]Agent mode turned OFF[/bold red]")
            cli.console.print("[dim]Regular chat mode - no tool access[/dim]")
        else:
            cli.console.print("[bold yellow]Usage: /agent [on|off][/bold yellow]")

        # Save settings after changing agent mode
        cli.save_settings()
        return False
