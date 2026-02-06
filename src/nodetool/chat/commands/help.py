"""Help command implementation."""


from rich.table import Table

from nodetool.chat.chat_cli import ChatCLI

from .base import Command


class HelpCommand(Command):
    def __init__(self):
        super().__init__("help", "Display available commands", ["h"])

    async def execute(self, cli: ChatCLI, args: list[str]) -> bool:
        cli.console.print("\n[bold]NodeTool Omnipotent Agent[/bold]", style="green")
        cli.console.print(
            "The agent has full control over NodeTool via MCP tools. "
            "It can create, run, debug, and manage workflows autonomously.\n",
            style="dim",
        )

        cli.console.print("[bold]Available Commands[/bold]", style="cyan")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Command", style="cyan")
        table.add_column("Description", style="green")

        # Get unique command objects to avoid printing aliases separately
        unique_commands = sorted(set(cli.commands.values()), key=lambda cmd: cmd.name)

        for cmd in unique_commands:
            aliases = f" ({', '.join(['/' + a for a in cmd.aliases])})" if cmd.aliases else ""
            table.add_row(f"/{cmd.name}{aliases}", cmd.description)

        cli.console.print(table)

        # Show MCP tools if in agent mode
        if cli.agent_mode:
            from nodetool.agents.tools.mcp_tools import get_all_mcp_tools

            mcp_tools = get_all_mcp_tools()
            cli.console.print(f"\n[bold]MCP Tools Available ({len(mcp_tools)})[/bold]", style="cyan")
            cli.console.print(
                "These tools are automatically available to the agent for controlling NodeTool:",
                style="dim",
            )
            tools_table = Table(show_header=True, header_style="bold magenta")
            tools_table.add_column("Tool", style="cyan")
            tools_table.add_column("Description", style="green", max_width=60)
            for tool in mcp_tools:
                # Get first line of description
                desc = tool.description.strip().split("\n")[0][:60]
                tools_table.add_row(tool.name, desc)
            cli.console.print(tools_table)

        cli.console.print("\n[bold]File System Commands[/bold]", style="cyan")

        workspace_table = Table(show_header=True, header_style="bold magenta")
        workspace_table.add_column("Command", style="cyan")
        workspace_table.add_column("Description", style="green")

        workspace_commands = [
            ("pwd", "Print current workspace directory"),
            ("ls [path]", "List contents of workspace directory"),
            ("cd [path]", "Change directory within workspace"),
            ("mkdir [dir]", "Create new directory in workspace"),
            ("rm [path]", "Remove file or directory in workspace"),
            ("open [file]", "Open file in system default application"),
            ("cat [file]", "Display the content of a file"),
            ("cp <src> <dest>", "Copy file or directory within workspace"),
            ("mv <src> <dest>", "Move/rename file or directory within workspace"),
            ("grep <pattern> [path]", "Search for pattern in files within workspace"),
            ("cdw", "Change directory to the defined workspace root"),
        ]

        for cmd, desc in workspace_commands:
            workspace_table.add_row(cmd, desc)

        cli.console.print(workspace_table)
        return False
