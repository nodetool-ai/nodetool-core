"""Tool management commands."""

import asyncio
import os
import sys
import termios
import tty

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from nodetool.chat.chat_cli import ChatCLI

from .base import Command


class ToolsCommand(Command):
    def __init__(self):
        super().__init__("tools", "List available tools or show details about a specific tool", ["t"])

    async def execute(self, cli: ChatCLI, args: list[str]) -> bool:
        if not args:
            table = Table(title="Available Tools", show_header=True)
            table.add_column("Tool Name", style="cyan")
            table.add_column("Status", style="magenta")
            table.add_column("Description", style="green")

            for tool in cli.all_tools:
                tool_name = tool.name
                status = (
                    "[bold green]ENABLED[/bold green]"
                    if cli.enabled_tools.get(tool.name, False)
                    else "[bold red]DISABLED[/bold red]"
                )
                table.add_row(tool_name, status, tool.description)

            cli.console.print(table)
            cli.console.print(
                f"\n[bold cyan]Currently enabled tools:[/bold cyan] {len([t for t in cli.enabled_tools.values() if t])}/{len(cli.all_tools)}"
            )
        else:
            tool_name = args[0]
            found = False

            for tool in cli.all_tools:
                if tool.name.lower() == tool_name.lower():
                    found = True
                    status = (
                        "[bold green]ENABLED[/bold green]"
                        if cli.enabled_tools.get(tool.name, False)
                        else "[bold red]DISABLED[/bold red]"
                    )
                    panel = Panel(
                        f"[bold]Status:[/bold] {status}\n[bold]Description:[/bold] {tool.description}\n\n",
                        title=f"Tool: {tool.name}",
                        border_style="green",
                    )
                    cli.console.print(panel)
                    break

            if not found:
                cli.console.print(f"[bold red]Tool '{tool_name}' not found[/bold red]")

        return False


class ToolEnableCommand(Command):
    def __init__(self):
        super().__init__("enable", "Enable a specific tool or all tools", ["en"])

    async def execute(self, cli: "ChatCLI", args: list[str]) -> bool:
        if not args:
            cli.console.print("[bold red]Usage:[/bold red] /enable <tool_name> | all")
            return False

        if args[0].lower() == "all":
            for tool in cli.all_tools:
                cli.enabled_tools[tool.name] = True
            cli.refresh_tools()
            cli.console.print(f"[bold green]Enabled all {len(cli.all_tools)} tools[/bold green]")
            cli.save_settings()
            return False

        tool_name = args[0]
        found = False

        for tool in cli.all_tools:
            if tool.name.lower() == tool_name.lower():
                cli.enabled_tools[tool.name] = True
                cli.refresh_tools()
                cli.console.print(f"[bold green]Enabled tool:[/bold green] {tool.name}")
                found = True
                break

        if not found:
            cli.console.print(f"[bold red]Tool '{tool_name}' not found[/bold red]")
            return False

        cli.save_settings()
        return False


class ToolDisableCommand(Command):
    def __init__(self):
        super().__init__("disable", "Disable a specific tool or all tools", ["dis"])

    async def execute(self, cli: "ChatCLI", args: list[str]) -> bool:
        if not args:
            cli.console.print("[bold red]Usage:[/bold red] /disable <tool_name> | all")
            return False

        if args[0].lower() == "all":
            for tool in cli.all_tools:
                cli.enabled_tools[tool.name] = False
            cli.refresh_tools()
            cli.console.print(f"[bold red]Disabled all {len(cli.all_tools)} tools[/bold red]")
            cli.save_settings()
            return False

        tool_name = args[0]
        found = False

        for tool in cli.all_tools:
            if tool.name.lower() == tool_name.lower():
                cli.enabled_tools[tool.name] = False
                cli.refresh_tools()
                cli.console.print(f"[bold red]Disabled tool:[/bold red] {tool.name}")
                found = True
                break

        if not found:
            cli.console.print(f"[bold red]Tool '{tool_name}' not found[/bold red]")
            return False

        cli.save_settings()
        return False


class ToolSearchCommand(Command):
    """Command to open an interactive tool search modal."""

    def __init__(self):
        super().__init__(
            "search",
            "Open interactive tool search modal",
            ["s"],
        )

    async def execute(self, cli: "ChatCLI", args: list[str]) -> bool:
        """Execute the tool search modal."""
        await self.show_tool_search_modal(cli)
        return False

    async def show_tool_search_modal(self, cli: "ChatCLI"):
        """Show an interactive tool search modal using Rich Live."""
        # Get terminal size
        terminal_size = os.get_terminal_size()
        max_height = terminal_size.lines
        max_width = terminal_size.columns

        tools_per_page = max_height - 2  # Reserve lines for title

        # State variables
        search_query = ""
        selected_index = 0
        scroll_offset = 0
        filtered_tools = list(cli.all_tools)

        def filter_tools(query: str):
            """Filter tools based on search query."""
            if not query:
                return list(cli.all_tools)

            query_lower = query.lower()
            matching_tools = []

            for tool in cli.all_tools:
                # Search in name and description
                if query_lower in tool.name.lower() or query_lower in tool.description.lower():
                    matching_tools.append(tool)

            return matching_tools

        def highlight_match(text: str, query: str) -> str:
            """Highlight matching text."""
            if not query:
                return text

            # Simple case-insensitive highlighting
            query_lower = query.lower()
            text_lower = text.lower()

            if query_lower in text_lower:
                start = text_lower.find(query_lower)
                end = start + len(query)
                return f"{text[:start]}[bold yellow]{text[start:end]}[/bold yellow]{text[end:]}"
            return text

        def update_scroll_offset():
            """Update scroll offset to keep selected item visible."""
            nonlocal scroll_offset
            if selected_index < scroll_offset:
                scroll_offset = selected_index
            elif selected_index >= scroll_offset + tools_per_page:
                scroll_offset = selected_index - tools_per_page + 1

        def create_display(
            search_query: str,
            filtered_tools: list,
            selected_index: int,
            scroll_offset: int,
        ):
            """Create the display content."""
            start_idx = scroll_offset
            end_idx = min(start_idx + tools_per_page, len(filtered_tools))
            visible_tools = filtered_tools[start_idx:end_idx]

            lines = []

            # Title line
            title = "Tools Search"
            if search_query:
                title += f" - '{search_query}'"
            if len(filtered_tools) > tools_per_page:
                title += f" ({start_idx + 1}-{end_idx} of {len(filtered_tools)})"
            lines.append(Text.from_markup(f"[bold blue]{title}[/bold blue]"))

            for i, tool in enumerate(visible_tools):
                actual_index = start_idx + i
                indicator = "►" if actual_index == selected_index else " "
                status = "[green]ON[/green]" if cli.enabled_tools.get(tool.name, False) else "[red]OFF[/red]"

                # Highlight matches in name and description
                tool_name = highlight_match(tool.name, search_query)
                desc = (tool.description or "").strip().split("\n")[0]
                if len(desc) > max_width - 40:
                    desc = desc[: max_width - 43] + "..."
                desc = highlight_match(desc, search_query)

                lines.append(Text.from_markup(f"{indicator} [cyan]{tool_name}[/cyan] [{status}] {desc}"))

            if not visible_tools:
                lines.append(Text.from_markup("[dim]No tools found[/dim]"))

            return Group(*lines)

        # Set up terminal for raw input
        old_settings = None
        if sys.stdin.isatty():
            old_settings = termios.tcgetattr(sys.stdin.fileno())
            tty.setraw(sys.stdin.fileno())

        try:
            with Live(
                create_display(search_query, filtered_tools, selected_index, scroll_offset),
                refresh_per_second=10,
                screen=True,
            ) as live:
                while True:
                    # Read single character
                    if sys.stdin.isatty():
                        char = sys.stdin.read(1)

                        if char == "\x1b":  # ESC sequence
                            next_char = sys.stdin.read(1)
                            if next_char == "[":
                                arrow_char = sys.stdin.read(1)
                                if arrow_char == "A":  # Up arrow
                                    if selected_index > 0:
                                        selected_index -= 1
                                        update_scroll_offset()
                                elif arrow_char == "B":  # Down arrow
                                    if selected_index < len(filtered_tools) - 1:
                                        selected_index += 1
                                        update_scroll_offset()
                                elif arrow_char == "5":  # Page Up
                                    next_char = sys.stdin.read(1)  # Read the '~'
                                    if next_char == "~":
                                        selected_index = max(0, selected_index - tools_per_page)
                                        update_scroll_offset()
                                elif arrow_char == "6":  # Page Down
                                    next_char = sys.stdin.read(1)  # Read the '~'
                                    if next_char == "~":
                                        selected_index = min(
                                            len(filtered_tools) - 1,
                                            selected_index + tools_per_page,
                                        )
                                        update_scroll_offset()
                            else:
                                # ESC pressed - exit
                                break
                        elif char == "\r" or char == "\n":  # Enter
                            if filtered_tools and selected_index < len(filtered_tools):
                                selected_tool = filtered_tools[selected_index]
                                # Toggle tool status
                                current_status = cli.enabled_tools.get(selected_tool.name, False)
                                cli.enabled_tools[selected_tool.name] = not current_status
                                cli.refresh_tools()
                                cli.save_settings()
                        elif char == "\x7f":  # Backspace
                            if search_query:
                                search_query = search_query[:-1]
                                filtered_tools = filter_tools(search_query)
                                selected_index = min(0, len(filtered_tools) - 1) if filtered_tools else 0
                                update_scroll_offset()
                        elif char == "\x03":  # Ctrl+C
                            break
                        elif char.isprintable():
                            search_query += char
                            filtered_tools = filter_tools(search_query)
                            selected_index = min(0, len(filtered_tools) - 1) if filtered_tools else 0
                            update_scroll_offset()

                    # Update the display
                    live.update(create_display(search_query, filtered_tools, selected_index, scroll_offset))

                    # Small delay to prevent excessive CPU usage
                    await asyncio.sleep(0.05)

        finally:
            # Restore terminal settings
            if old_settings and sys.stdin.isatty():
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)

        cli.console.print("[bold green]Tool search closed[/bold green]")
