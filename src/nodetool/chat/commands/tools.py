"""Tool management commands."""

import asyncio
from typing import List
import sys
import termios
import tty
import os
from nodetool.chat.chat_cli import ChatCLI
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from .base import Command


class ToolsCommand(Command):
    def __init__(self):
        super().__init__(
            "tools", "List available tools or show details about a specific tool", ["t"]
        )

    async def execute(self, cli: ChatCLI, args: List[str]) -> bool:
        if not args:
            table = Table(title="Available Tools", show_header=True)
            table.add_column("Tool Name", style="cyan")
            table.add_column("Status", style="magenta")
            table.add_column("Description", style="green")

            for tool in cli.all_tools:
                tool_name = tool.name
                status = "[bold green]ENABLED[/bold green]" if cli.enabled_tools.get(tool.name, False) else "[bold red]DISABLED[/bold red]"
                table.add_row(tool_name, status, tool.description)

            cli.console.print(table)
            cli.console.print(f"\n[bold cyan]Currently enabled tools:[/bold cyan] {len([t for t in cli.enabled_tools.values() if t])}/{len(cli.all_tools)}")
        else:
            tool_name = args[0]
            found = False

            for tool in cli.all_tools:
                if tool.name.lower() == tool_name.lower():
                    found = True
                    status = "[bold green]ENABLED[/bold green]" if cli.enabled_tools.get(tool.name, False) else "[bold red]DISABLED[/bold red]"
                    panel = Panel(
                        f"[bold]Status:[/bold] {status}\n"
                        f"[bold]Description:[/bold] {tool.description}\n\n",
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
        super().__init__(
            "enable", "Enable a specific tool or all tools", ["en"]
        )

    async def execute(self, cli: "ChatCLI", args: List[str]) -> bool:
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
        super().__init__(
            "disable", "Disable a specific tool or all tools", ["dis"]
        )

    async def execute(self, cli: "ChatCLI", args: List[str]) -> bool:
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

    async def execute(self, cli: "ChatCLI", args: List[str]) -> bool:
        """Execute the tool search modal."""
        await self.show_tool_search_modal(cli)
        return False

    async def show_tool_search_modal(self, cli: "ChatCLI"):
        """Show an interactive tool search modal using Rich Live."""
        # Get terminal size
        terminal_size = os.get_terminal_size()
        max_height = terminal_size.lines
        max_width = terminal_size.columns
        
        # Calculate available space for tools list (subtract header, search, footer)
        available_height = max_height - 12  # Reserve space for header, search, footer, borders
        tools_per_page = max(5, available_height)  # Minimum 5 tools visible
        
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
                if (query_lower in tool.name.lower() or 
                    query_lower in tool.description.lower()):
                    matching_tools.append(tool)
            
            return matching_tools
        
        def update_scroll_offset():
            """Update scroll offset to keep selected item visible."""
            nonlocal scroll_offset
            if selected_index < scroll_offset:
                scroll_offset = selected_index
            elif selected_index >= scroll_offset + tools_per_page:
                scroll_offset = selected_index - tools_per_page + 1
        
        def create_modal_layout(search_query: str, filtered_tools: list, selected_index: int, scroll_offset: int):
            """Create the modal layout with scrolling support."""
            layout = Layout()
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="search", size=3),
                Layout(name="tools", ratio=1),
                Layout(name="footer", size=4)
            )
            
            # Header
            header_text = Text("Tool Search", style="bold cyan", justify="center")
            layout["header"].update(Panel(header_text, border_style="cyan"))
            
            # Search input box with better visual styling
            cursor = "█" if len(search_query) % 2 == 0 else " "  # Blinking effect
            search_display = f"🔍 {search_query}{cursor}"
            if not search_query:
                search_display = f"🔍 Type to search tools...{cursor}"
            
            search_panel = Panel(
                search_display, 
                title="[bold]Search Tools[/bold]",
                border_style="bright_green",
                padding=(0, 1)
            )
            layout["search"].update(search_panel)
            
            # Tools table with scrolling
            table = Table(show_header=True, header_style="bold magenta", show_lines=False)
            table.add_column("", width=3)  # Selection indicator
            table.add_column("Tool Name", style="cyan", width=min(30, max_width // 4))
            table.add_column("Status", style="magenta", width=10) 
            table.add_column("Description", style="green")
            
            # Calculate visible tools
            start_idx = scroll_offset
            end_idx = min(start_idx + tools_per_page, len(filtered_tools))
            visible_tools = filtered_tools[start_idx:end_idx]
            
            for i, tool in enumerate(visible_tools):
                actual_index = start_idx + i
                indicator = "►" if actual_index == selected_index else " "
                status_style = "[bold green]ENABLED[/bold green]" if cli.enabled_tools.get(tool.name, False) else "[bold red]DISABLED[/bold red]"
                
                # Get only first line of description and truncate based on terminal width
                desc_width = max_width - 50  # Reserve space for other columns
                desc = tool.description.split('\n')[0]  # Take only first line
                if len(desc) > desc_width:
                    desc = desc[:desc_width-3] + "..."
                
                table.add_row(indicator, tool.name, status_style, desc)
            
            # Add scroll indicators
            title_suffix = ""
            if len(filtered_tools) > tools_per_page:
                scroll_info = f" ({start_idx + 1}-{end_idx} of {len(filtered_tools)})"
                if scroll_offset > 0:
                    title_suffix += " ↑"
                if end_idx < len(filtered_tools):
                    title_suffix += " ↓"
                title_suffix = scroll_info + title_suffix
            
            layout["tools"].update(Panel(table, border_style="blue", title=f"Tools{title_suffix}"))
            
            # Footer with enhanced controls
            controls = [
                "↑/↓ Navigate",
                "PgUp/PgDn Scroll",
                "ENTER Toggle",
                "ESC Exit",
                "Type to search"
            ]
            footer_text = f"[bold]Controls:[/bold] {' | '.join(controls)}"
            layout["footer"].update(Panel(footer_text, border_style="yellow"))
            
            return layout

        # Set up terminal for raw input
        old_settings = None
        if sys.stdin.isatty():
            old_settings = termios.tcgetattr(sys.stdin.fileno())
            tty.setraw(sys.stdin.fileno())
        
        try:
            with Live(create_modal_layout(search_query, filtered_tools, selected_index, scroll_offset), 
                     refresh_per_second=10, screen=True) as live:
                
                while True:
                    # Read single character
                    if sys.stdin.isatty():
                        char = sys.stdin.read(1)
                        
                        if char == '\x1b':  # ESC sequence
                            next_char = sys.stdin.read(1)
                            if next_char == '[':
                                arrow_char = sys.stdin.read(1)
                                if arrow_char == 'A':  # Up arrow
                                    if selected_index > 0:
                                        selected_index -= 1
                                        update_scroll_offset()
                                elif arrow_char == 'B':  # Down arrow
                                    if selected_index < len(filtered_tools) - 1:
                                        selected_index += 1
                                        update_scroll_offset()
                                elif arrow_char == '5':  # Page Up
                                    next_char = sys.stdin.read(1)  # Read the '~'
                                    if next_char == '~':
                                        selected_index = max(0, selected_index - tools_per_page)
                                        update_scroll_offset()
                                elif arrow_char == '6':  # Page Down
                                    next_char = sys.stdin.read(1)  # Read the '~'
                                    if next_char == '~':
                                        selected_index = min(len(filtered_tools) - 1, selected_index + tools_per_page)
                                        update_scroll_offset()
                            else:
                                # ESC pressed - exit
                                break
                        elif char == '\r' or char == '\n':  # Enter
                            if filtered_tools and selected_index < len(filtered_tools):
                                selected_tool = filtered_tools[selected_index]
                                # Toggle tool status
                                current_status = cli.enabled_tools.get(selected_tool.name, False)
                                cli.enabled_tools[selected_tool.name] = not current_status
                                cli.refresh_tools()
                                cli.save_settings()
                        elif char == '\x7f':  # Backspace
                            if search_query:
                                search_query = search_query[:-1]
                                filtered_tools = filter_tools(search_query)
                                selected_index = 0
                                scroll_offset = 0
                        elif char == '\x03':  # Ctrl+C
                            break
                        elif char.isprintable():
                            search_query += char
                            filtered_tools = filter_tools(search_query)
                            selected_index = 0
                            scroll_offset = 0
                    
                    # Update the display
                    live.update(create_modal_layout(search_query, filtered_tools, selected_index, scroll_offset))
                    
                    # Small delay to prevent excessive CPU usage
                    await asyncio.sleep(0.05)
        
        finally:
            # Restore terminal settings
            if old_settings and sys.stdin.isatty():
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)
        
        cli.console.print("[bold green]Tool search closed[/bold green]")