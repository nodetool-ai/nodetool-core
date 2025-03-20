#!/usr/bin/env python3
import curses
import json
import os
from pathlib import Path
import textwrap
import re


class FileExplorer:
    """Text UI for exploring files with preview functionality."""

    def __init__(self, start_dir: str = "."):
        self.current_dir = Path(start_dir).resolve()
        self.files = []
        self.current_file_idx = 0
        self.top_file_idx = 0
        self.active_panel = 0  # 0: files, 1: preview

    def load_directory(self, dir_path=None):
        """Load all files and directories from the current path."""
        if dir_path:
            self.current_dir = Path(dir_path).resolve()

        self.files = []

        # Add parent directory option
        self.files.append(
            {"name": "..", "type": "dir", "path": self.current_dir.parent}
        )

        # Add directories and files
        for item in sorted(
            self.current_dir.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())
        ):
            item_type = "dir" if item.is_dir() else "file"
            file_entry = {
                "name": item.name,
                "type": item_type,
                "path": item,
            }
            self.files.append(file_entry)

        # Reset selection
        self.current_file_idx = 0
        self.top_file_idx = 0

    def run(self, stdscr):
        """Run the UI main loop."""
        curses.curs_set(0)  # Hide cursor
        stdscr.clear()

        # Load initial directory
        self.load_directory()

        # Set up color pairs
        curses.start_color()
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)  # Selected item
        curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)  # Header
        curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Key/Directory
        curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Value/File
        curses.init_pair(5, curses.COLOR_CYAN, curses.COLOR_BLACK)  # Markdown headers
        curses.init_pair(6, curses.COLOR_BLACK, curses.COLOR_GREEN)  # Active panel
        curses.init_pair(7, curses.COLOR_RED, curses.COLOR_BLACK)  # Markdown emphasis
        curses.init_pair(8, curses.COLOR_WHITE, curses.COLOR_BLACK)  # Normal text

        # Main loop
        while True:
            # Get screen dimensions
            height, width = stdscr.getmaxyx()

            # Calculate panel dimensions
            files_width = max(width // 3, 25)
            preview_width = width - files_width

            # Create panels
            files_panel = stdscr.derwin(height, files_width, 0, 0)
            preview_panel = stdscr.derwin(height, preview_width, 0, files_width)

            # Draw panels
            self._draw_file_list(files_panel, height, files_width)
            self._draw_preview(preview_panel, height, preview_width)

            # Refresh panels
            files_panel.refresh()
            preview_panel.refresh()

            # Handle input
            key = stdscr.getch()
            if key == ord("q"):
                break
            elif key == ord("\t") or key == 9:  # Tab key
                self.active_panel = (self.active_panel + 1) % 2
            elif key == curses.KEY_UP and self.active_panel == 0:
                if self.current_file_idx > 0:
                    self.current_file_idx -= 1
                    if self.current_file_idx < self.top_file_idx:
                        self.top_file_idx -= 1
            elif key == curses.KEY_DOWN and self.active_panel == 0:
                if self.current_file_idx < len(self.files) - 1:
                    self.current_file_idx += 1
                    if self.current_file_idx >= self.top_file_idx + height - 4:
                        self.top_file_idx += 1
            elif key == curses.KEY_RIGHT or key == 10 or key == 13:  # Enter key
                if self.files and self.current_file_idx < len(self.files):
                    selected = self.files[self.current_file_idx]
                    if selected["type"] == "dir":
                        self.load_directory(selected["path"])
            elif key == curses.KEY_LEFT:
                # Go to parent directory
                self.load_directory(self.current_dir.parent)
            elif key == ord("r"):
                # Reload current directory
                self.load_directory()

    def _draw_file_list(self, panel, height, width):
        """Draw the file list panel."""
        panel.clear()
        panel.box()

        # Draw header with highlight if active
        header_style = (
            curses.color_pair(6) if self.active_panel == 0 else curses.color_pair(2)
        )
        panel.addstr(0, 2, f" {self.current_dir} ", header_style)

        # Draw help text
        help_text = (
            "[Tab]:Switch [↑/↓]:Navigate [Enter]:Open [←]:Parent [r]:Reload [q]:Quit"
        )
        if width > 10:
            panel.addstr(height - 1, 2, help_text[: width - 4])

        # Draw files
        max_files = height - 4

        if self.current_file_idx < self.top_file_idx:
            self.top_file_idx = self.current_file_idx
        elif self.current_file_idx >= self.top_file_idx + max_files:
            self.top_file_idx = self.current_file_idx - max_files + 1

        for i in range(min(max_files, len(self.files))):
            file_idx = i + self.top_file_idx
            if file_idx < len(self.files):
                file_entry = self.files[file_idx]
                file_name = file_entry["name"]
                file_type = file_entry["type"]

                # Choose icon and color for file type
                icon = "📁 " if file_type == "dir" else "📄 "
                type_color = (
                    curses.color_pair(3) if file_type == "dir" else curses.color_pair(4)
                )

                # Set selection highlight
                line_attr = (
                    curses.color_pair(1)
                    if file_idx == self.current_file_idx
                    else type_color
                )

                # Truncate if necessary
                display_name = icon + file_name
                if len(display_name) > width - 6:
                    display_name = display_name[: width - 9] + "..."

                panel.addstr(i + 2, 2, display_name, line_attr)

    def _draw_preview(self, panel, height, width):
        """Draw the file preview panel."""
        panel.clear()
        panel.box()

        # Draw header with highlight if active
        header_style = (
            curses.color_pair(6) if self.active_panel == 1 else curses.color_pair(2)
        )
        panel.addstr(0, 2, " File Preview ", header_style)

        # If no files, show message
        if not self.files:
            panel.addstr(2, 2, "No files found.")
            return

        # Show current file preview
        if self.current_file_idx < len(self.files):
            file_entry = self.files[self.current_file_idx]
            file_path = file_entry["path"]
            file_type = file_entry["type"]
            file_name = file_entry["name"]

            # Display file name
            panel.addstr(1, 2, f"File: {file_name}")

            # If directory, show contents list
            if file_type == "dir":
                panel.addstr(3, 2, "Directory contents:", curses.A_BOLD)
                try:
                    contents = list(file_path.iterdir())
                    for i, item in enumerate(
                        sorted(contents, key=lambda x: (not x.is_dir(), x.name.lower()))
                    ):
                        if i + 5 >= height:
                            panel.addstr(i + 5, 2, "... more items")
                            break
                        icon = "📁 " if item.is_dir() else "📄 "
                        item_color = (
                            curses.color_pair(3)
                            if item.is_dir()
                            else curses.color_pair(4)
                        )
                        panel.addstr(i + 5, 2, f"{icon}{item.name}", item_color)
                except Exception as e:
                    panel.addstr(3, 2, f"Error: {str(e)}")

            # If file, try to preview based on extension
            elif file_type == "file":
                try:
                    # Check if it's a text file by trying to read it
                    if file_path.suffix.lower() in [".md", ".markdown"]:
                        self._render_markdown(
                            panel, file_path, 3, 2, width - 4, height - 5
                        )
                    elif file_path.suffix.lower() == ".json":
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        self._render_json(panel, data, 3, 2, width - 4, height - 5)
                    else:
                        # Generic text preview
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                lines = f.readlines()
                            for i, line in enumerate(lines):
                                if i + 3 >= height:
                                    panel.addstr(i + 3, 2, "... more lines")
                                    break
                                # Truncate long lines
                                if len(line) > width - 6:
                                    line = line[: width - 9] + "..."
                                panel.addstr(i + 3, 2, line.rstrip())
                        except UnicodeDecodeError:
                            panel.addstr(3, 2, "Binary file - preview not available")
                except Exception as e:
                    panel.addstr(3, 2, f"Error: {str(e)}")

    def _render_markdown(self, panel, file_path, y, x, max_width, max_height):
        """Render markdown content with basic formatting."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            line_num = 0
            for i, line in enumerate(lines):
                if line_num >= max_height:
                    panel.addstr(y + line_num, x, "... more lines")
                    break

                # Detect heading
                if line.startswith("#"):
                    heading_level = 0
                    for char in line:
                        if char == "#":
                            heading_level += 1
                        else:
                            break

                    heading_text = line[heading_level:].strip()

                    # Style based on heading level
                    if heading_level <= 2:
                        panel.addstr(
                            y + line_num,
                            x,
                            heading_text,
                            curses.color_pair(5) | curses.A_BOLD | curses.A_UNDERLINE,
                        )
                    else:
                        panel.addstr(
                            y + line_num,
                            x,
                            heading_text,
                            curses.color_pair(5) | curses.A_BOLD,
                        )

                # Detect lists
                elif line.strip().startswith(("- ", "* ", "+ ", "1. ")):
                    indent = len(line) - len(line.lstrip())
                    marker_end = line.find(" ", indent) + 1
                    panel.addstr(
                        y + line_num,
                        x + indent,
                        line[indent:marker_end],
                        curses.color_pair(3),
                    )
                    panel.addstr(
                        y + line_num, x + marker_end, line[marker_end:].rstrip()
                    )

                # Detect emphasis (basic)
                elif "*" in line or "_" in line:
                    # Very basic emphasis detection - not comprehensive
                    parts = re.split(r"(\*\*.*?\*\*|\*.*?\*|__.*?__|_.*?_)", line)
                    current_x = x

                    for part in parts:
                        # Bold
                        if part.startswith("**") and part.endswith("**"):
                            text = part[2:-2]
                            panel.addstr(y + line_num, current_x, text, curses.A_BOLD)
                            current_x += len(text)
                        # Italic with *
                        elif part.startswith("*") and part.endswith("*"):
                            text = part[1:-1]
                            panel.addstr(
                                y + line_num,
                                current_x,
                                text,
                                curses.color_pair(7) | curses.A_ITALIC,
                            )
                            current_x += len(text)
                        # Bold with _
                        elif part.startswith("__") and part.endswith("__"):
                            text = part[2:-2]
                            panel.addstr(y + line_num, current_x, text, curses.A_BOLD)
                            current_x += len(text)
                        # Italic with _
                        elif part.startswith("_") and part.endswith("_"):
                            text = part[1:-1]
                            panel.addstr(
                                y + line_num,
                                current_x,
                                text,
                                curses.color_pair(7) | curses.A_ITALIC,
                            )
                            current_x += len(text)
                        # Regular text
                        else:
                            if len(part) > 0:
                                panel.addstr(y + line_num, current_x, part.rstrip())
                                current_x += len(part)

                # Regular line
                else:
                    panel.addstr(y + line_num, x, line.rstrip())

                line_num += 1

        except Exception as e:
            panel.addstr(y, x, f"Error rendering markdown: {str(e)}")

    def _render_json(self, panel, data, y, x, max_width, max_height, depth=0):
        """Recursively render JSON data with proper indentation and colors."""
        indent = "  " * depth
        line = 0

        if isinstance(data, dict):
            for key, value in data.items():
                if line >= max_height:
                    panel.addstr(y + line, x, f"{indent}...")
                    break

                key_str = f"{indent}{key}: "
                panel.addstr(y + line, x, key_str, curses.color_pair(3))

                if isinstance(value, (dict, list)) and value:
                    panel.addstr(y + line, x + len(key_str), "")
                    line += 1
                    line += self._render_json(
                        panel,
                        value,
                        y + line,
                        x,
                        max_width,
                        max_height - line,
                        depth + 1,
                    )
                else:
                    value_str = self._format_value(value)
                    # Wrap long values
                    if len(key_str) + len(value_str) > max_width:
                        panel.addstr(
                            y + line,
                            x + len(key_str),
                            value_str[: max_width - len(key_str)],
                            curses.color_pair(4),
                        )
                        wrapped = textwrap.wrap(
                            value_str[max_width - len(key_str) :],
                            max_width - len(indent) - 2,
                        )
                        line += 1
                        for wrap_line in wrapped:
                            if line >= max_height:
                                break
                            panel.addstr(
                                y + line,
                                x + len(indent) + 2,
                                wrap_line,
                                curses.color_pair(4),
                            )
                            line += 1
                    else:
                        panel.addstr(
                            y + line, x + len(key_str), value_str, curses.color_pair(4)
                        )
                        line += 1
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if line >= max_height:
                    panel.addstr(y + line, x, f"{indent}...")
                    line += 1
                    break

                item_prefix = f"{indent}[{i}] "
                panel.addstr(y + line, x, item_prefix, curses.color_pair(3))

                if isinstance(item, (dict, list)) and item:
                    panel.addstr(y + line, x + len(item_prefix), "")
                    line += 1
                    line += self._render_json(
                        panel,
                        item,
                        y + line,
                        x,
                        max_width,
                        max_height - line,
                        depth + 1,
                    )
                else:
                    value_str = self._format_value(item)
                    if len(item_prefix) + len(value_str) > max_width:
                        panel.addstr(
                            y + line,
                            x + len(item_prefix),
                            value_str[: max_width - len(item_prefix)],
                            curses.color_pair(4),
                        )
                        wrapped = textwrap.wrap(
                            value_str[max_width - len(item_prefix) :],
                            max_width - len(indent) - 2,
                        )
                        line += 1
                        for wrap_line in wrapped:
                            if line >= max_height:
                                break
                            panel.addstr(
                                y + line,
                                x + len(indent) + 2,
                                wrap_line,
                                curses.color_pair(4),
                            )
                            line += 1
                    else:
                        panel.addstr(
                            y + line,
                            x + len(item_prefix),
                            value_str,
                            curses.color_pair(4),
                        )
                        line += 1
        else:
            value_str = self._format_value(data)
            panel.addstr(y + line, x, f"{indent}{value_str}", curses.color_pair(4))
            line += 1

        return line

    def _format_value(self, value):
        """Format a value for display."""
        if isinstance(value, str):
            if len(value) > 1000:  # Truncate very long strings
                return f'"{value[:997]}..."'
            return f'"{value}"'
        elif value is None:
            return "null"
        else:
            return str(value)


def main():
    """Main entry point."""
    import sys

    if len(sys.argv) > 1:
        start_dir = sys.argv[1]
    else:
        start_dir = "."

    explorer = FileExplorer(start_dir)
    curses.wrapper(explorer.run)


if __name__ == "__main__":
    main()
