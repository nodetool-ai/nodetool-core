#!/usr/bin/env python3
import curses
import json
import os
from pathlib import Path
import textwrap


class TraceExplorer:
    """Text UI for exploring log trace files."""

    def __init__(self, traces_dir: str):
        self.traces_dir = Path(traces_dir)
        self.trace_files = []
        self.current_file_idx = 0
        self.current_entry_idx = 0
        self.entries = []
        self.top_file_idx = 0
        self.top_entry_idx = 0
        self.active_panel = 0  # 0: files, 1: events, 2: payload

    def load_trace_files(self):
        """Load all available trace files."""
        if not self.traces_dir.exists():
            return

        self.trace_files = sorted(
            [f for f in self.traces_dir.glob("*.jsonl")],
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )

    def load_trace_entries(self, file_idx):
        """Load all entries from the selected trace file."""
        if not self.trace_files or file_idx >= len(self.trace_files):
            self.entries = []
            return

        self.entries = []
        with open(self.trace_files[file_idx], "r") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    self.entries.append(entry)
                except json.JSONDecodeError:
                    pass

    def run(self, stdscr):
        """Run the UI main loop."""
        curses.curs_set(0)  # Hide cursor
        stdscr.clear()

        # Load trace files
        self.load_trace_files()

        # Set up color pairs
        curses.start_color()
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)  # Selected item
        curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)  # Header
        curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Key
        curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Value
        curses.init_pair(5, curses.COLOR_CYAN, curses.COLOR_BLACK)  # Event type
        curses.init_pair(6, curses.COLOR_BLACK, curses.COLOR_GREEN)  # Active panel

        # Main loop
        while True:
            # Get screen dimensions
            height, width = stdscr.getmaxyx()

            # Calculate panel dimensions
            files_width = max(width // 5, 25)
            events_width = max(width // 5, 25)
            payload_width = width - files_width - events_width

            # Create panels
            files_panel = stdscr.derwin(height, files_width, 0, 0)
            events_panel = stdscr.derwin(height, events_width, 0, files_width)
            payload_panel = stdscr.derwin(
                height, payload_width, 0, files_width + events_width
            )

            # Draw panels
            self._draw_file_list(files_panel, height, files_width)
            self._draw_events_list(events_panel, height, events_width)
            self._draw_payload(payload_panel, height, payload_width)

            # Refresh panels
            files_panel.refresh()
            events_panel.refresh()
            payload_panel.refresh()

            # Handle input
            key = stdscr.getch()
            if key == ord("q"):
                break
            elif key == ord("\t") or key == 9:  # Tab key
                self.active_panel = (self.active_panel + 1) % 3
            elif key == curses.KEY_UP:
                if self.active_panel == 0 and self.current_file_idx > 0:
                    self.current_file_idx -= 1
                    self.current_entry_idx = 0
                    self.top_entry_idx = 0
                    self.load_trace_entries(self.current_file_idx)
                elif self.active_panel == 1 and self.current_entry_idx > 0:
                    self.current_entry_idx -= 1
                    if self.current_entry_idx < self.top_entry_idx:
                        self.top_entry_idx -= 1
            elif key == curses.KEY_DOWN:
                if (
                    self.active_panel == 0
                    and self.current_file_idx < len(self.trace_files) - 1
                ):
                    self.current_file_idx += 1
                    self.current_entry_idx = 0
                    self.top_entry_idx = 0
                    self.load_trace_entries(self.current_file_idx)
                elif (
                    self.active_panel == 1
                    and self.current_entry_idx < len(self.entries) - 1
                ):
                    self.current_entry_idx += 1
                    if self.current_entry_idx >= self.top_entry_idx + height - 4:
                        self.top_entry_idx += 1
            elif key == ord("n") and self.active_panel == 1:
                if self.current_entry_idx < len(self.entries) - 1:
                    self.current_entry_idx += 1
                    if self.current_entry_idx >= self.top_entry_idx + height - 4:
                        self.top_entry_idx += 1
            elif key == ord("p") and self.active_panel == 1:
                if self.current_entry_idx > 0:
                    self.current_entry_idx -= 1
                    if self.current_entry_idx < self.top_entry_idx:
                        self.top_entry_idx -= 1
            elif key == ord("r"):
                self.load_trace_files()
                if self.trace_files:
                    self.load_trace_entries(self.current_file_idx)

    def _draw_file_list(self, panel, height, width):
        """Draw the file list panel."""
        panel.clear()
        panel.box()

        # Draw header with highlight if active
        header_style = (
            curses.color_pair(6) if self.active_panel == 0 else curses.color_pair(2)
        )
        panel.addstr(0, 2, " Trace Files ", header_style)

        # Draw help text
        help_text = "[Tab]:Switch Panel [↑/↓]:Navigate [r]:Reload [q]:Quit"
        panel.addstr(height - 1, 2, help_text[: width - 4])

        # Draw files
        max_files = height - 4

        if self.current_file_idx < self.top_file_idx:
            self.top_file_idx = self.current_file_idx
        elif self.current_file_idx >= self.top_file_idx + max_files:
            self.top_file_idx = self.current_file_idx - max_files + 1

        for i in range(min(max_files, len(self.trace_files))):
            file_idx = i + self.top_file_idx
            if file_idx < len(self.trace_files):
                file_name = self.trace_files[file_idx].name
                line_attr = (
                    curses.color_pair(1) if file_idx == self.current_file_idx else 0
                )

                # Truncate if necessary
                display_name = file_name
                if len(display_name) > width - 6:
                    display_name = display_name[: width - 9] + "..."

                panel.addstr(i + 2, 2, display_name, line_attr)

    def _draw_events_list(self, panel, height, width):
        """Draw the events list panel."""
        panel.clear()
        panel.box()

        # Draw header with highlight if active
        header_style = (
            curses.color_pair(6) if self.active_panel == 1 else curses.color_pair(2)
        )
        panel.addstr(0, 2, " Events ", header_style)

        # If no files or entries, show message
        if not self.trace_files:
            panel.addstr(2, 2, "No trace files found.")
            return

        if not self.entries:
            self.load_trace_entries(self.current_file_idx)
            if not self.entries:
                panel.addstr(2, 2, "No entries in this trace file.")
                return

        # Show event count
        panel.addstr(1, 2, f"Events: {len(self.entries)}")

        # Display events list
        max_events = height - 4

        if self.current_entry_idx < self.top_entry_idx:
            self.top_entry_idx = self.current_entry_idx
        elif self.current_entry_idx >= self.top_entry_idx + max_events:
            self.top_entry_idx = self.current_entry_idx - max_events + 1

        for i in range(min(max_events, len(self.entries))):
            entry_idx = i + self.top_entry_idx
            if entry_idx < len(self.entries):
                entry = self.entries[entry_idx]
                event_type = entry.get("event", "unknown")
                line_attr = (
                    curses.color_pair(1) if entry_idx == self.current_entry_idx else 0
                )

                # Truncate if necessary
                display_event = event_type
                if len(display_event) > width - 6:
                    display_event = display_event[: width - 9] + "..."

                panel.addstr(i + 2, 2, display_event, line_attr)

    def _draw_payload(self, panel, height, width):
        """Draw the event payload panel."""
        panel.clear()
        panel.box()

        # Draw header with highlight if active
        header_style = (
            curses.color_pair(6) if self.active_panel == 2 else curses.color_pair(2)
        )
        panel.addstr(0, 2, " Event Payload ", header_style)

        # If no files or entries, show message
        if not self.trace_files:
            panel.addstr(2, 2, "No trace files found.")
            return

        if not self.entries:
            self.load_trace_entries(self.current_file_idx)
            if not self.entries:
                panel.addstr(2, 2, "No entries in this trace file.")
                return

        # Show current entry
        if self.current_entry_idx < len(self.entries):
            entry = self.entries[self.current_entry_idx]

            # Display entry index and timestamp
            panel.addstr(
                1, 2, f"Entry {self.current_entry_idx + 1}/{len(self.entries)}"
            )

            if "timestamp" in entry:
                timestamp = entry["timestamp"]
                timestamp_str = f"Timestamp: {timestamp:.2f}"
                if len(timestamp_str) < width - 4:
                    panel.addstr(1, width - len(timestamp_str) - 2, timestamp_str)

            # Display event type
            if "event" in entry:
                event_type = entry["event"]
                panel.addstr(2, 2, "Event: ", curses.A_BOLD)
                panel.addstr(2, 9, event_type, curses.color_pair(5) | curses.A_BOLD)

            # Pretty print the entry data
            if "data" in entry:
                data = entry["data"]
                y = 4

                # Render the data as a pretty-printed JSON with indentation
                self._render_json(panel, data, y, 2, width - 4, height - 6)

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
    # Create and run the explorer
    import sys

    if len(sys.argv) > 1:
        traces_dir = sys.argv[1]
    else:
        traces_dir = "./traces"

    explorer = TraceExplorer(traces_dir)
    curses.wrapper(explorer.run)


if __name__ == "__main__":
    main()
