from __future__ import annotations

import sys
from typing import Any, Dict, Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)


class ProgressManager:
    """Manages persistent progress bars for different operations."""

    def __init__(self, console: Console | None = None):
        self.console: Console = console or Console()
        self.progress: Progress | None = None
        self.tasks: dict[str, TaskID] = {}  # Maps operation IDs to task IDs
        self.current_operations: dict[str, dict[str, Any]] = {}  # operation info

    def start(self) -> None:
        """Start the progress display."""
        if self.progress is None:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=self.console,
                transient=False,
            )
            self.progress.start()

    def stop(self) -> None:
        """Stop the progress display."""
        if self.progress:
            self.progress.stop()
            self.progress = None
            self.tasks.clear()
            self.current_operations.clear()

    def add_task(self, operation_id: str, description: str, total: float | None = None) -> int:
        """Add a new progress task."""
        self.start()
        assert self.progress is not None
        if operation_id not in self.tasks:
            task_id = self.progress.add_task(description, total=total)
            self.tasks[operation_id] = task_id
            self.current_operations[operation_id] = {
                "description": description,
                "total": total,
                "completed": 0,
            }
        return self.tasks[operation_id]

    def update_task(
        self,
        operation_id: str,
        completed: float | None = None,
        description: str | None = None,
    ) -> None:
        """Update a progress task."""
        if operation_id in self.tasks and self.progress:
            assert self.progress is not None
            task_id = self.tasks[operation_id]
            update_kwargs: dict[str, Any] = {}

            if completed is not None:
                # Calculate advance amount
                current_completed = self.current_operations[operation_id]["completed"]
                advance = completed - current_completed
                if advance > 0:
                    update_kwargs["advance"] = advance
                self.current_operations[operation_id]["completed"] = completed

            if description is not None:
                update_kwargs["description"] = description
                self.current_operations[operation_id]["description"] = description

            if update_kwargs:
                self.progress.update(task_id, **update_kwargs)

    def complete_task(self, operation_id: str) -> None:
        """Mark a task as completed."""
        if operation_id in self.tasks and self.progress:
            assert self.progress is not None
            task_id = self.tasks[operation_id]
            op_info = self.current_operations[operation_id]
            if op_info["total"]:
                # Complete the progress bar
                self.progress.update(task_id, completed=op_info["total"])
            # Remove the task after a brief moment
            self.progress.remove_task(task_id)
            del self.tasks[operation_id]
            del self.current_operations[operation_id]

            # If no more tasks, stop the progress display
            if not self.tasks:
                self.stop()

    def remove_task(self, operation_id: str) -> None:
        """Remove a task without completing it."""
        if operation_id in self.tasks and self.progress:
            assert self.progress is not None
            task_id = self.tasks[operation_id]
            self.progress.remove_task(task_id)
            del self.tasks[operation_id]
            del self.current_operations[operation_id]

            # If no more tasks, stop the progress display
            if not self.tasks:
                self.stop()

    def _display_progress_update(self, progress_update):
        """Shared function to display progress updates using Rich progress bars."""
        status = progress_update.get("status", "unknown")
        message = progress_update.get("message", "")

        if status == "starting":
            self.console.print(f"[blue]🚀 {message}[/]")

        elif status == "progress":
            # Handle different types of progress updates with proper progress bars

            if "current_file" in progress_update:
                current_file = progress_update["current_file"]

                if "file_progress" in progress_update and "total_files" in progress_update:
                    # File-based progress with progress bar
                    file_num = progress_update["file_progress"]
                    total_files = progress_update["total_files"]
                    operation_id = f"files_{current_file}"

                    description = f"📁 Downloading files ({file_num}/{total_files}): {current_file}"

                    # Add or update file progress task
                    if operation_id not in self.tasks:
                        self.add_task(operation_id, description, total=total_files)
                    self.update_task(operation_id, completed=file_num, description=description)
                else:
                    # Single file progress without known total
                    self.console.print(f"[yellow]📁 {current_file}[/]")

            # Handle download progress with size information
            if "downloaded_size" in progress_update and "total_size" in progress_update:
                downloaded = progress_update["downloaded_size"]
                total = progress_update["total_size"]

                if total > 0:
                    # Create a unique operation ID for this download
                    operation_id = progress_update.get("operation_id", "download")
                    current_file = progress_update.get("current_file", "")

                    downloaded_mb = downloaded / (1024 * 1024)
                    total_mb = total / (1024 * 1024)

                    description = "📊 Downloading"
                    if current_file:
                        description += f" {current_file}"
                    description += f" ({downloaded_mb:.1f}/{total_mb:.1f} MB)"

                    # Add or update download progress task
                    if operation_id not in self.tasks:
                        self.add_task(operation_id, description, total=total)
                    self.update_task(operation_id, completed=downloaded, description=description)

            # Handle general progress messages
            if not any(key in progress_update for key in ["current_file", "downloaded_size"]):
                self.console.print(f"[yellow]⚙️ {message}[/]")

        elif status == "completed":
            self.console.print(f"[green]✅ {message}[/]")

            # Complete any active progress tasks related to downloads/files
            for operation_id in list(self.tasks.keys()):
                if "download" in operation_id or "files_" in operation_id:
                    self.complete_task(operation_id)

            if "downloaded_files" in progress_update:
                self.console.print(f"[green]📋 Downloaded {progress_update['downloaded_files']} files[/]")

        elif status.startswith("pulling"):
            # Handle Docker/Ollama pulling status with progress bars
            digest = progress_update.get("digest", "")
            total = progress_update.get("total")
            completed = progress_update.get("completed")

            # Extract the layer ID from status (e.g., "pulling aeda25e63ebd")
            layer_id = status.replace("pulling ", "") if " " in status else "unknown"
            operation_id = f"pull_{layer_id}"

            # Create description with layer info
            description = f"🐋 Pulling layer {layer_id}"
            if digest and "sha256:" in digest:
                short_digest = digest.split(":")[-1][:12] if ":" in digest else digest[:12]
                description += f" (sha256:{short_digest})"

            # Show progress with progress bar if size information is available
            if total and completed is not None:
                total_mb = total / (1024 * 1024)
                completed_mb = completed / (1024 * 1024)
                description += f" ({completed_mb:.1f}/{total_mb:.1f} MB)"

                # Add or update pulling progress task
                if operation_id not in self.tasks:
                    self.add_task(operation_id, description, total=total)
                self.update_task(operation_id, completed=completed, description=description)
            elif total:
                # Just show size without progress bar
                total_mb = total / (1024 * 1024)
                description += f" ({total_mb:.1f} MB)"
                self.console.print(f"[yellow]{description}[/]")
            else:
                # No size info available
                self.console.print(f"[yellow]{description}[/]")

        elif status == "error":
            error = progress_update.get("error", "Unknown error")
            self.console.print(f"[red]❌ Error: {error}[/]")

            # Stop any active progress bars on error
            self.stop()
            sys.exit(1)

        elif status == "healthy":
            self.console.print("[green]✅ System is healthy[/]")

            # Display system information for health checks
            self.console.print(f"[cyan]🖥️ Platform: {progress_update.get('platform', 'Unknown')}[/]")
            self.console.print(f"[cyan]🐍 Python: {progress_update.get('python_version', 'Unknown')}[/]")
            self.console.print(f"[cyan]🏠 Hostname: {progress_update.get('hostname', 'Unknown')}[/]")

            # Memory info
            memory = progress_update.get("memory", {})
            if isinstance(memory, dict):
                self.console.print(
                    f"[cyan]💾 Memory: {memory.get('available_gb', 0):.1f}GB available / {memory.get('total_gb', 0):.1f}GB total ({memory.get('used_percent', 0)}% used)[/]"
                )

            # Disk info
            disk = progress_update.get("disk", {})
            if isinstance(disk, dict):
                self.console.print(
                    f"[cyan]💿 Disk: {disk.get('free_gb', 0):.1f}GB free / {disk.get('total_gb', 0):.1f}GB total ({disk.get('used_percent', 0)}% used)[/]"
                )

            # GPU info
            gpus = progress_update.get("gpus", [])
            if isinstance(gpus, list) and gpus:
                self.console.print("[cyan]🎮 GPUs:[/]")
                for i, gpu in enumerate(gpus):
                    name = gpu.get("name", "Unknown")
                    used_mb = gpu.get("memory_used_mb", 0)
                    total_mb = gpu.get("memory_total_mb", 0)
                    used_pct = (used_mb / total_mb * 100) if total_mb > 0 else 0
                    self.console.print(f"[cyan]  GPU {i}: {name} - {used_mb}MB/{total_mb}MB ({used_pct:.1f}% used)[/]")
            elif gpus == "unavailable":
                self.console.print("[yellow]🎮 GPUs: Not available[/]")
