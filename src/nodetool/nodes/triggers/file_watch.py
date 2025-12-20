"""
File Watch Trigger Node
=======================

This module provides a trigger that monitors filesystem changes using
the watchdog library. Changes to files or directories emit events.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import Field

from nodetool.nodes.triggers.base import TriggerEvent, TriggerNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


class FileWatchTrigger(TriggerNode):
    """
    Trigger node that monitors filesystem changes.

    This trigger uses the watchdog library to monitor a directory or file
    for changes. When a change is detected, an event is emitted containing:
    - The path of the changed file
    - The type of change (created, modified, deleted, moved)
    - Timestamp of the event

    This trigger is useful for:
    - Processing files as they arrive in a directory
    - Triggering workflows on configuration changes
    - Building file-based automation pipelines

    Example:
        Monitor /data/input for new files:
        - When a file is created, the workflow processes it
        - Each file triggers a separate workflow execution
    """

    path: str = Field(
        default=".",
        description="Path to watch (file or directory)",
    )
    recursive: bool = Field(
        default=False,
        description="Watch subdirectories recursively",
    )
    patterns: list[str] = Field(
        default=["*"],
        description="File patterns to watch (e.g., ['*.txt', '*.json'])",
    )
    ignore_patterns: list[str] = Field(
        default=[],
        description="File patterns to ignore",
    )
    events: list[str] = Field(
        default=["created", "modified", "deleted", "moved"],
        description="Types of events to watch for",
    )
    debounce_seconds: float = Field(
        default=0.5,
        description="Debounce time to avoid duplicate events",
        ge=0,
    )

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._observer = None
        self._last_events: dict[str, float] = {}

    async def setup_trigger(self, context: ProcessingContext) -> None:
        """Start the filesystem watcher."""
        try:
            from watchdog.observers import Observer
            from watchdog.events import (
                FileSystemEventHandler,
                FileCreatedEvent,
                FileModifiedEvent,
                FileDeletedEvent,
                FileMovedEvent,
                DirCreatedEvent,
                DirModifiedEvent,
                DirDeletedEvent,
                DirMovedEvent,
            )
            import fnmatch
        except ImportError:
            raise ImportError(
                "watchdog is required for FileWatchTrigger. "
                "Install it with: pip install watchdog"
            )

        watch_path = Path(self.path).expanduser().resolve()
        if not watch_path.exists():
            raise ValueError(f"Watch path does not exist: {watch_path}")

        log.info(f"Setting up file watch trigger on {watch_path}")

        trigger = self

        class EventHandler(FileSystemEventHandler):
            """Handler for filesystem events."""

            def _should_process(self, path: str) -> bool:
                """Check if the path matches the configured patterns."""
                name = Path(path).name

                # Check ignore patterns
                for pattern in trigger.ignore_patterns:
                    if fnmatch.fnmatch(name, pattern):
                        return False

                # Check include patterns
                for pattern in trigger.patterns:
                    if fnmatch.fnmatch(name, pattern):
                        return True

                return False

            def _debounce(self, path: str) -> bool:
                """Check if event should be debounced."""
                now = datetime.now(timezone.utc).timestamp()
                last = trigger._last_events.get(path, 0)

                if now - last < trigger.debounce_seconds:
                    return True

                trigger._last_events[path] = now
                return False

            def _emit_event(self, event_type: str, src_path: str, dest_path: str | None = None, is_directory: bool = False):
                """Emit a filesystem event."""
                if event_type not in trigger.events:
                    return

                if not self._should_process(src_path):
                    return

                if self._debounce(src_path):
                    return

                event: TriggerEvent = {
                    "data": {
                        "path": src_path,
                        "dest_path": dest_path,
                        "is_directory": is_directory,
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": str(watch_path),
                    "event_type": event_type,
                }

                log.debug(f"File event: {event_type} - {src_path}")
                trigger.push_event(event)

            def on_created(self, event):
                if isinstance(event, (FileCreatedEvent, DirCreatedEvent)):
                    self._emit_event("created", event.src_path, is_directory=isinstance(event, DirCreatedEvent))

            def on_modified(self, event):
                if isinstance(event, (FileModifiedEvent, DirModifiedEvent)):
                    self._emit_event("modified", event.src_path, is_directory=isinstance(event, DirModifiedEvent))

            def on_deleted(self, event):
                if isinstance(event, (FileDeletedEvent, DirDeletedEvent)):
                    self._emit_event("deleted", event.src_path, is_directory=isinstance(event, DirDeletedEvent))

            def on_moved(self, event):
                if isinstance(event, (FileMovedEvent, DirMovedEvent)):
                    self._emit_event("moved", event.src_path, event.dest_path, is_directory=isinstance(event, DirMovedEvent))

        # Create and start the observer
        self._observer = Observer()
        self._observer.schedule(
            EventHandler(),
            str(watch_path),
            recursive=self.recursive,
        )
        self._observer.start()
        log.info(f"File watcher started on {watch_path}")

    async def wait_for_event(self, context: ProcessingContext) -> TriggerEvent | None:
        """Wait for the next filesystem event."""
        return await self.get_event_from_queue()

    async def cleanup_trigger(self, context: ProcessingContext) -> None:
        """Stop the filesystem watcher."""
        log.info("Cleaning up file watch trigger")

        if self._observer:
            try:
                self._observer.stop()
                self._observer.join(timeout=2.0)
            except Exception as e:
                log.warning(f"Error stopping observer: {e}")
            finally:
                self._observer = None
