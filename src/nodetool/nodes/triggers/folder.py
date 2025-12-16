"""
Folder Watch Trigger Node
=========================

This module provides the FolderWatchTrigger node that monitors a folder
for file system changes and triggers workflow execution when changes occur.

The folder watch trigger:
1. Monitors a specified directory path for changes
2. Detects file creation, modification, deletion, and moves
3. Emits file event data to downstream nodes

Usage:
    Configure the folder path to watch and optionally specify which types
    of events to trigger on. The trigger will emit FileEvent objects for
    each detected change.
"""

from datetime import datetime
from enum import Enum
from typing import Literal, TypedDict

from pydantic import Field

from nodetool.metadata.types import AssetRef, BaseType, Datetime, FolderPath
from nodetool.nodes.triggers.base import TriggerNode
from nodetool.workflows.processing_context import ProcessingContext


class FileChangeType(str, Enum):
    """Types of file system changes that can be detected."""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"


class FileEvent(BaseType):
    """
    Represents a file system change event.
    
    Attributes:
        file_path: The path to the affected file
        change_type: The type of change (created, modified, deleted, moved)
        old_path: For moved files, the original path
        file_name: The name of the affected file
        is_directory: Whether the affected item is a directory
    """
    type: Literal["file_event"] = "file_event"
    timestamp: Datetime = Field(default_factory=Datetime)
    file_path: str = Field(default="", description="Path to the affected file")
    change_type: FileChangeType = Field(
        default=FileChangeType.CREATED,
        description="Type of file system change"
    )
    old_path: str = Field(default="", description="Original path for moved files")
    file_name: str = Field(default="", description="Name of the affected file")
    is_directory: bool = Field(default=False, description="Whether the item is a directory")


class FolderWatchTrigger(TriggerNode):
    """
    Trigger node that monitors a folder for file system changes.
    
    This node watches a specified directory and triggers when files are
    created, modified, deleted, or moved within that directory. It's useful
    for building file-driven workflows like automatic processing of new files.
    
    folder, file, watch, filesystem, directory, trigger, event
    
    Attributes:
        folder_path: The path to the folder to monitor
        recursive: Whether to monitor subdirectories
        patterns: File patterns to match (e.g., ["*.jpg", "*.png"])
        watch_created: Trigger on file creation
        watch_modified: Trigger on file modification
        watch_deleted: Trigger on file deletion
        watch_moved: Trigger on file moves/renames
    """
    
    folder_path: FolderPath = Field(
        default_factory=FolderPath,
        description="Path to the folder to monitor"
    )
    recursive: bool = Field(
        default=False,
        description="Watch subdirectories recursively"
    )
    patterns: list[str] = Field(
        default=["*"],
        description="File patterns to match (e.g., ['*.jpg', '*.png'])"
    )
    watch_created: bool = Field(
        default=True,
        description="Trigger when files are created"
    )
    watch_modified: bool = Field(
        default=True,
        description="Trigger when files are modified"
    )
    watch_deleted: bool = Field(
        default=False,
        description="Trigger when files are deleted"
    )
    watch_moved: bool = Field(
        default=False,
        description="Trigger when files are moved or renamed"
    )
    
    # Input fields populated when triggered
    file_path: str = Field(default="", description="Path to the changed file")
    file_name: str = Field(default="", description="Name of the changed file")
    change_type: FileChangeType = Field(
        default=FileChangeType.CREATED,
        description="Type of change detected"
    )
    old_path: str = Field(default="", description="Original path for moved files")
    is_directory: bool = Field(default=False, description="Whether the item is a directory")

    class OutputType(TypedDict):
        event: FileEvent
        file: AssetRef

    async def process(self, context: ProcessingContext) -> OutputType:
        """
        Process the folder watch trigger and emit the event data.
        
        The input fields are expected to be populated by the workflow runner
        when a file system change is detected.
        """
        event = FileEvent(
            timestamp=Datetime.from_datetime(datetime.now()),
            file_path=self.file_path,
            change_type=self.change_type,
            old_path=self.old_path,
            file_name=self.file_name,
            is_directory=self.is_directory,
        )
        
        # Create a file reference for the affected file
        file_ref = AssetRef.from_file(self.file_path) if self.file_path else AssetRef()
        
        return {"event": event, "file": file_ref}
