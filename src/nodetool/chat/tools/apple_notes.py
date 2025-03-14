"""
Apple Notes tools module.

This module provides tools for working with Apple Notes:
- CreateAppleNoteTool: Create notes in Apple Notes
- ReadAppleNotesTool: Read from Apple Notes
"""

from typing import Any

from nodetool.workflows.processing_context import ProcessingContext
from .base import Tool


class CreateAppleNoteTool(Tool):
    def __init__(self):
        super().__init__(
            name="create_apple_note",
            description="Create a new note in Apple Notes on macOS",
        )
        self.input_schema = {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Title of the note",
                },
                "body": {
                    "type": "string",
                    "description": "Content of the note",
                },
                "folder": {
                    "type": "string",
                    "description": "Notes folder to save to (defaults to 'Notes')",
                    "default": "Notes",
                },
            },
            "required": ["title", "body"],
        }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        try:
            from nodetool.nodes.apple.notes import CreateNote  # type: ignore

            create_note = CreateNote(
                title=params["title"],
                body=params["body"],
                folder=params.get("folder", "Notes"),
            )
            await create_note.process(context)
            return {
                "success": True,
                "message": f"Note '{params['title']}' created in folder '{params.get('folder', 'Notes')}'",
            }
        except Exception as e:
            return {"error": str(e)}


class ReadAppleNotesTool(Tool):
    def __init__(self):
        super().__init__(
            name="read_apple_notes",
            description="Read notes from Apple Notes on macOS",
        )
        self.input_schema = {
            "type": "object",
            "properties": {
                "note_limit": {
                    "type": "integer",
                    "description": "Maximum number of notes to read (0 for unlimited)",
                    "default": 10,
                },
                "note_limit_per_folder": {
                    "type": "integer",
                    "description": "Maximum notes per folder (0 for unlimited)",
                    "default": 10,
                },
            },
        }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        try:
            from nodetool.nodes.apple.notes import ReadNotes  # type: ignore

            read_notes = ReadNotes(
                note_limit=params.get("note_limit", 10),
                note_limit_per_folder=params.get("note_limit_per_folder", 10),
            )
            notes = await read_notes.process(context)
            return {
                "notes": notes,
                "count": len(notes),
            }
        except Exception as e:
            return {"error": str(e)}
