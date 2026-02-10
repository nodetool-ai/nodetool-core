"""
Migration: Add encrypted_content column to messages table
Version: 20260126_000000

This migration adds support for encrypting chat message content at rest.
The encrypted_content column stores the encrypted version of the content field.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodetool.migrations.db_adapter import MigrationDBAdapter

version = "20260126_000000"
name = "add_encrypted_content_to_messages"

creates_tables = []
modifies_tables = ["nodetool_messages"]


async def up(db: "MigrationDBAdapter") -> None:
    """Add encrypted_content column to messages table."""
    await db.execute("""
        ALTER TABLE nodetool_messages
        ADD COLUMN encrypted_content TEXT
    """)


async def down(db: "MigrationDBAdapter") -> None:
    """Remove encrypted_content column from messages table."""
    # SQLite doesn't support DROP COLUMN directly in older versions,
    # but modern SQLite (3.35.0+) does support it
    await db.execute("""
        ALTER TABLE nodetool_messages
        DROP COLUMN encrypted_content
    """)
