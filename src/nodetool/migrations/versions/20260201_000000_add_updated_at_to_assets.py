"""
Migration: Add updated_at column to assets table
Version: 20260201_000000
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodetool.migrations.db_adapter import MigrationDBAdapter

version = "20260201_000000"
name = "add_updated_at_to_assets"

creates_tables = []
modifies_tables = ["nodetool_assets"]


async def up(db: "MigrationDBAdapter") -> None:
    """Add updated_at column to assets table."""
    columns = await db.get_columns("nodetool_assets")

    if "updated_at" not in columns:
        await db.execute("""
            ALTER TABLE nodetool_assets ADD COLUMN updated_at TEXT
        """)
        # Set updated_at to created_at for existing rows
        await db.execute("""
            UPDATE nodetool_assets SET updated_at = created_at WHERE updated_at IS NULL
        """)


async def down(db: "MigrationDBAdapter") -> None:
    """Remove updated_at column from assets table.

    Note: SQLite doesn't support DROP COLUMN directly in older versions.
    This is a no-op for safety.
    """
    pass
