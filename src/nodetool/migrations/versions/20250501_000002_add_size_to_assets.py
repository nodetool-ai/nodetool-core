"""
Migration: Add size column to assets
Version: 20250501_000002
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodetool.migrations.db_adapter import MigrationDBAdapter

version = "20250501_000002"
name = "add_size_to_assets"

creates_tables = []
modifies_tables = ["nodetool_assets"]


async def up(db: "MigrationDBAdapter") -> None:
    """Add size column to assets table."""
    columns = await db.get_columns("nodetool_assets")

    if "size" not in columns:
        await db.execute("""
            ALTER TABLE nodetool_assets ADD COLUMN size INTEGER
        """)


async def down(db: "MigrationDBAdapter") -> None:
    """Remove size column from assets table.

    Note: SQLite doesn't support DROP COLUMN directly in older versions.
    This is a no-op for safety.
    """
    pass
