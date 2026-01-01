"""
Migration: Create assets table
Version: 20250428_212009_002
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodetool.migrations.db_adapter import MigrationDBAdapter

version = "20250428_212009_002"
name = "create_assets"

creates_tables = ["nodetool_assets"]
modifies_tables = []


async def up(db: "MigrationDBAdapter") -> None:
    """Create the assets table."""
    await db.execute("""
        CREATE TABLE IF NOT EXISTS nodetool_assets (
            id TEXT PRIMARY KEY,
            type TEXT,
            user_id TEXT,
            workflow_id TEXT,
            parent_id TEXT,
            file_id TEXT,
            name TEXT,
            content_type TEXT,
            metadata TEXT,
            created_at TEXT,
            duration REAL
        )
    """)

    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_nodetool_assets_user_id_parent_id
        ON nodetool_assets (user_id, parent_id)
    """)


async def down(db: "MigrationDBAdapter") -> None:
    """Drop the assets table."""
    await db.execute("DROP INDEX IF EXISTS idx_nodetool_assets_user_id_parent_id")
    await db.execute("DROP TABLE IF EXISTS nodetool_assets")
