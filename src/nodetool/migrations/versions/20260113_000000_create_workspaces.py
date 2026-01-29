"""
Migration: Create workspaces table and add workspace_id to workflows
Version: 20260113_000000
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodetool.migrations.db_adapter import MigrationDBAdapter

version = "20260113_000000"
name = "create_workspaces"

creates_tables = ["nodetool_workspaces"]
modifies_tables = ["nodetool_workflows"]


async def up(db: "MigrationDBAdapter") -> None:
    """Create the workspaces table and add workspace_id to workflows."""
    # Create the workspaces table
    await db.execute("""
        CREATE TABLE IF NOT EXISTS nodetool_workspaces (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            path TEXT NOT NULL,
            is_default INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    # Create index on user_id
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_nodetool_workspaces_user_id
        ON nodetool_workspaces (user_id)
    """)

    # Add workspace_id column to workflows table
    columns = await db.get_columns("nodetool_workflows")
    if "workspace_id" not in columns:
        await db.execute("""
            ALTER TABLE nodetool_workflows ADD COLUMN workspace_id TEXT
        """)


async def down(db: "MigrationDBAdapter") -> None:
    """Drop the workspaces table and remove workspace_id from workflows.

    Note: SQLite doesn't support DROP COLUMN directly in older versions.
    The workspace_id column removal is a no-op for safety.
    """
    await db.execute("DROP INDEX IF EXISTS idx_nodetool_workspaces_user_id")
    await db.execute("DROP TABLE IF EXISTS nodetool_workspaces")
