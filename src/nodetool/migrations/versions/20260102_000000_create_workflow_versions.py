"""
Migration: Create workflow_versions table
Version: 20260102_000000
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodetool.migrations.db_adapter import MigrationDBAdapter

version = "20260102_000000"
name = "create_workflow_versions"

creates_tables = ["nodetool_workflow_versions"]
modifies_tables = []


async def up(db: "MigrationDBAdapter") -> None:
    """Create the workflow_versions table for storing workflow version history."""
    await db.execute("""
        CREATE TABLE IF NOT EXISTS nodetool_workflow_versions (
            id TEXT PRIMARY KEY,
            workflow_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            name TEXT DEFAULT '',
            description TEXT DEFAULT '',
            graph TEXT DEFAULT '{}'
        )
    """)

    # Index for efficient lookup by workflow_id
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_nodetool_workflow_versions_workflow_id
        ON nodetool_workflow_versions (workflow_id)
    """)

    # Unique constraint to ensure version numbers are unique per workflow
    await db.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_nodetool_workflow_versions_workflow_version
        ON nodetool_workflow_versions (workflow_id, version)
    """)


async def down(db: "MigrationDBAdapter") -> None:
    """Drop the workflow_versions table."""
    await db.execute("DROP INDEX IF EXISTS idx_nodetool_workflow_versions_workflow_id")
    await db.execute("DROP INDEX IF EXISTS idx_nodetool_workflow_versions_workflow_version")
    await db.execute("DROP TABLE IF EXISTS nodetool_workflow_versions")
