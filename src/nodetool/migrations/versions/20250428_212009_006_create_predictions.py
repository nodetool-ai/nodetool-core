"""
Migration: Create predictions table
Version: 20250428_212009_006
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodetool.migrations.db_adapter import MigrationDBAdapter

version = "20250428_212009_006"
name = "create_predictions"

creates_tables = ["nodetool_predictions"]
modifies_tables = []


async def up(db: "MigrationDBAdapter") -> None:
    """Create the predictions table."""
    await db.execute("""
        CREATE TABLE IF NOT EXISTS nodetool_predictions (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            node_id TEXT,
            provider TEXT,
            model TEXT,
            workflow_id TEXT,
            error TEXT,
            logs TEXT,
            status TEXT,
            created_at TEXT,
            started_at TEXT,
            completed_at TEXT,
            cost REAL,
            duration REAL,
            hardware TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER
        )
    """)


async def down(db: "MigrationDBAdapter") -> None:
    """Drop the predictions table."""
    await db.execute("DROP TABLE IF EXISTS nodetool_predictions")
