"""
Migration: Create run_state table
Version: 20251228_000000

Authoritative source of truth for workflow run status.
Replaces event log as source of correctness.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodetool.migrations.db_adapter import MigrationDBAdapter

version = "20251228_000000"
name = "create_run_state"

creates_tables = ["run_state"]
modifies_tables = []


async def up(db: "MigrationDBAdapter") -> None:
    """Create the run_state table."""
    await db.execute("""
        CREATE TABLE IF NOT EXISTS run_state (
            run_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,

            suspended_node_id TEXT,
            suspension_reason TEXT,
            suspension_state_json TEXT,
            suspension_metadata_json TEXT,

            completed_at TEXT,
            failed_at TEXT,
            error_message TEXT,

            version INTEGER NOT NULL DEFAULT 0
        )
    """)

    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_run_state_status
        ON run_state(status)
    """)

    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_run_state_updated
        ON run_state(updated_at)
    """)


async def down(db: "MigrationDBAdapter") -> None:
    """Drop the run_state table."""
    await db.execute("DROP INDEX IF EXISTS idx_run_state_status")
    await db.execute("DROP INDEX IF EXISTS idx_run_state_updated")
    await db.execute("DROP TABLE IF EXISTS run_state")
