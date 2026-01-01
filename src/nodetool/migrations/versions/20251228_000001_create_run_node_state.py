"""
Migration: Create run_node_state table
Version: 20251228_000001

Authoritative source of truth for per-node execution state.
Replaces event log projection as source of correctness.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodetool.migrations.db_adapter import MigrationDBAdapter

version = "20251228_000001"
name = "create_run_node_state"

creates_tables = ["run_node_state"]
modifies_tables = []


async def up(db: "MigrationDBAdapter") -> None:
    """Create the run_node_state table."""
    await db.execute("""
        CREATE TABLE IF NOT EXISTS run_node_state (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            node_id TEXT NOT NULL,

            status TEXT NOT NULL,
            attempt INTEGER NOT NULL DEFAULT 1,

            scheduled_at TEXT,
            started_at TEXT,
            completed_at TEXT,
            failed_at TEXT,
            suspended_at TEXT,
            updated_at TEXT NOT NULL,

            last_error TEXT,
            retryable INTEGER NOT NULL DEFAULT 0,

            suspension_reason TEXT,
            resume_state_json TEXT,

            outputs_json TEXT
        )
    """)

    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_run_node_state_run_status
        ON run_node_state(run_id, status)
    """)

    await db.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_run_node_state_run_node
        ON run_node_state(run_id, node_id)
    """)


async def down(db: "MigrationDBAdapter") -> None:
    """Drop the run_node_state table."""
    await db.execute("DROP INDEX IF EXISTS idx_run_node_state_run_status")
    await db.execute("DROP INDEX IF EXISTS idx_run_node_state_run_node")
    await db.execute("DROP TABLE IF EXISTS run_node_state")
