"""
Migration: Create run_events table
Version: 20260103_000000

Append-only event log for workflow run audit trail.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodetool.migrations.db_adapter import MigrationDBAdapter

version = "20260103_000000"
name = "create_run_events"

creates_tables = ["run_events"]
modifies_tables = []


async def up(db: "MigrationDBAdapter") -> None:
    """Create the run_events table."""
    await db.execute("""
        CREATE TABLE IF NOT EXISTS run_events (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            seq INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            event_time TEXT NOT NULL,
            node_id TEXT,
            payload TEXT
        )
    """)

    await db.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_run_events_run_seq
        ON run_events(run_id, seq)
    """)

    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_run_events_run_node
        ON run_events(run_id, node_id)
    """)

    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_run_events_run_type
        ON run_events(run_id, event_type)
    """)


async def down(db: "MigrationDBAdapter") -> None:
    """Drop the run_events table."""
    await db.execute("DROP INDEX IF EXISTS idx_run_events_run_seq")
    await db.execute("DROP INDEX IF EXISTS idx_run_events_run_node")
    await db.execute("DROP INDEX IF EXISTS idx_run_events_run_type")
    await db.execute("DROP TABLE IF EXISTS run_events")
