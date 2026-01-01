"""
Migration: Create trigger_inputs table
Version: 20251228_000003

Durable storage for trigger events that wake up workflows.
Provides idempotent delivery and cross-process coordination.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodetool.migrations.db_adapter import MigrationDBAdapter

version = "20251228_000003"
name = "create_trigger_inputs"

creates_tables = ["trigger_inputs"]
modifies_tables = []


async def up(db: "MigrationDBAdapter") -> None:
    """Create the trigger_inputs table."""
    await db.execute("""
        CREATE TABLE IF NOT EXISTS trigger_inputs (
            id TEXT PRIMARY KEY,
            
            input_id TEXT NOT NULL UNIQUE,
            run_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            
            payload_json TEXT,
            
            processed INTEGER NOT NULL DEFAULT 0,
            processed_at TEXT,
            
            cursor TEXT,
            
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_trigger_input_run_node_processed 
        ON trigger_inputs(run_id, node_id, processed)
    """)

    await db.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_trigger_input_id 
        ON trigger_inputs(input_id)
    """)


async def down(db: "MigrationDBAdapter") -> None:
    """Drop the trigger_inputs table."""
    await db.execute("DROP INDEX IF EXISTS idx_trigger_input_run_node_processed")
    await db.execute("DROP INDEX IF EXISTS idx_trigger_input_id")
    await db.execute("DROP TABLE IF EXISTS trigger_inputs")
