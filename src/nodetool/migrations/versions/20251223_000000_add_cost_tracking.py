"""
Migration: Add cost tracking fields to predictions
Version: 20251223_000000
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

version = "20251223_000000"
name = "add_cost_tracking_to_predictions"

# Tables this migration modifies
creates_tables = []
modifies_tables = ["nodetool_predictions"]


async def up(db: "aiosqlite.Connection") -> None:
    """Add cost tracking fields to predictions table."""
    cursor = await db.execute("PRAGMA table_info(nodetool_predictions)")
    columns = await cursor.fetchall()
    column_names = [col[1] for col in columns]

    # Add new columns if they don't exist
    new_columns = [
        ("total_tokens", "INTEGER"),
        ("cached_tokens", "INTEGER"),
        ("reasoning_tokens", "INTEGER"),
        ("input_size", "INTEGER"),
        ("output_size", "INTEGER"),
        ("parameters", "TEXT"),
        ("metadata", "TEXT"),
    ]

    for col_name, col_type in new_columns:
        if col_name not in column_names:
            await db.execute(f"""
                ALTER TABLE nodetool_predictions ADD COLUMN {col_name} {col_type}
            """)

    # Create indexes for cost aggregation queries
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_prediction_user_provider
        ON nodetool_predictions(user_id, provider)
    """)

    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_prediction_user_model
        ON nodetool_predictions(user_id, model)
    """)

    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_prediction_created_at
        ON nodetool_predictions(created_at)
    """)


async def down(db: "aiosqlite.Connection") -> None:
    """Remove cost tracking fields from predictions table.

    Note: We only drop the indexes since SQLite doesn't support DROP COLUMN
    in older versions. The columns will remain but be unused.
    """
    await db.execute("DROP INDEX IF EXISTS idx_prediction_user_provider")
    await db.execute("DROP INDEX IF EXISTS idx_prediction_user_model")
    await db.execute("DROP INDEX IF EXISTS idx_prediction_created_at")
