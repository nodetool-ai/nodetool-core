"""
Migration: Add FTS search index for messages
Version: 20260127_000000
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodetool.migrations.db_adapter import MigrationDBAdapter

version = "20260127_000000"
name = "add_message_search_fts"

creates_tables = ["messages_fts"]
modifies_tables = ["nodetool_messages"]


async def up(db: "MigrationDBAdapter") -> None:
    """Create FTS5 table and triggers for message search (SQLite only)."""
    if db.db_type != "sqlite":
        return

    await db.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
        USING fts5(
            message_id UNINDEXED,
            user_id UNINDEXED,
            thread_id UNINDEXED,
            content
        )
    """)

    await db.execute("""
        INSERT INTO messages_fts(message_id, user_id, thread_id, content)
        SELECT id, user_id, thread_id, content
        FROM nodetool_messages
        WHERE content IS NOT NULL
    """)

    await db.execute("""
        CREATE TRIGGER IF NOT EXISTS messages_fts_insert
        AFTER INSERT ON nodetool_messages
        BEGIN
            INSERT INTO messages_fts(message_id, user_id, thread_id, content)
            SELECT new.id, new.user_id, new.thread_id, new.content
            WHERE new.content IS NOT NULL;
        END;
    """)

    await db.execute("""
        CREATE TRIGGER IF NOT EXISTS messages_fts_update
        AFTER UPDATE ON nodetool_messages
        BEGIN
            DELETE FROM messages_fts WHERE message_id = old.id;
            INSERT INTO messages_fts(message_id, user_id, thread_id, content)
            SELECT new.id, new.user_id, new.thread_id, new.content
            WHERE new.content IS NOT NULL;
        END;
    """)

    await db.execute("""
        CREATE TRIGGER IF NOT EXISTS messages_fts_delete
        AFTER DELETE ON nodetool_messages
        BEGIN
            DELETE FROM messages_fts WHERE message_id = old.id;
        END;
    """)


async def down(db: "MigrationDBAdapter") -> None:
    """Drop FTS5 table and triggers (SQLite only)."""
    if db.db_type != "sqlite":
        return

    await db.execute("DROP TRIGGER IF EXISTS messages_fts_insert")
    await db.execute("DROP TRIGGER IF EXISTS messages_fts_update")
    await db.execute("DROP TRIGGER IF EXISTS messages_fts_delete")
    await db.execute("DROP TABLE IF EXISTS messages_fts")
