"""
Migration versions package.

This package contains all database migration files in Python format.
Each migration file must contain:
- version: Unique version identifier (timestamp-based)
- name: Human-readable migration name
- creates_tables: List of tables this migration creates (optional)
- modifies_tables: List of tables this migration modifies (optional)
- async def up(db): Function to apply the migration
- async def down(db): Function to rollback the migration
"""
