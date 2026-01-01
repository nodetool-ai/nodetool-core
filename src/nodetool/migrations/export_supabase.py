"""
Export migrations to Supabase format.

Creates SQL migration files in supabase/migrations/ that can be
applied using the Supabase CLI (supabase db push).
"""

import asyncio
import hashlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from nodetool.config.logging_config import get_logger
from nodetool.migrations.db_adapter import MigrationDBAdapter
from nodetool.migrations.runner import MIGRATIONS_DIR, Migration, MigrationRunner

log = get_logger(__name__)


def python_type_to_sql(python_type: str) -> str:
    """Convert Python type annotation to SQL type."""
    type_map = {
        "TEXT": "TEXT",
        "str": "TEXT",
        "INTEGER": "INTEGER",
        "int": "INTEGER",
        "REAL": "REAL",
        "float": "REAL",
        "bool": "INTEGER",
        "datetime": "TIMESTAMPTZ",
    }
    return type_map.get(python_type, "TEXT")


def extract_table_name(sql: str) -> str | None:
    """Extract table name from CREATE TABLE statement."""
    import re

    match = re.search(r"CREATE TABLE.*?IF NOT EXISTS\s+(\w+)", sql, re.IGNORECASE)
    return match.group(1) if match else None


def extract_columns(sql: str) -> list[tuple[str, str, bool]] | None:
    """Extract column definitions from CREATE TABLE statement."""
    import re

    table_match = re.search(r"CREATE TABLE.*?\(([^;]+)\)", sql, re.IGNORECASE | re.DOTALL)
    if not table_match:
        return None

    columns = []
    column_defs = table_match.group(1)

    for line in column_defs.split(","):
        line = line.strip()
        if not line or line.upper().startswith(("PRIMARY", "FOREIGN", "CONSTRAINT", "CREATE", "INDEX")):
            continue

        col_match = re.match(r"(\w+)\s+(\w+(?:\([^)]+\))?)", line, re.IGNORECASE)
        if col_match:
            name = col_match.group(1)
            sql_type = col_match.group(2).upper()
            nullable = "NOT NULL" not in line.upper()
            columns.append((name, sql_type, nullable))

    return columns if columns else None


def extract_alter_columns(sql: str) -> list[tuple[str, str]] | None:
    """Extract ADD COLUMN operations from ALTER TABLE statement."""
    import re

    columns = []
    add_matches = re.findall(r"ALTER TABLE.*?ADD COLUMN.*?(\w+)\s+(\w+[^;]+)", sql, re.IGNORECASE)

    for name, sql_type in add_matches:
        columns.append((name, sql_type.strip()))

    return columns if columns else None


async def export_migration_to_sql(
    migration: Migration,
    adapter: MigrationDBAdapter | None = None,
) -> str:
    """Convert a Python migration to SQL format for Supabase."""

    source = migration.file_path.read_text()

    import ast

    tree = ast.parse(source)

    sql_statements: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "up":
            # Walk all nodes in the function body to find db.execute() calls
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Await):
                    call = stmt.value
                    if isinstance(call, ast.Call):
                        if isinstance(call.func, ast.Attribute):
                            func_name = call.func.attr
                            if func_name == "execute" and call.args:
                                arg = call.args[0]
                                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                                    sql = arg.value.strip()
                                    if sql:
                                        sql_statements.append(sql)

    sql = "\n\n".join(sql_statements)

    lines = sql.split("\n")
    formatted_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        line = line.replace("?", "%s")

        formatted_lines.append(line)

    return "\n".join(formatted_lines)


def generate_supabase_migration_filename(migration: Migration) -> str:
    """Generate Supabase-style migration filename."""
    # Convert version format: 20250428_212009_001 -> 20250428212009_001
    version = migration.version.replace("_", "")

    # Clean name for filename
    safe_name = migration.name.replace("_", "-")

    return f"{version}_{safe_name}.sql"


async def export_all_migrations(
    output_dir: Path,
    migrations_dir: Path | None = None,
    latest_only: bool = False,
) -> list[Path]:
    """Export all migrations to Supabase format.

    Args:
        output_dir: Directory to write Supabase migration files
        migrations_dir: Directory containing Python migrations
        latest_only: If True, only export migrations after the last Supabase migration

    Returns:
        List of created migration files
    """

    # Find existing Supabase migrations to determine baseline
    existing_migrations: dict[str, bool] = {}
    if latest_only and output_dir.exists():
        for f in output_dir.glob("*.sql"):
            # Extract version from filename
            version_match = f.stem.split("_")[0]
            if version_match.isdigit():
                existing_migrations[version_match] = True

    # Load Python migrations
    runner = MigrationRunner(None, migrations_dir=migrations_dir or MIGRATIONS_DIR)
    migrations = runner.discover_migrations()

    # Filter migrations
    if latest_only:
        migrations = [m for m in migrations if m.version.replace("_", "") not in existing_migrations]

    created_files: list[Path] = []

    for migration in migrations:
        try:
            sql = await export_migration_to_sql(migration)

            if sql.strip():
                filename = generate_supabase_migration_filename(migration)
                filepath = output_dir / filename
                filepath.write_text(sql)
                created_files.append(filepath)
                log.info(f"Exported: {filename}")
            else:
                log.warning(f"No SQL content for migration: {migration.version}")

        except Exception as e:
            log.error(f"Failed to export migration {migration.version}: {e}")
            raise

    return created_files


async def sync_migrations_to_supabase(
    supabase_dir: Path | None = None,
) -> list[Path]:
    """Sync migrations to Supabase format.

    This is the main entry point for exporting migrations to Supabase.
    It uses the project root supabase/ directory.

    Args:
        supabase_dir: Path to supabase directory (defaults to project root/supabase)

    Returns:
        List of created migration files
    """
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    supabase_dir = supabase_dir or (project_root / "supabase")

    migrations_dir = supabase_dir / "migrations"
    migrations_dir.mkdir(parents=True, exist_ok=True)

    return await export_all_migrations(migrations_dir)


def main():
    """CLI entry point for exporting migrations."""
    import argparse

    parser = argparse.ArgumentParser(description="Export migrations to Supabase format")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output directory for Supabase migrations",
    )
    parser.add_argument(
        "--migrations-dir",
        "-m",
        type=Path,
        help="Source directory for Python migrations",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Only export migrations not already in output directory",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if there are unexported migrations",
    )

    args = parser.parse_args()

    output_dir = args.output
    migrations_dir = args.migrations_dir

    try:
        if args.check:
            # Just check if there are pending exports
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            supabase_dir = project_root / "supabase"
            output_dir = supabase_dir / "migrations"

            existing: dict[str, bool] = {}
            if output_dir.exists():
                for f in output_dir.glob("*.sql"):
                    version_match = f.stem.split("_")[0]
                    if version_match.isdigit():
                        existing[version_match] = True

            runner = MigrationRunner(None, migrations_dir=migrations_dir or MIGRATIONS_DIR)
            migrations = runner.discover_migrations()
            pending = [m for m in migrations if m.version.replace("_", "") not in existing]

            if pending:
                print(f"Pending migrations to export: {len(pending)}")
                for m in pending:
                    print(f"  - {m.version}: {m.name}")
                sys.exit(1)
            else:
                print("All migrations are exported to Supabase format")
                sys.exit(0)

        else:
            created = asyncio.run(
                export_all_migrations(
                    output_dir=output_dir,
                    migrations_dir=migrations_dir,
                    latest_only=args.latest_only,
                )
            )

            if created:
                print(f"Exported {len(created)} migration(s) to {output_dir}")
            else:
                print("No new migrations to export")

    except Exception as e:
        log.error(f"Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
