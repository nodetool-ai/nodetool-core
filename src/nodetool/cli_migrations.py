"""
CLI commands for database migrations.

Provides commands for managing database schema migrations including:
- upgrade: Apply pending migrations
- downgrade: Rollback migrations
- status: Show migration status
- create: Create a new migration file
- validate: Validate migration checksums
- baseline: Manually baseline migrations for legacy databases
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group("migrations")
def migrations():
    """Manage database migrations.

    Commands for applying, rolling back, and managing database schema migrations.
    """
    pass


@migrations.command("upgrade")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be done without making changes",
)
@click.option(
    "--target",
    type=str,
    default=None,
    help="Target version to migrate to",
)
@click.option(
    "--skip-checksum-validation",
    is_flag=True,
    help="Skip checksum validation of applied migrations",
)
def upgrade(dry_run: bool, target: Optional[str], skip_checksum_validation: bool):
    """Apply pending migrations.

    Detects the database state and applies the appropriate migrations:
    - Fresh install: Creates tables and runs all migrations
    - Legacy database: Baselines existing tables and runs new migrations
    - Normal operation: Runs only pending migrations

    Examples:
        # Apply all pending migrations
        nodetool migrations upgrade

        # Preview what would be done
        nodetool migrations upgrade --dry-run

        # Migrate to a specific version
        nodetool migrations upgrade --target 20250501_000000
    """

    async def run_upgrade():
        from nodetool.config.environment import Environment
        from nodetool.migrations.runner import MigrationRunner
        from nodetool.runtime.db_sqlite import SQLiteConnectionPool

        try:
            db_path = Environment.get("DB_PATH", "~/.config/nodetool/nodetool.sqlite3")
            db_path = str(Path(db_path).expanduser())

            pool = await SQLiteConnectionPool.get_shared(db_path)
            conn = await pool.acquire()

            try:
                runner = MigrationRunner(conn)

                if dry_run:
                    console.print("[cyan]DRY RUN - No changes will be made[/]")
                    console.print()

                applied = await runner.migrate(
                    target=target,
                    dry_run=dry_run,
                    validate_checksums=not skip_checksum_validation,
                )

                if applied:
                    console.print(f"[green]✅ Applied {len(applied)} migration(s):[/]")
                    for version in applied:
                        console.print(f"  • {version}")
                else:
                    console.print("[yellow]No migrations to apply - database is up to date[/]")

            finally:
                await pool.release(conn)

        except Exception as e:
            console.print(f"[red]❌ Migration failed: {e}[/]")
            raise SystemExit(1) from e

    asyncio.run(run_upgrade())


@migrations.command("downgrade")
@click.option(
    "--steps",
    "-n",
    type=int,
    default=1,
    help="Number of migrations to rollback (default: 1)",
)
@click.option(
    "--force",
    is_flag=True,
    help="Skip confirmation prompt",
)
def downgrade(steps: int, force: bool):
    """Rollback migrations.

    Rolls back the specified number of most recently applied migrations.
    Use with caution in production environments.

    Examples:
        # Rollback the last migration
        nodetool migrations downgrade

        # Rollback the last 3 migrations
        nodetool migrations downgrade --steps 3
    """
    if not force:
        if not click.confirm(f"Are you sure you want to rollback {steps} migration(s)? This may cause data loss."):
            console.print("[yellow]Operation cancelled[/]")
            return

    async def run_downgrade():
        from nodetool.config.environment import Environment
        from nodetool.migrations.runner import MigrationRunner
        from nodetool.runtime.db_sqlite import SQLiteConnectionPool

        try:
            db_path = Environment.get("DB_PATH", "~/.config/nodetool/nodetool.sqlite3")
            db_path = str(Path(db_path).expanduser())

            pool = await SQLiteConnectionPool.get_shared(db_path)
            conn = await pool.acquire()

            try:
                runner = MigrationRunner(conn)
                rolled_back = await runner.rollback(steps=steps)

                if rolled_back:
                    console.print(f"[green]✅ Rolled back {len(rolled_back)} migration(s):[/]")
                    for version in rolled_back:
                        console.print(f"  • {version}")
                else:
                    console.print("[yellow]No migrations to rollback[/]")

            finally:
                await pool.release(conn)

        except Exception as e:
            console.print(f"[red]❌ Rollback failed: {e}[/]")
            raise SystemExit(1) from e

    asyncio.run(run_downgrade())


@migrations.command("status")
def status():
    """Show migration status.

    Displays the current database state, applied migrations, and pending migrations.

    Examples:
        nodetool migrations status
    """

    async def run_status():
        from nodetool.config.environment import Environment
        from nodetool.migrations.runner import MigrationRunner
        from nodetool.runtime.db_sqlite import SQLiteConnectionPool

        try:
            db_path = Environment.get("DB_PATH", "~/.config/nodetool/nodetool.sqlite3")
            db_path = str(Path(db_path).expanduser())

            pool = await SQLiteConnectionPool.get_shared(db_path)
            conn = await pool.acquire()

            try:
                runner = MigrationRunner(conn)
                result = await runner.status()

                console.print()
                console.print(f"[bold cyan]Database State:[/] {result['state']}")
                console.print(f"[bold cyan]Current Version:[/] {result['current_version'] or 'None'}")
                console.print()

                # Applied migrations table
                if result["applied"]:
                    table = Table(title="Applied Migrations")
                    table.add_column("Version", style="cyan")
                    table.add_column("Name", style="green")
                    table.add_column("Applied At", style="yellow")
                    table.add_column("Time (ms)", style="blue")
                    table.add_column("Baselined", style="magenta")

                    for m in result["applied"]:
                        table.add_row(
                            m["version"],
                            m["name"],
                            m["applied_at"],
                            str(m["execution_time_ms"]),
                            "Yes" if m["baselined"] else "No",
                        )

                    console.print(table)
                    console.print()

                # Pending migrations table
                if result["pending"]:
                    table = Table(title="Pending Migrations")
                    table.add_column("Version", style="cyan")
                    table.add_column("Name", style="green")

                    for m in result["pending"]:
                        table.add_row(m["version"], m["name"])

                    console.print(table)
                else:
                    console.print("[green]No pending migrations - database is up to date[/]")

            finally:
                await pool.release(conn)

        except Exception as e:
            console.print(f"[red]❌ Error getting status: {e}[/]")
            raise SystemExit(1) from e

    asyncio.run(run_status())


@migrations.command("create")
@click.argument("name")
def create(name: str):
    """Create a new migration file.

    Creates a new migration file with a timestamp-based version and the
    provided name. The file will be created in the migrations/versions directory.

    Examples:
        nodetool migrations create add_user_preferences
    """
    from nodetool.migrations.runner import MIGRATIONS_DIR

    # Generate timestamp-based version
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    version = timestamp

    # Sanitize name (replace spaces with underscores, lowercase)
    safe_name = name.lower().replace(" ", "_").replace("-", "_")
    filename = f"{version}_{safe_name}.py"
    filepath = MIGRATIONS_DIR / filename

    # Template for new migration
    template = f'''"""
Migration: {name.replace("_", " ").title()}
Version: {version}
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodetool.migrations.db_adapter import MigrationDBAdapter

version = "{version}"
name = "{safe_name}"

# Tables this migration creates (if creating new tables)
creates_tables = []
# Tables this migration modifies (if altering existing tables)
modifies_tables = []


async def up(db: "MigrationDBAdapter") -> None:
    """Apply the migration.

    TODO: Add your migration logic here.
    Use db.execute() for SQL statements.
    Use db.get_columns() to check existing columns.
    """
    pass


async def down(db: "MigrationDBAdapter") -> None:
    """Rollback the migration.

    TODO: Add your rollback logic here.
    """
    pass
'''

    try:
        # Ensure directory exists
        MIGRATIONS_DIR.mkdir(parents=True, exist_ok=True)

        # Write migration file
        filepath.write_text(template)

        console.print("[green]✅ Created migration file:[/]")
        console.print(f"  {filepath}")
        console.print()
        console.print("[cyan]Next steps:[/]")
        console.print("  1. Edit the migration file to add your schema changes")
        console.print("  2. Run: nodetool migrations upgrade")

    except Exception as e:
        console.print(f"[red]❌ Failed to create migration: {e}[/]")
        raise SystemExit(1) from e


@migrations.command("validate")
def validate():
    """Validate migration checksums.

    Checks that the checksums of applied migrations match the current
    migration files. This detects if migration files have been modified
    after they were applied.

    Examples:
        nodetool migrations validate
    """

    async def run_validate():
        from nodetool.config.environment import Environment
        from nodetool.migrations.runner import MigrationRunner
        from nodetool.runtime.db_sqlite import SQLiteConnectionPool

        try:
            db_path = Environment.get("DB_PATH", "~/.config/nodetool/nodetool.sqlite3")
            db_path = str(Path(db_path).expanduser())

            pool = await SQLiteConnectionPool.get_shared(db_path)
            conn = await pool.acquire()

            try:
                runner = MigrationRunner(conn)
                mismatches = await runner.validate_checksums()

                if mismatches:
                    console.print("[red]❌ Checksum validation failed![/]")
                    console.print()
                    console.print("[yellow]The following migrations have been modified after application:[/]")
                    for version in mismatches:
                        console.print(f"  • {version}")
                    console.print()
                    console.print("[yellow]This could indicate:[/]")
                    console.print("  - Migration files were manually edited")
                    console.print("  - Version control conflict resolution")
                    console.print("  - Corruption")
                    raise SystemExit(1)
                else:
                    console.print("[green]✅ All migration checksums are valid[/]")

            finally:
                await pool.release(conn)

        except SystemExit:
            raise
        except Exception as e:
            console.print(f"[red]❌ Validation error: {e}[/]")
            raise SystemExit(1) from e

    asyncio.run(run_validate())


@migrations.command("baseline")
@click.option(
    "--force",
    is_flag=True,
    help="Force re-baseline even if tracking already exists",
)
def baseline(force: bool):
    """Manually baseline migrations.

    Marks migrations as applied without executing them, based on which
    tables already exist. This is useful for:
    - Upgrading from a pre-migration system
    - Recovering from migration state issues
    - Setting up a database that was created manually

    Examples:
        # Baseline migrations for existing database
        nodetool migrations baseline

        # Force re-baseline
        nodetool migrations baseline --force
    """
    if not force:
        if not click.confirm("This will mark migrations as applied based on existing tables. Continue?"):
            console.print("[yellow]Operation cancelled[/]")
            return

    async def run_baseline():
        from nodetool.config.environment import Environment
        from nodetool.migrations.runner import MigrationRunner
        from nodetool.runtime.db_sqlite import SQLiteConnectionPool

        try:
            db_path = Environment.get("DB_PATH", "~/.config/nodetool/nodetool.sqlite3")
            db_path = str(Path(db_path).expanduser())

            pool = await SQLiteConnectionPool.get_shared(db_path)
            conn = await pool.acquire()

            try:
                runner = MigrationRunner(conn)
                baselined = await runner.baseline(force=force)

                console.print(f"[green]✅ Baselined {baselined} migration(s)[/]")

            finally:
                await pool.release(conn)

        except Exception as e:
            console.print(f"[red]❌ Baseline failed: {e}[/]")
            raise SystemExit(1) from e

    asyncio.run(run_baseline())


@migrations.command("export")
@click.option(
    "--output",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True),
    default=None,
    help="Output directory for Supabase migrations",
)
@click.option(
    "--migrations-dir",
    "-m",
    type=click.Path(file_okay=False, dir_okay=True),
    default=None,
    help="Source directory for Python migrations",
)
@click.option(
    "--latest-only",
    is_flag=True,
    help="Only export migrations not already in output directory",
)
def export(output: str | None, migrations_dir: str | None, latest_only: bool):
    """Export migrations to Supabase format.

    Creates SQL migration files in supabase/migrations/ that can be
    applied using the Supabase CLI (supabase db push).

    Examples:
        # Export all migrations to supabase/migrations/
        nodetool migrations export

        # Export only new migrations
        nodetool migrations export --latest-only

        # Export to custom directory
        nodetool migrations export --output ./custom/supabase/migrations
    """
    import asyncio
    from pathlib import Path

    async def run_export():
        from nodetool.config.environment import Environment
        from nodetool.migrations.export_supabase import export_all_migrations

        try:
            output_path = Path(output) if output else None
            migrations_path = Path(migrations_dir) if migrations_dir else None

            if not output_path:
                output_path = Path.cwd() / "supabase" / "migrations"

            console.print(f"[cyan]Exporting migrations to: {output_path}[/]")

            created = await export_all_migrations(
                output_dir=output_path,
                migrations_dir=migrations_path,
                latest_only=latest_only,
            )

            if created:
                console.print(f"[green]✅ Exported {len(created)} migration(s):[/]")
                for f in created:
                    console.print(f"  • {f.name}")
                console.print()
                console.print("[cyan]Next steps:[/]")
                console.print("  1. Review the generated SQL files")
                console.print("  2. Check them into version control")
                console.print("  3. Run: supabase db push")
            else:
                console.print("[yellow]No new migrations to export[/]")

        except Exception as e:
            console.print(f"[red]❌ Export failed: {e}[/]")
            raise SystemExit(1) from e

    asyncio.run(run_export())


@migrations.command("check-export")
def check_export():
    """Check if all migrations are exported to Supabase format.

    Exits with code 1 if there are pending migrations to export.

    Examples:
        nodetool migrations check-export
    """
    import asyncio
    import sys
    from pathlib import Path

    async def run_check():
        from nodetool.config.environment import Environment
        from nodetool.migrations.export_supabase import export_all_migrations
        from nodetool.migrations.runner import MIGRATIONS_DIR, MigrationRunner

        try:
            supabase_dir = Path.cwd() / "supabase"
            output_dir = supabase_dir / "migrations"

            # Find existing Supabase migrations
            existing: dict[str, bool] = {}
            if output_dir.exists():
                for f in output_dir.glob("*.sql"):
                    version_match = f.stem.split("_")[0]
                    if version_match.isdigit():
                        existing[version_match] = True

            # Load Python migrations
            runner = MigrationRunner(None, migrations_dir=MIGRATIONS_DIR)
            migrations = runner.discover_migrations()

            # Find pending
            pending = [m for m in migrations if m.version.replace("_", "") not in existing]

            if pending:
                console.print(f"[yellow]⚠️  {len(pending)} migration(s) not exported to Supabase:[/]")
                for m in pending:
                    console.print(f"  • {m.version}: {m.name}")
                console.print()
                console.print("Run: nodetool migrations export")
                raise SystemExit(1)
            else:
                console.print("[green]✅ All migrations are exported to Supabase format[/]")

        except SystemExit:
            raise
        except Exception as e:
            console.print(f"[red]❌ Check failed: {e}[/]")
            raise SystemExit(1) from e

    asyncio.run(run_check())
