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
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

console = Console()


def _ensure_settings_loaded():
    """Ensure environment settings are loaded before command processing.

    This must be called before Click processes envvar options, otherwise
    environment variables from .env files won't be available.
    """
    from nodetool.config.environment import Environment

    Environment.load_settings()


async def _get_db_connection(postgres_url: Optional[str] = None):
    """Get a database connection based on configuration.

    Args:
        postgres_url: Optional PostgreSQL connection URL. If provided, connects to PostgreSQL.
                      Otherwise checks POSTGRES_* env vars, Supabase API, or falls back to SQLite.

    Returns:
        Tuple of (connection_or_pool, cleanup_func, db_type)

    Raises:
        ImportError: If psycopg_pool is not installed when using PostgreSQL.
    """
    from nodetool.config.environment import Environment
    from nodetool.config.logging_config import get_logger

    logger = get_logger(__name__)

    # Check for Supabase configuration first
    supabase_url = Environment.get_supabase_url()
    db_source = None
    if not postgres_url and supabase_url:
        postgres_url = await Environment.get_supabase_postgres_uri()
        if postgres_url:
            db_source = "supabase"
            logger.info(f"Using Supabase database: {supabase_url}")

    # Check for PostgreSQL configuration
    postgres_db = Environment.get("POSTGRES_DB")

    if postgres_url or postgres_db:
        if not db_source:
            db_source = "postgres" if postgres_url else "postgres-env"

        try:
            from psycopg_pool import AsyncConnectionPool
        except ImportError as e:
            raise ImportError(
                "psycopg-pool is required for PostgreSQL migrations. Install it with: pip install psycopg psycopg-pool"
            ) from e

        if not postgres_url and postgres_db:
            params = Environment.get_postgres_params()
            postgres_url = (
                f"dbname={params['database']} user={params['user']} "
                f"password={params['password']} host={params['host']} port={params['port']}"
            )

        if postgres_url is None:
            raise RuntimeError("postgres_url should be set at this point")
        pool = AsyncConnectionPool(postgres_url, min_size=1, max_size=5)
        await pool.open()

        async def cleanup():
            await pool.close()

        return pool, cleanup, db_source or "postgres"
    else:
        db_path = Environment.get("DB_PATH", "~/.config/nodetool/nodetool.sqlite3")
        db_path = str(Path(db_path).expanduser())

        logger.info(f"Using SQLite database: {db_path}")

        from nodetool.runtime.db_sqlite import SQLiteConnectionPool

        pool = await SQLiteConnectionPool.get_shared(db_path)
        conn = await pool.acquire()

        async def cleanup():
            await pool.release(conn)

        return conn, cleanup, "sqlite"


@click.group("migrations")
def migrations():
    """Manage database migrations.

    Commands for applying, rolling back, and managing database schema migrations.
    Supports both SQLite (default), PostgreSQL, and Supabase databases.

    To migrate a Supabase database, set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
        export SUPABASE_URL="https://your-project.supabase.co"
        export SUPABASE_SERVICE_ROLE_KEY="your-service-role-key"
        nodetool migrations upgrade

    To migrate a PostgreSQL database, use the --postgres-url option:
        nodetool migrations upgrade --postgres-url "postgresql://user:pass@host:5432/db"
    """
    _ensure_settings_loaded()
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
@click.option(
    "--postgres-url",
    type=str,
    default=None,
    envvar="POSTGRES_URL",
    help="PostgreSQL connection URL. Use for PostgreSQL/Supabase migrations.",
)
def upgrade(dry_run: bool, target: Optional[str], skip_checksum_validation: bool, postgres_url: Optional[str]):
    """Apply pending migrations.

    Detects the database state and applies the appropriate migrations:
    - Fresh install: Creates tables and runs all migrations
    - Legacy database: Baselines existing tables and runs new migrations
    - Normal operation: Runs only pending migrations

    Examples:
        # Apply all pending migrations to SQLite (default)
        nodetool migrations upgrade

        # Apply migrations to PostgreSQL/Supabase
        nodetool migrations upgrade --postgres-url "postgresql://user:pass@host:5432/db"

        # Preview what would be done
        nodetool migrations upgrade --dry-run

        # Migrate to a specific version
        nodetool migrations upgrade --target 20250501_000000
    """

    async def run_upgrade():
        from nodetool.migrations.runner import MigrationRunner

        cleanup = None
        try:
            conn_or_pool, cleanup, db_type = await _get_db_connection(postgres_url)
            console.print(f"[cyan]Using {db_type} database[/]")

            runner = MigrationRunner(conn_or_pool)

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

        except Exception as e:
            console.print(f"[red]❌ Migration failed: {e}[/]")
            raise SystemExit(1) from e
        finally:
            if cleanup:
                await cleanup()

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
@click.option(
    "--postgres-url",
    type=str,
    default=None,
    envvar="POSTGRES_URL",
    help="PostgreSQL connection URL. Use for PostgreSQL/Supabase migrations.",
)
def downgrade(steps: int, force: bool, postgres_url: Optional[str]):
    """Rollback migrations.

    Rolls back the specified number of most recently applied migrations.
    Use with caution in production environments.

    Examples:
        # Rollback the last migration (SQLite)
        nodetool migrations downgrade

        # Rollback the last migration (PostgreSQL/Supabase)
        nodetool migrations downgrade --postgres-url "postgresql://user:pass@host:5432/db"

        # Rollback the last 3 migrations
        nodetool migrations downgrade --steps 3
    """
    if not force:
        if not click.confirm(f"Are you sure you want to rollback {steps} migration(s)? This may cause data loss."):
            console.print("[yellow]Operation cancelled[/]")
            return

    async def run_downgrade():
        from nodetool.migrations.runner import MigrationRunner

        cleanup = None
        try:
            conn_or_pool, cleanup, db_type = await _get_db_connection(postgres_url)
            console.print(f"[cyan]Using {db_type} database[/]")

            runner = MigrationRunner(conn_or_pool)
            rolled_back = await runner.rollback(steps=steps)

            if rolled_back:
                console.print(f"[green]✅ Rolled back {len(rolled_back)} migration(s):[/]")
                for version in rolled_back:
                    console.print(f"  • {version}")
            else:
                console.print("[yellow]No migrations to rollback[/]")

        except Exception as e:
            console.print(f"[red]❌ Rollback failed: {e}[/]")
            raise SystemExit(1) from e
        finally:
            if cleanup:
                await cleanup()

    asyncio.run(run_downgrade())


@migrations.command("status")
@click.option(
    "--postgres-url",
    type=str,
    default=None,
    envvar="POSTGRES_URL",
    help="PostgreSQL connection URL. Use for PostgreSQL/Supabase migrations.",
)
def status(postgres_url: Optional[str]):
    """Show migration status.

    Displays the current database state, applied migrations, and pending migrations.

    Examples:
        # Show SQLite status (default)
        nodetool migrations status

        # Show PostgreSQL/Supabase status
        nodetool migrations status --postgres-url "postgresql://user:pass@host:5432/db"
    """

    async def run_status():
        from nodetool.migrations.runner import MigrationRunner

        cleanup = None
        try:
            conn_or_pool, cleanup, db_type = await _get_db_connection(postgres_url)
            console.print(f"[cyan]Using {db_type} database[/]")
            console.print()

            runner = MigrationRunner(conn_or_pool)
            result = await runner.status()

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

        except Exception as e:
            console.print(f"[red]❌ Error getting status: {e}[/]")
            raise SystemExit(1) from e
        finally:
            if cleanup:
                await cleanup()

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
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
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
@click.option(
    "--postgres-url",
    type=str,
    default=None,
    envvar="POSTGRES_URL",
    help="PostgreSQL connection URL. Use for PostgreSQL/Supabase migrations.",
)
def validate(postgres_url: Optional[str]):
    """Validate migration checksums.

    Checks that the checksums of applied migrations match the current
    migration files. This detects if migration files have been modified
    after they were applied.

    Examples:
        # Validate SQLite migrations (default)
        nodetool migrations validate

        # Validate PostgreSQL/Supabase migrations
        nodetool migrations validate --postgres-url "postgresql://user:pass@host:5432/db"
    """

    async def run_validate():
        from nodetool.migrations.runner import MigrationRunner

        cleanup = None
        try:
            conn_or_pool, cleanup, db_type = await _get_db_connection(postgres_url)
            console.print(f"[cyan]Using {db_type} database[/]")

            runner = MigrationRunner(conn_or_pool)
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

        except SystemExit:
            raise
        except Exception as e:
            console.print(f"[red]❌ Validation error: {e}[/]")
            raise SystemExit(1) from e
        finally:
            if cleanup:
                await cleanup()

    asyncio.run(run_validate())


@migrations.command("baseline")
@click.option(
    "--force",
    is_flag=True,
    help="Force re-baseline even if tracking already exists",
)
@click.option(
    "--postgres-url",
    type=str,
    default=None,
    envvar="POSTGRES_URL",
    help="PostgreSQL connection URL. Use for PostgreSQL/Supabase migrations.",
)
def baseline(force: bool, postgres_url: Optional[str]):
    """Manually baseline migrations.

    Marks migrations as applied without executing them, based on which
    tables already exist. This is useful for:
    - Upgrading from a pre-migration system
    - Recovering from migration state issues
    - Setting up a database that was created manually

    Examples:
        # Baseline SQLite migrations (default)
        nodetool migrations baseline

        # Baseline PostgreSQL/Supabase migrations
        nodetool migrations baseline --postgres-url "postgresql://user:pass@host:5432/db"

        # Force re-baseline
        nodetool migrations baseline --force
    """
    if not force:
        if not click.confirm("This will mark migrations as applied based on existing tables. Continue?"):
            console.print("[yellow]Operation cancelled[/]")
            return

    async def run_baseline():
        from nodetool.migrations.runner import MigrationRunner

        cleanup = None
        try:
            conn_or_pool, cleanup, db_type = await _get_db_connection(postgres_url)
            console.print(f"[cyan]Using {db_type} database[/]")

            runner = MigrationRunner(conn_or_pool)
            baselined = await runner.baseline(force=force)

            console.print(f"[green]✅ Baselined {baselined} migration(s)[/]")

        except Exception as e:
            console.print(f"[red]❌ Baseline failed: {e}[/]")
            raise SystemExit(1) from e
        finally:
            if cleanup:
                await cleanup()

    asyncio.run(run_baseline())
