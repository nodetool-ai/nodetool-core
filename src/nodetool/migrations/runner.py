"""
Migration runner for the NodeTool database migration system.

Provides the core MigrationRunner class that handles:
- Migration discovery and ordering
- Migration execution with transaction safety
- Migration tracking and version management
- Database-level locking for multi-instance deployments
- Rollback support
- Checksum validation
- Baselining for legacy databases

The migration system is database-agnostic and works with SQLite, PostgreSQL,
and other databases through the MigrationDBAdapter interface.
"""

import asyncio
import hashlib
import importlib.util
import socket
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Coroutine

from nodetool.config.logging_config import get_logger
from nodetool.migrations.db_adapter import (
    MigrationDBAdapter,
    create_migration_adapter,
)
from nodetool.migrations.exceptions import (
    BaselineError,
    LockError,
    MigrationDiscoveryError,
    MigrationError,
    RollbackError,
)
from nodetool.migrations.state import (
    MIGRATION_LOCK_TABLE,
    MIGRATION_TRACKING_TABLE,
    DatabaseState,
)

log = get_logger(__name__)

# Default location for migration files
MIGRATIONS_DIR = Path(__file__).parent / "versions"


@dataclass
class Migration:
    """Represents a database migration.

    Attributes:
        version: Unique version identifier (timestamp-based, e.g., 20250428_212009)
        name: Human-readable migration name
        checksum: SHA256 hash of the migration file content
        file_path: Path to the migration file
        up: Async function to apply the migration
        down: Async function to rollback the migration
        creates_tables: List of tables this migration creates
        modifies_tables: List of tables this migration modifies
    """

    version: str
    name: str
    checksum: str
    file_path: Path
    up: Callable[[MigrationDBAdapter], Coroutine[Any, Any, None]]
    down: Callable[[MigrationDBAdapter], Coroutine[Any, Any, None]]
    creates_tables: list[str]
    modifies_tables: list[str]


@dataclass
class AppliedMigration:
    """Represents a migration that has been applied to the database.

    Attributes:
        version: Migration version identifier
        name: Human-readable migration name
        checksum: SHA256 hash recorded when migration was applied
        applied_at: When the migration was applied
        execution_time_ms: How long the migration took to execute
        baselined: Whether this migration was baselined (not actually executed)
    """

    version: str
    name: str
    checksum: str
    applied_at: datetime
    execution_time_ms: int
    baselined: bool


class MigrationRunner:
    """Core migration runner for the NodeTool database migration system.

    This class handles all aspects of database migrations including:
    - Discovery and loading of migration files
    - Version ordering and dependency management
    - Transaction-safe migration execution
    - Migration tracking and status
    - Database-level locking for multi-instance safety
    - Rollback support
    - Checksum validation for integrity
    - Baselining for legacy database upgrades

    The runner is database-agnostic and works with any database that has
    a MigrationDBAdapter implementation (SQLite, PostgreSQL, etc.).

    Example:
        # Using with raw connection (auto-creates adapter)
        runner = MigrationRunner(sqlite_connection)
        await runner.migrate()

        # Using with explicit adapter
        adapter = PostgresMigrationAdapter(pool)
        runner = MigrationRunner(adapter)
        await runner.migrate()

        status = await runner.status()
        await runner.rollback(steps=1)
    """

    def __init__(
        self,
        connection_or_adapter: Any = None,
        migrations_dir: Path | None = None,
    ):
        """Initialize the migration runner.

        Args:
            connection_or_adapter: Either a MigrationDBAdapter instance or
                                   a raw database connection (will be wrapped
                                   in an appropriate adapter). Can be None for
                                   discovery-only operations.
            migrations_dir: Optional custom migrations directory
        """
        # Accept either an adapter or a raw connection, or None for discovery-only
        if connection_or_adapter is None:
            self._adapter = None
        elif isinstance(connection_or_adapter, MigrationDBAdapter):
            self._adapter = connection_or_adapter
        else:
            # Try to create an adapter from the connection
            self._adapter = create_migration_adapter(connection_or_adapter)

        self._migrations_dir = migrations_dir or MIGRATIONS_DIR
        self._migrations_cache: list[Migration] | None = None

    @property
    def adapter(self) -> MigrationDBAdapter | None:
        """Get the database adapter."""
        return self._adapter

    @property
    def db_type(self) -> str:
        """Get the database type."""
        if self._adapter is None:
            return "unknown"
        return self._adapter.db_type

    # -------------------------------------------------------------------------
    # Migration tracking table management
    # -------------------------------------------------------------------------

    async def _create_tracking_tables(self) -> None:
        """Create the migration tracking and lock tables if they don't exist."""
        db_type = self._adapter.db_type

        # Create migrations tracking table
        await self._adapter.execute(f"""
            CREATE TABLE IF NOT EXISTS {MIGRATION_TRACKING_TABLE} (
                version TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                checksum TEXT NOT NULL,
                applied_at TEXT NOT NULL,
                execution_time_ms INTEGER NOT NULL,
                baselined INTEGER DEFAULT 0
            )
        """)

        # Create migration lock table - syntax varies slightly by database
        if db_type == "sqlite":
            await self._adapter.execute(f"""
                CREATE TABLE IF NOT EXISTS {MIGRATION_LOCK_TABLE} (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    locked_at TEXT,
                    locked_by TEXT
                )
            """)
            # Insert initial lock row if not exists (SQLite uses INSERT OR IGNORE)
            await self._adapter.execute(f"""
                INSERT OR IGNORE INTO {MIGRATION_LOCK_TABLE} (id, locked_at, locked_by)
                VALUES (1, NULL, NULL)
            """)
        else:
            # PostgreSQL/MySQL style
            await self._adapter.execute(f"""
                CREATE TABLE IF NOT EXISTS {MIGRATION_LOCK_TABLE} (
                    id INTEGER PRIMARY KEY,
                    locked_at TEXT,
                    locked_by TEXT,
                    CONSTRAINT single_row CHECK (id = 1)
                )
            """)
            # PostgreSQL uses ON CONFLICT DO NOTHING
            await self._adapter.execute(f"""
                INSERT INTO {MIGRATION_LOCK_TABLE} (id, locked_at, locked_by)
                VALUES (1, NULL, NULL)
                ON CONFLICT (id) DO NOTHING
            """)

        await self._adapter.commit()

    async def _get_applied_migrations(self) -> dict[str, AppliedMigration]:
        """Get all applied migrations from the tracking table.

        Returns:
            Dict mapping version to AppliedMigration objects
        """
        if not await self._adapter.table_exists(MIGRATION_TRACKING_TABLE):
            return {}

        rows = await self._adapter.fetchall(f"""
            SELECT version, name, checksum, applied_at, execution_time_ms, baselined
            FROM {MIGRATION_TRACKING_TABLE}
            ORDER BY version
        """)

        applied = {}
        for row in rows:
            applied[row["version"]] = AppliedMigration(
                version=row["version"],
                name=row["name"],
                checksum=row["checksum"],
                applied_at=datetime.fromisoformat(row["applied_at"]),
                execution_time_ms=row["execution_time_ms"],
                baselined=bool(row["baselined"]),
            )
        return applied

    async def _record_migration(
        self,
        migration: Migration,
        execution_time_ms: int,
        baselined: bool = False,
    ) -> None:
        """Record a migration as applied in the tracking table.

        Args:
            migration: The migration that was applied
            execution_time_ms: How long the migration took
            baselined: Whether this was a baseline (not actually executed)
        """
        await self._adapter.execute(
            f"""
            INSERT INTO {MIGRATION_TRACKING_TABLE}
            (version, name, checksum, applied_at, execution_time_ms, baselined)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                migration.version,
                migration.name,
                migration.checksum,
                datetime.now(UTC).isoformat(),
                execution_time_ms,
                1 if baselined else 0,
            ),
        )
        await self._adapter.commit()

    async def _remove_migration_record(self, version: str) -> None:
        """Remove a migration record from the tracking table.

        Args:
            version: The migration version to remove
        """
        await self._adapter.execute(
            f"DELETE FROM {MIGRATION_TRACKING_TABLE} WHERE version = ?",
            (version,),
        )
        await self._adapter.commit()

    # -------------------------------------------------------------------------
    # Locking mechanism
    # -------------------------------------------------------------------------

    async def _acquire_lock(self, timeout: float = 30.0) -> bool:
        """Acquire the migration lock.

        Uses optimistic locking with retry to prevent concurrent migrations.

        Args:
            timeout: Maximum time to wait for lock in seconds

        Returns:
            True if lock was acquired

        Raises:
            LockError: If lock cannot be acquired within timeout
        """
        lock_id = f"{socket.gethostname()}:{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Try to acquire lock using atomic UPDATE
            await self._adapter.execute(
                f"""
                UPDATE {MIGRATION_LOCK_TABLE}
                SET locked_at = ?, locked_by = ?
                WHERE id = 1 AND locked_at IS NULL
                """,
                (datetime.now(UTC).isoformat(), lock_id),
            )
            await self._adapter.commit()

            if self._adapter.get_rowcount() > 0:
                log.debug(f"Migration lock acquired by {lock_id}")
                return True

            # Check if lock is stale (older than 5 minutes)
            row = await self._adapter.fetchone(f"SELECT locked_at, locked_by FROM {MIGRATION_LOCK_TABLE} WHERE id = 1")
            if row and row["locked_at"]:
                locked_at = datetime.fromisoformat(row["locked_at"])
                if (datetime.now(UTC) - locked_at).total_seconds() > 300:
                    # Stale lock, try to take it over
                    log.warning(f"Taking over stale migration lock from {row['locked_by']}")
                    await self._adapter.execute(
                        f"""
                        UPDATE {MIGRATION_LOCK_TABLE}
                        SET locked_at = ?, locked_by = ?
                        WHERE id = 1 AND locked_at = ?
                        """,
                        (datetime.now(UTC).isoformat(), lock_id, row["locked_at"]),
                    )
                    await self._adapter.commit()
                    if self._adapter.get_rowcount() > 0:
                        return True

            # Wait and retry
            await asyncio.sleep(0.5)

        raise LockError(f"Could not acquire migration lock within {timeout}s. Another migration may be in progress.")

    async def _release_lock(self) -> None:
        """Release the migration lock."""
        await self._adapter.execute(
            f"""
            UPDATE {MIGRATION_LOCK_TABLE}
            SET locked_at = NULL, locked_by = NULL
            WHERE id = 1
            """
        )
        await self._adapter.commit()
        log.debug("Migration lock released")

    # -------------------------------------------------------------------------
    # Migration discovery
    # -------------------------------------------------------------------------

    def _compute_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum of a migration file.

        Args:
            file_path: Path to the migration file

        Returns:
            SHA256 hex digest of file contents
        """
        content = file_path.read_bytes()
        return hashlib.sha256(content).hexdigest()

    def _load_migration(self, file_path: Path) -> Migration:
        """Load a migration from a Python file.

        Args:
            file_path: Path to the migration file

        Returns:
            Migration object

        Raises:
            MigrationDiscoveryError: If migration file is invalid
        """
        try:
            spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
            if spec is None or spec.loader is None:
                raise MigrationDiscoveryError(f"Cannot load migration: {file_path}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Validate required attributes
            if not hasattr(module, "version"):
                raise MigrationDiscoveryError(f"Migration {file_path} missing 'version' attribute")
            if not hasattr(module, "name"):
                raise MigrationDiscoveryError(f"Migration {file_path} missing 'name' attribute")
            if not hasattr(module, "up"):
                raise MigrationDiscoveryError(f"Migration {file_path} missing 'up' function")
            if not hasattr(module, "down"):
                raise MigrationDiscoveryError(f"Migration {file_path} missing 'down' function")

            return Migration(
                version=module.version,
                name=module.name,
                checksum=self._compute_checksum(file_path),
                file_path=file_path,
                up=module.up,
                down=module.down,
                creates_tables=getattr(module, "creates_tables", []),
                modifies_tables=getattr(module, "modifies_tables", []),
            )
        except Exception as e:
            if isinstance(e, MigrationDiscoveryError):
                raise
            raise MigrationDiscoveryError(f"Failed to load migration {file_path}: {e}") from e

    def discover_migrations(self) -> list[Migration]:
        """Discover and load all migrations from the migrations directory.

        Migrations are sorted by version (timestamp-based naming ensures order).

        Returns:
            List of Migration objects sorted by version

        Raises:
            MigrationDiscoveryError: If migration discovery fails
        """
        if self._migrations_cache is not None:
            return self._migrations_cache

        if not self._migrations_dir.exists():
            log.warning(f"Migrations directory not found: {self._migrations_dir}")
            return []

        migrations = []
        for file_path in sorted(self._migrations_dir.glob("*.py")):
            if file_path.name.startswith("__"):
                continue

            try:
                migration = self._load_migration(file_path)
                migrations.append(migration)
            except MigrationDiscoveryError:
                raise
            except Exception as e:
                raise MigrationDiscoveryError(f"Failed to discover migration {file_path}: {e}") from e

        # Sort by version
        migrations.sort(key=lambda m: m.version)
        self._migrations_cache = migrations

        log.debug(f"Discovered {len(migrations)} migrations")
        return migrations

    # -------------------------------------------------------------------------
    # Checksum validation
    # -------------------------------------------------------------------------

    async def validate_checksums(self) -> list[str]:
        """Validate checksums of applied migrations.

        Compares the recorded checksums of applied migrations with the
        current checksums of migration files to detect tampering.

        Returns:
            List of migration versions with checksum mismatches
        """
        applied = await self._get_applied_migrations()
        migrations = {m.version: m for m in self.discover_migrations()}
        mismatches = []

        for version, applied_migration in applied.items():
            if version in migrations:
                current_checksum = migrations[version].checksum
                if current_checksum != applied_migration.checksum:
                    mismatches.append(version)
                    log.warning(
                        f"Checksum mismatch for migration {version}: "
                        f"expected {applied_migration.checksum}, "
                        f"got {current_checksum}"
                    )

        return mismatches

    # -------------------------------------------------------------------------
    # Database state detection
    # -------------------------------------------------------------------------

    async def _detect_database_state(self) -> DatabaseState:
        """Detect the current state of the database.

        Returns:
            DatabaseState enum value
        """
        from nodetool.migrations.state import APPLICATION_TABLES

        # First, check if migration tracking table exists
        if await self._adapter.table_exists(MIGRATION_TRACKING_TABLE):
            return DatabaseState.MIGRATION_TRACKED

        # Check if any application tables exist
        for table_name in APPLICATION_TABLES:
            if await self._adapter.table_exists(table_name):
                return DatabaseState.LEGACY_DATABASE

        # No tables exist - fresh install
        return DatabaseState.FRESH_INSTALL

    # -------------------------------------------------------------------------
    # Migration execution
    # -------------------------------------------------------------------------

    async def migrate(
        self,
        target: str | None = None,
        dry_run: bool = False,
        validate_checksums: bool = True,
    ) -> list[str]:
        """Apply pending migrations.

        Handles three database states:
        1. FRESH_INSTALL: Create tracking tables and run all migrations
        2. LEGACY_DATABASE: Create tracking tables and baseline existing migrations
        3. MIGRATION_TRACKED: Run only pending migrations

        Args:
            target: Optional target version (apply migrations up to and including this version)
            dry_run: If True, show what would be done without making changes
            validate_checksums: If True, validate checksums of applied migrations

        Returns:
            List of migration versions that were applied

        Raises:
            MigrationError: If migration fails
            ChecksumError: If checksum validation fails
            LockError: If lock cannot be acquired
        """
        # Detect database state
        db_state = await self._detect_database_state()
        log.info(f"Database state: {db_state.value}")

        # Create tracking tables if needed
        if db_state != DatabaseState.MIGRATION_TRACKED:
            if dry_run:
                log.info("[DRY RUN] Would create migration tracking tables")
            else:
                await self._create_tracking_tables()

        # Acquire lock
        if not dry_run:
            await self._acquire_lock()

        try:
            # Handle baselining for legacy databases
            if db_state == DatabaseState.LEGACY_DATABASE:
                if dry_run:
                    log.info("[DRY RUN] Would baseline existing migrations")
                else:
                    await self._baseline_migrations()

            # Validate checksums
            if validate_checksums and not dry_run:
                mismatches = await self.validate_checksums()
                if mismatches:
                    log.warning(
                        f"Checksum mismatch for migrations: {', '.join(mismatches)}. "
                        "Migration files may have been modified after application. "
                        "Continuing since this is a development environment."
                    )

            # Repair incorrectly baselined ALTER TABLE migrations
            # This fixes databases where an older version incorrectly baselined
            # ALTER TABLE migrations without actually executing them.
            if not dry_run:
                await self._repair_baselined_alter_migrations()

            # Get pending migrations
            applied = await self._get_applied_migrations()
            migrations = self.discover_migrations()
            pending = [m for m in migrations if m.version not in applied]

            # Filter by target if specified
            if target:
                pending = [m for m in pending if m.version <= target]

            if not pending:
                log.info("No pending migrations")
                return []

            log.info(f"Found {len(pending)} pending migration(s)")

            # Apply migrations
            applied_versions = []
            for migration in pending:
                if dry_run:
                    log.info(f"[DRY RUN] Would apply migration: {migration.version} ({migration.name})")
                    applied_versions.append(migration.version)
                else:
                    await self._apply_migration(migration)
                    applied_versions.append(migration.version)

            return applied_versions

        finally:
            if not dry_run:
                await self._release_lock()

    async def _apply_migration(self, migration: Migration) -> None:
        """Apply a single migration with transaction safety.

        Args:
            migration: The migration to apply

        Raises:
            MigrationError: If migration fails
        """
        log.info(f"Applying migration: {migration.version} ({migration.name})")
        start_time = time.time()

        try:
            # Execute migration's up function with the adapter
            assert self._adapter is not None, "Database adapter must be initialized"
            await migration.up(self._adapter)
            await self._adapter.commit()

            # Record successful migration
            execution_time_ms = int((time.time() - start_time) * 1000)
            await self._record_migration(migration, execution_time_ms)

            log.info(f"Migration {migration.version} applied successfully in {execution_time_ms}ms")

        except Exception as e:
            # Rollback on failure
            await self._adapter.rollback()
            raise MigrationError(
                f"Migration {migration.version} failed: {e}",
                migration_version=migration.version,
            ) from e

    # -------------------------------------------------------------------------
    # Repair incorrectly baselined migrations
    # -------------------------------------------------------------------------

    async def _repair_baselined_alter_migrations(self) -> None:
        """Repair ALTER TABLE migrations that were incorrectly baselined.

        Earlier versions of the migration runner would baseline ALTER TABLE
        migrations if the target table existed, without checking whether the
        actual column changes had been applied. This left databases with
        missing columns but the migration marked as applied.

        This method detects such cases and re-executes the migrations.
        The individual migration scripts contain safeguards (e.g., checking
        if a column exists before adding it), so it is safe to re-run them.
        """
        applied = await self._get_applied_migrations()
        migrations_map = {m.version: m for m in self.discover_migrations()}
        repaired = 0

        for version, applied_migration in applied.items():
            if not applied_migration.baselined:
                continue

            migration = migrations_map.get(version)
            if migration is None:
                continue

            if not migration.modifies_tables:
                continue

            # This ALTER TABLE migration was baselined — re-execute it
            log.info(
                f"Repairing incorrectly baselined ALTER TABLE migration: "
                f"{migration.version} ({migration.name})"
            )

            try:
                await migration.up(self._adapter)
                await self._adapter.commit()

                # Update the tracking record: mark as no longer baselined
                await self._adapter.execute(
                    f"""
                    UPDATE {MIGRATION_TRACKING_TABLE}
                    SET baselined = 0, applied_at = ?
                    WHERE version = ?
                    """,
                    (datetime.now(UTC).isoformat(), migration.version),
                )
                await self._adapter.commit()
                repaired += 1

            except Exception as e:
                await self._adapter.rollback()
                log.error(
                    f"Failed to repair migration {migration.version}: {e}. "
                    "The migration's own safeguards should prevent this — "
                    "please report this as a bug."
                )

        if repaired:
            log.info(f"Repaired {repaired} incorrectly baselined ALTER TABLE migration(s)")

    # -------------------------------------------------------------------------
    # Baselining
    # -------------------------------------------------------------------------

    async def _baseline_migrations(self) -> None:
        """Baseline migrations for a legacy database.

        For each migration, checks if the tables it creates already exist.
        If they do, marks the migration as applied without executing it.
        If they don't, executes the migration normally.

        For migrations that MODIFY tables (ALTER TABLE), we always execute
        them since the migration code has safeguards (e.g., checking if columns
        exist before adding them). This ensures legacy databases get all schema updates.
        """
        migrations = self.discover_migrations()
        baselined = 0
        executed = 0

        log.info("Detected existing database without migration tracking. Performing baseline...")

        for migration in migrations:
            # Check if the tables this migration creates already exist
            should_baseline = False
            if migration.creates_tables:
                # Check each table individually
                all_exist = True
                for table in migration.creates_tables:
                    if not await self._adapter.table_exists(table):
                        all_exist = False
                        break
                should_baseline = all_exist
            elif migration.modifies_tables:
                # For ALTER TABLE migrations, always execute them.
                # The migration code has safeguards like checking if columns exist
                # before adding them, so it's safe to run on legacy databases.
                # We can't just check if the table exists because the table might
                # be missing columns that this migration adds.
                should_baseline = False

            if should_baseline:
                # Mark as applied without executing
                await self._record_migration(migration, 0, baselined=True)
                baselined += 1
                log.debug(f"Baselined migration: {migration.version}")
            else:
                # Execute the migration
                await self._apply_migration(migration)
                executed += 1

        log.info(f"Baselining complete: {baselined} migrations baselined, {executed} migrations executed")

    async def baseline(self, force: bool = False) -> int:
        """Manually trigger baselining for the current database state.

        This is useful when you need to re-baseline a database or when
        automatic detection doesn't work correctly.

        Args:
            force: If True, baseline even if tracking table exists

        Returns:
            Number of migrations that were baselined

        Raises:
            BaselineError: If baselining fails
        """
        await self._acquire_lock()

        try:
            # Check current state
            has_tracking = await self._adapter.table_exists(MIGRATION_TRACKING_TABLE)

            if has_tracking and not force:
                raise BaselineError("Migration tracking already exists. Use --force to re-baseline.")

            if not has_tracking:
                await self._create_tracking_tables()

            # Clear existing records if forcing
            if force and has_tracking:
                await self._adapter.execute(f"DELETE FROM {MIGRATION_TRACKING_TABLE}")
                await self._adapter.commit()

            # Baseline all migrations
            migrations = self.discover_migrations()
            baselined = 0

            for migration in migrations:
                # Check if tables exist
                should_baseline = False
                if migration.creates_tables:
                    all_exist = True
                    for table in migration.creates_tables:
                        if not await self._adapter.table_exists(table):
                            all_exist = False
                            break
                    should_baseline = all_exist
                elif migration.modifies_tables:
                    # For ALTER TABLE migrations, always execute them.
                    # The migration code has safeguards like checking if columns
                    # exist before adding them, so it's safe to re-run.
                    should_baseline = False

                if should_baseline:
                    await self._record_migration(migration, 0, baselined=True)
                    baselined += 1
                    log.info(f"Baselined: {migration.version} ({migration.name})")
                else:
                    await self._apply_migration(migration)
                    log.info(f"Executed: {migration.version} ({migration.name})")

            return baselined

        finally:
            await self._release_lock()

    # -------------------------------------------------------------------------
    # Rollback
    # -------------------------------------------------------------------------

    async def rollback(self, steps: int = 1) -> list[str]:
        """Rollback the last N migrations.

        Args:
            steps: Number of migrations to rollback

        Returns:
            List of migration versions that were rolled back

        Raises:
            RollbackError: If rollback fails
        """
        await self._acquire_lock()

        try:
            applied = await self._get_applied_migrations()
            migrations_map = {m.version: m for m in self.discover_migrations()}

            # Get migrations to rollback (most recent first)
            versions_to_rollback = sorted(applied.keys(), reverse=True)[:steps]

            if not versions_to_rollback:
                log.info("No migrations to rollback")
                return []

            rolled_back = []
            for version in versions_to_rollback:
                applied_migration = applied[version]

                # Skip baselined migrations - they weren't actually applied
                if applied_migration.baselined:
                    log.warning(f"Skipping rollback of baselined migration: {version}")
                    await self._remove_migration_record(version)
                    rolled_back.append(version)
                    continue

                if version not in migrations_map:
                    raise RollbackError(
                        f"Migration {version} not found in migrations directory. Cannot rollback.",
                        migration_version=version,
                    )

                migration = migrations_map[version]
                await self._rollback_migration(migration)
                rolled_back.append(version)

            return rolled_back

        finally:
            await self._release_lock()

    async def _rollback_migration(self, migration: Migration) -> None:
        """Rollback a single migration.

        Args:
            migration: The migration to rollback

        Raises:
            RollbackError: If rollback fails
        """
        log.info(f"Rolling back migration: {migration.version} ({migration.name})")

        try:
            assert self._adapter is not None, "Database adapter must be initialized"
            await migration.down(self._adapter)
            await self._adapter.commit()

            await self._remove_migration_record(migration.version)

            log.info(f"Migration {migration.version} rolled back successfully")

        except Exception as e:
            await self._adapter.rollback()
            raise RollbackError(
                f"Rollback of migration {migration.version} failed: {e}",
                migration_version=migration.version,
            ) from e

    # -------------------------------------------------------------------------
    # Status and info
    # -------------------------------------------------------------------------

    async def status(self) -> dict[str, Any]:
        """Get the current migration status.

        Returns:
            Dictionary with:
            - state: Current database state
            - current_version: Latest applied migration version
            - applied: List of applied migrations with details
            - pending: List of pending migrations
        """
        db_state = await self._detect_database_state()

        if db_state == DatabaseState.FRESH_INSTALL:
            return {
                "state": db_state.value,
                "current_version": None,
                "applied": [],
                "pending": [{"version": m.version, "name": m.name} for m in self.discover_migrations()],
            }

        applied = await self._get_applied_migrations()
        all_migrations = self.discover_migrations()
        pending = [m for m in all_migrations if m.version not in applied]

        return {
            "state": db_state.value,
            "current_version": max(applied.keys()) if applied else None,
            "applied": [
                {
                    "version": m.version,
                    "name": m.name,
                    "applied_at": m.applied_at.isoformat(),
                    "execution_time_ms": m.execution_time_ms,
                    "baselined": m.baselined,
                }
                for m in applied.values()
            ],
            "pending": [{"version": m.version, "name": m.name} for m in pending],
        }

    async def get_current_version(self) -> str | None:
        """Get the current (latest applied) migration version.

        Returns:
            Latest applied migration version, or None if no migrations applied
        """
        applied = await self._get_applied_migrations()
        if not applied:
            return None
        return max(applied.keys())

    def get_pending_count(self) -> int:
        """Get the number of pending migrations.

        Returns:
            Number of migrations that haven't been applied yet
        """
        # This is a sync method since it only looks at files
        migrations = self.discover_migrations()
        # We can't easily check applied without async, so return total
        return len(migrations)
