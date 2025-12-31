"""
Tests for the MigrationRunner class.

Tests cover:
- Migration discovery
- Migration execution
- Rollback functionality
- Status reporting
- Checksum validation
- Locking mechanism
"""

import pytest
import pytest_asyncio
import tempfile
from pathlib import Path

import aiosqlite


@pytest_asyncio.fixture
async def test_db():
    """Create a temporary in-memory SQLite database for testing."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    try:
        yield conn
    finally:
        await conn.close()


@pytest_asyncio.fixture
async def temp_migrations_dir(tmp_path):
    """Create a temporary migrations directory with test migration files."""
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()

    # Create a test migration file
    migration1 = migrations_dir / "20250101_000000_create_test_table.py"
    migration1.write_text('''"""
Migration: Create test table
Version: 20250101_000000
"""

version = "20250101_000000"
name = "create_test_table"
creates_tables = ["test_table"]
modifies_tables = []

async def up(db):
    await db.execute("""
        CREATE TABLE IF NOT EXISTS test_table (
            id TEXT PRIMARY KEY,
            name TEXT,
            value INTEGER
        )
    """)

async def down(db):
    await db.execute("DROP TABLE IF EXISTS test_table")
''')

    # Create a second test migration
    migration2 = migrations_dir / "20250101_000001_add_description.py"
    migration2.write_text('''"""
Migration: Add description column
Version: 20250101_000001
"""

version = "20250101_000001"
name = "add_description"
creates_tables = []
modifies_tables = ["test_table"]

async def up(db):
    await db.execute("""
        ALTER TABLE test_table ADD COLUMN description TEXT
    """)

async def down(db):
    # SQLite doesn't support DROP COLUMN in older versions
    pass
''')

    return migrations_dir


class TestMigrationRunner:
    """Tests for the MigrationRunner class."""

    @pytest.mark.asyncio
    async def test_discover_migrations(self, test_db, temp_migrations_dir):
        """Test that migration files are discovered and loaded correctly."""
        from nodetool.migrations.runner import MigrationRunner

        runner = MigrationRunner(test_db, migrations_dir=temp_migrations_dir)
        migrations = runner.discover_migrations()

        assert len(migrations) == 2
        assert migrations[0].version == "20250101_000000"
        assert migrations[0].name == "create_test_table"
        assert migrations[1].version == "20250101_000001"
        assert migrations[1].name == "add_description"

    @pytest.mark.asyncio
    async def test_migrate_fresh_install(self, test_db, temp_migrations_dir):
        """Test migration on a fresh database (no tables)."""
        from nodetool.migrations.runner import MigrationRunner
        from nodetool.migrations.state import DatabaseState, detect_database_state_sqlite

        runner = MigrationRunner(test_db, migrations_dir=temp_migrations_dir)

        # Verify fresh install state
        state = await detect_database_state_sqlite(test_db)
        assert state == DatabaseState.FRESH_INSTALL

        # Run migrations
        applied = await runner.migrate()

        assert len(applied) == 2
        assert "20250101_000000" in applied
        assert "20250101_000001" in applied

        # Verify table was created
        cursor = await test_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='test_table'"
        )
        result = await cursor.fetchone()
        assert result is not None

    @pytest.mark.asyncio
    async def test_migrate_idempotent(self, test_db, temp_migrations_dir):
        """Test that running migrate twice doesn't cause errors."""
        from nodetool.migrations.runner import MigrationRunner

        runner = MigrationRunner(test_db, migrations_dir=temp_migrations_dir)

        # First migration
        applied1 = await runner.migrate()
        assert len(applied1) == 2

        # Second migration (should be no-op)
        applied2 = await runner.migrate()
        assert len(applied2) == 0

    @pytest.mark.asyncio
    async def test_status(self, test_db, temp_migrations_dir):
        """Test the status method."""
        from nodetool.migrations.runner import MigrationRunner

        runner = MigrationRunner(test_db, migrations_dir=temp_migrations_dir)

        # Before migration
        status1 = await runner.status()
        assert status1["state"] == "fresh_install"
        assert len(status1["pending"]) == 2
        assert len(status1["applied"]) == 0

        # Run migrations
        await runner.migrate()

        # After migration
        status2 = await runner.status()
        assert status2["state"] == "migration_tracked"
        assert len(status2["pending"]) == 0
        assert len(status2["applied"]) == 2
        assert status2["current_version"] == "20250101_000001"

    @pytest.mark.asyncio
    async def test_rollback(self, test_db, temp_migrations_dir):
        """Test rolling back migrations."""
        from nodetool.migrations.runner import MigrationRunner

        runner = MigrationRunner(test_db, migrations_dir=temp_migrations_dir)

        # Run migrations
        await runner.migrate()

        # Rollback one migration
        rolled_back = await runner.rollback(steps=1)
        assert len(rolled_back) == 1
        assert "20250101_000001" in rolled_back

        # Check status
        status = await runner.status()
        assert len(status["applied"]) == 1
        assert len(status["pending"]) == 1

    @pytest.mark.asyncio
    async def test_dry_run(self, test_db, temp_migrations_dir):
        """Test dry run mode doesn't make changes."""
        from nodetool.migrations.runner import MigrationRunner

        runner = MigrationRunner(test_db, migrations_dir=temp_migrations_dir)

        # Dry run
        applied = await runner.migrate(dry_run=True)
        assert len(applied) == 2

        # Verify no tables were created
        cursor = await test_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='test_table'"
        )
        result = await cursor.fetchone()
        assert result is None

    @pytest.mark.asyncio
    async def test_target_version(self, test_db, temp_migrations_dir):
        """Test migrating to a specific target version."""
        from nodetool.migrations.runner import MigrationRunner

        runner = MigrationRunner(test_db, migrations_dir=temp_migrations_dir)

        # Migrate to first version only
        applied = await runner.migrate(target="20250101_000000")
        assert len(applied) == 1
        assert "20250101_000000" in applied

        # Check status - one pending
        status = await runner.status()
        assert len(status["applied"]) == 1
        assert len(status["pending"]) == 1

    @pytest.mark.asyncio
    async def test_checksum_validation(self, test_db, temp_migrations_dir):
        """Test that checksum validation detects modified migrations."""
        from nodetool.migrations.runner import MigrationRunner

        runner = MigrationRunner(test_db, migrations_dir=temp_migrations_dir)

        # Run migrations
        await runner.migrate()

        # Modify a migration file
        migration_file = temp_migrations_dir / "20250101_000000_create_test_table.py"
        original_content = migration_file.read_text()
        migration_file.write_text(original_content + "\n# Modified")

        # Clear migration cache to force reload
        runner._migrations_cache = None

        # Validate checksums - should detect mismatch
        mismatches = await runner.validate_checksums()
        assert len(mismatches) == 1
        assert "20250101_000000" in mismatches


class TestDatabaseStateDetection:
    """Tests for database state detection."""

    @pytest.mark.asyncio
    async def test_detect_fresh_install(self, test_db):
        """Test detection of fresh install (no tables)."""
        from nodetool.migrations.state import DatabaseState, detect_database_state_sqlite

        state = await detect_database_state_sqlite(test_db)
        assert state == DatabaseState.FRESH_INSTALL

    @pytest.mark.asyncio
    async def test_detect_legacy_database(self, test_db):
        """Test detection of legacy database (tables but no tracking)."""
        from nodetool.migrations.state import DatabaseState, detect_database_state_sqlite

        # Create an application table without migration tracking
        await test_db.execute("""
            CREATE TABLE nodetool_workflows (
                id TEXT PRIMARY KEY,
                name TEXT
            )
        """)
        await test_db.commit()

        state = await detect_database_state_sqlite(test_db)
        assert state == DatabaseState.LEGACY_DATABASE

    @pytest.mark.asyncio
    async def test_detect_migration_tracked(self, test_db):
        """Test detection of database with migration tracking."""
        from nodetool.migrations.state import (
            DatabaseState,
            MIGRATION_TRACKING_TABLE,
            detect_database_state_sqlite,
        )

        # Create the migration tracking table
        await test_db.execute(f"""
            CREATE TABLE {MIGRATION_TRACKING_TABLE} (
                version TEXT PRIMARY KEY,
                name TEXT,
                checksum TEXT,
                applied_at TEXT,
                execution_time_ms INTEGER,
                baselined INTEGER DEFAULT 0
            )
        """)
        await test_db.commit()

        state = await detect_database_state_sqlite(test_db)
        assert state == DatabaseState.MIGRATION_TRACKED


class TestBaselining:
    """Tests for migration baselining."""

    @pytest.mark.asyncio
    async def test_baseline_legacy_database(self, test_db, temp_migrations_dir):
        """Test baselining a legacy database."""
        from nodetool.migrations.runner import MigrationRunner
        from nodetool.migrations.state import DatabaseState, detect_database_state_sqlite

        # Create a nodetool_* table manually (simulating legacy database)
        # This triggers LEGACY_DATABASE detection
        await test_db.execute("""
            CREATE TABLE nodetool_workflows (
                id TEXT PRIMARY KEY,
                name TEXT
            )
        """)
        # Also create the test_table that our migration creates
        await test_db.execute("""
            CREATE TABLE test_table (
                id TEXT PRIMARY KEY,
                name TEXT,
                value INTEGER
            )
        """)
        await test_db.commit()

        # Verify legacy state
        state = await detect_database_state_sqlite(test_db)
        assert state == DatabaseState.LEGACY_DATABASE

        runner = MigrationRunner(test_db, migrations_dir=temp_migrations_dir)

        # Run migrate - should baseline the first migration since test_table exists
        applied = await runner.migrate()

        # First migration should be baselined, second should be executed
        status = await runner.status()
        assert len(status["applied"]) == 2

        # Find the baselined migration
        baselined = [m for m in status["applied"] if m["baselined"]]
        assert len(baselined) >= 1


class TestLocking:
    """Tests for migration locking mechanism."""

    @pytest.mark.asyncio
    async def test_lock_acquired_and_released(self, test_db, temp_migrations_dir):
        """Test that lock is properly acquired and released."""
        from nodetool.migrations.runner import MigrationRunner
        from nodetool.migrations.state import MIGRATION_LOCK_TABLE

        runner = MigrationRunner(test_db, migrations_dir=temp_migrations_dir)

        # Run migration (acquires and releases lock)
        await runner.migrate()

        # Check lock is released
        cursor = await test_db.execute(
            f"SELECT locked_at, locked_by FROM {MIGRATION_LOCK_TABLE} WHERE id = 1"
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] is None  # locked_at should be NULL
        assert row[1] is None  # locked_by should be NULL
