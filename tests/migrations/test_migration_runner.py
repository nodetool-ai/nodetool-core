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

import tempfile
from pathlib import Path

import aiosqlite
import pytest
import pytest_asyncio


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
        cursor = await test_db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test_table'")
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
        cursor = await test_db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test_table'")
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
            MIGRATION_TRACKING_TABLE,
            DatabaseState,
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
        await runner.migrate()

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
        cursor = await test_db.execute(f"SELECT locked_at, locked_by FROM {MIGRATION_LOCK_TABLE} WHERE id = 1")
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] is None  # locked_at should be NULL
        assert row[1] is None  # locked_by should be NULL


@pytest_asyncio.fixture
async def isolated_db():
    """Create a truly isolated file-based database for tests that need isolation."""
    import os

    with tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False) as temp_db:
        db_path = temp_db.name

    conn = await aiosqlite.connect(db_path)
    conn.row_factory = aiosqlite.Row
    try:
        yield conn
    finally:
        await conn.close()
        os.unlink(db_path)


class TestMigrationFailure:
    """Tests for migration failure scenarios."""

    @pytest.mark.asyncio
    async def test_migration_rollback_on_failure(self, isolated_db, tmp_path):
        """Test that failed migration's state is handled correctly.

        Note: SQLite DDL statements are auto-committing, so CREATE TABLE cannot
        be rolled back in the same way as DML statements. However, we verify that:
        1. The migration error is properly raised
        2. The migration is NOT recorded in the tracking table
        3. The database remains in a consistent state
        """
        from nodetool.migrations.exceptions import MigrationError
        from nodetool.migrations.runner import MigrationRunner
        from nodetool.migrations.state import MIGRATION_TRACKING_TABLE

        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()

        # Create a migration that fails after creating a table
        bad_migration = migrations_dir / "20250101_000000_failing_migration.py"
        bad_migration.write_text('''"""
Migration: Failing migration
Version: 20250101_000000
"""

version = "20250101_000000"
name = "failing_migration"
creates_tables = ["test_table"]
modifies_tables = []

async def up(db):
    # Create a table first (SQLite DDL is auto-committing)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS test_table (
            id TEXT PRIMARY KEY,
            name TEXT
        )
    """)
    # Then explicitly fail
    raise RuntimeError("Migration failed intentionally")

async def down(db):
    await db.execute("DROP TABLE IF EXISTS test_table")
''')

        runner = MigrationRunner(isolated_db, migrations_dir=migrations_dir)

        # Migration should fail with MigrationError wrapping our RuntimeError
        with pytest.raises(MigrationError):
            await runner.migrate()

        # Verify NO migration records were created (even if table exists due to DDL auto-commit)
        cursor = await isolated_db.execute(f"SELECT * FROM {MIGRATION_TRACKING_TABLE}")
        rows = await cursor.fetchall()
        assert len(rows) == 0, "Failed migration should not be recorded"

        # Note: The table may still exist due to SQLite DDL auto-commit behavior
        # This is expected - SQLite DDL cannot be rolled back like DML

    @pytest.mark.asyncio
    async def test_partial_migration_rollback(self, isolated_db, tmp_path):
        """Test that successful migrations persist even if later ones fail.

        Each migration runs in its own transaction. When migration 2 fails,
        migration 1's committed changes persist (correct behavior -
        don't undo successful work).
        """
        from nodetool.migrations.exceptions import MigrationError
        from nodetool.migrations.runner import MigrationRunner
        from nodetool.migrations.state import MIGRATION_TRACKING_TABLE

        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()

        # Create first migration that succeeds
        migration1 = migrations_dir / "20250101_000000_first_migration.py"
        migration1.write_text('''"""
Migration: First migration
Version: 20250101_000000
"""

version = "20250101_000000"
name = "first_migration"
creates_tables = ["first_table"]
modifies_tables = []

async def up(db):
    await db.execute("""
        CREATE TABLE IF NOT EXISTS first_table (
            id TEXT PRIMARY KEY,
            name TEXT
        )
    """)

async def down(db):
    await db.execute("DROP TABLE IF EXISTS first_table")
''')

        # Create second migration that fails
        migration2 = migrations_dir / "20250101_000001_second_migration.py"
        migration2.write_text('''"""
Migration: Second migration
Version: 20250101_000001
"""

version = "20250101_000001"
name = "second_migration"
creates_tables = []
modifies_tables = []

async def up(db):
    await db.execute("""
        CREATE TABLE IF NOT EXISTS second_table (
            id TEXT PRIMARY KEY
        )
    """)
    raise RuntimeError("Simulated migration failure")

async def down(db):
    await db.execute("DROP TABLE IF EXISTS second_table")
''')

        runner = MigrationRunner(isolated_db, migrations_dir=migrations_dir)

        # Migration should fail (MigrationError wraps RuntimeError)
        with pytest.raises(MigrationError):
            await runner.migrate()

        # First migration succeeded and committed, so first_table should exist
        cursor = await isolated_db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='first_table'")
        result = await cursor.fetchone()
        assert result is not None, "First migration should have committed (don't undo successful work)"

        # Second migration failed, so second_table should NOT exist (or may exist due to DDL auto-commit)
        # But importantly, second_table should NOT be recorded in migrations

        # Only first migration should be recorded
        cursor = await isolated_db.execute(f"SELECT * FROM {MIGRATION_TRACKING_TABLE}")
        rows = await cursor.fetchall()
        assert len(rows) == 1
        assert rows[0]["version"] == "20250101_000000"

    @pytest.mark.asyncio
    async def test_failed_migration_leaves_db_consistent(self, isolated_db, tmp_path):
        """Test that failed migration leaves database in consistent state."""
        from nodetool.migrations.runner import MigrationRunner
        from nodetool.migrations.state import MIGRATION_TRACKING_TABLE

        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()

        # Create a migration that fails
        bad_migration = migrations_dir / "20250101_000000_bad_migration.py"
        bad_migration.write_text('''"""
Migration: Bad migration
Version: 20250101_000000
"""

version = "20250101_000000"
name = "bad_migration"
creates_tables = []
modifies_tables = []

async def up(db):
    await db.execute("SELECT * FROM nonexistent_table_xyz")

async def down(db):
    pass
''')

        runner = MigrationRunner(isolated_db, migrations_dir=migrations_dir)

        # Try to migrate - should fail
        try:
            await runner.migrate()
        except Exception:
            pass

        # Verify the connection is still usable
        cursor = await isolated_db.execute("SELECT 1")
        result = await cursor.fetchone()
        assert result[0] == 1

        # Verify no migration records were created
        cursor = await isolated_db.execute(f"SELECT * FROM {MIGRATION_TRACKING_TABLE}")
        rows = await cursor.fetchall()
        assert len(rows) == 0


class TestConcurrentMigrations:
    """Tests for concurrent migration scenarios."""

    @pytest.mark.asyncio
    async def test_migration_on_isolated_db(self, isolated_db, tmp_path):
        """Test that migrations work correctly on isolated database."""
        from nodetool.migrations.runner import MigrationRunner

        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()

        migration = migrations_dir / "20250101_000000_test_migration.py"
        migration.write_text('''"""
Migration: Test
Version: 20250101_000000
"""

version = "20250101_000000"
name = "test_migration"
creates_tables = ["test_table"]
modifies_tables = []

async def up(db):
    await db.execute("""
        CREATE TABLE IF NOT EXISTS test_table (
            id TEXT PRIMARY KEY,
            name TEXT
        )
    """)

async def down(db):
    await db.execute("DROP TABLE IF EXISTS test_table")
''')

        runner = MigrationRunner(isolated_db, migrations_dir=migrations_dir)

        # Run migration
        applied = await runner.migrate()

        # Migration should succeed
        assert len(applied) == 1

        # Verify table was created
        cursor = await isolated_db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test_table'")
        result = await cursor.fetchone()
        assert result is not None

        # Verify lock table was created
        cursor = await isolated_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='_nodetool_migration_lock'"
        )
        result = await cursor.fetchone()
        assert result is not None


class TestEdgeCases:
    """Tests for various edge cases."""

    @pytest.mark.asyncio
    async def test_migration_with_empty_creates_tables(self, isolated_db, tmp_path):
        """Test migration that doesn't create tables (e.g., only alters)."""
        from nodetool.migrations.runner import MigrationRunner

        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()

        # Create a table first (simulating existing schema)
        await isolated_db.execute("""
            CREATE TABLE nodetool_custom_table (
                id TEXT PRIMARY KEY,
                name TEXT
            )
        """)
        await isolated_db.commit()

        # Create migration that only modifies (no creates_tables)
        migration = migrations_dir / "20250101_000000_alter_migration.py"
        migration.write_text('''"""
Migration: Alter migration
Version: 20250101_000000
"""

version = "20250101_000000"
name = "alter_migration"
creates_tables = []
modifies_tables = ["nodetool_custom_table"]

async def up(db):
    await db.execute("""
        ALTER TABLE nodetool_custom_table ADD COLUMN description TEXT
    """)

async def down(db):
    pass
''')

        runner = MigrationRunner(isolated_db, migrations_dir=migrations_dir)
        applied = await runner.migrate()

        assert len(applied) == 1

        # Verify column was added
        cursor = await isolated_db.execute("PRAGMA table_info(nodetool_custom_table)")
        columns = await cursor.fetchall()
        column_names = [col[1] for col in columns]
        assert "description" in column_names

    @pytest.mark.asyncio
    async def test_duplicate_migration_versions(self, test_db, tmp_path):
        """Test that duplicate version numbers are handled correctly."""
        from nodetool.migrations.runner import MigrationRunner

        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()

        # Create two migrations with same version (should be sorted by name)
        migration1 = migrations_dir / "20250101_000000_first.py"
        migration1.write_text('''"""
Migration: First
Version: 20250101_000000
"""

version = "20250101_000000"
name = "first"
creates_tables = ["table_a"]
modifies_tables = []

async def up(db):
    await db.execute("CREATE TABLE IF NOT EXISTS table_a (id TEXT PRIMARY KEY)")

async def down(db):
    await db.execute("DROP TABLE IF EXISTS table_a")
''')

        migration2 = migrations_dir / "20250101_000000_second.py"
        migration2.write_text('''"""
Migration: Second
Version: 20250101_000000
"""

version = "20250101_000000"
name = "second"
creates_tables = ["table_b"]
modifies_tables = []

async def up(db):
    await db.execute("CREATE TABLE IF NOT EXISTS table_b (id TEXT PRIMARY KEY)")

async def down(db):
    await db.execute("DROP TABLE IF EXISTS table_b")
''')

        runner = MigrationRunner(test_db, migrations_dir=migrations_dir)
        migrations = runner.discover_migrations()

        # Both migrations should be discovered
        assert len(migrations) == 2

    @pytest.mark.asyncio
    async def test_migration_with_no_changes(self, test_db, tmp_path):
        """Test migration that has no effect (idempotent)."""
        from nodetool.migrations.runner import MigrationRunner

        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()

        migration = migrations_dir / "20250101_000000_no_op.py"
        migration.write_text('''"""
Migration: No-op
Version: 20250101_000000
"""

version = "20250101_000000"
name = "no_op"
creates_tables = []
modifies_tables = []

async def up(db):
    pass  # No changes

async def down(db):
    pass
''')

        runner = MigrationRunner(test_db, migrations_dir=migrations_dir)
        applied = await runner.migrate()

        assert len(applied) == 1

        # Verify tracking table was created
        status = await runner.status()
        assert status["state"] == "migration_tracked"
        assert len(status["applied"]) == 1
