"""
Tests for the MigrationRunner with PostgreSQL.

These tests require a running PostgreSQL database and are marked with the 'postgres' marker.
Run them with: pytest -m postgres tests/migrations/test_postgres_migration_runner.py

To start a PostgreSQL test database:
    docker run -d --name nodetool-test-postgres \
        -e POSTGRES_PASSWORD=testpass \
        -e POSTGRES_USER=testuser \
        -e POSTGRES_DB=testdb \
        -p 5433:5432 postgres:15-alpine
"""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

# Skip all tests in this file if psycopg is not installed or postgres is not available
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.postgres,
]

# Test PostgreSQL connection URL (can be overridden via environment variable)
POSTGRES_URL = os.environ.get("POSTGRES_TEST_URL", "postgresql://testuser:testpass@localhost:5433/testdb")


def postgres_available():
    """Check if PostgreSQL psycopg_pool is available for testing."""
    try:
        from psycopg_pool import AsyncConnectionPool

        # Verify the import works - use the imported class
        _ = AsyncConnectionPool
        return True
    except ImportError:
        return False


@pytest_asyncio.fixture
async def postgres_pool():
    """Create a PostgreSQL connection pool for testing."""
    if not postgres_available():
        pytest.skip("psycopg_pool not installed")

    from psycopg_pool import AsyncConnectionPool

    try:
        pool = AsyncConnectionPool(POSTGRES_URL, min_size=1, max_size=5)
        await pool.open()
    except Exception as e:
        pytest.skip(f"PostgreSQL not available: {e}")
        return

    # Clean up any existing migration tables
    try:
        async with pool.connection() as conn:
            await conn.execute("DROP TABLE IF EXISTS _nodetool_migration_lock CASCADE")
            await conn.execute("DROP TABLE IF EXISTS _nodetool_migrations CASCADE")
            await conn.execute("DROP TABLE IF EXISTS test_table CASCADE")
            await conn.execute("DROP TABLE IF EXISTS nodetool_workflows CASCADE")
            await conn.commit()
    except Exception as e:
        pytest.skip(f"Failed to setup test database: {e}")
        await pool.close()
        return

    try:
        yield pool
    finally:
        # Clean up after tests
        try:
            async with pool.connection() as conn:
                await conn.execute("DROP TABLE IF EXISTS _nodetool_migration_lock CASCADE")
                await conn.execute("DROP TABLE IF EXISTS _nodetool_migrations CASCADE")
                await conn.execute("DROP TABLE IF EXISTS test_table CASCADE")
                await conn.execute("DROP TABLE IF EXISTS nodetool_workflows CASCADE")
                await conn.commit()
        except Exception:
            pass
        await pool.close()


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
    await db.execute("""
        ALTER TABLE test_table DROP COLUMN description
    """)
''')

    return migrations_dir


class TestPostgresMigrationRunner:
    """Tests for the MigrationRunner with PostgreSQL."""

    async def test_postgres_migration_fresh_install(self, postgres_pool, temp_migrations_dir):
        """Test migration on a fresh PostgreSQL database."""
        from nodetool.migrations.runner import MigrationRunner

        runner = MigrationRunner(postgres_pool, migrations_dir=temp_migrations_dir)

        # Run migrations
        applied = await runner.migrate()

        assert len(applied) == 2
        assert "20250101_000000" in applied
        assert "20250101_000001" in applied

        # Verify table was created
        async with postgres_pool.connection() as conn, conn.cursor() as cursor:
            await cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'test_table'")
            columns = [row[0] for row in await cursor.fetchall()]
            assert "id" in columns
            assert "name" in columns
            assert "value" in columns
            assert "description" in columns

    async def test_postgres_migration_idempotent(self, postgres_pool, temp_migrations_dir):
        """Test that running migrate twice doesn't cause errors."""
        from nodetool.migrations.runner import MigrationRunner

        runner = MigrationRunner(postgres_pool, migrations_dir=temp_migrations_dir)

        # First migration
        applied1 = await runner.migrate()
        assert len(applied1) == 2

        # Second migration (should be no-op)
        applied2 = await runner.migrate()
        assert len(applied2) == 0

    async def test_postgres_status(self, postgres_pool, temp_migrations_dir):
        """Test the status method with PostgreSQL."""
        from nodetool.migrations.runner import MigrationRunner

        runner = MigrationRunner(postgres_pool, migrations_dir=temp_migrations_dir)

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

    async def test_postgres_rollback(self, postgres_pool, temp_migrations_dir):
        """Test rolling back migrations on PostgreSQL."""
        from nodetool.migrations.runner import MigrationRunner

        runner = MigrationRunner(postgres_pool, migrations_dir=temp_migrations_dir)

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

        # Verify column was removed
        async with postgres_pool.connection() as conn, conn.cursor() as cursor:
            await cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'test_table'")
            columns = [row[0] for row in await cursor.fetchall()]
            assert "description" not in columns

    async def test_postgres_checksum_validation(self, postgres_pool, temp_migrations_dir):
        """Test that checksum validation works with PostgreSQL."""
        from nodetool.migrations.runner import MigrationRunner

        runner = MigrationRunner(postgres_pool, migrations_dir=temp_migrations_dir)

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
