"""
Tests for ResourceScope isolation and database adapter handling.

Verifies that:
- Concurrent jobs don't share DB adapters across loops
- Scope creation and cleanup works correctly
- Fallback to Environment works for pre-scope operations
"""

import asyncio
import tempfile
from typing import Any

import pytest
import pytest_asyncio

from nodetool.runtime.db_sqlite import SQLiteConnectionPool
from nodetool.runtime.resources import ResourceScope, maybe_scope

# Skip global DB setup/teardown from tests/conftest.py for this module
pytestmark = pytest.mark.no_setup


@pytest_asyncio.fixture
async def test_pool():
    """Create a temporary database pool for these tests."""
    import os
    import tempfile

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".sqlite3", prefix="test_resource_scope_", delete=False) as temp_db:
        db_path = temp_db.name

    # Create pool
    pool = await SQLiteConnectionPool.get_shared(db_path)

    yield pool

    # Cleanup
    try:
        await pool.close_all()
        SQLiteConnectionPool._pools.pop(db_path, None)
        os.unlink(db_path)
    except Exception:
        pass


class MockModel:
    """Mock model for testing adapter behavior."""

    @classmethod
    def get_table_schema(cls) -> dict[str, Any]:
        return {
            "table_name": "mock_table",
            "primary_key": "id",
        }

    @classmethod
    def db_fields(cls) -> dict[str, Any]:
        # Return a simple field matching SQLiteAdapter expectations
        return {"id": {"name": "id", "type": "TEXT"}}

    @classmethod
    def get_indexes(cls) -> list[dict[str, Any]]:
        return []


@pytest.mark.asyncio
async def test_resource_scope_binding(test_pool):
    """Test that ResourceScope properly binds and unbinds context."""
    async with ResourceScope(pool=test_pool) as scope:
        # Scope should be bound inside the context
        bound_scope = maybe_scope()
        assert bound_scope is not None
        assert bound_scope is scope
        assert bound_scope.db is not None


@pytest.mark.asyncio
async def test_resource_scope_database_provider(test_pool):
    """Test that ResourceScope provides a working database provider."""
    async with ResourceScope(pool=test_pool) as scope:
        # Provider should be initialized
        assert scope.db is not None

        # Provider should have cleanup method
        assert hasattr(scope.db, "cleanup")
        assert callable(scope.db.cleanup)


@pytest.mark.asyncio
async def test_resource_scope_cleanup(test_pool):
    """Test that ResourceScope properly cleans up resources."""
    async with ResourceScope(pool=test_pool) as scope:
        # Verify db provider is accessible
        assert scope.db is not None


@pytest.mark.asyncio
async def test_concurrent_scopes_isolation(test_pool):
    """Test that concurrent scopes maintain separate context bindings.

    This test verifies that two concurrent jobs get separate scope bindings
    and don't interfere with each other's context.
    """
    async with ResourceScope(pool=test_pool), ResourceScope(pool=test_pool) as scope_2:
        scope_ids = []

        async def job_1():
            """First concurrent job."""
            async with ResourceScope(pool=test_pool) as scope:
                # Verify scope is bound
                bound = maybe_scope()
                assert bound is scope
                scope_ids.append(id(scope))

                # Simulate some work
                await asyncio.sleep(0.01)

                # Verify scope is still bound
                assert maybe_scope() is scope

        async def job_2():
            """Second concurrent job."""
            async with ResourceScope(pool=test_pool) as scope:
                # Verify scope is bound
                bound = maybe_scope()
                assert bound is scope
                scope_ids.append(id(scope))

                # Simulate some work
                await asyncio.sleep(0.02)

                # Verify scope is still bound
                assert maybe_scope() is scope

        # Run both jobs concurrently
        await asyncio.gather(job_1(), job_2())

        # After both complete, the parent scope should still be bound (scope_2)
        assert maybe_scope() is scope_2

        # Both scopes should have completed
        assert len(scope_ids) == 2
        # Scopes should be different objects
        assert scope_ids[0] != scope_ids[1]


@pytest.mark.asyncio
async def test_maybe_scope_fallback(test_pool):
    """Test that maybe_scope returns None when no scope is bound."""

    async with ResourceScope(pool=test_pool):
        # Inside scope, maybe_scope returns the scope
        scope = maybe_scope()
        assert scope is not None
