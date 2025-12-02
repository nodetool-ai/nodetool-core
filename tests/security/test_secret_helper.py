"""
Tests for secret helper functions.
"""

import os

import pytest

from nodetool.models.secret import Secret
from nodetool.security.secret_helper import (
    get_secret,
    get_secret_required,
    get_secret_sync,
    has_secret,
)


@pytest.mark.asyncio
class TestSecretHelper:
    """Tests for secret helper functions."""

    async def test_get_secret_from_database(self):
        """Test getting secret from database."""
        user_id = "test_user_helper"
        key = "TEST_DB_SECRET"
        value = "test_value_123"

        # Create secret in database
        await Secret.create(user_id=user_id, key=key, value=value)

        # Retrieve using helper
        result = await get_secret(key, user_id)

        assert result == value

    async def test_get_secret_from_env(self):
        """Test that environment variables take priority."""
        user_id = "test_user_env"
        key = "TEST_ENV_SECRET"
        env_value = "env_value_123"
        db_value = "db_value_456"

        # Create secret in database
        await Secret.create(user_id=user_id, key=key, value=db_value)

        # Set environment variable
        os.environ[key] = env_value

        try:
            # Retrieve using helper
            result = await get_secret(key, user_id)

            # Should return env value (higher priority)
            assert result == env_value
        finally:
            # Clean up
            del os.environ[key]

    async def test_get_secret_not_found(self):
        """Test getting secret that doesn't exist."""
        result = await get_secret("NONEXISTENT_SECRET", "test_user")

        assert result is None

    async def test_get_secret_with_default(self):
        """Test getting secret with default value."""
        default_value = "default_123"
        result = await get_secret("NONEXISTENT_SECRET", "test_user", default=default_value)

        assert result == default_value

    async def test_get_secret_required_success(self):
        """Test get_secret_required with existing secret."""
        user_id = "test_user_required"
        key = "REQUIRED_SECRET"
        value = "required_value"

        await Secret.create(user_id=user_id, key=key, value=value)

        result = await get_secret_required(key, user_id)

        assert result == value

    async def test_get_secret_required_not_found(self):
        """Test get_secret_required raises exception when not found."""
        with pytest.raises(ValueError, match=r"Required secret.*not found"):
            await get_secret_required("NONEXISTENT_REQUIRED", "test_user")

    def test_get_secret_sync_from_env(self):
        """Test synchronous secret retrieval from environment."""
        key = "SYNC_SECRET"
        value = "sync_value_123"

        os.environ[key] = value

        try:
            result = get_secret_sync(key)
            assert result == value
        finally:
            del os.environ[key]

    def test_get_secret_sync_not_found(self):
        """Test synchronous secret retrieval when not found."""
        result = get_secret_sync("NONEXISTENT_SYNC_SECRET")
        assert result is None

    def test_get_secret_sync_with_default(self):
        """Test synchronous secret retrieval with default."""
        default_value = "sync_default"
        result = get_secret_sync("NONEXISTENT_SYNC_SECRET", default=default_value)
        assert result == default_value

    @pytest.mark.asyncio
    async def test_get_secret_sync_from_database(self):
        """Test that get_secret_sync can retrieve from database when user_id is provided."""
        user_id = "test_user_sync_db"
        key = "TEST_SYNC_DB_SECRET"
        value = "test_sync_db_value_123"

        # Create secret in database
        await Secret.create(user_id=user_id, key=key, value=value)

        # Remove env var to ensure we check database
        env_value = os.environ.pop(key, None)

        try:
            # Retrieve using sync function with user_id
            result = get_secret_sync(key, user_id=user_id)

            assert result == value
        finally:
            # Restore env var if it existed
            if env_value:
                os.environ[key] = env_value

    @pytest.mark.asyncio
    async def test_get_secret_sync_env_overrides_database(self):
        """Test that environment variables take priority over database in sync function."""
        user_id = "test_user_sync_priority"
        key = "TEST_SYNC_PRIORITY_SECRET"
        env_value = "env_value_sync"
        db_value = "db_value_sync"

        # Create secret in database
        await Secret.create(user_id=user_id, key=key, value=db_value)

        # Set environment variable
        os.environ[key] = env_value

        try:
            # Should return env value (higher priority)
            result = get_secret_sync(key, user_id=user_id)
            assert result == env_value
            assert result != db_value
        finally:
            if key in os.environ:
                del os.environ[key]

    def test_get_secret_sync_no_user_id_skips_database(self):
        """Test that get_secret_sync skips database lookup when no user_id is provided."""
        # Without user_id, should only check environment
        result = get_secret_sync("NONEXISTENT_SECRET_NO_USER")
        assert result is None

        # With default but no user_id
        result_with_default = get_secret_sync("NONEXISTENT_SECRET_NO_USER", default="default")
        assert result_with_default == "default"

    async def test_has_secret_in_database(self):
        """Test has_secret returns True for database secret."""
        user_id = "test_user_has"
        key = "HAS_SECRET_DB"
        value = "has_value"

        await Secret.create(user_id=user_id, key=key, value=value)

        result = await has_secret(key, user_id)

        assert result is True

    async def test_has_secret_in_env(self):
        """Test has_secret returns True for environment secret."""
        key = "HAS_SECRET_ENV"
        os.environ[key] = "env_value"

        try:
            result = await has_secret(key, "test_user")
            assert result is True
        finally:
            del os.environ[key]

    async def test_has_secret_not_found(self):
        """Test has_secret returns False when not found."""
        result = await has_secret("NONEXISTENT_HAS_SECRET", "test_user")

        assert result is False

    async def test_priority_order(self):
        """Test that priority order is: env > database."""
        user_id = "test_user_priority"
        key = "PRIORITY_SECRET"
        env_value = "from_env"
        db_value = "from_db"

        # Create in database
        await Secret.create(user_id=user_id, key=key, value=db_value)

        # Without env, should get from database
        result1 = await get_secret(key, user_id)
        assert result1 == db_value

        # With env, should get from env
        os.environ[key] = env_value
        try:
            result2 = await get_secret(key, user_id)
            assert result2 == env_value
        finally:
            del os.environ[key]

        # After removing env, should get from database again
        result3 = await get_secret(key, user_id)
        assert result3 == db_value

    async def test_different_users_different_secrets(self):
        """Test that different users can have different values for same key."""
        user1 = "user_1"
        user2 = "user_2"
        key = "SHARED_KEY"
        value1 = "user1_value"
        value2 = "user2_value"

        await Secret.create(user_id=user1, key=key, value=value1)
        await Secret.create(user_id=user2, key=key, value=value2)

        result1 = await get_secret(key, user1)
        result2 = await get_secret(key, user2)

        assert result1 == value1
        assert result2 == value2
