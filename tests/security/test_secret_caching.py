"""
Tests for secret caching.
"""

from unittest.mock import MagicMock, patch

import pytest

from nodetool.models.secret import Secret
from nodetool.security.secret_helper import (
    _SECRET_CACHE,
    clear_secret_cache,
    get_secret,
    get_secrets_batch,
)


@pytest.mark.asyncio
class TestSecretCaching:
    """Tests for secret caching."""

    def setup_method(self):
        """Clear cache before each test."""
        _SECRET_CACHE.clear()

    async def test_get_secret_caches_value(self):
        """Test that get_secret caches the value."""
        user_id = "test_user_cache"
        key = "CACHE_TEST_KEY"
        value = "cache_test_value"

        # Create secret
        await Secret.create(user_id=user_id, key=key, value=value)

        # First retrieval should hit DB and cache it
        result1 = await get_secret(key, user_id)
        assert result1 == value
        assert (user_id, key) in _SECRET_CACHE

        # Modify DB directly to prove we are reading from cache
        # (simulating a change that bypassed the cache invalidation for some reason,
        # or just to prove we are not hitting the DB)
        await Secret.find(user_id, key)
        # We manually change the encrypted value to something junk without updating timestamp
        # or just mock the find method to return None

        with patch("nodetool.models.secret.Secret.find") as mock_find:
            mock_find.return_value = None

            # Second retrieval should use cache
            result2 = await get_secret(key, user_id)
            assert result2 == value
            mock_find.assert_not_called()

    async def test_upsert_invalidates_cache(self):
        """Test that upsert invalidates the cache."""
        user_id = "test_user_upsert"
        key = "UPSERT_TEST_KEY"
        value1 = "value1"
        value2 = "value2"

        # 1. Create and cache
        await Secret.upsert(user_id, key, value1)
        await get_secret(key, user_id)
        assert _SECRET_CACHE[(user_id, key)] == value1

        # 2. Upsert new value
        await Secret.upsert(user_id, key, value2)

        # Cache should be cleared
        assert (user_id, key) not in _SECRET_CACHE

        # 3. Retrieve should get new value and re-cache
        result = await get_secret(key, user_id)
        assert result == value2
        assert _SECRET_CACHE[(user_id, key)] == value2

    async def test_update_value_invalidates_cache(self):
        """Test that update_value invalidates the cache."""
        user_id = "test_user_update"
        key = "UPDATE_TEST_KEY"
        value1 = "value1"
        value2 = "value2"

        # Create
        secret = await Secret.create(user_id=user_id, key=key, value=value1)

        # Cache it
        await get_secret(key, user_id)
        assert (user_id, key) in _SECRET_CACHE

        # Update value
        await secret.update_value(value2)

        # Cache should be cleared
        assert (user_id, key) not in _SECRET_CACHE

        # Fetch again
        result = await get_secret(key, user_id)
        assert result == value2

    async def test_delete_invalidates_cache(self):
        """Test that deletion invalidates the cache."""
        user_id = "test_user_delete"
        key = "DELETE_TEST_KEY"
        value = "value"

        await Secret.create(user_id=user_id, key=key, value=value)
        await get_secret(key, user_id)

        assert (user_id, key) in _SECRET_CACHE

        await Secret.delete_secret(user_id, key)

        assert (user_id, key) not in _SECRET_CACHE

        result = await get_secret(key, user_id)
        assert result is None

    async def test_get_secrets_batch_uses_cache(self):
        """Test that get_secrets_batch uses and populates cache."""
        user_id = "test_user_batch"
        keys = ["BATCH_1", "BATCH_2", "BATCH_3"]

        # Create secrets
        for k in keys:
            await Secret.create(user_id=user_id, key=k, value=f"val_{k}")

        # Pre-populate one item in cache
        _SECRET_CACHE[(user_id, "BATCH_1")] = "val_BATCH_1"

        # Mock Secret.query to ensure we only query for the missing ones
        original_query = Secret.query

        with patch("nodetool.models.secret.Secret.query", side_effect=original_query) as mock_query:
            results = await get_secrets_batch(keys, user_id)

            # Verify results
            for k in keys:
                assert results[k] == f"val_{k}"

            # Verify cache is fully populated
            assert len(_SECRET_CACHE) == 3

            # Verify we queried DB but only for proper keys if our implementation is optimal
            # Note: Checking query args might be brittle, but we can check calls
            assert mock_query.call_count == 1
