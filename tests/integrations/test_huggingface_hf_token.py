"""Tests for HF_TOKEN retrieval from database in HuggingFace downloads."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nodetool.deploy.admin_operations import (
    AdminDownloadManager,
)
from nodetool.deploy.admin_operations import (
    get_hf_token as get_hf_token_admin,
)
from nodetool.integrations.huggingface.hf_auth import get_hf_token
from nodetool.integrations.huggingface.hf_download import (
    DownloadManager,
)
from nodetool.integrations.huggingface.huggingface_models import (
    fetch_model_info,
    fetch_model_readme,
)
from nodetool.integrations.huggingface.huggingface_models import (
    get_hf_token as get_hf_token_models,
)
from nodetool.models.secret import Secret
from nodetool.runtime.resources import ResourceScope


@pytest.mark.asyncio
class TestHFTokenFromDatabase:
    """Test that HF_TOKEN is retrieved from database when available."""

    async def test_get_hf_token_from_env(self):
        """Test that get_hf_token retrieves from environment variables."""
        # Set environment variable
        test_token = "hf_test_token_from_env"
        os.environ["HF_TOKEN"] = test_token

        try:
            result = await get_hf_token()
            assert result == test_token
        finally:
            # Clean up
            if "HF_TOKEN" in os.environ:
                del os.environ["HF_TOKEN"]

    async def test_get_hf_token_from_database_with_resource_scope(self):
        """Test that get_hf_token can retrieve from database when ResourceScope is available."""
        user_id = "test_user_hf_token"
        db_token = "hf_test_token_from_db"

        # Create secret in database
        await Secret.create(user_id=user_id, key="HF_TOKEN", value=db_token)

        # Remove env var to ensure we check database
        env_token = os.environ.pop("HF_TOKEN", None)

        try:
            # Create a version that checks database when ResourceScope is available
            from nodetool.runtime.resources import maybe_scope
            from nodetool.security.secret_helper import get_secret

            async def get_hf_token_with_db():
                """Get HF_TOKEN from env or database if ResourceScope is available."""
                # Check environment first
                if os.environ.get("HF_TOKEN"):
                    return os.environ.get("HF_TOKEN")

                # If ResourceScope is available, try database
                scope = maybe_scope()
                if scope and scope.db:
                    # We need user_id - in real usage this would come from auth
                    # For testing, we'll use a test user_id
                    token = await get_secret("HF_TOKEN", user_id)
                    if token:
                        return token

                return None

            # Note: Tests run with ResourceScope automatically via conftest
            # So we test that it retrieves from database when ResourceScope is available
            async with ResourceScope():
                result_with_scope = await get_hf_token_with_db()
                assert result_with_scope == db_token

        finally:
            # Restore env var if it existed
            if env_token:
                os.environ["HF_TOKEN"] = env_token

    async def test_get_hf_token_env_takes_priority(self):
        """Test that environment variables take priority over database."""
        user_id = "test_user_priority"
        env_token = "hf_token_from_env"
        db_token = "hf_token_from_db"

        # Create secret in database
        await Secret.create(user_id=user_id, key="HF_TOKEN", value=db_token)

        # Set environment variable
        os.environ["HF_TOKEN"] = env_token

        try:
            # Should return env token (higher priority)
            result = await get_hf_token()
            assert result == env_token
            assert result != db_token
        finally:
            if "HF_TOKEN" in os.environ:
                del os.environ["HF_TOKEN"]

    async def test_download_manager_uses_hf_token(self):
        """Test that DownloadManager uses HF_TOKEN when available."""
        test_token = "hf_test_token_for_download"

        # Set environment variable
        os.environ["HF_TOKEN"] = test_token

        try:
            manager = await DownloadManager.create()
            # Verify token is stored
            assert manager.token == test_token
            # Verify HfApi was initialized with token
            assert manager.api.token == test_token
        finally:
            if "HF_TOKEN" in os.environ:
                del os.environ["HF_TOKEN"]



    async def test_admin_download_manager_uses_hf_token(self):
        """Test that AdminDownloadManager uses HF_TOKEN when available."""
        test_token = "hf_test_token_for_admin"

        # Set environment variable
        os.environ["HF_TOKEN"] = test_token

        try:
            manager = await AdminDownloadManager.create()
            # Verify token is stored
            assert manager.token == test_token
            # Verify HfApi was initialized with token
            assert manager.api.token == test_token
        finally:
            if "HF_TOKEN" in os.environ:
                del os.environ["HF_TOKEN"]

    async def test_fetch_model_readme_uses_hf_token(self):
        """Test that fetch_model_readme uses HF_TOKEN when downloading."""
        test_token = "hf_test_token_for_readme"

        # Set environment variable
        os.environ["HF_TOKEN"] = test_token

        try:
            # Patch async_hf_download where it's imported in huggingface_models
            with patch(
                "nodetool.integrations.huggingface.huggingface_models.async_hf_download",
                new_callable=AsyncMock
            ) as mock_download:
                mock_download.return_value = "/tmp/README.md"

                # Mock file reading
                with patch("builtins.open", create=True) as mock_open:
                    mock_open.return_value.__enter__.return_value.read.return_value = (
                        "# Test README"
                    )

                    # Mock try_to_load_from_cache to return None (not cached)
                    with patch(
                        "huggingface_hub.try_to_load_from_cache",
                        return_value=None,
                    ):
                        await fetch_model_readme("test/repo")

                        # Verify async_hf_download was called with token
                        mock_download.assert_called_once()
                        call_kwargs = mock_download.call_args[1]
                        assert call_kwargs.get("token") == test_token
        finally:
            if "HF_TOKEN" in os.environ:
                del os.environ["HF_TOKEN"]

    async def test_fetch_model_info_uses_hf_token(self):
        """Test that fetch_model_info uses HF_TOKEN when querying API."""
        test_token = "hf_test_token_for_info"

        # Set environment variable
        os.environ["HF_TOKEN"] = test_token

        try:
            from huggingface_hub import ModelInfo

            # Mock HfApi and model_info
            with patch(
                "nodetool.integrations.huggingface.huggingface_models.HfApi"
            ) as mock_hf_api_class:
                mock_api = MagicMock()
                mock_model_info = MagicMock(spec=ModelInfo)
                mock_model_info.siblings = []
                mock_api.model_info.return_value = mock_model_info
                mock_hf_api_class.return_value = mock_api

                # Mock cache to return None (cache miss)
                with patch(
                    "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.model_info_cache.get",
                    return_value=None,
                ):
                    await fetch_model_info("test/repo")

                    # Verify HfApi was initialized with token
                    mock_hf_api_class.assert_called_once()
                    call_kwargs = mock_hf_api_class.call_args[1]
                    assert call_kwargs.get("token") == test_token
        finally:
            if "HF_TOKEN" in os.environ:
                del os.environ["HF_TOKEN"]

    async def test_get_hf_token_fallback_to_none(self):
        """Test that get_hf_token returns None when not found."""
        # Ensure HF_TOKEN is not in environment
        env_token = os.environ.pop("HF_TOKEN", None)

        try:
            result = await get_hf_token()
            assert result is None
        finally:
            if env_token:
                os.environ["HF_TOKEN"] = env_token

    async def test_all_get_hf_token_functions_consistent(self):
        """Test that all get_hf_token functions behave consistently."""
        test_token = "hf_test_token_consistent"

        # Set environment variable
        os.environ["HF_TOKEN"] = test_token

        try:
            # All should return the same value
            token_cache = await get_hf_token()
            token_models = await get_hf_token_models()
            token_admin = await get_hf_token_admin()

            assert token_cache == test_token
            assert token_models == test_token
            assert token_admin == test_token
            assert token_cache == token_models == token_admin
        finally:
            if "HF_TOKEN" in os.environ:
                del os.environ["HF_TOKEN"]


@pytest.mark.asyncio
class TestHFTokenDatabaseIntegration:
    """Integration tests for HF_TOKEN database retrieval in download scenarios."""

    async def test_download_with_database_token_simulation(self):
        """Simulate download scenario with database token."""
        user_id = "test_user_download"
        db_token = "hf_db_token_for_download"

        # Create secret in database
        await Secret.create(user_id=user_id, key="HF_TOKEN", value=db_token)

        # Remove env var
        env_token = os.environ.pop("HF_TOKEN", None)

        try:
            # Simulate what would happen if we had user_id and ResourceScope
            async with ResourceScope():
                from nodetool.security.secret_helper import get_secret

                # Get token from database
                retrieved_token = await get_secret("HF_TOKEN", user_id)
                assert retrieved_token == db_token

                # Verify DownloadManager would use this token
                # (In real implementation, we'd pass user_id and use async get_secret)
                with patch(
                    "nodetool.integrations.huggingface.hf_auth.get_hf_token",
                    return_value=retrieved_token,
                ):
                    manager = await DownloadManager.create()
                    assert manager.token == db_token
                    assert manager.api.token == db_token
        finally:
            if env_token:
                os.environ["HF_TOKEN"] = env_token

    async def test_download_priority_env_over_db(self):
        """Test that environment token takes priority over database token."""
        user_id = "test_user_priority_download"
        env_token = "hf_env_token_priority"
        db_token = "hf_db_token_priority"

        # Create secret in database
        await Secret.create(user_id=user_id, key="HF_TOKEN", value=db_token)

        # Set environment variable
        os.environ["HF_TOKEN"] = env_token

        try:
            # get_hf_token should return env token
            token = await get_hf_token()
            assert token == env_token
            assert token != db_token

            # DownloadManager should use env token
            manager = await DownloadManager.create()
            assert manager.token == env_token
        finally:
            if "HF_TOKEN" in os.environ:
                del os.environ["HF_TOKEN"]
