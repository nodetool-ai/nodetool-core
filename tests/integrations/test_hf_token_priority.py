"""
Tests for HF_TOKEN priority.
"""

import os
import pytest
from unittest.mock import MagicMock, patch

from nodetool.models.secret import Secret
from nodetool.integrations.huggingface.hf_auth import get_hf_token
from nodetool.security.secret_helper import clear_secret_cache, _SECRET_CACHE


@pytest.mark.asyncio
class TestHFTokenPriority:
    """Tests for HF_TOKEN priority."""

    async def setup_method(self):
        """Clear cache before each test."""
        _SECRET_CACHE.clear()

    async def test_get_hf_token_prioritizes_db_over_env(self):
        """Test that get_hf_token prioritizes DB secret over environment variable."""
        user_id = "test_user_hf_priority"
        db_token = "db_token_value"
        env_token = "env_token_value"

        # Create secret in database
        await Secret.create(user_id=user_id, key="HF_TOKEN", value=db_token)
        
        # Set environment variable
        os.environ["HF_TOKEN"] = env_token

        try:
            # Should return DB token (custom priority for HF_TOKEN)
            result = await get_hf_token(user_id)
            assert result == db_token
        finally:
            del os.environ["HF_TOKEN"]

    async def test_get_hf_token_falls_back_to_env(self):
        """Test that get_hf_token falls back to env if not in DB."""
        user_id = "test_user_hf_fallback"
        env_token = "env_token_value"

        # Ensure no DB secret
        await Secret.delete_secret(user_id, "HF_TOKEN")
        
        # Set environment variable
        os.environ["HF_TOKEN"] = env_token

        try:
            # Should return env token
            result = await get_hf_token(user_id)
            assert result == env_token
        finally:
            del os.environ["HF_TOKEN"]

    async def test_get_hf_token_no_user_id_uses_env(self):
        """Test that without user_id, it uses env var."""
        env_token = "env_token_val_no_user"
        os.environ["HF_TOKEN"] = env_token
        
        try:
            result = await get_hf_token(None)
            assert result == env_token
        finally:
            del os.environ["HF_TOKEN"]
