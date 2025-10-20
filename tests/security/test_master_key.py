"""
Tests for master key management.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from nodetool.security.master_key import MasterKeyManager
from nodetool.security.crypto import SecretCrypto


class TestMasterKeyManager:
    """Tests for MasterKeyManager class."""

    def setup_method(self):
        """Clear cached key before each test."""
        MasterKeyManager.clear_cache()

    def teardown_method(self):
        """Clean up after each test."""
        MasterKeyManager.clear_cache()
        # Clean up environment variable if set
        if "SECRETS_MASTER_KEY" in os.environ:
            del os.environ["SECRETS_MASTER_KEY"]
        if "AWS_SECRETS_MASTER_KEY_NAME" in os.environ:
            del os.environ["AWS_SECRETS_MASTER_KEY_NAME"]

    def test_get_master_key_from_env(self):
        """Test getting master key from environment variable."""
        test_key = SecretCrypto.generate_master_key()
        os.environ["SECRETS_MASTER_KEY"] = test_key

        key = MasterKeyManager.get_master_key()

        assert key == test_key
        assert MasterKeyManager.is_using_env_key()

    def test_get_master_key_caching(self):
        """Test that master key is cached."""
        test_key = SecretCrypto.generate_master_key()
        os.environ["SECRETS_MASTER_KEY"] = test_key

        # First call
        key1 = MasterKeyManager.get_master_key()

        # Second call should return cached value
        key2 = MasterKeyManager.get_master_key()

        assert key1 == key2 == test_key

    def test_clear_cache(self):
        """Test clearing the cache."""
        test_key = SecretCrypto.generate_master_key()
        os.environ["SECRETS_MASTER_KEY"] = test_key

        # Get key (caches it)
        MasterKeyManager.get_master_key()

        # Clear cache
        MasterKeyManager.clear_cache()

        # Should have cleared
        assert MasterKeyManager._cached_master_key is None

    @patch("nodetool.security.master_key.keyring")
    def test_get_master_key_from_keyring(self, mock_keyring):
        """Test getting master key from system keyring."""
        test_key = SecretCrypto.generate_master_key()
        mock_keyring.get_password.return_value = test_key

        key = MasterKeyManager.get_master_key()

        assert key == test_key
        mock_keyring.get_password.assert_called_once()

    @patch("nodetool.security.master_key.keyring")
    def test_get_master_key_generates_new_if_not_found(self, mock_keyring):
        """Test that a new key is generated if not found in keyring."""
        mock_keyring.get_password.return_value = None

        key = MasterKeyManager.get_master_key()

        # Should have generated a valid key
        assert key is not None
        assert isinstance(key, str)

        # Should have stored it in keyring
        mock_keyring.set_password.assert_called_once()

    @patch("nodetool.security.master_key.keyring")
    def test_set_master_key(self, mock_keyring):
        """Test setting a custom master key."""
        test_key = SecretCrypto.generate_master_key()

        MasterKeyManager.set_master_key(test_key)

        # Should be cached
        assert MasterKeyManager._cached_master_key == test_key

        # Should be stored in keyring
        mock_keyring.set_password.assert_called_once_with(
            "nodetool", "secrets_master_key", test_key
        )

    @patch("nodetool.security.master_key.keyring")
    def test_delete_master_key(self, mock_keyring):
        """Test deleting master key."""
        MasterKeyManager.delete_master_key()

        # Cache should be cleared
        assert MasterKeyManager._cached_master_key is None

        # Should be deleted from keyring
        mock_keyring.delete_password.assert_called_once_with(
            "nodetool", "secrets_master_key"
        )

    def test_export_master_key(self):
        """Test exporting master key."""
        test_key = SecretCrypto.generate_master_key()
        os.environ["SECRETS_MASTER_KEY"] = test_key

        exported = MasterKeyManager.export_master_key()

        assert exported == test_key

    def test_is_using_env_key(self):
        """Test checking if environment key is being used."""
        # No env key set
        assert not MasterKeyManager.is_using_env_key()

        # Set env key
        os.environ["SECRETS_MASTER_KEY"] = "test_key"
        assert MasterKeyManager.is_using_env_key()

    def test_is_using_aws_key(self):
        """Test checking if AWS key is configured."""
        # No AWS key configured
        assert not MasterKeyManager.is_using_aws_key()

        # Set AWS key name
        os.environ["AWS_SECRETS_MASTER_KEY_NAME"] = "my-secret"
        assert MasterKeyManager.is_using_aws_key()

    @patch("nodetool.security.master_key.boto3")
    def test_get_from_aws_secrets(self, mock_boto3):
        """Test getting master key from AWS Secrets Manager."""
        test_key = SecretCrypto.generate_master_key()
        os.environ["AWS_SECRETS_MASTER_KEY_NAME"] = "nodetool-master-key"

        # Mock AWS response
        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {"SecretString": test_key}
        mock_boto3.session.Session.return_value.client.return_value = mock_client

        key = MasterKeyManager.get_master_key()

        assert key == test_key
        mock_client.get_secret_value.assert_called_once_with(
            SecretId="nodetool-master-key"
        )

    @patch("nodetool.security.master_key.boto3")
    @patch("nodetool.security.master_key.keyring")
    def test_aws_fallback_to_keyring(self, mock_keyring, mock_boto3):
        """Test fallback to keyring when AWS fails."""
        test_key = SecretCrypto.generate_master_key()
        os.environ["AWS_SECRETS_MASTER_KEY_NAME"] = "nodetool-master-key"

        # Mock AWS to fail
        mock_client = MagicMock()
        from botocore.exceptions import ClientError

        mock_client.get_secret_value.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException"}}, "GetSecretValue"
        )
        mock_boto3.session.Session.return_value.client.return_value = mock_client

        # Mock keyring to return a key
        mock_keyring.get_password.return_value = test_key

        key = MasterKeyManager.get_master_key()

        # Should have fallen back to keyring
        assert key == test_key
        mock_keyring.get_password.assert_called_once()
