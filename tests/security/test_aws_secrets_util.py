"""
Tests for AWS Secrets Manager utility.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from botocore.exceptions import ClientError

from nodetool.security.aws_secrets_util import AWSSecretsUtil
from nodetool.security.crypto import SecretCrypto


class TestAWSSecretsUtil:
    """Tests for AWSSecretsUtil class."""

    def test_get_aws_client_exists(self):
        """
        Test that get_aws_client method exists and is callable.
        Verifies the method is defined in the utility class.
        """
        assert hasattr(AWSSecretsUtil, "get_aws_client")
        assert callable(AWSSecretsUtil.get_aws_client)

    def test_aws_client_methods_exist(self):
        """
        Test that all AWS client methods exist.
        Verifies the API surface of AWSSecretsUtil.
        """
        assert hasattr(AWSSecretsUtil, "store_master_key")
        assert hasattr(AWSSecretsUtil, "retrieve_master_key")
        assert hasattr(AWSSecretsUtil, "delete_master_key")
        assert hasattr(AWSSecretsUtil, "generate_and_store")

        assert callable(AWSSecretsUtil.store_master_key)
        assert callable(AWSSecretsUtil.retrieve_master_key)
        assert callable(AWSSecretsUtil.delete_master_key)
        assert callable(AWSSecretsUtil.generate_and_store)

    @patch("nodetool.security.aws_secrets_util.AWSSecretsUtil.get_aws_client")
    def test_store_master_key_create_new(self, mock_get_client):
        """
        Test storing a new master key to AWS.
        Verifies that create_secret is called.
        """
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        master_key = SecretCrypto.generate_master_key()
        success = AWSSecretsUtil.store_master_key(
            secret_name="test-master-key", master_key=master_key, region="us-east-1"
        )

        assert success is True
        mock_client.create_secret.assert_called_once_with(
            Name="test-master-key", SecretString=master_key, Description="NodeTool master encryption key for secrets"
        )

    @patch("nodetool.security.aws_secrets_util.AWSSecretsUtil.get_aws_client")
    def test_store_master_key_update_existing(self, mock_get_client):
        """
        Test updating an existing master key.
        Verifies that put_secret_value is called when secret exists.
        """
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Mock ResourceExistsException
        error_response = {"Error": {"Code": "ResourceExistsException"}}
        mock_client.create_secret.side_effect = ClientError(error_response, "CreateSecret")

        master_key = SecretCrypto.generate_master_key()
        success = AWSSecretsUtil.store_master_key(
            secret_name="test-master-key", master_key=master_key, region="us-east-1"
        )

        assert success is True
        mock_client.put_secret_value.assert_called_once_with(SecretId="test-master-key", SecretString=master_key)

    @patch("nodetool.security.aws_secrets_util.AWSSecretsUtil.get_aws_client")
    def test_store_master_key_error(self, mock_get_client):
        """
        Test handling of errors when storing master key.
        Verifies that False is returned on error.
        """
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Mock a generic error
        error_response = {"Error": {"Code": "InternalServerError"}}
        mock_client.create_secret.side_effect = ClientError(error_response, "CreateSecret")

        master_key = SecretCrypto.generate_master_key()
        success = AWSSecretsUtil.store_master_key(secret_name="test-master-key", master_key=master_key)

        assert success is False

    @patch("nodetool.security.aws_secrets_util.AWSSecretsUtil.get_aws_client")
    def test_retrieve_master_key_string_secret(self, mock_get_client):
        """
        Test retrieving a master key stored as string.
        Verifies that SecretString is correctly extracted.
        """
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        master_key = SecretCrypto.generate_master_key()
        mock_client.get_secret_value.return_value = {"SecretString": master_key}

        retrieved_key = AWSSecretsUtil.retrieve_master_key(secret_name="test-master-key", region="us-east-1")

        assert retrieved_key == master_key
        mock_client.get_secret_value.assert_called_once_with(SecretId="test-master-key")

    @patch("nodetool.security.aws_secrets_util.AWSSecretsUtil.get_aws_client")
    def test_retrieve_master_key_binary_secret(self, mock_get_client):
        """
        Test retrieving a master key stored as binary.
        Verifies that SecretBinary is correctly base64-decoded.
        """
        import base64

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        master_key = SecretCrypto.generate_master_key()
        master_key_bytes = master_key.encode()
        base64_encoded = base64.b64encode(master_key_bytes).decode()

        mock_client.get_secret_value.return_value = {"SecretBinary": base64_encoded}

        retrieved_key = AWSSecretsUtil.retrieve_master_key(secret_name="test-master-key")

        assert retrieved_key == master_key

    @patch("nodetool.security.aws_secrets_util.AWSSecretsUtil.get_aws_client")
    def test_retrieve_master_key_not_found(self, mock_get_client):
        """
        Test handling of secret not found error.
        Verifies that None is returned.
        """
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        error_response = {"Error": {"Code": "ResourceNotFoundException"}}
        mock_client.get_secret_value.side_effect = ClientError(error_response, "GetSecretValue")

        retrieved_key = AWSSecretsUtil.retrieve_master_key(secret_name="nonexistent")

        assert retrieved_key is None

    @patch("nodetool.security.aws_secrets_util.AWSSecretsUtil.get_aws_client")
    def test_delete_master_key_with_recovery_window(self, mock_get_client):
        """
        Test deleting a master key with recovery window.
        Verifies that RecoveryWindowInDays is set to 30 by default.
        """
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        success = AWSSecretsUtil.delete_master_key(secret_name="test-master-key", region="us-east-1", force=False)

        assert success is True
        mock_client.delete_secret.assert_called_once_with(SecretId="test-master-key", RecoveryWindowInDays=30)

    @patch("nodetool.security.aws_secrets_util.AWSSecretsUtil.get_aws_client")
    def test_delete_master_key_force(self, mock_get_client):
        """
        Test force deleting a master key.
        Verifies that ForceDeleteWithoutRecovery is used.
        """
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        success = AWSSecretsUtil.delete_master_key(secret_name="test-master-key", force=True)

        assert success is True
        mock_client.delete_secret.assert_called_once_with(SecretId="test-master-key", ForceDeleteWithoutRecovery=True)

    @patch("nodetool.security.aws_secrets_util.AWSSecretsUtil.get_aws_client")
    def test_delete_master_key_error(self, mock_get_client):
        """
        Test handling of errors when deleting a key.
        Verifies that False is returned on error.
        """
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_client.delete_secret.side_effect = Exception("AWS Error")

        success = AWSSecretsUtil.delete_master_key(secret_name="test-master-key")

        assert success is False

    def test_generate_and_store(self):
        """
        Test generating and storing a new master key.
        Verifies that generate_master_key and store_master_key are called.
        """
        with (
            patch("nodetool.security.crypto.SecretCrypto.generate_master_key") as mock_generate,
            patch.object(AWSSecretsUtil, "store_master_key") as mock_store,
        ):
            mock_key = "test_generated_key"
            mock_generate.return_value = mock_key
            mock_store.return_value = True

            result = AWSSecretsUtil.generate_and_store(secret_name="test-master-key", region="us-east-1")

            assert result == mock_key
            mock_generate.assert_called_once()
            # Just verify store_master_key was called with the key
            assert mock_store.called
            call_args = mock_store.call_args
            assert "test_generated_key" in str(call_args)

    def test_generate_and_store_failure(self):
        """
        Test failure when storing generated key.
        Verifies that None is returned if storage fails.
        """
        with (
            patch("nodetool.security.crypto.SecretCrypto.generate_master_key") as mock_generate,
            patch.object(AWSSecretsUtil, "store_master_key") as mock_store,
        ):
            mock_key = "test_generated_key"
            mock_generate.return_value = mock_key
            mock_store.return_value = False  # Storage fails

            result = AWSSecretsUtil.generate_and_store(secret_name="test-master-key")

            assert result is None


class TestAWSSecretsUtilIntegration:
    """Integration tests for AWS Secrets Manager utility."""

    def test_store_and_retrieve_roundtrip(self):
        """
        Test storing and retrieving a master key.
        Verifies the complete roundtrip flow.
        """
        with patch("nodetool.security.aws_secrets_util.AWSSecretsUtil.get_aws_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            # Store
            test_key = "test_stored_key"
            mock_client.create_secret.return_value = {}
            store_result = AWSSecretsUtil.store_master_key(secret_name="test-key", master_key=test_key)
            assert store_result is True

            # Retrieve
            mock_client.get_secret_value.return_value = {"SecretString": test_key}
            retrieved = AWSSecretsUtil.retrieve_master_key(secret_name="test-key")
            assert retrieved == test_key

    def test_delete_recovery_window_default(self):
        """
        Test that delete without force uses 30-day recovery window.
        Verifies recovery window is the default.
        """
        with patch("nodetool.security.aws_secrets_util.AWSSecretsUtil.get_aws_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            AWSSecretsUtil.delete_master_key(secret_name="test-key", force=False)

            # Verify recovery window was set
            mock_client.delete_secret.assert_called_once()
            call_args = mock_client.delete_secret.call_args
            assert "RecoveryWindowInDays" in call_args[1]
            assert call_args[1]["RecoveryWindowInDays"] == 30
