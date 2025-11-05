"""
Master key management with keychain/keyring integration.

This module manages the master encryption key for secrets, storing it securely
in the system keychain (macOS Keychain, Windows Credential Manager, Linux Secret Service)
or AWS Secrets Manager.
"""

import os
import logging
from typing import Optional
import keyring
from keyring.errors import KeyringError
import boto3
from botocore.exceptions import ClientError
from nodetool.security.crypto import SecretCrypto

# Keyring service name for storing the master key
KEYRING_SERVICE = "nodetool"
KEYRING_USERNAME = "secrets_master_key"


class MasterKeyManager:
    """
    Manages the master encryption key for secrets.

    The master key is retrieved from (in order of precedence):
    1. SECRETS_MASTER_KEY environment variable (explicit key)
    2. AWS Secrets Manager (if AWS_SECRETS_MASTER_KEY_NAME is set)
    3. System keychain/keyring
    4. Auto-generated and stored in keychain if not found
    """

    _cached_master_key: Optional[str] = None
    _logger: Optional[logging.Logger] = None

    @classmethod
    def _get_logger(cls) -> logging.Logger:
        """Get or initialize the logger lazily to avoid circular imports."""
        if cls._logger is None:
            from nodetool.config.logging_config import get_logger
            cls._logger = get_logger(__name__)
        return cls._logger

    @classmethod
    def _get_from_aws_secrets(cls, secret_name: str) -> Optional[str]:
        """
        Retrieve master key from AWS Secrets Manager.

        Args:
            secret_name: The name of the secret in AWS Secrets Manager.

        Returns:
            The master key if found, None otherwise.
        """
        try:
            # Get AWS region from environment or default to us-east-1
            region = os.environ.get("AWS_REGION", "us-east-1")

            # Create Secrets Manager client
            session = boto3.session.Session()
            client = session.client(
                service_name="secretsmanager",
                region_name=region
            )

            # Retrieve secret
            try:
                response = client.get_secret_value(SecretId=secret_name)

                # Secret can be either a string or binary
                if "SecretString" in response:
                    return response["SecretString"]
                else:
                    # Binary secrets are base64-encoded
                    import base64
                    return base64.b64decode(response["SecretBinary"]).decode()

            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code == "ResourceNotFoundException":
                    cls._get_logger().warning(f"AWS secret '{secret_name}' not found")
                elif error_code == "AccessDeniedException":
                    cls._get_logger().error(f"Access denied to AWS secret '{secret_name}'")
                else:
                    cls._get_logger().error(f"Error retrieving AWS secret: {e}")
                return None

        except Exception as e:
            cls._get_logger().error(f"Unexpected error retrieving AWS secret: {e}")
            return None

    @classmethod
    async def get_master_key(cls) -> str:
        """
        Get the master encryption key.

        Returns:
            The master key as a base64-encoded string.

        Raises:
            RuntimeError: If the master key cannot be retrieved or generated.
        """
        import asyncio

        # Return cached key if available
        if cls._cached_master_key is not None:
            return cls._cached_master_key

        # 1. Check environment variable first (explicit key)
        env_key = os.environ.get("SECRETS_MASTER_KEY")
        if env_key:
            cls._get_logger().info("Using master key from SECRETS_MASTER_KEY environment variable")
            cls._cached_master_key = env_key
            return env_key

        # 2. Check AWS Secrets Manager if configured
        aws_secret_name = os.environ.get("AWS_SECRETS_MASTER_KEY_NAME")
        if aws_secret_name:
            cls._get_logger().info(f"Attempting to retrieve master key from AWS Secrets Manager: {aws_secret_name}")
            aws_key = cls._get_from_aws_secrets(aws_secret_name)
            if aws_key:
                cls._get_logger().info("Using master key from AWS Secrets Manager")
                cls._cached_master_key = aws_key
                return aws_key
            else:
                cls._get_logger().warning("Failed to retrieve master key from AWS Secrets Manager, falling back to keychain")

        # 3. Try to get from system keychain (using asyncio.to_thread to avoid blocking event loop)
        try:
            stored_key = await asyncio.to_thread(keyring.get_password, KEYRING_SERVICE, KEYRING_USERNAME)
            if stored_key:
                cls._get_logger().info("Using master key from system keychain")
                cls._cached_master_key = stored_key
                return stored_key
        except KeyringError as e:
            cls._get_logger().warning(f"Could not access system keychain: {e}")

        # 4. Generate new master key and store in keychain
        cls._get_logger().info("Generating new master key and storing in system keychain")
        new_key = SecretCrypto.generate_master_key()

        try:
            await asyncio.to_thread(keyring.set_password, KEYRING_SERVICE, KEYRING_USERNAME, new_key)
            cls._get_logger().info("Master key successfully stored in system keychain")
        except KeyringError as e:
            cls._get_logger().error(f"Failed to store master key in system keychain: {e}")
            raise RuntimeError(
                "Failed to store master key in system keychain. "
                "Please set SECRETS_MASTER_KEY environment variable manually or configure AWS_SECRETS_MASTER_KEY_NAME."
            ) from e

        cls._cached_master_key = new_key
        return new_key

    @classmethod
    def set_master_key(cls, master_key: str) -> None:
        """
        Set a custom master key and store it in the keychain.

        This is useful for:
        - Migrating from one master key to another
        - Restoring a master key backup
        - Setting up a shared master key across multiple instances

        Args:
            master_key: The master key to set (base64-encoded string).

        Raises:
            RuntimeError: If the master key cannot be stored.
        """
        try:
            keyring.set_password(KEYRING_SERVICE, KEYRING_USERNAME, master_key)
            cls._cached_master_key = master_key
            cls._get_logger().info("Master key updated successfully in system keychain")
        except KeyringError as e:
            cls._get_logger().error(f"Failed to update master key in system keychain: {e}")
            raise RuntimeError(
                "Failed to update master key in system keychain"
            ) from e

    @classmethod
    def delete_master_key(cls) -> None:
        """
        Delete the master key from the keychain.

        WARNING: This will make all encrypted secrets inaccessible!
        Only use this if you're sure you want to reset the encryption.

        Note: This does NOT delete the key from AWS Secrets Manager.

        Raises:
            RuntimeError: If the master key cannot be deleted.
        """
        try:
            keyring.delete_password(KEYRING_SERVICE, KEYRING_USERNAME)
            cls._cached_master_key = None
            cls._get_logger().warning("Master key deleted from system keychain")
        except KeyringError as e:
            cls._get_logger().error(f"Failed to delete master key from system keychain: {e}")
            raise RuntimeError(
                "Failed to delete master key from system keychain"
            ) from e

    @classmethod
    def clear_cache(cls) -> None:
        """
        Clear the cached master key.

        This forces the next call to get_master_key() to retrieve from
        environment, AWS, or keychain again.
        """
        cls._cached_master_key = None

    @classmethod
    def is_using_env_key(cls) -> bool:
        """
        Check if the master key is being sourced from environment variable.

        Returns:
            True if SECRETS_MASTER_KEY environment variable is set.
        """
        return os.environ.get("SECRETS_MASTER_KEY") is not None

    @classmethod
    def is_using_aws_key(cls) -> bool:
        """
        Check if the master key should be sourced from AWS Secrets Manager.

        Returns:
            True if AWS_SECRETS_MASTER_KEY_NAME environment variable is set.
        """
        return os.environ.get("AWS_SECRETS_MASTER_KEY_NAME") is not None

    @classmethod
    async def export_master_key(cls) -> str:
        """
        Export the current master key for backup purposes.

        WARNING: Store this key securely! Anyone with this key can decrypt
        all secrets in the database.

        Returns:
            The master key as a base64-encoded string.
        """
        return await cls.get_master_key()
