"""
Cryptographic utilities for secret encryption and decryption.

This module provides utilities for encrypting and decrypting secrets using
Fernet symmetric encryption (AES-128 in CBC mode with PKCS7 padding).
"""

import base64

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class SecretCrypto:
    """
    Handles encryption and decryption of secrets using Fernet symmetric encryption.

    The master key is combined with user_id (as salt) to derive user-specific encryption keys.
    """

    @staticmethod
    def generate_master_key() -> str:
        """
        Generate a new master key for encrypting secrets.

        Returns:
            A base64-encoded master key string.
        """
        return Fernet.generate_key().decode()

    @staticmethod
    def derive_encryption_key(master_key: str, user_id: str) -> Fernet:
        """
        Derive an encryption key from the master key using user_id as salt.

        This ensures each user's secrets are encrypted with a unique derived key,
        providing isolation between users even if the master key is compromised.

        Args:
            master_key: The master key to derive from.
            user_id: The user ID to use as salt for key derivation.

        Returns:
            A Fernet cipher instance.
        """
        # Use user_id as salt to create per-user encryption keys
        salt = user_id.encode()

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
        return Fernet(key)

    @staticmethod
    def encrypt(value: str, master_key: str, user_id: str) -> str:
        """
        Encrypt a secret value using the master key and user_id.

        Args:
            value: The plaintext value to encrypt.
            master_key: The master key to use for encryption.
            user_id: The user ID to use as salt.

        Returns:
            The encrypted value as a base64-encoded string.
        """
        fernet = SecretCrypto.derive_encryption_key(master_key, user_id)
        encrypted = fernet.encrypt(value.encode())
        return encrypted.decode()

    @staticmethod
    def decrypt(encrypted_value: str, master_key: str, user_id: str) -> str:
        """
        Decrypt an encrypted secret value using the master key and user_id.

        Args:
            encrypted_value: The encrypted value as a base64-encoded string.
            master_key: The master key to use for decryption.
            user_id: The user ID to use as salt.

        Returns:
            The decrypted plaintext value.

        Raises:
            cryptography.fernet.InvalidToken: If the master key is incorrect or the data is corrupted.
        """
        fernet = SecretCrypto.derive_encryption_key(master_key, user_id)
        try:
            decrypted = fernet.decrypt(encrypted_value.encode())
        except InvalidToken as exc:
            raise ValueError("Failed to decrypt secret") from exc
        return decrypted.decode()

    @staticmethod
    def is_valid_master_key(master_key: str, test_encrypted_value: str, user_id: str) -> bool:
        """
        Check if a master key is valid by attempting to decrypt a test value.

        Args:
            master_key: The master key to validate.
            test_encrypted_value: An encrypted test value to decrypt.
            user_id: The user ID used as salt for the test value.

        Returns:
            True if the master key is valid, False otherwise.
        """
        try:
            SecretCrypto.decrypt(test_encrypted_value, master_key, user_id)
            return True
        except Exception:
            return False
