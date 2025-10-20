"""
Tests for cryptographic utilities.
"""

import pytest
from nodetool.security.crypto import SecretCrypto


class TestSecretCrypto:
    """Tests for SecretCrypto class."""

    def test_generate_master_key(self):
        """Test master key generation."""
        key1 = SecretCrypto.generate_master_key()
        key2 = SecretCrypto.generate_master_key()

        # Keys should be strings
        assert isinstance(key1, str)
        assert isinstance(key2, str)

        # Keys should be unique
        assert key1 != key2

        # Keys should be base64-encoded (should not raise)
        import base64
        base64.urlsafe_b64decode(key1)
        base64.urlsafe_b64decode(key2)

    def test_encrypt_decrypt(self):
        """Test encryption and decryption."""
        master_key = SecretCrypto.generate_master_key()
        user_id = "test_user_123"
        plaintext = "my_secret_api_key_12345"

        # Encrypt
        encrypted = SecretCrypto.encrypt(plaintext, master_key, user_id)

        # Should be different from plaintext
        assert encrypted != plaintext

        # Decrypt
        decrypted = SecretCrypto.decrypt(encrypted, master_key, user_id)

        # Should match original
        assert decrypted == plaintext

    def test_decrypt_with_wrong_key_fails(self):
        """Test that decryption fails with wrong master key."""
        master_key1 = SecretCrypto.generate_master_key()
        master_key2 = SecretCrypto.generate_master_key()
        user_id = "test_user_123"
        plaintext = "secret_value"

        encrypted = SecretCrypto.encrypt(plaintext, master_key1, user_id)

        # Should fail with wrong key
        with pytest.raises(Exception):
            SecretCrypto.decrypt(encrypted, master_key2, user_id)

    def test_decrypt_with_wrong_user_id_fails(self):
        """Test that decryption fails with wrong user_id."""
        master_key = SecretCrypto.generate_master_key()
        user_id1 = "user_123"
        user_id2 = "user_456"
        plaintext = "secret_value"

        encrypted = SecretCrypto.encrypt(plaintext, master_key, user_id1)

        # Should fail with wrong user_id (used as salt)
        with pytest.raises(Exception):
            SecretCrypto.decrypt(encrypted, master_key, user_id2)

    def test_user_isolation(self):
        """Test that different users get different encrypted values."""
        master_key = SecretCrypto.generate_master_key()
        user_id1 = "user_123"
        user_id2 = "user_456"
        plaintext = "same_secret"

        encrypted1 = SecretCrypto.encrypt(plaintext, master_key, user_id1)
        encrypted2 = SecretCrypto.encrypt(plaintext, master_key, user_id2)

        # Same plaintext encrypted for different users should be different
        assert encrypted1 != encrypted2

        # Each should decrypt correctly with their own user_id
        assert SecretCrypto.decrypt(encrypted1, master_key, user_id1) == plaintext
        assert SecretCrypto.decrypt(encrypted2, master_key, user_id2) == plaintext

    def test_is_valid_master_key(self):
        """Test master key validation."""
        master_key = SecretCrypto.generate_master_key()
        wrong_key = SecretCrypto.generate_master_key()
        user_id = "test_user"
        plaintext = "test_value"

        encrypted = SecretCrypto.encrypt(plaintext, master_key, user_id)

        # Correct key should validate
        assert SecretCrypto.is_valid_master_key(master_key, encrypted, user_id)

        # Wrong key should not validate
        assert not SecretCrypto.is_valid_master_key(wrong_key, encrypted, user_id)

    def test_encrypt_empty_string(self):
        """Test encrypting empty string."""
        master_key = SecretCrypto.generate_master_key()
        user_id = "test_user"
        plaintext = ""

        encrypted = SecretCrypto.encrypt(plaintext, master_key, user_id)
        decrypted = SecretCrypto.decrypt(encrypted, master_key, user_id)

        assert decrypted == plaintext

    def test_encrypt_unicode(self):
        """Test encrypting Unicode strings."""
        master_key = SecretCrypto.generate_master_key()
        user_id = "test_user"
        plaintext = "Hello 世界 🔐 мир"

        encrypted = SecretCrypto.encrypt(plaintext, master_key, user_id)
        decrypted = SecretCrypto.decrypt(encrypted, master_key, user_id)

        assert decrypted == plaintext

    def test_encrypt_large_value(self):
        """Test encrypting large values."""
        master_key = SecretCrypto.generate_master_key()
        user_id = "test_user"
        plaintext = "x" * 10000

        encrypted = SecretCrypto.encrypt(plaintext, master_key, user_id)
        decrypted = SecretCrypto.decrypt(encrypted, master_key, user_id)

        assert decrypted == plaintext
