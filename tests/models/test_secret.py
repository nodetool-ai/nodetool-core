"""
Tests for Secret database model.
"""

import pytest

from nodetool.models.secret import Secret
from nodetool.security.crypto import SecretCrypto
from nodetool.security.master_key import MasterKeyManager


@pytest.mark.asyncio
class TestSecretModel:
    """Tests for Secret model."""

    async def test_create_table(self):
        """Test creating the secrets table."""
        await Secret.create_table()

    async def test_create_secret(self):
        """Test creating a new secret."""
        user_id = "test_user_123"
        key = "OPENAI_API_KEY"
        value = "sk-test-12345"
        description = "API key for OpenAI"

        secret = await Secret.create(user_id=user_id, key=key, value=value, description=description)

        assert secret.id is not None
        assert secret.user_id == user_id
        assert secret.key == key
        assert secret.description == description
        assert secret.encrypted_value != value  # Should be encrypted
        assert secret.created_at is not None
        assert secret.updated_at is not None

    async def test_find_secret(self):
        """Test finding a secret by user_id and key."""
        user_id = "test_user_456"
        key = "TEST_SECRET"
        value = "test_value_123"

        # Create
        await Secret.create(user_id=user_id, key=key, value=value)

        # Find
        found = await Secret.find(user_id, key)

        assert found is not None
        assert found.user_id == user_id
        assert found.key == key

    async def test_find_nonexistent_secret(self):
        """Test finding a secret that doesn't exist."""
        found = await Secret.find("nonexistent_user", "NONEXISTENT_KEY")

        assert found is None

    async def test_get_decrypted_value(self):
        """Test decrypting a secret value."""
        user_id = "test_user_789"
        key = "MY_SECRET"
        value = "my_secret_value_12345"

        secret = await Secret.create(user_id=user_id, key=key, value=value)

        decrypted = await secret.get_decrypted_value()

        assert decrypted == value

    async def test_update_value(self):
        """Test updating a secret value."""
        user_id = "test_user_update"
        key = "UPDATE_TEST"
        old_value = "old_value_123"
        new_value = "new_value_456"

        secret = await Secret.create(user_id=user_id, key=key, value=old_value)

        # Update
        await secret.update_value(new_value)

        # Verify
        decrypted = await secret.get_decrypted_value()
        assert decrypted == new_value

        # Reload from DB and verify
        reloaded = await Secret.find(user_id, key)
        assert reloaded is not None
        assert await reloaded.get_decrypted_value() == new_value

    async def test_list_for_user(self):
        """Test listing all secrets for a user."""
        user_id = "test_user_list"

        # Create multiple secrets
        await Secret.create(user_id=user_id, key="SECRET_1", value="value_1")
        await Secret.create(user_id=user_id, key="SECRET_2", value="value_2")
        await Secret.create(user_id=user_id, key="SECRET_3", value="value_3")

        # Create a secret for a different user (should not be included)
        await Secret.create(user_id="other_user", key="OTHER_SECRET", value="other_value")

        # List
        secrets, _ = await Secret.list_for_user(user_id)

        assert len(secrets) >= 3
        user_ids = [s.user_id for s in secrets]
        assert all(uid == user_id for uid in user_ids)

    async def test_upsert_create(self):
        """Test upsert when secret doesn't exist (creates new)."""
        user_id = "test_user_upsert"
        key = "UPSERT_TEST"
        value = "initial_value"

        secret = await Secret.upsert(user_id=user_id, key=key, value=value)

        assert secret.user_id == user_id
        assert secret.key == key
        assert await secret.get_decrypted_value() == value

    async def test_upsert_update(self):
        """Test upsert when secret exists (updates existing)."""
        user_id = "test_user_upsert2"
        key = "UPSERT_TEST2"
        old_value = "old_value"
        new_value = "new_value"

        # Create
        await Secret.create(user_id=user_id, key=key, value=old_value)

        # Upsert with new value
        secret = await Secret.upsert(user_id=user_id, key=key, value=new_value)

        assert await secret.get_decrypted_value() == new_value

        # Verify only one secret exists
        all_secrets, _ = await Secret.list_for_user(user_id)
        matching_secrets = [s for s in all_secrets if s.key == key]
        assert len(matching_secrets) == 1

    async def test_delete_secret(self):
        """Test deleting a secret."""
        user_id = "test_user_delete"
        key = "DELETE_TEST"
        value = "value_to_delete"

        # Create
        await Secret.create(user_id=user_id, key=key, value=value)

        # Delete
        success = await Secret.delete_secret(user_id, key)

        assert success is True

        # Verify deleted
        found = await Secret.find(user_id, key)
        assert found is None

    async def test_delete_nonexistent_secret(self):
        """Test deleting a secret that doesn't exist."""
        success = await Secret.delete_secret("nonexistent_user", "NONEXISTENT_KEY")

        assert success is False

    async def test_to_dict_safe(self):
        """Test safe dictionary representation."""
        user_id = "test_user_safe"
        key = "SAFE_TEST"
        value = "secret_value_should_not_appear"

        secret = await Secret.create(user_id=user_id, key=key, value=value)

        safe_dict = secret.to_dict_safe()

        assert "id" in safe_dict
        assert "user_id" in safe_dict
        assert "key" in safe_dict
        assert "description" in safe_dict
        assert "created_at" in safe_dict
        assert "updated_at" in safe_dict

        # Should NOT contain encrypted_value or decrypted value
        assert "encrypted_value" not in safe_dict
        assert "value" not in safe_dict
        assert value not in str(safe_dict)

    async def test_user_isolation(self):
        """Test that users can't access each other's secrets."""
        user1 = "user_isolation_1"
        user2 = "user_isolation_2"
        key = "SHARED_KEY_NAME"
        value1 = "user1_secret"
        value2 = "user2_secret"

        # Create secrets with same key for different users
        await Secret.create(user_id=user1, key=key, value=value1)
        await Secret.create(user_id=user2, key=key, value=value2)

        # Each user should find their own secret
        found1 = await Secret.find(user1, key)
        found2 = await Secret.find(user2, key)

        assert found1 is not None
        assert found2 is not None
        assert found1.id != found2.id
        assert await found1.get_decrypted_value() == value1
        assert await found2.get_decrypted_value() == value2

    async def test_encrypted_values_are_different_for_same_plaintext(self):
        """Test that encryption produces different ciphertexts (includes IV/nonce)."""
        user_id = "test_user_different"
        value = "same_plaintext"

        # Create two secrets with the same plaintext value
        secret1 = await Secret.create(user_id=user_id, key="KEY1", value=value)
        secret2 = await Secret.create(user_id=user_id, key="KEY2", value=value)

        # Encrypted values should be different (Fernet includes random IV)
        assert secret1.encrypted_value != secret2.encrypted_value

        # But both should decrypt to the same value
        assert await secret1.get_decrypted_value() == value
        assert await secret2.get_decrypted_value() == value

    async def test_update_description(self):
        """Test updating secret description."""
        user_id = "test_user_desc"
        key = "DESC_TEST"
        value = "test_value"
        old_desc = "Old description"
        new_desc = "New description"

        # Create with description
        secret = await Secret.create(user_id=user_id, key=key, value=value, description=old_desc)

        assert secret.description == old_desc

        # Update description
        secret.description = new_desc
        await secret.save()

        # Reload and verify
        reloaded = await Secret.find(user_id, key)
        assert reloaded is not None
        assert reloaded.description == new_desc

    async def test_pagination(self):
        """Test pagination of secrets list."""
        user_id = "test_user_pagination"

        # Create multiple secrets
        for i in range(5):
            await Secret.create(user_id=user_id, key=f"PAGINATED_SECRET_{i}", value=f"value_{i}")

        # Get first page
        secrets_page1, next_key = await Secret.list_for_user(user_id, limit=2)

        assert len(secrets_page1) == 2
        assert next_key is not None

        # Get second page
        secrets_page2, _next_key2 = await Secret.list_for_user(user_id, limit=2, start_key=next_key)

        assert len(secrets_page2) == 2

        # Verify no overlap
        page1_keys = [s.key for s in secrets_page1]
        page2_keys = [s.key for s in secrets_page2]
        assert len(set(page1_keys) & set(page2_keys)) == 0
