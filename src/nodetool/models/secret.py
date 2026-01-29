"""
Defines the Secret database model for encrypted secret storage.

Secrets are encrypted using a master key and user_id as salt, providing
per-user encryption isolation.
"""

from datetime import UTC, datetime
from typing import Optional

from nodetool.models.base_model import DBField, DBIndex, DBModel, create_time_ordered_uuid
from nodetool.models.condition_builder import Field


@DBIndex(["user_id", "key"], unique=True)
@DBIndex(["user_id"])
class Secret(DBModel):
    """Database model for encrypted secrets."""

    @classmethod
    def get_table_schema(cls):
        """Returns the database table schema for secrets."""
        return {"table_name": "nodetool_secrets"}

    id: str = DBField(default_factory=create_time_ordered_uuid)
    user_id: str = DBField()
    key: str = DBField()
    encrypted_value: str = DBField()
    description: Optional[str] = DBField(default=None)
    created_at: datetime = DBField(default_factory=datetime.now)
    updated_at: datetime = DBField(default_factory=datetime.now)

    def before_save(self):
        """Updates the `updated_at` timestamp before saving."""
        self.updated_at = datetime.now()

    @classmethod
    async def create(cls, user_id: str, key: str, value: str, description: Optional[str] = None, **kwargs):  # type: ignore[override]
        """
        Create a new encrypted secret.

        Args:
            user_id: The ID of the user who owns this secret.
            key: The key/name for this secret (e.g., "OPENAI_API_KEY").
            value: The plaintext value to encrypt and store.
            description: Optional description of the secret.
            **kwargs: Additional fields to set on the model.

        Returns:
            The newly created Secret instance.
        """
        from nodetool.security.crypto import SecretCrypto
        from nodetool.security.master_key import MasterKeyManager

        # Get master key and encrypt the value
        master_key = await MasterKeyManager.get_master_key()
        encrypted_value = SecretCrypto.encrypt(value, master_key, user_id)

        return await super().create(
            id=create_time_ordered_uuid(),
            user_id=user_id,
            key=key,
            encrypted_value=encrypted_value,
            description=description,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            **kwargs,
        )

    @classmethod
    async def find(cls, user_id: str, key: str) -> Optional["Secret"]:
        """
        Find a secret by user_id and key.

        Args:
            user_id: The user ID.
            key: The secret key.

        Returns:
            The Secret instance if found, None otherwise.
        """
        condition = Field("user_id").equals(user_id).and_(Field("key").equals(key))
        results, _ = await cls.query(condition, limit=1)
        return results[0] if results else None

    @classmethod
    async def list_for_user(
        cls, user_id: str, limit: int = 100, start_key: Optional[str] = None
    ) -> tuple[list["Secret"], str]:
        """
        List all secrets for a user.

        Args:
            user_id: The user ID.
            limit: Maximum number of secrets to return.
            start_key: Pagination key for continuation.

        Returns:
            A tuple of (list of Secret instances, next pagination key).
        """
        condition = Field("user_id").equals(user_id)

        if start_key:
            condition = condition.and_(Field("id").greater_than(start_key))

        return await cls.query(condition, limit=limit)

    @classmethod
    async def list_all(cls, limit: int = 1000) -> list["Secret"]:
        """Return all secrets (best effort, limited by the provided limit)."""
        secrets, _ = await cls.query(limit=limit)
        return secrets

    async def get_decrypted_value(self) -> str:
        """
        Decrypt and return the secret value.

        Returns:
            The decrypted plaintext value.

        Raises:
            Exception: If decryption fails (e.g., wrong master key).
        """
        from nodetool.security.crypto import SecretCrypto
        from nodetool.security.master_key import MasterKeyManager

        master_key = await MasterKeyManager.get_master_key()
        return SecretCrypto.decrypt(self.encrypted_value, master_key, self.user_id)

    async def update_value(self, new_value: str) -> None:
        """
        Update the secret value.

        Args:
            new_value: The new plaintext value to encrypt and store.
        """
        from nodetool.security.crypto import SecretCrypto
        from nodetool.security.master_key import MasterKeyManager

        master_key = await MasterKeyManager.get_master_key()
        self.encrypted_value = SecretCrypto.encrypt(new_value, master_key, self.user_id)

        # Invalidate cache
        from nodetool.security.secret_helper import clear_secret_cache

        clear_secret_cache(self.user_id, self.key)

        await self.save()

    @classmethod
    async def upsert_encrypted(
        cls,
        user_id: str,
        key: str,
        encrypted_value: str,
        description: Optional[str] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
    ) -> "Secret":
        """
        Create or update a secret using a pre-encrypted value.

        This is used when migrating secrets between instances that share the same
        master key. The encrypted blob is persisted as-is without decrypting.
        """
        existing = await cls.find(user_id, key)
        if created_at is None:
            created_at = datetime.now(UTC)
        if updated_at is None:
            updated_at = datetime.now(UTC)

        if existing:
            existing.encrypted_value = encrypted_value
            existing.updated_at = updated_at
            if description is not None:
                existing.description = description

            # Invalidate cache
            from nodetool.security.secret_helper import clear_secret_cache

            clear_secret_cache(user_id, key)

            await existing.save()
            return existing

        return await super().create(
            id=create_time_ordered_uuid(),
            user_id=user_id,
            key=key,
            encrypted_value=encrypted_value,
            description=description,
            created_at=created_at,
            updated_at=updated_at,
        )

    @classmethod
    async def upsert(cls, user_id: str, key: str, value: str, description: Optional[str] = None) -> "Secret":
        """
        Create or update a secret.

        If a secret with the same user_id and key exists, update it.
        Otherwise, create a new one.

        Args:
            user_id: The user ID.
            key: The secret key.
            value: The plaintext value to encrypt and store.
            description: Optional description.

        Returns:
            The Secret instance (created or updated).
        """

        existing = await cls.find(user_id, key)

        if existing:
            await existing.update_value(value)
            if description is not None:
                existing.description = description
                await existing.save()

            # Invalidate cache
            from nodetool.security.secret_helper import clear_secret_cache

            clear_secret_cache(user_id, key)

            return existing
        else:
            return await cls.create(user_id=user_id, key=key, value=value, description=description)

    @classmethod
    async def delete_secret(cls, user_id: str, key: str) -> bool:
        """
        Delete a secret.

        Args:
            user_id: The user ID.
            key: The secret key.

        Returns:
            True if the secret was deleted, False if not found.
        """
        secret = await cls.find(user_id, key)
        if secret:
            await secret.delete()

            # Invalidate cache
            from nodetool.security.secret_helper import clear_secret_cache

            clear_secret_cache(user_id, key)

            return True
        return False

    def to_dict_safe(self) -> dict:
        """
        Return a dictionary representation without the encrypted value.

        This is safe to return in API responses.

        Returns:
            A dictionary with id, user_id, key, description, created_at, updated_at.
        """
        return {
            "id": self.id,
            "user_id": self.user_id,
            "key": self.key,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
