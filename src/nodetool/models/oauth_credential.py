"""
OAuth Credential Model for storing OAuth tokens.

This module provides a database model for storing OAuth credentials from
various providers (e.g., Hugging Face). Tokens are encrypted using the
master key and user_id as salt.
"""

from datetime import UTC, datetime
from typing import Optional

from nodetool.config.logging_config import get_logger
from nodetool.models.base_model import DBField, DBIndex, DBModel, create_time_ordered_uuid
from nodetool.models.condition_builder import Field

log = get_logger(__name__)


@DBIndex(["user_id", "provider", "account_id"], unique=True)
@DBIndex(["user_id", "provider"])
@DBIndex(["user_id"])
class OAuthCredential(DBModel):
    """Database model for encrypted OAuth credentials."""

    @classmethod
    def get_table_schema(cls):
        """Returns the database table schema for OAuth credentials."""
        return {"table_name": "nodetool_oauth_credentials"}

    id: str = DBField(default_factory=create_time_ordered_uuid)
    user_id: str = DBField()
    provider: str = DBField()  # e.g., "huggingface"
    account_id: str = DBField()  # Provider's unique account identifier
    username: Optional[str] = DBField(default=None)
    encrypted_access_token: str = DBField()
    encrypted_refresh_token: Optional[str] = DBField(default=None)
    token_type: str = DBField(default="Bearer")
    scope: Optional[str] = DBField(default=None)
    received_at: datetime = DBField(default_factory=lambda: datetime.now(UTC))
    expires_at: Optional[datetime] = DBField(default=None)
    created_at: datetime = DBField(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = DBField(default_factory=lambda: datetime.now(UTC))

    def before_save(self):
        """Updates the `updated_at` timestamp before saving."""
        self.updated_at = datetime.now(UTC)

    @classmethod
    async def create_encrypted(
        cls,
        user_id: str,
        provider: str,
        account_id: str,
        access_token: str,
        username: Optional[str] = None,
        refresh_token: Optional[str] = None,
        token_type: str = "Bearer",
        scope: Optional[str] = None,
        received_at: Optional[datetime] = None,
        expires_at: Optional[datetime] = None,
        **kwargs,
    ):
        """
        Create a new encrypted OAuth credential.

        Args:
            user_id: The ID of the user who owns this credential.
            provider: The OAuth provider name (e.g., "huggingface").
            account_id: Provider's unique account identifier.
            access_token: The plaintext access token to encrypt and store.
            username: Optional username from the provider.
            refresh_token: Optional plaintext refresh token to encrypt and store.
            token_type: Token type (default: "Bearer").
            scope: OAuth scope string.
            received_at: When the token was received.
            expires_at: When the token expires.
            **kwargs: Additional fields to set on the model.

        Returns:
            The newly created OAuthCredential instance.
        """
        from nodetool.security.crypto import SecretCrypto
        from nodetool.security.master_key import MasterKeyManager

        # Get master key and encrypt the tokens
        master_key = await MasterKeyManager.get_master_key()
        encrypted_access_token = SecretCrypto.encrypt(access_token, master_key, user_id)
        encrypted_refresh_token = None
        if refresh_token:
            encrypted_refresh_token = SecretCrypto.encrypt(refresh_token, master_key, user_id)

        if received_at is None:
            received_at = datetime.now(UTC)

        return await super().create(
            id=create_time_ordered_uuid(),
            user_id=user_id,
            provider=provider,
            account_id=account_id,
            username=username,
            encrypted_access_token=encrypted_access_token,
            encrypted_refresh_token=encrypted_refresh_token,
            token_type=token_type,
            scope=scope,
            received_at=received_at,
            expires_at=expires_at,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            **kwargs,
        )

    @classmethod
    async def find_by_account(
        cls, user_id: str, provider: str, account_id: str
    ) -> Optional["OAuthCredential"]:
        """
        Find an OAuth credential by user_id, provider, and account_id.

        Args:
            user_id: The user ID.
            provider: The OAuth provider name.
            account_id: Provider's unique account identifier.

        Returns:
            The OAuthCredential instance if found, None otherwise.
        """
        condition = (
            Field("user_id")
            .equals(user_id)
            .and_(Field("provider").equals(provider))
            .and_(Field("account_id").equals(account_id))
        )
        results, _ = await cls.query(condition, limit=1)
        return results[0] if results else None

    @classmethod
    async def list_for_user_and_provider(
        cls, user_id: str, provider: str, limit: int = 100
    ) -> list["OAuthCredential"]:
        """
        List all OAuth credentials for a user and provider.

        Args:
            user_id: The user ID.
            provider: The OAuth provider name.
            limit: Maximum number of credentials to return.

        Returns:
            A list of OAuthCredential instances.
        """
        condition = Field("user_id").equals(user_id).and_(Field("provider").equals(provider))
        results, _ = await cls.query(condition, limit=limit)
        return results

    async def get_decrypted_access_token(self) -> str:
        """
        Decrypt and return the access token.

        Returns:
            The decrypted plaintext access token.

        Raises:
            Exception: If decryption fails (e.g., wrong master key).
        """
        from nodetool.security.crypto import SecretCrypto
        from nodetool.security.master_key import MasterKeyManager

        master_key = await MasterKeyManager.get_master_key()
        return SecretCrypto.decrypt(self.encrypted_access_token, master_key, self.user_id)

    async def get_decrypted_refresh_token(self) -> Optional[str]:
        """
        Decrypt and return the refresh token.

        Returns:
            The decrypted plaintext refresh token, or None if not set.

        Raises:
            Exception: If decryption fails (e.g., wrong master key).
        """
        if not self.encrypted_refresh_token:
            return None

        from nodetool.security.crypto import SecretCrypto
        from nodetool.security.master_key import MasterKeyManager

        master_key = await MasterKeyManager.get_master_key()
        return SecretCrypto.decrypt(self.encrypted_refresh_token, master_key, self.user_id)

    async def update_tokens(
        self,
        access_token: str,
        refresh_token: Optional[str] = None,
        token_type: Optional[str] = None,
        scope: Optional[str] = None,
        received_at: Optional[datetime] = None,
        expires_at: Optional[datetime] = None,
    ) -> None:
        """
        Update the OAuth tokens.

        Args:
            access_token: The new plaintext access token to encrypt and store.
            refresh_token: Optional new plaintext refresh token to encrypt and store.
            token_type: Optional new token type.
            scope: Optional new scope.
            received_at: Optional new received_at timestamp.
            expires_at: Optional new expires_at timestamp.
        """
        from nodetool.security.crypto import SecretCrypto
        from nodetool.security.master_key import MasterKeyManager

        master_key = await MasterKeyManager.get_master_key()
        self.encrypted_access_token = SecretCrypto.encrypt(access_token, master_key, self.user_id)

        if refresh_token is not None:
            self.encrypted_refresh_token = SecretCrypto.encrypt(
                refresh_token, master_key, self.user_id
            )

        if token_type is not None:
            self.token_type = token_type

        if scope is not None:
            self.scope = scope

        if received_at is not None:
            self.received_at = received_at
        else:
            self.received_at = datetime.now(UTC)

        if expires_at is not None:
            self.expires_at = expires_at

        await self.save()

    @classmethod
    async def upsert(
        cls,
        user_id: str,
        provider: str,
        account_id: str,
        access_token: str,
        username: Optional[str] = None,
        refresh_token: Optional[str] = None,
        token_type: str = "Bearer",
        scope: Optional[str] = None,
        received_at: Optional[datetime] = None,
        expires_at: Optional[datetime] = None,
    ) -> "OAuthCredential":
        """
        Create or update an OAuth credential.

        If a credential with the same user_id, provider, and account_id exists, update it.
        Otherwise, create a new one.

        Args:
            user_id: The user ID.
            provider: The OAuth provider name.
            account_id: Provider's unique account identifier.
            access_token: The plaintext access token to encrypt and store.
            username: Optional username from the provider.
            refresh_token: Optional plaintext refresh token to encrypt and store.
            token_type: Token type (default: "Bearer").
            scope: OAuth scope string.
            received_at: When the token was received.
            expires_at: When the token expires.

        Returns:
            The OAuthCredential instance (created or updated).
        """
        existing = await cls.find_by_account(user_id, provider, account_id)

        if existing:
            await existing.update_tokens(
                access_token=access_token,
                refresh_token=refresh_token,
                token_type=token_type,
                scope=scope,
                received_at=received_at,
                expires_at=expires_at,
            )
            if username is not None:
                existing.username = username
                await existing.save()
            return existing
        else:
            return await cls.create_encrypted(
                user_id=user_id,
                provider=provider,
                account_id=account_id,
                access_token=access_token,
                username=username,
                refresh_token=refresh_token,
                token_type=token_type,
                scope=scope,
                received_at=received_at,
                expires_at=expires_at,
            )

    def to_dict_safe(self) -> dict:
        """
        Return a dictionary representation without the encrypted tokens.

        This is safe to return in API responses.

        Returns:
            A dictionary with metadata but not the encrypted tokens.
        """
        return {
            "id": self.id,
            "user_id": self.user_id,
            "provider": self.provider,
            "account_id": self.account_id,
            "username": self.username,
            "token_type": self.token_type,
            "scope": self.scope,
            "received_at": self.received_at.isoformat() if self.received_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
