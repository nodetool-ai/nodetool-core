"""
OAuth token database model for storing OAuth2 credentials.

This model stores OAuth tokens with support for refresh tokens,
expiration tracking, and multi-account scenarios.
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
class OAuthToken(DBModel):
    """Database model for OAuth tokens."""

    @classmethod
    def get_table_schema(cls):
        """Returns the database table schema for OAuth tokens."""
        return {"table_name": "nodetool_oauth_tokens"}

    id: str = DBField(default_factory=create_time_ordered_uuid)
    user_id: str = DBField()
    provider: str = DBField()  # e.g., "github", "google", etc.
    account_id: str = DBField()  # Provider-specific account identifier
    access_token: str = DBField()
    refresh_token: Optional[str] = DBField(default=None)
    token_type: str = DBField(default="bearer")
    scope: str = DBField(default="")
    received_at: datetime = DBField(default_factory=datetime.now)
    expires_at: Optional[datetime] = DBField(default=None)
    created_at: datetime = DBField(default_factory=datetime.now)
    updated_at: datetime = DBField(default_factory=datetime.now)

    def before_save(self):
        """Updates the `updated_at` timestamp before saving."""
        self.updated_at = datetime.now()

    @classmethod
    async def create_token(
        cls,
        user_id: str,
        provider: str,
        account_id: str,
        access_token: str,
        refresh_token: Optional[str] = None,
        token_type: str = "bearer",
        scope: str = "",
        expires_in: Optional[int] = None,
        **kwargs,
    ):
        """
        Create a new OAuth token.

        Args:
            user_id: The ID of the user who owns this token.
            provider: The OAuth provider (e.g., "github").
            account_id: Provider-specific account identifier.
            access_token: The OAuth access token.
            refresh_token: Optional refresh token.
            token_type: Token type (default: "bearer").
            scope: Space-separated list of granted scopes.
            expires_in: Token expiration time in seconds from now.
            **kwargs: Additional fields to set on the model.

        Returns:
            The newly created OAuthToken instance.
        """
        received_at = datetime.now()
        expires_at = None
        if expires_in is not None:
            from datetime import timedelta

            expires_at = received_at + timedelta(seconds=expires_in)

        return await super().create(
            id=create_time_ordered_uuid(),
            user_id=user_id,
            provider=provider,
            account_id=account_id,
            access_token=access_token,
            refresh_token=refresh_token,
            token_type=token_type,
            scope=scope,
            received_at=received_at,
            expires_at=expires_at,
            created_at=received_at,
            updated_at=received_at,
            **kwargs,
        )

    @classmethod
    async def find_by_account(cls, user_id: str, provider: str, account_id: str) -> Optional["OAuthToken"]:
        """
        Find a token by user_id, provider, and account_id.

        Args:
            user_id: The user ID.
            provider: The OAuth provider.
            account_id: Provider-specific account identifier.

        Returns:
            The OAuthToken instance if found, None otherwise.
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
    ) -> tuple[list["OAuthToken"], str]:
        """
        List all tokens for a user and provider.

        Args:
            user_id: The user ID.
            provider: The OAuth provider.
            limit: Maximum number of tokens to return.

        Returns:
            Tuple of (list of tokens, next cursor).
        """
        condition = Field("user_id").equals(user_id).and_(Field("provider").equals(provider))
        results, cursor = await cls.query(condition, limit=limit)
        return results, cursor

    @classmethod
    async def update_token(
        cls,
        user_id: str,
        provider: str,
        account_id: str,
        access_token: str,
        refresh_token: Optional[str] = None,
        expires_in: Optional[int] = None,
        scope: Optional[str] = None,
    ) -> Optional["OAuthToken"]:
        """
        Update an existing token or create if it doesn't exist.

        Args:
            user_id: The user ID.
            provider: The OAuth provider.
            account_id: Provider-specific account identifier.
            access_token: New access token.
            refresh_token: New refresh token (if provided).
            expires_in: Token expiration time in seconds from now.
            scope: New scope (if provided).

        Returns:
            The updated OAuthToken instance.
        """
        token = await cls.find_by_account(user_id, provider, account_id)

        if not token:
            return await cls.create_token(
                user_id=user_id,
                provider=provider,
                account_id=account_id,
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=expires_in,
                scope=scope or "",
            )

        # Update existing token
        token.access_token = access_token
        if refresh_token is not None:
            token.refresh_token = refresh_token
        if scope is not None:
            token.scope = scope
        token.received_at = datetime.now()
        if expires_in is not None:
            from datetime import timedelta

            token.expires_at = datetime.now() + timedelta(seconds=expires_in)
        else:
            token.expires_at = None

        await token.save()
        return token

    @classmethod
    async def delete_token(cls, user_id: str, provider: str, account_id: str) -> bool:
        """
        Delete a token.

        Args:
            user_id: The user ID.
            provider: The OAuth provider.
            account_id: Provider-specific account identifier.

        Returns:
            True if deleted, False if not found.
        """
        token = await cls.find_by_account(user_id, provider, account_id)
        if token:
            await token.delete()
            return True
        return False

    def is_expired(self) -> bool:
        """
        Check if the token is expired.

        Returns:
            True if expired, False otherwise.
        """
        if self.expires_at is None:
            return False
        return datetime.now() >= self.expires_at

    def to_dict_safe(self) -> dict:
        """
        Convert to dictionary without exposing sensitive tokens.

        Returns:
            Dictionary with metadata only (no tokens).
        """
        return {
            "id": self.id,
            "user_id": self.user_id,
            "provider": self.provider,
            "account_id": self.account_id,
            "token_type": self.token_type,
            "scope": self.scope,
            "has_refresh_token": self.refresh_token is not None,
            "received_at": self.received_at.isoformat() if self.received_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_expired": self.is_expired(),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
