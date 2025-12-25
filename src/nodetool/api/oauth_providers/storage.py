"""Secure token storage abstraction for OAuth tokens.

This module provides a secure way to store OAuth tokens with:
- Encryption at rest (planned - currently in-memory)
- Support for multiple accounts per provider
- Never returning raw tokens after storage
- Provider-agnostic storage format
"""

import time
from dataclasses import asdict, dataclass
from typing import Optional

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class StoredToken:
    """Normalized stored token format.

    All providers store tokens in this unified format.

    Attributes:
        provider: Provider name (e.g., "google", "github")
        account_id: Optional account/user identifier
        access_token: OAuth access token (encrypted at rest)
        refresh_token: Optional refresh token (encrypted at rest)
        scope: OAuth scopes granted
        token_type: Token type (usually "Bearer")
        received_at: Unix timestamp when token was received
        expires_at: Optional Unix timestamp when token expires
        metadata: Optional provider-specific metadata
    """

    provider: str
    account_id: Optional[str]
    access_token: str
    refresh_token: Optional[str]
    scope: str
    token_type: str
    received_at: int
    expires_at: Optional[int] = None
    metadata: Optional[dict] = None

    def is_expired(self) -> bool:
        """Check if token is expired."""
        if not self.expires_at:
            return False
        return time.time() >= self.expires_at

    def needs_refresh(self, buffer_seconds: int = 300) -> bool:
        """Check if token needs refresh soon.

        Args:
            buffer_seconds: Number of seconds before expiry to consider "needs refresh"

        Returns:
            True if token expires within buffer_seconds
        """
        if not self.expires_at:
            return False
        return time.time() >= (self.expires_at - buffer_seconds)

    def to_safe_dict(self) -> dict:
        """Convert to dictionary without exposing tokens.

        Returns a sanitized version suitable for API responses.
        """
        return {
            "provider": self.provider,
            "account_id": self.account_id,
            "scope": self.scope,
            "token_type": self.token_type,
            "received_at": self.received_at,
            "expires_at": self.expires_at,
            "is_expired": self.is_expired(),
            "needs_refresh": self.needs_refresh(),
        }


class TokenStore:
    """In-memory token storage.

    TODO: Replace with persistent encrypted storage (database or keychain).

    Storage key: (provider, account_id) -> StoredToken
    """

    def __init__(self):
        self._store: dict[tuple[str, Optional[str]], StoredToken] = {}

    def store(
        self,
        provider: str,
        token_data: dict,
        account_id: Optional[str] = None,
    ) -> StoredToken:
        """Store OAuth token.

        Args:
            provider: Provider name
            token_data: Token data from provider
            account_id: Optional account identifier

        Returns:
            Stored token object
        """
        stored_token = StoredToken(
            provider=provider,
            account_id=account_id,
            access_token=token_data["access_token"],
            refresh_token=token_data.get("refresh_token"),
            scope=token_data.get("scope", ""),
            token_type=token_data.get("token_type", "Bearer"),
            received_at=token_data.get("received_at", int(time.time())),
            expires_at=token_data.get("expires_at"),
            metadata=token_data.get("metadata"),
        )

        key = (provider, account_id)
        self._store[key] = stored_token

        log.info(f"Stored token for provider={provider}, account_id={account_id}")
        return stored_token

    def get(
        self,
        provider: str,
        account_id: Optional[str] = None,
    ) -> Optional[StoredToken]:
        """Retrieve stored token.

        Args:
            provider: Provider name
            account_id: Optional account identifier

        Returns:
            StoredToken if found, None otherwise
        """
        key = (provider, account_id)
        return self._store.get(key)

    def list_by_provider(self, provider: str) -> list[StoredToken]:
        """List all stored tokens for a provider.

        Args:
            provider: Provider name

        Returns:
            List of stored tokens
        """
        return [
            token
            for (p, _), token in self._store.items()
            if p == provider
        ]

    def delete(
        self,
        provider: str,
        account_id: Optional[str] = None,
    ) -> bool:
        """Delete stored token.

        Args:
            provider: Provider name
            account_id: Optional account identifier

        Returns:
            True if token was deleted, False if not found
        """
        key = (provider, account_id)
        if key in self._store:
            del self._store[key]
            log.info(f"Deleted token for provider={provider}, account_id={account_id}")
            return True
        return False

    def clear_provider(self, provider: str) -> int:
        """Clear all tokens for a provider.

        Args:
            provider: Provider name

        Returns:
            Number of tokens deleted
        """
        keys_to_delete = [key for key in self._store if key[0] == provider]
        for key in keys_to_delete:
            del self._store[key]

        log.info(f"Cleared {len(keys_to_delete)} tokens for provider={provider}")
        return len(keys_to_delete)

    def clear_all(self) -> int:
        """Clear all stored tokens.

        Returns:
            Number of tokens deleted
        """
        count = len(self._store)
        self._store.clear()
        log.info(f"Cleared all {count} tokens")
        return count


# Global token store instance
# TODO: Replace with database-backed storage
_token_store = TokenStore()


def get_token_store() -> TokenStore:
    """Get global token store instance."""
    return _token_store
