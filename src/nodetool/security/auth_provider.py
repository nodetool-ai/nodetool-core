from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


class TokenType(Enum):
    STATIC = auto()
    USER = auto()


@dataclass(slots=True)
class AuthResult:
    ok: bool
    user_id: str | None = None
    token_type: TokenType | None = None
    error: str | None = None


class AuthProvider(ABC):
    """Common interface for all authentication providers."""

    @staticmethod
    def prefer_header() -> str:
        """Return the preferred HTTP header name for bearer tokens."""
        return "authorization"

    def extract_token_from_headers(self, headers: Mapping[str, str]) -> str | None:
        """Extract a bearer token from HTTP headers."""
        header_name = self.prefer_header()
        auth_header = headers.get(header_name) or headers.get(header_name.title())
        if not auth_header:
            return None
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None
        token = parts[1].strip()
        return token or None

    def extract_token_from_ws(self, headers: Mapping[str, str], query_params: Mapping[str, str]) -> str | None:
        """
        Extract a bearer token for WebSocket connections.

        Compatibility fallback: allow ?api_key=<token> query parameter.
        """
        token = self.extract_token_from_headers(headers)
        if token:
            return token
        api_key = query_params.get("api_key")
        if api_key:
            api_key = api_key.strip()
            return api_key or None
        return None

    @abstractmethod
    async def verify_token(self, token: str) -> AuthResult:
        """Validate a token and return the associated authentication result."""

    def clear_caches(self) -> None:
        """Clear any internal caches held by the provider."""
        return None
