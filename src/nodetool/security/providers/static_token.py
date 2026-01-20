from __future__ import annotations

from dataclasses import dataclass

from nodetool.security.auth_provider import AuthProvider, AuthResult, TokenType


@dataclass(slots=True)
class StaticTokenAuthProvider(AuthProvider):
    """Simple provider that validates against a pre-shared static token."""

    static_token: str
    user_id: str = "1"

    async def verify_token(self, token: str) -> AuthResult:
        if token and token == self.static_token:
            return AuthResult(ok=True, user_id=self.user_id, token_type=TokenType.STATIC)
        return AuthResult(ok=False, error="Invalid static token")

    def clear_caches(self) -> None:
        # No caches to clear for the static provider, but keep method for interface compatibility.
        return None
