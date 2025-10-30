from __future__ import annotations

from nodetool.security.auth_provider import AuthProvider, AuthResult, TokenType


class LocalAuthProvider(AuthProvider):
    """Development auth provider that always returns user '1'.

    This provider accepts any token (including missing) and maps requests to
    the fixed user ID "1". Intended for local development only.
    """

    async def verify_token(self, token: str | None = None) -> AuthResult:  # type: ignore[override]
        return AuthResult(ok=True, user_id="1", token_type=TokenType.USER)

